"""
Redis-based sliding window rate limiter middleware.

Applies per-tenant and per-IP rate limits before the request reaches
route handlers. Uses a sorted-set sliding window algorithm in Redis
for accurate, distributed rate limiting.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)

# ── Default Limits ───────────────────────────────────────────────────────
DEFAULT_REQUESTS_PER_MINUTE = 600  # High for dev; lower in production
DEFAULT_WINDOW_SECONDS = 60

# Stricter limit for unauthenticated / anonymous traffic
ANON_REQUESTS_PER_MINUTE = 300  # High for dev; lower in production

# Paths exempt from rate limiting
EXEMPT_PATHS = {"/api/v1/health", "/docs", "/redoc", "/openapi.json"}


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter backed by Redis sorted sets.

    The middleware identifies the caller by tenant_id (if authenticated)
    or by client IP. Rate-limit headers are included in every response.
    """

    def __init__(
        self,
        app,
        *,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        anon_requests_per_minute: int = ANON_REQUESTS_PER_MINUTE,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.anon_requests_per_minute = anon_requests_per_minute
        self.window_seconds = window_seconds

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip exempt paths and preflight requests
        if request.method == "OPTIONS" or request.url.path.rstrip("/") in EXEMPT_PATHS:
            return await call_next(request)

        # Build the rate-limit key
        tenant_id: Optional[str] = getattr(request.state, "tenant_id", None)
        if tenant_id:
            key = f"ratelimit:tenant:{tenant_id}"
            limit = self.requests_per_minute
        else:
            client_ip = self._get_client_ip(request)
            key = f"ratelimit:ip:{client_ip}"
            limit = self.anon_requests_per_minute

        # Check the limit
        try:
            allowed = await redis_client.rate_limit_check(
                key=key,
                max_requests=limit,
                window_seconds=self.window_seconds,
            )
        except RuntimeError:
            # Redis not connected -- fail open to avoid blocking all traffic
            logger.warning("Redis unavailable; rate limiter failing open.")
            allowed = True

        if not allowed:
            logger.warning("Rate limit exceeded for key=%s", key)
            return JSONResponse(
                status_code=429,
                content={
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "detail": "Rate limit exceeded. Please retry later.",
                },
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(limit),
                },
            )

        response = await call_next(request)

        # Attach rate-limit headers for client visibility
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Window"] = f"{self.window_seconds}s"

        return response

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract the originating client IP, respecting reverse-proxy headers."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
