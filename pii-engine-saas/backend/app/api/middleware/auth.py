"""
JWT and API Key authentication middleware.

Inspects every incoming request for either a ``Bearer`` JWT token
or an ``X-API-Key`` header and populates ``request.state`` with
the authenticated identity before the route handler executes.

Public paths (health checks, docs) bypass authentication.
"""

from __future__ import annotations

import logging
from typing import Set

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.exceptions import UnauthorizedError
from app.core.security import decode_token, hash_api_key

logger = logging.getLogger(__name__)

# Paths that do not require authentication
PUBLIC_PATHS: Set[str] = {
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/health",
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/auth/refresh",
    "/api/v1/auth/forgot-password",
    "/api/v1/auth/reset-password",
    "/api/v1/auth/verify-email",
    "/api/v1/notifications",
    "/api/v1/pii-types",
    "/api/v1/regex-rules",
    "/api/v1/field-patterns",
    "/api/v1/context-rules",
    "/api/v1/masking-rules",
    "/api/v1/analytics",
    "/api/v1/detection",
    "/api/v1/learning",
    "/api/v1/models",
    "/api/v1/api",
    "/api/v1/users",
    "/api/v1/tenant",
    "/api/v1/audit",
}


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that authenticates requests via JWT or API key."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Initialise state attributes
        request.state.current_user = None
        request.state.tenant_id = None
        request.state.api_key_scopes = None
        request.state.auth_method = None

        # Skip authentication for public paths and OPTIONS (CORS preflight)
        if request.method == "OPTIONS" or self._is_public(request.url.path):
            return await call_next(request)

        # Try JWT first, then API key
        auth_header = request.headers.get("Authorization")
        api_key_header = request.headers.get("X-API-Key")

        if auth_header and auth_header.startswith("Bearer "):
            await self._authenticate_jwt(request, auth_header)
        elif api_key_header:
            await self._authenticate_api_key(request, api_key_header)

        # Always let the request through - individual route dependencies
        # will enforce auth where needed
        return await call_next(request)

    # ── Internal Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _is_public(path: str) -> bool:
        """Return ``True`` if the path is in the public allow-list."""
        for public in PUBLIC_PATHS:
            if path.rstrip("/") == public.rstrip("/") or path.startswith(public + "/"):
                return True
        return False

    @staticmethod
    async def _authenticate_jwt(request: Request, auth_header: str) -> None:
        """Decode the JWT and set user info on request state."""
        token = auth_header.removeprefix("Bearer ").strip()
        try:
            payload = decode_token(token)
            request.state.auth_method = "jwt"
            request.state.user_id = payload.get("sub")
            request.state.tenant_id = payload.get("tenant_id")
            request.state.user_role = payload.get("role")
        except UnauthorizedError:
            # Let downstream dependencies handle the error for protected routes
            logger.debug("JWT authentication failed for %s", request.url.path)

    @staticmethod
    async def _authenticate_api_key(request: Request, api_key: str) -> None:
        """Hash the API key and set it on request state for downstream lookup."""
        request.state.auth_method = "api_key"
        request.state.api_key_hash = hash_api_key(api_key)
        # Actual DB lookup is deferred to the get_api_key_user dependency
        # to avoid importing ORM models at middleware level.
