"""
FastAPI application factory for the PII Engine SaaS API.

Creates and configures the application with:
- CORS middleware
- Custom authentication, rate-limiting, and tenant-context middleware
- Global exception handlers for the custom exception hierarchy
- Startup / shutdown lifecycle hooks for all infrastructure connections
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.middleware.auth import AuthMiddleware
from app.api.middleware.rate_limiter import RateLimiterMiddleware
from app.api.middleware.tenant_context import TenantContextMiddleware
from app.api.v1 import api_router as v1_router
from app.config import get_settings
from app.core.elasticsearch_client import elastic_client
from app.core.exceptions import PIIEngineException
from app.core.minio_client import minio_client
from app.core.mongodb import mongodb_client
from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)


# ── Lifecycle ────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown of all infrastructure connections."""
    settings = get_settings()
    _configure_logging(settings.LOG_LEVEL)
    logger.info("Starting PII Engine SaaS API v%s ...", settings.APP_VERSION)

    # ── Startup ──────────────────────────────────────────────────────
    # PostgreSQL
    from app.core.database import init_db

    await init_db()

    # MongoDB
    await mongodb_client.connect()

    # Elasticsearch
    await elastic_client.connect()

    # Redis
    await redis_client.connect()

    # MinIO (synchronous SDK)
    minio_client.connect()

    logger.info("All infrastructure connections established.")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down PII Engine SaaS API...")

    from app.core.database import close_db

    await close_db()
    await mongodb_client.close()
    await elastic_client.close()
    await redis_client.close()

    logger.info("All connections closed. Goodbye.")


# ── Factory ──────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Build and return the fully-configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Enterprise SaaS platform for PII detection and management.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware (executed bottom-to-top) ───────────────────────────
    # 1. CORS -- outermost
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Window", "Retry-After"],
    )

    # 2. Rate limiter
    app.add_middleware(RateLimiterMiddleware)

    # 3. Tenant context (needs auth to have run first)
    app.add_middleware(TenantContextMiddleware)

    # 4. Authentication -- innermost, runs first
    app.add_middleware(AuthMiddleware)

    # ── Routers ──────────────────────────────────────────────────────
    app.include_router(v1_router, prefix="/api/v1")

    # ── Exception Handlers ───────────────────────────────────────────
    _register_exception_handlers(app)

    return app


# ── Exception Handlers ───────────────────────────────────────────────────


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app."""

    @app.exception_handler(PIIEngineException)
    async def pii_engine_exception_handler(
        request: Request, exc: PIIEngineException
    ) -> JSONResponse:
        logger.warning(
            "PIIEngineException [%s] %s (path=%s)",
            exc.error_code,
            exc.detail,
            request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception(
            "Unhandled exception on %s %s", request.method, request.url.path
        )
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "INTERNAL_ERROR",
                "detail": f"{type(exc).__name__}: {exc}",
            },
        )


# ── Logging ──────────────────────────────────────────────────────────────


def _configure_logging(level: str) -> None:
    """Set up structured logging for the application."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Quieten noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("motor").setLevel(logging.WARNING)


# ── Module-level app instance (used by ``uvicorn app.main:app``) ─────────
app = create_app()
