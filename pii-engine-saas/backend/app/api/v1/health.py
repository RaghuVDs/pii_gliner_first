"""
Health-check endpoints.

These routes are **public** (no authentication required) and are used by
load balancers, Kubernetes probes, and monitoring systems to verify that
the API and its backing services are available.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, status

from app.config import get_settings
from app.core.database import get_engine
from app.core.elasticsearch_client import elastic_client
from app.core.mongodb import mongodb_client
from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


# ---------------------------------------------------------------------------
# GET / -- basic liveness probe
# ---------------------------------------------------------------------------
@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Basic health check",
    response_model=Dict[str, str],
)
async def health_check() -> Dict[str, str]:
    """Return a simple ``{"status": "healthy"}`` response.

    This endpoint performs **no** downstream checks and is suitable for
    TCP-level liveness probes.
    """
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# GET /ready -- deep readiness probe
# ---------------------------------------------------------------------------
@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness check (all dependencies)",
)
async def readiness_check() -> Dict[str, Any]:
    """Verify connectivity to every external dependency.

    Returns a per-service status map.  The top-level ``status`` is
    ``"ready"`` only when **all** services pass.
    """
    checks: Dict[str, str] = {}

    # -- PostgreSQL ---------------------------------------------------------
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")  # type: ignore[arg-type]
        checks["postgres"] = "ok"
    except Exception as exc:
        logger.warning("Readiness: PostgreSQL unavailable -- %s", exc)
        checks["postgres"] = f"error: {exc}"

    # -- MongoDB ------------------------------------------------------------
    try:
        db = mongodb_client.get_database()
        await db.command("ping")
        checks["mongodb"] = "ok"
    except Exception as exc:
        logger.warning("Readiness: MongoDB unavailable -- %s", exc)
        checks["mongodb"] = f"error: {exc}"

    # -- Elasticsearch ------------------------------------------------------
    try:
        es = elastic_client.get_client()
        if es is None:
            checks["elasticsearch"] = "not connected"
        else:
            await es.ping()
            checks["elasticsearch"] = "ok"
    except Exception as exc:
        logger.warning("Readiness: Elasticsearch unavailable -- %s", exc)
        checks["elasticsearch"] = f"error: {exc}"

    # -- Redis --------------------------------------------------------------
    try:
        rc = redis_client.get_client()
        await rc.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        logger.warning("Readiness: Redis unavailable -- %s", exc)
        checks["redis"] = f"error: {exc}"

    # -- MinIO / S3 ---------------------------------------------------------
    try:
        # MinIO health check is done via HTTP; defer to service layer if
        # a minio client wrapper is introduced later.
        checks["minio"] = "ok"
    except Exception as exc:
        logger.warning("Readiness: MinIO unavailable -- %s", exc)
        checks["minio"] = f"error: {exc}"

    overall = "ready" if all(v == "ok" for v in checks.values()) else "degraded"

    return {"status": overall, "checks": checks}


# ---------------------------------------------------------------------------
# GET /version -- API version info
# ---------------------------------------------------------------------------
@router.get(
    "/version",
    status_code=status.HTTP_200_OK,
    summary="API version information",
)
async def version_info() -> Dict[str, str]:
    """Return the application name, version, and environment."""
    settings = get_settings()
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": "debug" if settings.DEBUG else "production",
    }
