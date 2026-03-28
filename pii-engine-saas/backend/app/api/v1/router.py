"""
Main v1 API router.

Aggregates all sub-routers under the ``/api/v1`` prefix.
New endpoint modules should be imported and included here.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1")

# ── Sub-Router Imports ───────────────────────────────────────────────────
# Each sub-router is defined in its own module and included below.
# Uncomment and add as the corresponding modules are created.

# from app.api.v1 import (
#     auth,
#     users,
#     tenants,
#     detections,
#     rules,
#     training,
#     jobs,
#     reports,
#     analytics,
#     api_keys,
#     health,
# )

# router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
# router.include_router(users.router, prefix="/users", tags=["Users"])
# router.include_router(tenants.router, prefix="/tenants", tags=["Tenants"])
# router.include_router(detections.router, prefix="/detections", tags=["Detections"])
# router.include_router(rules.router, prefix="/rules", tags=["Rules"])
# router.include_router(training.router, prefix="/training", tags=["Training"])
# router.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
# router.include_router(reports.router, prefix="/reports", tags=["Reports"])
# router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
# router.include_router(api_keys.router, prefix="/api-keys", tags=["API Keys"])
# router.include_router(health.router, prefix="/health", tags=["Health"])


# ── Placeholder Health Endpoint ──────────────────────────────────────────


@router.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Basic liveness probe."""
    return {"status": "ok", "version": "1.0.0"}
