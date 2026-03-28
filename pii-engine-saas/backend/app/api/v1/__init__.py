"""
API v1 router -- aggregates all versioned endpoint routers.

Import this module's ``api_router`` and include it in the FastAPI app::

    from app.api.v1 import api_router
    app.include_router(api_router, prefix="/api/v1")
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.auth import router as auth_router
from app.api.v1.users import router as users_router
from app.api.v1.tenant import router as tenant_router
from app.api.v1.pii_types import router as pii_types_router
from app.api.v1.regex_rules import router as regex_rules_router
from app.api.v1.field_patterns import router as field_patterns_router
from app.api.v1.context_rules import router as context_rules_router
from app.api.v1.masking_rules import router as masking_rules_router
from app.api.v1.detection import router as detection_router
from app.api.v1.learning import router as learning_router
from app.api.v1.ml_models import router as ml_models_router
from app.api.v1.analytics import router as analytics_router
from app.api.v1.api_management import router as api_management_router
from app.api.v1.audit import router as audit_router
from app.api.v1.notifications import router as notifications_router

api_router = APIRouter()

api_router.include_router(health_router, prefix="/health")
api_router.include_router(auth_router, prefix="/auth")
api_router.include_router(users_router, prefix="/users")
api_router.include_router(tenant_router, prefix="/tenant")
api_router.include_router(pii_types_router, prefix="/pii-types")
api_router.include_router(regex_rules_router, prefix="/regex-rules")
api_router.include_router(field_patterns_router, prefix="/field-patterns")
api_router.include_router(context_rules_router, prefix="/context-rules")
api_router.include_router(masking_rules_router, prefix="/masking-rules")
api_router.include_router(detection_router, prefix="/detection")
api_router.include_router(learning_router, prefix="/learning")
api_router.include_router(ml_models_router, prefix="/models")
api_router.include_router(analytics_router, prefix="/analytics")
api_router.include_router(api_management_router, prefix="/api")
api_router.include_router(audit_router, prefix="/audit")
api_router.include_router(notifications_router, prefix="/notifications")
