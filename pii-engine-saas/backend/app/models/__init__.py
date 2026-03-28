"""
SQLAlchemy ORM models for the PII detection SaaS platform.

Import all models here so that Alembic and application code can use a single
import path:

    from app.models import User, Tenant, APIKey, ...
"""

from .base import Base, IntBase, SoftDeleteMixin

from .tenant import Tenant

from .user import User, UserSession

from .api_key import APIAccessRequest, APIKey, APIUsageLog

from .pii_type import (
    PIICategory,
    PIIType,
    TenantPIIConfig,
    CustomPIIType,
)

from .regex_rule import RegexRule, TenantRegexRule

from .field_pattern import FieldPattern, TenantFieldPattern

from .context_rule import ContextRule, TenantContextRule

from .masking_rule import MaskingStrategy, MaskingRule, TenantMaskingRule

from .ml_model_version import MLModelVersion

from .notification import Notification

__all__ = [
    # Base
    "Base",
    "IntBase",
    "SoftDeleteMixin",
    # Tenant
    "Tenant",
    # User
    "User",
    "UserSession",
    # API
    "APIAccessRequest",
    "APIKey",
    "APIUsageLog",
    # PII
    "PIICategory",
    "PIIType",
    "TenantPIIConfig",
    "CustomPIIType",
    # Regex
    "RegexRule",
    "TenantRegexRule",
    # Field Pattern
    "FieldPattern",
    "TenantFieldPattern",
    # Context Rule
    "ContextRule",
    "TenantContextRule",
    # Masking
    "MaskingStrategy",
    "MaskingRule",
    "TenantMaskingRule",
    # ML
    "MLModelVersion",
    # Notification
    "Notification",
]
