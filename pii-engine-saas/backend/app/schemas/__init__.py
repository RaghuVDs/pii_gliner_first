"""Pydantic v2 request/response schemas for the PII detection SaaS platform.

Re-exports the most commonly used schemas so consumers can do:

    from app.schemas import DetectRequest, DetectResponse, UserResponse, ...
"""

# -- common ----------------------------------------------------------------
from app.schemas.common import (
    ErrorResponse,
    PaginatedResponse,
    SuccessResponse,
    UUIDModel,
)

# -- auth ------------------------------------------------------------------
from app.schemas.auth import (
    ChangePasswordRequest,
    ForgotPasswordRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
)

# -- user ------------------------------------------------------------------
from app.schemas.user import (
    UserCreate,
    UserProfileUpdate,
    UserResponse,
    UserUpdate,
)

# -- tenant ----------------------------------------------------------------
from app.schemas.tenant import (
    TenantResponse,
    TenantUpdate,
    TenantUsageResponse,
)

# -- pii types -------------------------------------------------------------
from app.schemas.pii_type import (
    BatchPIIConfigUpdate,
    CustomPIITypeCreate,
    CustomPIITypeResponse,
    CustomPIITypeUpdate,
    PIICategoryResponse,
    PIIConfigUpdate,
    PIITypeResponse,
    TenantPIIConfigResponse,
)

# -- regex rules -----------------------------------------------------------
from app.schemas.regex_rule import (
    RegexRuleCreate,
    RegexRuleResponse,
    RegexRuleUpdate,
    RegexTestRequest,
    RegexTestResponse,
)

# -- field patterns --------------------------------------------------------
from app.schemas.field_pattern import (
    FieldPatternCreate,
    FieldPatternResponse,
    FieldPatternUpdate,
)

# -- context rules ---------------------------------------------------------
from app.schemas.context_rule import (
    ContextRuleCreate,
    ContextRuleResponse,
    ContextRuleUpdate,
)

# -- masking rules ---------------------------------------------------------
from app.schemas.masking_rule import (
    MaskingRuleResponse,
    MaskingRuleUpdate,
    MaskingStrategyResponse,
)

# -- detection -------------------------------------------------------------
from app.schemas.detection import (
    AsyncDetectRequest,
    DetectRequest,
    DetectResponse,
    DetectionItem,
    JobDetailResponse,
    JobResponse,
    RedactRequest,
    RedactResponse,
)

# -- learning --------------------------------------------------------------
from app.schemas.learning import (
    PendingRuleResponse,
    PendingRuleStatsResponse,
    PromoteRequest,
    TrainingDataStatsResponse,
    TrainingExampleResponse,
)

# -- ml model --------------------------------------------------------------
from app.schemas.ml_model import (
    ModelDetailResponse,
    ModelMetricsResponse,
    ModelVersionResponse,
    RetrainRequest,
    RetrainResponse,
)

# -- api keys --------------------------------------------------------------
from app.schemas.api_key import (
    APIAccessRequestCreate,
    APIAccessRequestResponse,
    APIAccessRequestReview,
    APIKeyCreate,
    APIKeyCreatedResponse,
    APIKeyResponse,
    APIKeyUsageResponse,
)

# -- analytics -------------------------------------------------------------
from app.schemas.analytics import (
    BySourceResponse,
    ByTypeResponse,
    HeatmapCell,
    HeatmapResponse,
    OverviewResponse,
    TimelineDataPoint,
    TimelineResponse,
)

# -- audit -----------------------------------------------------------------
from app.schemas.audit import (
    AuditLogFilter,
    AuditLogResponse,
)

# -- notifications ---------------------------------------------------------
from app.schemas.notification import (
    NotificationResponse,
    UnreadCountResponse,
)

__all__ = [
    # common
    "PaginatedResponse",
    "ErrorResponse",
    "SuccessResponse",
    "UUIDModel",
    # auth
    "RegisterRequest",
    "LoginRequest",
    "TokenResponse",
    "RefreshRequest",
    "ForgotPasswordRequest",
    "ResetPasswordRequest",
    "ChangePasswordRequest",
    # user
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "UserProfileUpdate",
    # tenant
    "TenantResponse",
    "TenantUpdate",
    "TenantUsageResponse",
    # pii types
    "PIICategoryResponse",
    "PIITypeResponse",
    "TenantPIIConfigResponse",
    "PIIConfigUpdate",
    "BatchPIIConfigUpdate",
    "CustomPIITypeCreate",
    "CustomPIITypeUpdate",
    "CustomPIITypeResponse",
    # regex rules
    "RegexRuleResponse",
    "RegexRuleCreate",
    "RegexRuleUpdate",
    "RegexTestRequest",
    "RegexTestResponse",
    # field patterns
    "FieldPatternResponse",
    "FieldPatternCreate",
    "FieldPatternUpdate",
    # context rules
    "ContextRuleResponse",
    "ContextRuleCreate",
    "ContextRuleUpdate",
    # masking rules
    "MaskingStrategyResponse",
    "MaskingRuleResponse",
    "MaskingRuleUpdate",
    # detection
    "DetectRequest",
    "DetectionItem",
    "DetectResponse",
    "RedactRequest",
    "RedactResponse",
    "AsyncDetectRequest",
    "JobResponse",
    "JobDetailResponse",
    # learning
    "PendingRuleResponse",
    "PromoteRequest",
    "PendingRuleStatsResponse",
    "TrainingExampleResponse",
    "TrainingDataStatsResponse",
    # ml model
    "ModelVersionResponse",
    "ModelDetailResponse",
    "RetrainRequest",
    "RetrainResponse",
    "ModelMetricsResponse",
    # api keys
    "APIAccessRequestCreate",
    "APIAccessRequestResponse",
    "APIAccessRequestReview",
    "APIKeyCreate",
    "APIKeyResponse",
    "APIKeyCreatedResponse",
    "APIKeyUsageResponse",
    # analytics
    "OverviewResponse",
    "ByTypeResponse",
    "BySourceResponse",
    "TimelineDataPoint",
    "TimelineResponse",
    "HeatmapCell",
    "HeatmapResponse",
    # audit
    "AuditLogResponse",
    "AuditLogFilter",
    # notifications
    "NotificationResponse",
    "UnreadCountResponse",
]
