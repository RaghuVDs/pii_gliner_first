"""API key management and access-request schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Access requests (approval workflow)
# ---------------------------------------------------------------------------
class APIAccessRequestCreate(BaseModel):
    """Request API access (submitted by an analyst/viewer)."""

    use_case: str = Field(
        ..., min_length=10, max_length=2000, description="Describe intended usage"
    )
    scopes_requested: list[str] | None = Field(
        None,
        description="Requested scopes (e.g. ['detect', 'redact'])",
    )
    rate_limit_requested: int | None = Field(
        None,
        ge=1,
        le=10000,
        description="Desired requests-per-minute limit",
    )


class APIAccessRequestResponse(BaseModel):
    """An API access request record."""

    id: UUID
    use_case: str
    scopes_requested: list[str] | None = None
    rate_limit_requested: int | None = None
    status: str = Field(
        ..., description="Request status: pending, approved, denied"
    )
    reviewed_by: UUID | None = None
    reviewed_at: datetime | None = None
    review_notes: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class APIAccessRequestReview(BaseModel):
    """Admin action to approve or deny an access request."""

    status: str = Field(
        ...,
        pattern=r"^(approved|denied)$",
        description="Must be 'approved' or 'denied'",
    )
    review_notes: str | None = Field(None, max_length=2000)
    rate_limit_override: int | None = Field(None, ge=1, le=10000)
    scopes_override: list[str] | None = None


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
class APIKeyCreate(BaseModel):
    """Create a new API key (post-approval or by admin)."""

    name: str = Field(
        ..., min_length=1, max_length=255, description="Human-friendly key name"
    )
    scopes: list[str] | None = Field(
        None,
        description="Permitted scopes (defaults to tenant default scopes)",
    )
    rate_limit_per_min: int | None = Field(
        None, ge=1, le=10000, description="Per-minute rate limit"
    )
    expires_in_days: int | None = Field(
        None,
        ge=1,
        le=365,
        description="Key TTL in days (None = non-expiring)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Production Scanner",
                "scopes": ["detect", "redact"],
                "rate_limit_per_min": 60,
                "expires_in_days": 90,
            }
        },
    )


class APIKeyResponse(BaseModel):
    """API key metadata (the secret itself is NOT included)."""

    id: UUID
    name: str
    key_prefix: str = Field(
        ..., description="First 8 chars of the key for identification"
    )
    scopes: list[str] = Field(default_factory=list)
    rate_limit_per_min: int | None = None
    is_active: bool
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    total_requests: int = Field(default=0, ge=0)
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class APIKeyCreatedResponse(APIKeyResponse):
    """Returned exactly once when a key is created -- includes the raw secret."""

    key: str = Field(
        ...,
        description="The full API key. Store it securely; it will not be shown again.",
    )


# ---------------------------------------------------------------------------
# Usage stats
# ---------------------------------------------------------------------------
class APIKeyUsageResponse(BaseModel):
    """Usage statistics for a single API key."""

    total_requests: int = Field(..., ge=0)
    requests_today: int = Field(..., ge=0)
    requests_this_month: int = Field(..., ge=0)
    by_endpoint: dict[str, int] = Field(
        default_factory=dict,
        description="Request counts keyed by endpoint path",
    )
    by_status_code: dict[str, int] = Field(
        default_factory=dict,
        description="Request counts keyed by HTTP status code",
    )
