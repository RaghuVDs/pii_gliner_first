"""Tenant (organisation) schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
class TenantResponse(BaseModel):
    """Full tenant record visible to tenant admins."""

    id: UUID
    name: str
    slug: str
    plan: str = Field(..., description="Subscription plan: free, pro, enterprise")
    settings: dict[str, Any] = Field(default_factory=dict)
    max_users: int
    max_api_keys: int
    max_monthly_scans: int
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------
class TenantUpdate(BaseModel):
    """Partial update for tenant settings (admin only)."""

    name: str | None = Field(None, min_length=1, max_length=255)
    settings: dict[str, Any] | None = None
    max_users: int | None = Field(None, ge=1)
    max_api_keys: int | None = Field(None, ge=0)
    max_monthly_scans: int | None = Field(None, ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Acme Corp Updated",
                "settings": {"default_masking": "redact"},
                "max_users": 25,
            }
        },
    )


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
class TenantUsageResponse(BaseModel):
    """Current resource-usage snapshot for a tenant."""

    current_month_scans: int = Field(..., ge=0)
    max_monthly_scans: int = Field(..., ge=0)
    active_users: int = Field(..., ge=0)
    max_users: int = Field(..., ge=0)
    active_api_keys: int = Field(..., ge=0)
    max_api_keys: int = Field(..., ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "current_month_scans": 1240,
                "max_monthly_scans": 10000,
                "active_users": 5,
                "max_users": 10,
                "active_api_keys": 2,
                "max_api_keys": 5,
            }
        },
    )
