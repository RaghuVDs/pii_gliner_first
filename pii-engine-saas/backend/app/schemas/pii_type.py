"""PII type and category configuration schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Category
# ---------------------------------------------------------------------------
class PIICategoryResponse(BaseModel):
    """A grouping category for PII types (e.g. Personal, Financial)."""

    id: int
    name: str
    display_name: str
    description: str | None = None
    tier: str = Field(..., description="Category tier: standard, sensitive, custom")
    sort_order: int

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Tenant-level PII config (nested)
# ---------------------------------------------------------------------------
class TenantPIIConfigResponse(BaseModel):
    """Tenant-specific overrides for a system PII type."""

    is_enabled: bool
    custom_threshold: float | None = None
    custom_aliases: list[str] | None = None
    notes: str | None = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# PII type response (system + tenant overlay)
# ---------------------------------------------------------------------------
class PIITypeResponse(BaseModel):
    """System PII type with optional per-tenant configuration overlay."""

    id: int
    category_id: int | None = None
    category_name: str | None = None
    name: str
    display_name: str
    description: str | None = None
    gliner_aliases: list[str] = Field(default_factory=list)
    default_threshold: float = Field(..., ge=0.0, le=1.0)
    is_system: bool
    is_sensitive: bool
    compliance_tags: list[str] = Field(default_factory=list)
    tenant_config: TenantPIIConfigResponse | None = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Tenant config mutation
# ---------------------------------------------------------------------------
class PIIConfigUpdate(BaseModel):
    """Update a tenant's configuration for a single system PII type."""

    is_enabled: bool | None = None
    custom_threshold: float | None = Field(None, ge=0.0, le=1.0)
    custom_aliases: list[str] | None = None
    notes: str | None = Field(None, max_length=1000)


class BatchPIIConfigUpdate(BaseModel):
    """Bulk enable/disable multiple PII types at once."""

    pii_type_ids: list[int] = Field(..., min_length=1, description="PII type IDs to update")
    is_enabled: bool


# ---------------------------------------------------------------------------
# Custom (tenant-owned) PII types
# ---------------------------------------------------------------------------
class CustomPIITypeCreate(BaseModel):
    """Create a new custom PII type owned by the tenant."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Machine name (snake_case, starts with letter)",
    )
    display_name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)
    category_id: int | None = None
    gliner_aliases: list[str] = Field(
        ..., min_length=1, description="At least one GLiNER alias required"
    )
    default_threshold: float = Field(0.5, ge=0.0, le=1.0)
    compliance_tags: list[str] | None = None

    @field_validator("gliner_aliases")
    @classmethod
    def _non_empty_aliases(cls, v: list[str]) -> list[str]:
        cleaned = [a.strip() for a in v if a.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty alias is required")
        return cleaned


class CustomPIITypeUpdate(BaseModel):
    """Partial update for a tenant-owned custom PII type."""

    display_name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)
    gliner_aliases: list[str] | None = None
    default_threshold: float | None = Field(None, ge=0.0, le=1.0)
    is_enabled: bool | None = None
    compliance_tags: list[str] | None = None

    @field_validator("gliner_aliases")
    @classmethod
    def _non_empty_aliases(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        cleaned = [a.strip() for a in v if a.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty alias is required")
        return cleaned


class CustomPIITypeResponse(BaseModel):
    """A custom PII type owned by a specific tenant."""

    id: int
    tenant_id: UUID
    name: str
    display_name: str
    description: str | None = None
    category_id: int | None = None
    gliner_aliases: list[str] = Field(default_factory=list)
    default_threshold: float
    is_enabled: bool
    compliance_tags: list[str] = Field(default_factory=list)
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
