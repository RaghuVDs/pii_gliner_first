"""Field (label) pattern schemas for structured-data PII detection."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
class FieldPatternResponse(BaseModel):
    """A label/field-name pattern that hints at PII in structured data."""

    id: int
    pii_type_name: str
    label_pattern: str
    is_system: bool
    is_enabled: bool
    sort_order: int

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------
class FieldPatternCreate(BaseModel):
    """Create a new tenant-owned field pattern."""

    pii_type_name: str = Field(..., min_length=1, max_length=100)
    label_pattern: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Regex or glob pattern matched against column/field names",
    )


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------
class FieldPatternUpdate(BaseModel):
    """Partial update for an existing field pattern."""

    label_pattern: str | None = Field(None, min_length=1, max_length=500)
    is_enabled: bool | None = None
