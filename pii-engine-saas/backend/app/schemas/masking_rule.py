"""Masking / redaction strategy schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Strategy catalogue
# ---------------------------------------------------------------------------
class MaskingStrategyResponse(BaseModel):
    """An available masking strategy (e.g. redact, hash, mask, tokenize)."""

    id: int
    name: str
    display_name: str
    description: str | None = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Rule (per PII type)
# ---------------------------------------------------------------------------
class MaskingRuleResponse(BaseModel):
    """The effective masking rule for a given PII type within a tenant."""

    pii_type_name: str
    strategy_name: str
    strategy_display_name: str
    custom_params: dict[str, Any] | None = Field(
        None,
        description="Strategy-specific params (e.g. mask_char, hash_algo)",
    )
    is_tenant_override: bool = Field(
        ..., description="True if the tenant has overridden the system default"
    )

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------
class MaskingRuleUpdate(BaseModel):
    """Set or override the masking strategy for a PII type."""

    strategy_id: int = Field(..., description="ID of the masking strategy to use")
    custom_params: dict[str, Any] | None = Field(
        None,
        description="Optional strategy-specific parameters",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "strategy_id": 2,
                "custom_params": {"mask_char": "*", "preserve_length": True},
            }
        },
    )
