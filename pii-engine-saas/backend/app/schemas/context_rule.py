"""Context-rule schemas for keyword-based PII boosting / negation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
class ContextRuleResponse(BaseModel):
    """A keyword/context rule that boosts or negates PII confidence."""

    id: int
    pii_type_name: str
    keyword_pattern: str
    is_negative: bool = Field(
        ..., description="True = negation rule (lowers confidence)"
    )
    is_system: bool
    is_enabled: bool
    source: str = Field(
        ..., description="Origin of the rule: system, user, learned"
    )

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------
class ContextRuleCreate(BaseModel):
    """Create a new tenant-owned context rule."""

    pii_type_name: str = Field(..., min_length=1, max_length=100)
    keyword_pattern: str = Field(..., min_length=1, max_length=500)
    is_negative: bool = Field(
        default=False,
        description="Set to True if the keyword should *reduce* confidence",
    )


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------
class ContextRuleUpdate(BaseModel):
    """Partial update for an existing context rule."""

    keyword_pattern: str | None = Field(None, min_length=1, max_length=500)
    is_negative: bool | None = None
    is_enabled: bool | None = None
