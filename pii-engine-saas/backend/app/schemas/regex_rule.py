"""Regex-based PII detection rule schemas."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
class RegexRuleResponse(BaseModel):
    """A regex rule used for deterministic PII detection."""

    id: int
    pii_type_name: str
    pattern: str
    description: str | None = None
    is_system: bool
    is_enabled: bool
    sort_order: int

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------
class RegexRuleCreate(BaseModel):
    """Create a new tenant-owned regex rule."""

    pii_type_name: str = Field(..., min_length=1, max_length=100)
    pattern: str = Field(..., min_length=1, max_length=2000)
    description: str | None = Field(None, max_length=500)

    @field_validator("pattern")
    @classmethod
    def _valid_regex(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {exc}") from exc
        return v


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------
class RegexRuleUpdate(BaseModel):
    """Partial update for an existing regex rule."""

    pattern: str | None = Field(None, min_length=1, max_length=2000)
    description: str | None = Field(None, max_length=500)
    is_enabled: bool | None = None
    sort_order: int | None = Field(None, ge=0)

    @field_validator("pattern")
    @classmethod
    def _valid_regex(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {exc}") from exc
        return v


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
class RegexTestRequest(BaseModel):
    """Test a regex pattern against sample text."""

    pattern: str = Field(..., min_length=1, max_length=2000)
    test_text: str = Field(..., min_length=1, max_length=50000)

    @field_validator("pattern")
    @classmethod
    def _valid_regex(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {exc}") from exc
        return v


class RegexMatch(BaseModel):
    """A single regex match result."""

    text: str
    start: int
    end: int


class RegexTestResponse(BaseModel):
    """Result of testing a regex pattern."""

    matches: list[RegexMatch] = Field(default_factory=list)
