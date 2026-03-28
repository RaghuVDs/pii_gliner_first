"""Self-learning / pending-rule and training-data schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Pending rules (auto-discovered candidates)
# ---------------------------------------------------------------------------
class PendingRuleResponse(BaseModel):
    """A pending rule candidate discovered by the self-learning pipeline."""

    id: int
    group_key: str = Field(
        ..., description="Deduplicated group identifier"
    )
    suggested_label: str
    seen_count: int = Field(..., ge=1)
    example_contexts: list[str] = Field(default_factory=list)
    suggested_keywords: list[str] = Field(default_factory=list)
    structure_patterns: list[str] = Field(default_factory=list)
    co_occurring_labels: list[str] = Field(default_factory=list)
    status: str = Field(
        ..., description="Status: pending, promoted, dismissed"
    )
    first_seen: datetime
    last_seen: datetime

    model_config = ConfigDict(from_attributes=True)


class PromoteRequest(BaseModel):
    """Promote a pending rule into an active context rule."""

    label: str = Field(
        ..., min_length=1, max_length=100, description="PII type label to assign"
    )
    keywords: list[str] | None = Field(
        None,
        description="Override suggested keywords (uses auto-suggested if omitted)",
    )


class PendingRuleStatsResponse(BaseModel):
    """Aggregate statistics for the pending-rules queue."""

    total_groups: int = Field(..., ge=0)
    ready_to_promote: int = Field(..., ge=0)
    total_sightings: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------
class TrainingExampleResponse(BaseModel):
    """A single training example stored for model fine-tuning."""

    id: int
    structure: str
    label: str
    length: int
    keywords: list[str] = Field(default_factory=list)
    co_labels: list[str] = Field(default_factory=list)
    source: str = Field(..., description="Origin: detection, manual, import")
    score: float = Field(..., ge=0.0, le=1.0)
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TrainingDataStatsResponse(BaseModel):
    """Aggregate statistics about the training-data corpus."""

    total_examples: int = Field(..., ge=0)
    label_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Count of training examples per label",
    )
    source_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Count of training examples per source",
    )
