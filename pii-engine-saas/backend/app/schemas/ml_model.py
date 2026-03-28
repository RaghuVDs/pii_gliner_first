"""ML model versioning and retraining schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Model version
# ---------------------------------------------------------------------------
class ModelVersionResponse(BaseModel):
    """Summary of a trained model version."""

    id: int
    version: str = Field(..., description="Semantic version string, e.g. 'v3'")
    num_labels: int = Field(..., ge=0)
    num_examples: int = Field(..., ge=0)
    val_accuracy: float | None = Field(None, ge=0.0, le=1.0)
    val_weighted_f1: float | None = Field(None, ge=0.0, le=1.0)
    is_active: bool = Field(
        ..., description="Whether this version is currently serving traffic"
    )
    training_trigger: str = Field(
        ..., description="What triggered training: manual, scheduled, threshold"
    )
    trained_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ModelDetailResponse(ModelVersionResponse):
    """Extended model version with full training metrics."""

    train_size: int = Field(..., ge=0)
    val_size: int = Field(..., ge=0)
    metrics_detail: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-label precision, recall, f1, support, etc.",
    )


# ---------------------------------------------------------------------------
# Retrain
# ---------------------------------------------------------------------------
class RetrainRequest(BaseModel):
    """Trigger a model retraining job."""

    epochs: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Number of training epochs",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"epochs": 50}
        },
    )


class RetrainResponse(BaseModel):
    """Acknowledgement that a retrain job has been queued."""

    job_id: UUID
    status: str = Field(
        ..., description="Initial status, typically 'pending'"
    )
    message: str


# ---------------------------------------------------------------------------
# Metrics overview
# ---------------------------------------------------------------------------
class ModelMetricsResponse(BaseModel):
    """Historical model training runs and aggregate stats."""

    runs: list[ModelDetailResponse] = Field(default_factory=list)
    total_retrains: int = Field(..., ge=0)
