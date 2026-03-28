"""PII detection and redaction request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Detection primitives
# ---------------------------------------------------------------------------
class DetectionItem(BaseModel):
    """A single PII detection within the input text."""

    label: str = Field(..., description="PII type label (e.g. EMAIL_ADDRESS)")
    text: str = Field(..., description="The matched text span")
    start: int = Field(..., ge=0, description="Start character offset")
    end: int = Field(..., ge=0, description="End character offset (exclusive)")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source: str = Field(
        ...,
        description="Detection source: gliner, regex, field_pattern, context, lstm",
    )
    replacement_tag: str | None = Field(
        None, description="Redaction tag, e.g. [EMAIL_ADDRESS]"
    )
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (regex name, model version, etc.)",
    )

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Synchronous detect
# ---------------------------------------------------------------------------
class DetectRequest(BaseModel):
    """Synchronous PII detection request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=100_000,
        description="Input text to scan (max 100 000 chars)",
    )
    pii_types: list[str] | None = Field(
        None,
        description="Optional filter: only detect these PII types",
    )


class DetectResponse(BaseModel):
    """Result of a synchronous PII detection scan."""

    detections: list[DetectionItem] = Field(default_factory=list)
    detection_count: int = Field(..., ge=0)
    pii_types_found: list[str] = Field(
        default_factory=list,
        description="Distinct PII type labels found",
    )
    processing_time_ms: float = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Synchronous redact
# ---------------------------------------------------------------------------
class RedactRequest(BaseModel):
    """Synchronous PII redaction request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=100_000,
        description="Input text to redact (max 100 000 chars)",
    )
    pii_types: list[str] | None = Field(
        None,
        description="Optional filter: only redact these PII types",
    )


class RedactResponse(BaseModel):
    """Result of a synchronous PII redaction."""

    redacted_text: str
    detections: list[DetectionItem] = Field(default_factory=list)
    detection_count: int = Field(..., ge=0)
    processing_time_ms: float = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Asynchronous (job-based) detection
# ---------------------------------------------------------------------------
class AsyncDetectRequest(BaseModel):
    """Submit an asynchronous detection/redaction job."""

    text: str | None = Field(
        None,
        max_length=100_000,
        description="Inline text (mutually exclusive with file_url)",
    )
    file_url: str | None = Field(
        None,
        max_length=2048,
        description="URL of a file to process (mutually exclusive with text)",
    )
    pii_types: list[str] | None = None
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Job priority (1 = highest, 10 = lowest)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "John Doe lives at 123 Main St.",
                "priority": 3,
            }
        },
    )


class JobResponse(BaseModel):
    """Status of an asynchronous job."""

    id: UUID
    job_type: str = Field(
        ..., description="Job type: detect, redact, batch_detect"
    )
    status: str = Field(
        ..., description="Job status: pending, processing, completed, failed"
    )
    detection_count: int | None = None
    processing_time_ms: float | None = None
    created_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None

    model_config = ConfigDict(from_attributes=True)


class JobDetailResponse(JobResponse):
    """Extended job response that includes the full result payload."""

    result: DetectResponse | RedactResponse | None = Field(
        None,
        description="Full result payload (available when status = completed)",
    )
