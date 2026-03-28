"""
PII detection and redaction endpoints.

Provides synchronous and asynchronous (job-based) PII detection and
redaction.  Async endpoints submit work to the Celery task queue and
return a job ID for polling.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.deps import get_current_user, get_current_tenant
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.common import ErrorResponse, PaginatedResponse, SuccessResponse
from app.services import detection_service

router = APIRouter(tags=["Detection"])


# ---------------------------------------------------------------------------
# Inline request / response schemas
# ---------------------------------------------------------------------------

class DetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000, description="Text to scan")
    pii_types: list[str] | None = Field(
        None, description="Limit detection to these PII types (null = all enabled)",
    )
    threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Override confidence threshold",
    )
    language: str = Field("en", max_length=10)


class DetectionEntity(BaseModel):
    entity_type: str
    value: str
    start: int
    end: int
    confidence: float
    source: str = Field(..., description="Detection source: gliner, regex, field_pattern, context")


class DetectionResponse(BaseModel):
    text_length: int
    entity_count: int
    entities: List[DetectionEntity]
    processing_time_ms: float


class RedactionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    pii_types: list[str] | None = None
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    masking_strategy: str | None = Field(
        None, description="Override default masking strategy",
    )
    language: str = Field("en", max_length=10)


class RedactionResponse(BaseModel):
    original_length: int
    redacted_text: str
    entity_count: int
    entities: List[DetectionEntity]
    processing_time_ms: float


class AsyncJobResponse(BaseModel):
    job_id: str
    status: str = "pending"
    message: str = "Job submitted successfully."


class BatchDetectionRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    pii_types: list[str] | None = None
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    language: str = Field("en", max_length=10)


class BatchDetectionResponse(BaseModel):
    total_texts: int
    results: List[DetectionResponse]
    total_processing_time_ms: float


class JobStatus(BaseModel):
    job_id: str
    status: str = Field(..., description="pending, processing, completed, failed, cancelled")
    progress: float | None = Field(None, ge=0.0, le=100.0)
    result: Dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    completed_at: str | None = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# POST /detect -- synchronous PII detection
# ---------------------------------------------------------------------------
@router.post(
    "/detect",
    status_code=status.HTTP_200_OK,

    summary="Detect PII (synchronous)",
)
async def detect_pii(
    body: DetectionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Run synchronous PII detection on the provided text."""
    tenant_id = getattr(request.state, "tenant_id", None)
    return await detection_service.detect(
        db=db,
        tenant_id=tenant_id,
        text=body.text,
    )


# ---------------------------------------------------------------------------
# POST /redact -- synchronous PII redaction
# ---------------------------------------------------------------------------
@router.post(
    "/redact",
    status_code=status.HTTP_200_OK,

    summary="Redact PII (synchronous)",
)
async def redact_pii(
    body: RedactionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Detect PII and return the text with detected entities masked."""
    tenant_id = getattr(request.state, "tenant_id", None)
    return await detection_service.redact(
        db=db,
        tenant_id=tenant_id,
        text=body.text,
    )


# ---------------------------------------------------------------------------
# POST /detect/async -- async detection job
# ---------------------------------------------------------------------------
@router.post(
    "/detect/async",
    status_code=status.HTTP_202_ACCEPTED,

    summary="Detect PII (asynchronous)",
)
async def detect_pii_async(
    body: DetectionRequest,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Submit an asynchronous PII detection job.  Returns a job ID that
    can be polled via ``GET /jobs/{job_id}``."""
    return await detection_service.detect_async(
        db=db,
        tenant_id=tenant.id,
        text=body.text,
        pii_types=body.pii_types,
        threshold=body.threshold,
        language=body.language,
        user_id=current_user.id,
    )


# ---------------------------------------------------------------------------
# POST /redact/async -- async redaction job
# ---------------------------------------------------------------------------
@router.post(
    "/redact/async",
    status_code=status.HTTP_202_ACCEPTED,

    summary="Redact PII (asynchronous)",
)
async def redact_pii_async(
    body: RedactionRequest,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Submit an asynchronous PII redaction job."""
    return await detection_service.redact_async(
        db=db,
        tenant_id=tenant.id,
        text=body.text,
        pii_types=body.pii_types,
        threshold=body.threshold,
        masking_strategy=body.masking_strategy,
        language=body.language,
        user_id=current_user.id,
    )


# ---------------------------------------------------------------------------
# POST /batch -- batch detection (multiple texts)
# ---------------------------------------------------------------------------
@router.post(
    "/batch",
    status_code=status.HTTP_200_OK,

    summary="Batch PII detection",
)
async def batch_detect(
    body: BatchDetectionRequest,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Run PII detection on multiple texts in a single request."""
    return await detection_service.batch_detect(
        db=db,
        tenant_id=tenant.id,
        texts=body.texts,
        pii_types=body.pii_types,
        threshold=body.threshold,
        language=body.language,
        user_id=current_user.id,
    )


# ---------------------------------------------------------------------------
# GET /jobs -- list detection jobs (paginated)
# ---------------------------------------------------------------------------
@router.get(
    "/jobs",
    status_code=status.HTTP_200_OK,

    summary="List detection jobs",
)
async def list_jobs(
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    job_status: str | None = Query(
        None,
        alias="status",
        description="Filter by status: pending, processing, completed, failed, cancelled",
    ),
):
    """Return a paginated list of detection jobs for the current tenant."""
    return await detection_service.list_jobs(
        tenant_id=tenant.id,
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        status_filter=job_status,
    )


# ---------------------------------------------------------------------------
# GET /jobs/{job_id} -- get job status + results
# ---------------------------------------------------------------------------
@router.get(
    "/jobs/{job_id}",
    status_code=status.HTTP_200_OK,

    responses={404: {"model": ErrorResponse}},
    summary="Get job status and results",
)
async def get_job(
    job_id: str,
    tenant: Tenant = Depends(get_current_tenant),
):
    """Return the current status and (if complete) results for a detection job."""
    return await detection_service.get_job(
        tenant_id=tenant.id,
        job_id=job_id,
    )


# ---------------------------------------------------------------------------
# DELETE /jobs/{job_id} -- cancel pending job
# ---------------------------------------------------------------------------
@router.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_200_OK,

    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
        409: {"model": ErrorResponse, "description": "Job is not cancellable"},
    },
    summary="Cancel a pending job",
)
async def cancel_job(
    job_id: str,
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Cancel a pending or processing detection job."""
    await detection_service.cancel_job(
        tenant_id=tenant.id,
        job_id=job_id,
        cancelled_by=current_user.id,
    )
    return SuccessResponse(message="Job cancelled successfully.")
