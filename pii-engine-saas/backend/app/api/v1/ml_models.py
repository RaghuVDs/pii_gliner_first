"""
ML model management endpoints.

Provides routes to inspect model versions, trigger retraining jobs,
activate specific model versions, and review training metrics.
"""

from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.dependencies import (
    get_current_tenant,
    get_current_user,
    require_role,
)
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.common import ErrorResponse, PaginatedResponse, SuccessResponse
from app.services import ml_model_service

router = APIRouter(tags=["ML Models"])


# ---------------------------------------------------------------------------
# Inline request / response schemas
# ---------------------------------------------------------------------------

class ModelVersionResponse(BaseModel):
    id: str
    tenant_id: str
    version: str
    base_model: str
    status: str = Field(..., description="training, ready, active, archived")
    known_labels: list[str] = Field(default_factory=list)
    training_examples_count: int = 0
    file_path: str | None = None
    created_at: str
    activated_at: str | None = None
    metrics_summary: Dict[str, Any] | None = None

    model_config = ConfigDict(from_attributes=True)


class ActiveModelResponse(BaseModel):
    version_id: str
    version: str
    base_model: str
    known_labels: list[str] = Field(default_factory=list)
    activated_at: str
    training_examples_count: int = 0
    metrics_summary: Dict[str, Any] | None = None

    model_config = ConfigDict(from_attributes=True)


class RetrainRequest(BaseModel):
    description: str | None = Field(None, description="Optional description for this training run")
    force: bool = Field(
        False,
        description="Force retrain even if training data hasn't changed",
    )


class RetrainJobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    version: str | None = None
    message: str = "Retraining job submitted."


class RetrainJobStatus(BaseModel):
    job_id: str
    status: str = Field(..., description="queued, training, completed, failed")
    progress: float | None = Field(None, ge=0.0, le=100.0)
    version: str | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None

    model_config = ConfigDict(from_attributes=True)


class ModelMetricsResponse(BaseModel):
    version_id: str
    version: str
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    per_label_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    training_loss: list[float] = Field(default_factory=list)
    validation_loss: list[float] = Field(default_factory=list)
    training_duration_seconds: float | None = None
    evaluated_at: str | None = None

    model_config = ConfigDict(from_attributes=True)


class MetricsHistoryEntry(BaseModel):
    version_id: str
    version: str
    f1_score: float | None = None
    precision: float | None = None
    recall: float | None = None
    training_examples_count: int = 0
    created_at: str

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# GET / -- list model versions
# ---------------------------------------------------------------------------
@router.get(
    "/",
    status_code=status.HTTP_200_OK,

    summary="List model versions",
)
async def list_model_versions(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    model_status: str | None = Query(
        None, alias="status", description="Filter: training, ready, active, archived",
    ),
):
    """Return a paginated list of model versions for the tenant."""
    return await ml_model_service.list_versions(db=db, 
        tenant_id=tenant.id,
        page=page,
        page_size=page_size,
        status_filter=model_status,
    )


# ---------------------------------------------------------------------------
# GET /active -- get active model info + known labels
# ---------------------------------------------------------------------------
@router.get(
    "/active",
    status_code=status.HTTP_200_OK,

    responses={404: {"model": ErrorResponse, "description": "No active model"}},
    summary="Get active model info",
)
async def get_active_model(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Return details about the currently active model version and its
    known PII labels."""
    return await ml_model_service.get_active(db=db, tenant_id=tenant.id)


# ---------------------------------------------------------------------------
# POST /retrain -- trigger manual retrain
# ---------------------------------------------------------------------------
@router.post(
    "/retrain",
    status_code=status.HTTP_202_ACCEPTED,

    responses={
        409: {"model": ErrorResponse, "description": "Retrain already in progress"},
    },
    summary="Trigger model retraining",
    dependencies=[Depends(require_role("admin"))],
)
async def trigger_retrain(
    body: RetrainRequest,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Submit a model retraining job using the tenant's current
    training data corpus."""
    return await ml_model_service.trigger_retrain(db=db, 
        tenant_id=tenant.id,
        description=body.description,
        force=body.force,
        triggered_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# GET /retrain/{job_id} -- get retrain job status
# ---------------------------------------------------------------------------
@router.get(
    "/retrain/{job_id}",
    status_code=status.HTTP_200_OK,

    responses={404: {"model": ErrorResponse}},
    summary="Get retrain job status",
)
async def get_retrain_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Return the status and progress of a retraining job."""
    return await ml_model_service.get_retrain_status(db=db, 
        tenant_id=tenant.id,
        job_id=job_id,
    )


# ---------------------------------------------------------------------------
# POST /{version_id}/activate -- activate a model version
# ---------------------------------------------------------------------------
@router.post(
    "/{version_id}/activate",
    status_code=status.HTTP_200_OK,

    responses={
        404: {"model": ErrorResponse, "description": "Version not found"},
        409: {"model": ErrorResponse, "description": "Version not ready"},
    },
    summary="Activate a model version",
    dependencies=[Depends(require_role("admin"))],
)
async def activate_version(
    version_id: str,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Set a specific model version as the active version for the tenant.
    The previously active version is archived."""
    await ml_model_service.activate_version(db=db, 
        tenant_id=tenant.id,
        version_id=version_id,
        activated_by=current_user.id,
    )
    return SuccessResponse(message="Model version activated successfully.")


# ---------------------------------------------------------------------------
# GET /metrics -- metrics history across training runs
# ---------------------------------------------------------------------------
@router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,

    summary="Metrics history across training runs",
)
async def metrics_history(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    limit: int = Query(20, ge=1, le=100, description="Max entries to return"),
):
    """Return a chronological summary of key metrics across all training
    runs, useful for tracking model improvement over time."""
    return await ml_model_service.metrics_history(db=db, 
        tenant_id=tenant.id,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# GET /metrics/{version_id} -- detailed metrics for a specific version
# ---------------------------------------------------------------------------
@router.get(
    "/metrics/{version_id}",
    status_code=status.HTTP_200_OK,

    responses={404: {"model": ErrorResponse}},
    summary="Detailed metrics for a model version",
)
async def get_version_metrics(
    version_id: str,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Return detailed training and evaluation metrics for a specific
    model version, including per-label breakdowns."""
    return await ml_model_service.get_version_metrics(db=db, 
        tenant_id=tenant.id,
        version_id=version_id,
    )
