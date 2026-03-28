"""
Self-learning / feedback-loop endpoints.

Manages pending rules (auto-discovered patterns awaiting human review),
promotion/rejection workflows, and the training-data corpus used to
fine-tune the GLiNER model.
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
from app.services import learning_service

router = APIRouter(tags=["Learning"])


# ---------------------------------------------------------------------------
# Inline request / response schemas
# ---------------------------------------------------------------------------

class PendingRuleResponse(BaseModel):
    id: str
    tenant_id: str
    entity_type: str
    pattern: str | None = None
    example_texts: list[str] = Field(default_factory=list)
    seen_count: int = 0
    confidence_avg: float = 0.0
    status: str = Field(..., description="pending, promoted, rejected")
    source: str | None = None
    created_at: str
    reviewed_at: str | None = None
    reviewed_by: str | None = None

    model_config = ConfigDict(from_attributes=True)


class PendingRuleStats(BaseModel):
    model_config = ConfigDict(extra="allow")
    total_pending: int = 0
    total_groups: int = 0
    ready_to_promote: int = 0
    total_sightings: int = 0


class PromoteRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=100, description="Confirmed PII type label")
    keywords: list[str] = Field(default_factory=list, description="Context keywords")
    notes: str | None = None


class AutoPromoteRequest(BaseModel):
    min_seen_count: int = Field(5, ge=1, description="Minimum occurrences to auto-promote")
    min_confidence: float = Field(0.7, ge=0.0, le=1.0, description="Minimum avg confidence")


class AutoPromoteResponse(BaseModel):
    promoted_count: int
    promoted_rules: List[str] = Field(
        default_factory=list, description="IDs of auto-promoted rules",
    )


class TrainingExampleResponse(BaseModel):
    id: str
    tenant_id: str
    text: str
    entity_type: str
    entity_value: str
    start: int
    end: int
    source: str = Field(..., description="manual, detection, promotion")
    is_validated: bool = False
    created_at: str

    model_config = ConfigDict(from_attributes=True)


class TrainingDataStats(BaseModel):
    total_examples: int
    by_entity_type: Dict[str, int] = Field(default_factory=dict)
    validated_count: int
    unvalidated_count: int


# ---------------------------------------------------------------------------
# GET /pending-rules -- list pending rules
# ---------------------------------------------------------------------------
@router.get(
    "/pending-rules",
    status_code=status.HTTP_200_OK,

    summary="List pending rules",
)
async def list_pending_rules(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    entity_type: str | None = Query(None, description="Filter by entity type"),
    sort_by: str = Query("seen_count", description="Sort field: seen_count, confidence_avg, created_at"),
    sort_order: str = Query("desc", description="Sort direction: asc or desc"),
):
    """Return pending rules sorted by occurrence count, paginated."""
    return await learning_service.list_pending_rules(db=db, 
        tenant_id=tenant.id,
        page=page,
        page_size=page_size,
        entity_type=entity_type,
        sort_by=sort_by,
        sort_order=sort_order,
    )


# ---------------------------------------------------------------------------
# GET /pending-rules/stats -- summary stats
# ---------------------------------------------------------------------------
@router.get(
    "/pending-rules/stats",
    status_code=status.HTTP_200_OK,

    summary="Pending rules summary stats",
)
async def pending_rules_stats(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Return aggregate statistics about pending rules for the tenant."""
    return await learning_service.pending_rules_stats(db=db, tenant_id=tenant.id)


# ---------------------------------------------------------------------------
# POST /pending-rules/{rule_id}/promote -- manually promote
# ---------------------------------------------------------------------------
@router.post(
    "/pending-rules/{rule_id}/promote",
    status_code=status.HTTP_200_OK,

    responses={
        404: {"model": ErrorResponse, "description": "Rule not found"},
        409: {"model": ErrorResponse, "description": "Rule already reviewed"},
    },
    summary="Promote a pending rule",
)
async def promote_rule(
    rule_id: str,
    body: PromoteRequest,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Manually promote a pending rule into an active detection rule
    with a confirmed label and optional keywords."""
    return await learning_service.promote_rule(db=db, 
        tenant_id=tenant.id,
        rule_id=rule_id,
        label=body.label,
        keywords=body.keywords,
        notes=body.notes,
        promoted_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# POST /pending-rules/{rule_id}/reject -- reject a pending rule
# ---------------------------------------------------------------------------
@router.post(
    "/pending-rules/{rule_id}/reject",
    status_code=status.HTTP_200_OK,

    responses={
        404: {"model": ErrorResponse, "description": "Rule not found"},
        409: {"model": ErrorResponse, "description": "Rule already reviewed"},
    },
    summary="Reject a pending rule",
)
async def reject_rule(
    rule_id: str,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Reject a pending rule, marking it as not useful for detection."""
    return await learning_service.reject_rule(db=db, 
        tenant_id=tenant.id,
        rule_id=rule_id,
        rejected_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# POST /pending-rules/auto-promote -- trigger auto-promotion
# ---------------------------------------------------------------------------
@router.post(
    "/pending-rules/auto-promote",
    status_code=status.HTTP_200_OK,

    summary="Auto-promote pending rules",
    dependencies=[Depends(require_role("admin"))],
)
async def auto_promote(
    body: AutoPromoteRequest,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Automatically promote all pending rules that exceed the specified
    seen-count and confidence thresholds."""
    return await learning_service.auto_promote(db=db, 
        tenant_id=tenant.id,
        min_seen_count=body.min_seen_count,
        min_confidence=body.min_confidence,
        promoted_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# GET /training-data -- list training examples
# ---------------------------------------------------------------------------
@router.get(
    "/training-data",
    status_code=status.HTTP_200_OK,

    summary="List training examples",
)
async def list_training_data(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    entity_type: str | None = Query(None, description="Filter by entity type"),
    is_validated: bool | None = Query(None, description="Filter by validation status"),
):
    """Return a paginated list of training examples for the tenant."""
    return await learning_service.list_training_data(db=db, 
        tenant_id=tenant.id,
        page=page,
        page_size=page_size,
        entity_type=entity_type,
        is_validated=is_validated,
    )


# ---------------------------------------------------------------------------
# GET /training-data/stats -- training data stats
# ---------------------------------------------------------------------------
@router.get(
    "/training-data/stats",
    status_code=status.HTTP_200_OK,

    summary="Training data stats",
)
async def training_data_stats(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Return aggregate statistics about the training data corpus."""
    return await learning_service.training_data_stats(db=db, tenant_id=tenant.id)


# ---------------------------------------------------------------------------
# DELETE /training-data/{example_id} -- remove bad training example
# ---------------------------------------------------------------------------
@router.delete(
    "/training-data/{example_id}",
    status_code=status.HTTP_200_OK,

    responses={404: {"model": ErrorResponse}},
    summary="Remove a training example",
)
async def delete_training_example(
    example_id: str,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Remove a bad or incorrect training example from the corpus."""
    await learning_service.delete_training_example(db=db, 
        tenant_id=tenant.id,
        example_id=example_id,
        deleted_by=current_user.id,
    )
    return SuccessResponse(message="Training example removed.")
