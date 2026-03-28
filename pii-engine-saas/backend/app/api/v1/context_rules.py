"""
Context rule management endpoints.

Context rules provide surrounding-text heuristics that boost or lower the
confidence of PII detections.  System rules are read-only; tenants can
add, update, and soft-delete their own.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.v1.dependencies import (
    get_current_tenant,
    get_current_user,
)
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.common import ErrorResponse, PaginatedResponse, SuccessResponse
from app.services import context_rule_service

router = APIRouter(tags=["Context Rules"])


# ---------------------------------------------------------------------------
# Inline request / response schemas
# ---------------------------------------------------------------------------

class ContextRuleResponse(BaseModel):
    id: int
    pii_type_name: str
    context_keywords: list[str] = Field(default_factory=list)
    boost_score: float = Field(0.0, description="Score adjustment when context matches")
    window_size: int = Field(50, description="Character window around detected entity")
    description: str | None = None
    is_system: bool
    is_enabled: bool = True
    tenant_id: str | None = None

    model_config = ConfigDict(from_attributes=True)


class ContextRuleCreate(BaseModel):
    pii_type_name: str = Field(..., min_length=1, max_length=100)
    context_keywords: list[str] = Field(
        ..., min_length=1, description="Keywords that indicate relevant context",
    )
    boost_score: float = Field(0.15, ge=-1.0, le=1.0)
    window_size: int = Field(50, ge=10, le=500)
    description: str | None = None
    is_enabled: bool = True


class ContextRuleUpdate(BaseModel):
    context_keywords: list[str] | None = None
    boost_score: float | None = Field(None, ge=-1.0, le=1.0)
    window_size: int | None = Field(None, ge=10, le=500)
    description: str | None = None
    is_enabled: bool | None = None


# ---------------------------------------------------------------------------
# GET / -- list all context rules (system + tenant)
# ---------------------------------------------------------------------------
@router.get(
    "/",
    status_code=status.HTTP_200_OK,

    summary="List all context rules",
)
async def list_context_rules(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    pii_type_name: str | None = Query(None, description="Filter by PII type name"),
    is_system: bool | None = Query(None, description="Filter system vs tenant rules"),
):
    """Return all context rules visible to the tenant."""
    return await context_rule_service.list_rules(
        db=db,
        tenant_id=tenant.id,
        page=page,
        page_size=page_size,
        pii_type_name=pii_type_name,
        is_system=is_system,
    )


# ---------------------------------------------------------------------------
# GET /{pii_type_name} -- get rules for a specific PII type
# ---------------------------------------------------------------------------
@router.get(
    "/{pii_type_name}",
    status_code=status.HTTP_200_OK,

    summary="Get context rules for a PII type",
)
async def get_rules_by_type(
    pii_type_name: str,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Return all context rules (system + tenant) for the given PII type."""
    return await context_rule_service.get_rules_by_type(
        db=db,
        tenant_id=tenant.id,
        pii_type_name=pii_type_name,
    )


# ---------------------------------------------------------------------------
# POST / -- add tenant context rule
# ---------------------------------------------------------------------------
@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,

    summary="Add a tenant context rule",
)
async def create_context_rule(
    body: ContextRuleCreate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Create a new tenant-specific context rule."""
    return await context_rule_service.create_rule(
        db=db,
        tenant_id=tenant.id,
        pii_type_name=body.pii_type_name,
        context_keywords=body.context_keywords,
        boost_score=body.boost_score,
        window_size=body.window_size,
        description=body.description,
        is_enabled=body.is_enabled,
        created_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# PATCH /{rule_id} -- update tenant context rule
# ---------------------------------------------------------------------------
@router.patch(
    "/{rule_id}",
    status_code=status.HTTP_200_OK,

    responses={
        404: {"model": ErrorResponse, "description": "Rule not found"},
        403: {"model": ErrorResponse, "description": "Cannot modify system rules"},
    },
    summary="Update a tenant context rule",
)
async def update_context_rule(
    rule_id: int,
    body: ContextRuleUpdate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Update an existing tenant-specific context rule."""
    return await context_rule_service.update_rule(
        db=db,
        tenant_id=tenant.id,
        rule_id=rule_id,
        data=body,
        updated_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# DELETE /{rule_id} -- soft-delete tenant context rule
# ---------------------------------------------------------------------------
@router.delete(
    "/{rule_id}",
    status_code=status.HTTP_200_OK,

    responses={
        404: {"model": ErrorResponse, "description": "Rule not found"},
        403: {"model": ErrorResponse, "description": "Cannot delete system rules"},
    },
    summary="Delete a tenant context rule",
)
async def delete_context_rule(
    rule_id: int,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Soft-delete a tenant-specific context rule."""
    await context_rule_service.delete_rule(
        db=db,
        tenant_id=tenant.id,
        rule_id=rule_id,
        deleted_by=current_user.id,
    )
    return SuccessResponse(message="Context rule deleted.")
