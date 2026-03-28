"""
Field pattern management endpoints.

Field patterns are used for structured-data PII detection (e.g. column
names, JSON keys, CSV headers).  System patterns are read-only; tenants
can add, update, and soft-delete their own.
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
from app.services import field_pattern_service

router = APIRouter(tags=["Field Patterns"])


# ---------------------------------------------------------------------------
# Inline request / response schemas
# ---------------------------------------------------------------------------

class FieldPatternResponse(BaseModel):
    id: int
    pii_type_name: str
    pattern: str
    pattern_type: str = Field(
        ..., description="One of: exact, prefix, suffix, contains, regex",
    )
    description: str | None = None
    is_system: bool
    is_enabled: bool = True
    sort_order: int = 0
    tenant_id: str | None = None

    model_config = ConfigDict(from_attributes=True)


class FieldPatternCreate(BaseModel):
    pii_type_name: str = Field(..., min_length=1, max_length=100)
    pattern: str = Field(..., min_length=1, description="Match pattern for field names")
    pattern_type: str = Field(
        "contains",
        description="One of: exact, prefix, suffix, contains, regex",
    )
    description: str | None = None
    is_enabled: bool = True
    sort_order: int = Field(0, ge=0)


class FieldPatternUpdate(BaseModel):
    pattern: str | None = Field(None, min_length=1)
    pattern_type: str | None = None
    description: str | None = None
    is_enabled: bool | None = None
    sort_order: int | None = Field(None, ge=0)


# ---------------------------------------------------------------------------
# GET / -- list all field patterns (system + tenant)
# ---------------------------------------------------------------------------
@router.get(
    "/",
    status_code=status.HTTP_200_OK,

    summary="List all field patterns",
)
async def list_field_patterns(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    pii_type_name: str | None = Query(None, description="Filter by PII type name"),
    is_system: bool | None = Query(None, description="Filter system vs tenant patterns"),
):
    """Return all field patterns visible to the tenant (system + tenant-specific)."""
    return await field_pattern_service.list_patterns(
        db=db,
        tenant_id=tenant.id,
        page=page,
        page_size=page_size,
        pii_type_name=pii_type_name,
        is_system=is_system,
    )


# ---------------------------------------------------------------------------
# GET /{pii_type_name} -- get patterns for a specific PII type
# ---------------------------------------------------------------------------
@router.get(
    "/{pii_type_name}",
    status_code=status.HTTP_200_OK,

    summary="Get field patterns for a PII type",
)
async def get_patterns_by_type(
    pii_type_name: str,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Return all field patterns (system + tenant) for the given PII type."""
    return await field_pattern_service.get_patterns_by_type(
        db=db,
        tenant_id=tenant.id,
        pii_type_name=pii_type_name,
    )


# ---------------------------------------------------------------------------
# POST / -- add tenant field pattern
# ---------------------------------------------------------------------------
@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,

    summary="Add a tenant field pattern",
)
async def create_field_pattern(
    body: FieldPatternCreate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Create a new tenant-specific field pattern."""
    return await field_pattern_service.create_pattern(
        db=db,
        tenant_id=tenant.id,
        pii_type_name=body.pii_type_name,
        pattern=body.pattern,
        pattern_type=body.pattern_type,
        description=body.description,
        is_enabled=body.is_enabled,
        sort_order=body.sort_order,
        created_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# PATCH /{pattern_id} -- update tenant field pattern
# ---------------------------------------------------------------------------
@router.patch(
    "/{pattern_id}",
    status_code=status.HTTP_200_OK,

    responses={
        404: {"model": ErrorResponse, "description": "Pattern not found"},
        403: {"model": ErrorResponse, "description": "Cannot modify system patterns"},
    },
    summary="Update a tenant field pattern",
)
async def update_field_pattern(
    pattern_id: int,
    body: FieldPatternUpdate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Update an existing tenant-specific field pattern."""
    return await field_pattern_service.update_pattern(
        db=db,
        tenant_id=tenant.id,
        pattern_id=pattern_id,
        data=body,
        updated_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# DELETE /{pattern_id} -- soft-delete tenant field pattern
# ---------------------------------------------------------------------------
@router.delete(
    "/{pattern_id}",
    status_code=status.HTTP_200_OK,

    responses={
        404: {"model": ErrorResponse, "description": "Pattern not found"},
        403: {"model": ErrorResponse, "description": "Cannot delete system patterns"},
    },
    summary="Delete a tenant field pattern",
)
async def delete_field_pattern(
    pattern_id: int,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Soft-delete a tenant-specific field pattern."""
    await field_pattern_service.delete_pattern(
        db=db,
        tenant_id=tenant.id,
        pattern_id=pattern_id,
        deleted_by=current_user.id,
    )
    return SuccessResponse(message="Field pattern deleted.")
