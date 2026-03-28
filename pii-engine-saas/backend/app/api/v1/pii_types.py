"""
PII type configuration endpoints.

System PII types are read-only; tenants can customise their own config
(enable/disable, set thresholds, add aliases) and create custom PII types.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.v1.dependencies import (
    get_current_tenant,
    get_current_user,
    require_role,
)
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.common import ErrorResponse, PaginatedResponse, SuccessResponse
from app.services import pii_type_service

router = APIRouter(tags=["PII Types"])


# ---------------------------------------------------------------------------
# Inline request / response schemas (will live in schemas/ later)
# ---------------------------------------------------------------------------

class PIITypeResponse(BaseModel):
    id: int
    category_id: int
    category_name: str | None = None
    name: str
    display_name: str
    description: str | None = None
    gliner_aliases: list[str] = Field(default_factory=list)
    default_threshold: float
    is_system: bool
    is_sensitive: bool
    compliance_tags: list[str] = Field(default_factory=list)
    # tenant-specific overlay
    is_enabled: bool | None = None
    custom_threshold: float | None = None
    custom_aliases: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)


class PIICategoryResponse(BaseModel):
    id: int
    name: str
    display_name: str
    description: str | None = None
    tier: int
    sort_order: int

    model_config = ConfigDict(from_attributes=True)


class TenantPIIConfigUpdate(BaseModel):
    is_enabled: bool | None = None
    custom_threshold: float | None = Field(None, ge=0.0, le=1.0)
    custom_aliases: list[str] | None = None
    notes: str | None = None


class BatchConfigRequest(BaseModel):
    category: str | None = None
    tier: int | None = None
    pii_type_ids: list[int] | None = None
    is_enabled: bool


class CustomPIITypeCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    category_id: int | None = None
    gliner_aliases: list[str] = Field(default_factory=list)
    default_threshold: float = Field(0.4, ge=0.0, le=1.0)
    compliance_tags: list[str] = Field(default_factory=list)


class CustomPIITypeUpdate(BaseModel):
    display_name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    category_id: int | None = None
    gliner_aliases: list[str] | None = None
    default_threshold: float | None = Field(None, ge=0.0, le=1.0)
    is_enabled: bool | None = None
    compliance_tags: list[str] | None = None


# ---------------------------------------------------------------------------
# GET / -- list all PII types with tenant configs
# ---------------------------------------------------------------------------
@router.get(
    "/",
    status_code=status.HTTP_200_OK,

    summary="List all PII types with tenant config overlays",
)
async def list_pii_types(
    request: Request,
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: str | None = Query(None, description="Search by name or display_name"),
    category: str | None = Query(None, description="Filter by category name"),
    tier: int | None = Query(None, description="Filter by tier"),
    is_enabled: bool | None = Query(None, description="Filter by enabled status"),
):
    """Return a paginated, searchable, filterable list of all PII types
    with the tenant's config overlays merged in."""
    # Try to get tenant_id from auth, fall back to None for unauthenticated listing
    tenant_id = None
    try:
        tenant = await get_current_tenant(request, db=db)
        tenant_id = tenant.id
    except Exception:
        pass
    return await pii_type_service.list_pii_types(
        db=db,
        tenant_id=tenant_id,
        page=page,
        page_size=page_size,
        search=search,
        category=category,
        tier=tier,
        is_enabled=is_enabled,
    )


# ---------------------------------------------------------------------------
# GET /categories -- list categories with tier mapping
# ---------------------------------------------------------------------------
@router.get(
    "/categories",
    status_code=status.HTTP_200_OK,

    summary="List PII categories",
)
async def list_categories(
    db: AsyncSession = Depends(get_db),
):
    """Return all PII categories with their tier assignments."""
    return await pii_type_service.list_categories(db=db)


# ---------------------------------------------------------------------------
# GET /{pii_type_id} -- get single PII type + tenant config
# ---------------------------------------------------------------------------
@router.get(
    "/{pii_type_id}",
    status_code=status.HTTP_200_OK,

    responses={404: {"model": ErrorResponse}},
    summary="Get a single PII type",
)
async def get_pii_type(
    pii_type_id: int,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Return a single PII type with the tenant's config overlay."""
    return await pii_type_service.get_pii_type(
        db=db,
        tenant_id=tenant.id,
        pii_type_id=pii_type_id,
    )


# ---------------------------------------------------------------------------
# PATCH /{pii_type_id}/config -- update tenant config
# ---------------------------------------------------------------------------
@router.patch(
    "/{pii_type_id}/config",
    status_code=status.HTTP_200_OK,

    responses={404: {"model": ErrorResponse}},
    summary="Update tenant PII config for a type",
)
async def update_pii_config(
    pii_type_id: int,
    body: TenantPIIConfigUpdate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Enable/disable a PII type, adjust its threshold, or set custom
    aliases for the current tenant."""
    return await pii_type_service.update_tenant_config(
        db=db,
        tenant_id=tenant.id,
        pii_type_id=pii_type_id,
        data=body,
        updated_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# POST /batch-config -- batch enable/disable
# ---------------------------------------------------------------------------
@router.post(
    "/batch-config",
    status_code=status.HTTP_200_OK,

    summary="Batch enable/disable PII types",
)
async def batch_config(
    body: BatchConfigRequest,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Batch enable or disable PII types by category, tier, or explicit IDs."""
    count = await pii_type_service.batch_config(
        db=db,
        tenant_id=tenant.id,
        category=body.category,
        tier=body.tier,
        pii_type_ids=body.pii_type_ids,
        is_enabled=body.is_enabled,
        updated_by=current_user.id,
    )
    return SuccessResponse(message=f"Updated {count} PII type configs.")


# ---------------------------------------------------------------------------
# POST /custom -- create custom PII type
# ---------------------------------------------------------------------------
@router.post(
    "/custom",
    status_code=status.HTTP_201_CREATED,

    responses={409: {"model": ErrorResponse, "description": "Name already exists"}},
    summary="Create a custom PII type",
)
async def create_custom_pii_type(
    body: CustomPIITypeCreate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Create a new tenant-specific custom PII type."""
    return await pii_type_service.create_custom(
        db=db,
        tenant_id=tenant.id,
        data=body,
        created_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# PATCH /custom/{custom_id} -- update custom PII type
# ---------------------------------------------------------------------------
@router.patch(
    "/custom/{custom_id}",
    status_code=status.HTTP_200_OK,

    responses={404: {"model": ErrorResponse}},
    summary="Update a custom PII type",
)
async def update_custom_pii_type(
    custom_id: int,
    body: CustomPIITypeUpdate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Update an existing tenant-specific custom PII type."""
    return await pii_type_service.update_custom(
        db=db,
        tenant_id=tenant.id,
        custom_id=custom_id,
        data=body,
        updated_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# DELETE /custom/{custom_id} -- soft-delete custom PII type
# ---------------------------------------------------------------------------
@router.delete(
    "/custom/{custom_id}",
    status_code=status.HTTP_200_OK,

    responses={404: {"model": ErrorResponse}},
    summary="Delete a custom PII type",
)
async def delete_custom_pii_type(
    custom_id: int,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Soft-delete a tenant-specific custom PII type."""
    await pii_type_service.delete_custom(
        db=db,
        tenant_id=tenant.id,
        custom_id=custom_id,
        deleted_by=current_user.id,
    )
    return SuccessResponse(message="Custom PII type deleted.")


# ---------------------------------------------------------------------------
# POST /reset-defaults -- reset tenant configs to system defaults
# ---------------------------------------------------------------------------
@router.post(
    "/reset-defaults",
    status_code=status.HTTP_200_OK,

    summary="Reset PII configs to system defaults",
    dependencies=[Depends(require_role("admin"))],
)
async def reset_defaults(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Remove all tenant PII config overrides and revert to system defaults."""
    await pii_type_service.reset_defaults(
        db=db,
        tenant_id=tenant.id,
        reset_by=current_user.id,
    )
    return SuccessResponse(message="PII type configs reset to system defaults.")
