"""
Regex rule management endpoints.

Manages system-level and tenant-level regex rules used during PII
detection.  System rules are read-only; tenants can add, update, and
soft-delete their own rules.
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
from app.services import regex_rule_service

router = APIRouter(tags=["Regex Rules"])


# ---------------------------------------------------------------------------
# Inline request / response schemas
# ---------------------------------------------------------------------------

class RegexRuleResponse(BaseModel):
    id: int
    pii_type_name: str
    pattern: str
    description: str | None = None
    sort_order: int = 0
    is_system: bool
    is_enabled: bool = True
    tenant_id: str | None = None

    model_config = ConfigDict(from_attributes=True)


class TenantRegexRuleCreate(BaseModel):
    pii_type_name: str = Field(..., min_length=1, max_length=100)
    pattern: str = Field(..., min_length=1, description="Regex pattern")
    description: str | None = None
    is_enabled: bool = True
    sort_order: int = Field(0, ge=0)


class TenantRegexRuleUpdate(BaseModel):
    pattern: str | None = Field(None, min_length=1)
    description: str | None = None
    is_enabled: bool | None = None
    sort_order: int | None = Field(None, ge=0)


class RegexTestRequest(BaseModel):
    pattern: str = Field(..., min_length=1, description="Regex pattern to test")
    text: str = Field(..., min_length=1, description="Sample text to match against")


class RegexTestMatch(BaseModel):
    match: str
    start: int
    end: int


class RegexTestResponse(BaseModel):
    pattern: str
    matches: List[RegexTestMatch]
    match_count: int


# ---------------------------------------------------------------------------
# GET / -- list all regex rules (system + tenant)
# ---------------------------------------------------------------------------
@router.get(
    "/",
    status_code=status.HTTP_200_OK,

    summary="List all regex rules",
)
async def list_regex_rules(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    pii_type_name: str | None = Query(None, description="Filter by PII type name"),
    is_system: bool | None = Query(None, description="Filter system vs tenant rules"),
):
    """Return all regex rules visible to the tenant (system + tenant-specific),
    optionally filtered by PII type name."""
    return await regex_rule_service.list_rules(
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

    summary="Get regex rules for a PII type",
)
async def get_rules_by_type(
    pii_type_name: str,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Return all regex rules (system + tenant) for the given PII type."""
    return await regex_rule_service.get_rules_by_type(
        db=db,
        tenant_id=tenant.id,
        pii_type_name=pii_type_name,
    )


# ---------------------------------------------------------------------------
# POST / -- add tenant regex rule
# ---------------------------------------------------------------------------
@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,

    summary="Add a tenant regex rule",
)
async def create_regex_rule(
    body: TenantRegexRuleCreate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Create a new tenant-specific regex rule."""
    return await regex_rule_service.create_rule(
        db=db,
        tenant_id=tenant.id,
        pii_type_name=body.pii_type_name,
        pattern=body.pattern,
        description=body.description,
        is_enabled=body.is_enabled,
        sort_order=body.sort_order,
        created_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# PATCH /{rule_id} -- update tenant regex rule
# ---------------------------------------------------------------------------
@router.patch(
    "/{rule_id}",
    status_code=status.HTTP_200_OK,

    responses={
        404: {"model": ErrorResponse, "description": "Rule not found"},
        403: {"model": ErrorResponse, "description": "Cannot modify system rules"},
    },
    summary="Update a tenant regex rule",
)
async def update_regex_rule(
    rule_id: int,
    body: TenantRegexRuleUpdate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Update an existing tenant-specific regex rule."""
    return await regex_rule_service.update_rule(
        db=db,
        tenant_id=tenant.id,
        rule_id=rule_id,
        data=body,
        updated_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# DELETE /{rule_id} -- soft-delete tenant regex rule
# ---------------------------------------------------------------------------
@router.delete(
    "/{rule_id}",
    status_code=status.HTTP_200_OK,

    responses={
        404: {"model": ErrorResponse, "description": "Rule not found"},
        403: {"model": ErrorResponse, "description": "Cannot delete system rules"},
    },
    summary="Delete a tenant regex rule",
)
async def delete_regex_rule(
    rule_id: int,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Soft-delete a tenant-specific regex rule."""
    await regex_rule_service.delete_rule(
        db=db,
        tenant_id=tenant.id,
        rule_id=rule_id,
        deleted_by=current_user.id,
    )
    return SuccessResponse(message="Regex rule deleted.")


# ---------------------------------------------------------------------------
# POST /test -- test a regex pattern against sample text
# ---------------------------------------------------------------------------
@router.post(
    "/test",
    status_code=status.HTTP_200_OK,

    summary="Test a regex pattern",
)
async def test_regex(
    body: RegexTestRequest,
):
    """Test a regex pattern against sample text and return all matches.
    This is a stateless utility endpoint that does not persist anything."""
    return await regex_rule_service.test_pattern(
        pattern=body.pattern,
        text=body.text,
    )
