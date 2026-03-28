"""
API key and access-request management endpoints.

Covers the full lifecycle of API keys -- generation, rotation, revocation,
usage tracking -- and the request/approval workflow for new API access.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query, status
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
from app.services import api_key_service as api_management_service

router = APIRouter(tags=["API Management"])


# ---------------------------------------------------------------------------
# Inline request / response schemas
# ---------------------------------------------------------------------------

class APIKeyResponse(BaseModel):
    id: str
    tenant_id: str
    user_id: str
    name: str
    key_prefix: str
    scopes: list[str] = Field(default_factory=list)
    rate_limit_per_min: int = 60
    is_active: bool = True
    expires_at: str | None = None
    last_used_at: str | None = None
    total_requests: int = 0
    created_at: str

    model_config = ConfigDict(from_attributes=True)


class APIKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Friendly key name")
    scopes: list[str] = Field(
        default_factory=lambda: ["detect", "redact"],
        description="Permitted scopes",
    )
    rate_limit_per_min: int = Field(60, ge=1, le=10000)
    expires_in_days: int | None = Field(
        None, ge=1, le=365, description="Key lifetime in days (null = no expiry)",
    )


class APIKeyCreateResponse(BaseModel):
    """Returned only once at creation time -- includes the raw key."""
    id: str
    name: str
    key_prefix: str
    api_key: str = Field(..., description="Full API key (shown only once)")
    scopes: list[str]
    rate_limit_per_min: int
    expires_at: str | None = None
    created_at: str


class APIKeyRotateResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    new_api_key: str = Field(..., description="New API key (shown only once)")
    old_key_revoked: bool = True


class APIKeyUsageResponse(BaseModel):
    key_id: str
    key_name: str
    total_requests: int
    requests_today: int
    requests_this_week: int
    requests_this_month: int
    avg_response_time_ms: float | None = None
    error_rate: float | None = None
    top_endpoints: List[Dict[str, Any]] = Field(default_factory=list)


class APIAccessRequestResponse(BaseModel):
    id: str
    tenant_id: str
    requested_by: str
    requester_name: str | None = None
    use_case: str
    scopes_requested: list[str]
    rate_limit_requested: int | None = None
    status: str
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    review_notes: str | None = None
    created_at: str

    model_config = ConfigDict(from_attributes=True)


class APIAccessRequestCreate(BaseModel):
    use_case: str = Field(..., min_length=10, max_length=2000)
    scopes_requested: list[str] = Field(
        default_factory=lambda: ["detect", "redact"],
    )
    rate_limit_requested: int | None = Field(None, ge=1, le=10000)


class APIAccessRequestReview(BaseModel):
    status: str = Field(..., pattern=r"^(approved|denied)$", description="approved or denied")
    review_notes: str | None = None
    approved_scopes: list[str] | None = None
    approved_rate_limit: int | None = Field(None, ge=1, le=10000)


class AggregateUsageResponse(BaseModel):
    total_requests: int
    total_requests_today: int
    total_requests_this_week: int
    total_requests_this_month: int
    requests_by_key: List[Dict[str, Any]] = Field(default_factory=list)
    requests_by_endpoint: List[Dict[str, Any]] = Field(default_factory=list)
    avg_response_time_ms: float | None = None


# ---------------------------------------------------------------------------
# GET /keys -- list API keys
# ---------------------------------------------------------------------------
@router.get(
    "/keys",
    status_code=status.HTTP_200_OK,

    summary="List API keys",
)
async def list_api_keys(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    is_active: bool | None = Query(None, description="Filter by active status"),
):
    """Return all API keys belonging to the current tenant."""
    return await api_management_service.list_keys(
        db=db,
        tenant_id=tenant.id,
        page=page,
        page_size=page_size,
        is_active=is_active,
    )


# ---------------------------------------------------------------------------
# POST /keys -- generate new API key
# ---------------------------------------------------------------------------
@router.post(
    "/keys",
    status_code=status.HTTP_201_CREATED,

    responses={
        429: {"model": ErrorResponse, "description": "API key limit reached"},
    },
    summary="Generate a new API key",
)
async def create_api_key(
    body: APIKeyCreateRequest,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Generate a new API key for the tenant.  The raw key value is only
    returned in this response and cannot be retrieved afterwards."""
    return await api_management_service.create_key(
        db=db,
        tenant_id=tenant.id,
        user_id=current_user.id,
        name=body.name,
        scopes=body.scopes,
        rate_limit_per_min=body.rate_limit_per_min,
        expires_in_days=body.expires_in_days,
    )


# ---------------------------------------------------------------------------
# DELETE /keys/{key_id} -- revoke API key
# ---------------------------------------------------------------------------
@router.delete(
    "/keys/{key_id}",
    status_code=status.HTTP_200_OK,

    responses={404: {"model": ErrorResponse}},
    summary="Revoke an API key",
)
async def revoke_api_key(
    key_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Permanently revoke an API key.  It will immediately stop working."""
    await api_management_service.revoke_key(
        db=db,
        tenant_id=tenant.id,
        key_id=key_id,
        revoked_by=current_user.id,
    )
    return SuccessResponse(message="API key revoked.")


# ---------------------------------------------------------------------------
# POST /keys/{key_id}/rotate -- rotate API key
# ---------------------------------------------------------------------------
@router.post(
    "/keys/{key_id}/rotate",
    status_code=status.HTTP_200_OK,

    responses={404: {"model": ErrorResponse}},
    summary="Rotate an API key",
)
async def rotate_api_key(
    key_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Rotate an API key: revoke the existing secret and issue a new one,
    preserving the key's metadata and scopes."""
    return await api_management_service.rotate_key(
        db=db,
        tenant_id=tenant.id,
        key_id=key_id,
        rotated_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# GET /keys/{key_id}/usage -- usage stats for a key
# ---------------------------------------------------------------------------
@router.get(
    "/keys/{key_id}/usage",
    status_code=status.HTTP_200_OK,

    responses={404: {"model": ErrorResponse}},
    summary="API key usage stats",
)
async def get_key_usage(
    key_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Return usage statistics for a specific API key."""
    return await api_management_service.get_key_usage(
        db=db,
        tenant_id=tenant.id,
        key_id=key_id,
    )


# ---------------------------------------------------------------------------
# GET /requests -- list API access requests
# ---------------------------------------------------------------------------
@router.get(
    "/requests",
    status_code=status.HTTP_200_OK,

    summary="List API access requests",
)
async def list_access_requests(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    request_status: str | None = Query(
        None, alias="status", description="Filter: pending, approved, denied",
    ),
):
    """Return a paginated list of API access requests for the tenant."""
    return await api_management_service.list_access_requests(
        db=db,
        tenant_id=tenant.id,
        page=page,
        page_size=page_size,
        status_filter=request_status,
    )


# ---------------------------------------------------------------------------
# POST /requests -- create new API access request
# ---------------------------------------------------------------------------
@router.post(
    "/requests",
    status_code=status.HTTP_201_CREATED,

    summary="Create API access request",
)
async def create_access_request(
    body: APIAccessRequestCreate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Submit a new API access request for review by a tenant admin."""
    return await api_management_service.create_access_request(
        db=db,
        tenant_id=tenant.id,
        requested_by=current_user.id,
        use_case=body.use_case,
        scopes_requested=body.scopes_requested,
        rate_limit_requested=body.rate_limit_requested,
    )


# ---------------------------------------------------------------------------
# PATCH /requests/{request_id} -- approve/deny API access request
# ---------------------------------------------------------------------------
@router.patch(
    "/requests/{request_id}",
    status_code=status.HTTP_200_OK,

    responses={
        404: {"model": ErrorResponse, "description": "Request not found"},
        409: {"model": ErrorResponse, "description": "Request already reviewed"},
    },
    summary="Review API access request",
    dependencies=[Depends(require_role("admin"))],
)
async def review_access_request(
    request_id: uuid.UUID,
    body: APIAccessRequestReview,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Approve or deny an API access request.  Approving may auto-generate
    an API key for the requester."""
    return await api_management_service.review_access_request(
        db=db,
        tenant_id=tenant.id,
        request_id=request_id,
        status=body.status,
        review_notes=body.review_notes,
        approved_scopes=body.approved_scopes,
        approved_rate_limit=body.approved_rate_limit,
        reviewed_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# GET /usage -- aggregate API usage analytics
# ---------------------------------------------------------------------------
@router.get(
    "/usage",
    status_code=status.HTTP_200_OK,

    summary="Aggregate API usage",
)
async def aggregate_usage(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    start_date: date | None = Query(None, description="Start date (inclusive)"),
    end_date: date | None = Query(None, description="End date (inclusive)"),
):
    """Return aggregate API usage analytics across all keys for the tenant."""
    return await api_management_service.aggregate_usage(
        db=db,
        tenant_id=tenant.id,
        start_date=start_date,
        end_date=end_date,
    )
