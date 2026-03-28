"""
User management endpoints.

Provides CRUD operations for users within a tenant.  Self-service profile
endpoints (``/me``) require only an authenticated user; admin-level
operations on arbitrary users require the ``admin`` role.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, status
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
from app.schemas.user import UserCreate, UserProfileUpdate, UserResponse, UserUpdate
from app.services import user_service

router = APIRouter(tags=["Users"])


# ---------------------------------------------------------------------------
# GET / -- list users in tenant (admin+)
# ---------------------------------------------------------------------------
@router.get(
    "/",
    status_code=status.HTTP_200_OK,

    summary="List users in tenant",
    dependencies=[Depends(require_role("admin"))],
)
async def list_users(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    search: str | None = Query(None, description="Search by name or email"),
    role: str | None = Query(None, description="Filter by role"),
    is_active: bool | None = Query(None, description="Filter by active status"),
):
    """Return a paginated list of users belonging to the current tenant."""
    return await user_service.list_users(
        db=db,
        tenant_id=tenant.id,
        page=page,
        page_size=page_size,
        search=search,
        role=role,
        is_active=is_active,
    )


# ---------------------------------------------------------------------------
# POST / -- invite user to tenant (admin+)
# ---------------------------------------------------------------------------
@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,

    responses={
        409: {"model": ErrorResponse, "description": "Email already exists"},
    },
    summary="Invite a user to the tenant",
    dependencies=[Depends(require_role("admin"))],
)
async def create_user(
    body: UserCreate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Create (invite) a new user within the current tenant."""
    return await user_service.create_user(
        db=db,
        tenant_id=tenant.id,
        email=body.email,
        full_name=body.full_name,
        role=body.role,
        invited_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# GET /me -- get current user profile
# ---------------------------------------------------------------------------
@router.get(
    "/me",
    status_code=status.HTTP_200_OK,

    summary="Get own profile",
)
async def get_my_profile(
    current_user: User = Depends(get_current_user),
):
    """Return the authenticated user's profile."""
    return UserResponse.model_validate(current_user)


# ---------------------------------------------------------------------------
# PATCH /me -- update own profile
# ---------------------------------------------------------------------------
@router.patch(
    "/me",
    status_code=status.HTTP_200_OK,

    summary="Update own profile",
)
async def update_my_profile(
    body: UserProfileUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update the authenticated user's profile (name, avatar, preferences)."""
    return await user_service.update_profile(
        db=db,
        user_id=current_user.id,
        data=body,
    )


# ---------------------------------------------------------------------------
# GET /{user_id} -- get user details (admin+)
# ---------------------------------------------------------------------------
@router.get(
    "/{user_id}",
    status_code=status.HTTP_200_OK,

    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
    },
    summary="Get user details",
    dependencies=[Depends(require_role("admin"))],
)
async def get_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Return details for a specific user within the tenant."""
    return await user_service.get_user(
        db=db,
        tenant_id=tenant.id,
        user_id=user_id,
    )


# ---------------------------------------------------------------------------
# PATCH /{user_id} -- update user role/status (admin+)
# ---------------------------------------------------------------------------
@router.patch(
    "/{user_id}",
    status_code=status.HTTP_200_OK,

    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
    },
    summary="Update user role or status",
    dependencies=[Depends(require_role("admin"))],
)
async def update_user(
    user_id: uuid.UUID,
    body: UserUpdate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Update a user's role, name, or active status (admin only)."""
    return await user_service.update_user(
        db=db,
        tenant_id=tenant.id,
        user_id=user_id,
        data=body,
        updated_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# DELETE /{user_id} -- soft-delete user (admin+)
# ---------------------------------------------------------------------------
@router.delete(
    "/{user_id}",
    status_code=status.HTTP_200_OK,

    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
    },
    summary="Soft-delete a user",
    dependencies=[Depends(require_role("admin"))],
)
async def delete_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Soft-delete a user within the tenant.  Cannot delete yourself."""
    await user_service.delete_user(
        db=db,
        tenant_id=tenant.id,
        user_id=user_id,
        deleted_by=current_user.id,
    )
    return SuccessResponse(message="User deleted successfully.")
