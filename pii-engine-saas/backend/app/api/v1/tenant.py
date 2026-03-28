"""
Tenant (organisation) management endpoints.

Provides routes to inspect and update the current tenant's profile,
trigger onboarding, and view usage statistics.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.v1.dependencies import (
    get_current_tenant,
    get_current_user,
    require_role,
)
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.common import ErrorResponse, SuccessResponse
from app.schemas.tenant import TenantResponse, TenantUpdate, TenantUsageResponse
from app.services import tenant_service

router = APIRouter(tags=["Tenant"])


# ---------------------------------------------------------------------------
# GET / -- get tenant details + settings
# ---------------------------------------------------------------------------
@router.get(
    "/",
    status_code=status.HTTP_200_OK,

    summary="Get tenant details",
)
async def get_tenant(
    tenant: Tenant = Depends(get_current_tenant),
):
    """Return the full profile and settings for the authenticated tenant."""
    return TenantResponse.model_validate(tenant)


# ---------------------------------------------------------------------------
# PATCH / -- update tenant settings (admin)
# ---------------------------------------------------------------------------
@router.patch(
    "/",
    status_code=status.HTTP_200_OK,

    summary="Update tenant settings",
    dependencies=[Depends(require_role("admin"))],
)
async def update_tenant(
    body: TenantUpdate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Update the tenant's name, settings, or quota limits (admin only)."""
    return await tenant_service.update_tenant(
        db=db,
        tenant_id=tenant.id,
        data=body,
        updated_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# POST /onboard -- trigger onboarding (clone defaults)
# ---------------------------------------------------------------------------
@router.post(
    "/onboard",
    status_code=status.HTTP_200_OK,

    responses={
        409: {"model": ErrorResponse, "description": "Already onboarded"},
    },
    summary="Trigger tenant onboarding",
    dependencies=[Depends(require_role("admin"))],
)
async def onboard_tenant(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Run the onboarding workflow: clone system-default PII configs,
    masking rules, and seed data into the tenant's workspace."""
    await tenant_service.onboard(
        db=db,
        tenant_id=tenant.id,
        triggered_by=current_user.id,
    )
    return SuccessResponse(message="Tenant onboarding completed successfully.")


# ---------------------------------------------------------------------------
# GET /usage -- current billing period usage
# ---------------------------------------------------------------------------
@router.get(
    "/usage",
    status_code=status.HTTP_200_OK,

    summary="Get current usage stats",
)
async def get_usage(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Return the tenant's resource-usage snapshot for the current
    billing period (scans, users, API keys)."""
    return await tenant_service.get_usage(
        db=db,
        tenant_id=tenant.id,
    )
