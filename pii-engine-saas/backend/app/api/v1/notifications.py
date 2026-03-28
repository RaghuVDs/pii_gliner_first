"""
Notification endpoints.

Provides in-app notification management: listing, marking as read,
and unread counts.  Notifications are stored in MongoDB.
"""

from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.dependencies import (
    get_current_tenant,
    get_current_user,
)
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.common import PaginatedResponse, SuccessResponse
from app.services import notification_service

router = APIRouter(tags=["Notifications"])


# ---------------------------------------------------------------------------
# Inline response schemas
# ---------------------------------------------------------------------------

class NotificationResponse(BaseModel):
    id: str
    tenant_id: str
    user_id: str
    title: str
    message: str
    category: str = Field(
        "info",
        description="Notification category: info, warning, error, success",
    )
    action_url: str | None = Field(
        None, description="Optional deep-link to relevant resource",
    )
    is_read: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    read_at: str | None = None

    model_config = ConfigDict(from_attributes=True)


class UnreadCountResponse(BaseModel):
    count: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# GET / -- list notifications (unread first, paginated)
# ---------------------------------------------------------------------------
@router.get(
    "/",
    status_code=status.HTTP_200_OK,

    summary="List notifications",
)
async def list_notifications(
    request: Request,
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    is_read: bool | None = Query(None, description="Filter by read status"),
    category: str | None = Query(None, description="Filter by category"),
):
    """Return the current user's notifications. Returns empty list if not authenticated."""
    # Gracefully handle missing auth - return empty list
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        return []
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        return []
    return await notification_service.list_notifications(db=db,
        tenant_id=tenant_id,
        user_id=user_id,
        page=page,
        page_size=page_size,
        is_read=is_read,
        category=category,
    )


# ---------------------------------------------------------------------------
# PATCH /{notification_id}/read -- mark as read
# ---------------------------------------------------------------------------
@router.patch(
    "/{notification_id}/read",
    status_code=status.HTTP_200_OK,

    summary="Mark notification as read",
)
async def mark_as_read(
    notification_id: str,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Mark a single notification as read."""
    return await notification_service.mark_as_read(db=db, 
        tenant_id=tenant.id,
        user_id=current_user.id,
        notification_id=notification_id,
    )


# ---------------------------------------------------------------------------
# POST /mark-all-read -- mark all as read
# ---------------------------------------------------------------------------
@router.post(
    "/mark-all-read",
    status_code=status.HTTP_200_OK,

    summary="Mark all notifications as read",
)
async def mark_all_read(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Mark all of the current user's unread notifications as read."""
    count = await notification_service.mark_all_read(db=db, 
        tenant_id=tenant.id,
        user_id=current_user.id,
    )
    return SuccessResponse(message=f"Marked {count} notifications as read.")


# ---------------------------------------------------------------------------
# GET /unread-count -- count of unread notifications
# ---------------------------------------------------------------------------
@router.get(
    "/unread-count",
    status_code=status.HTTP_200_OK,

    summary="Unread notification count",
)
async def unread_count(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Return the count of unread notifications for the current user."""
    count = await notification_service.unread_count(db=db, 
        tenant_id=tenant.id,
        user_id=current_user.id,
    )
    return UnreadCountResponse(count=count)
