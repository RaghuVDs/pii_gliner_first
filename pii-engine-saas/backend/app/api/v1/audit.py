"""
Audit log endpoints.

Provides read-only access to the immutable audit trail stored in MongoDB.
Supports rich filtering and CSV export for compliance reporting.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.dependencies import (
    get_current_tenant,
    get_current_user,
    require_role,
)
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.common import PaginatedResponse
from app.services import audit_service

router = APIRouter(tags=["Audit"])


# ---------------------------------------------------------------------------
# Inline response schemas
# ---------------------------------------------------------------------------

class AuditLogEntry(BaseModel):
    id: str
    tenant_id: str
    user_id: str | None = None
    user_email: str | None = None
    action: str = Field(..., description="e.g. user.login, pii.detect, tenant.update")
    resource_type: str | None = Field(None, description="e.g. user, tenant, api_key, detection")
    resource_id: str | None = None
    details: Dict[str, Any] = Field(default_factory=dict)
    ip_address: str | None = None
    user_agent: str | None = None
    timestamp: str

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# GET /logs -- paginated audit log with filters
# ---------------------------------------------------------------------------
@router.get(
    "/logs",
    status_code=status.HTTP_200_OK,

    summary="List audit logs",
    dependencies=[Depends(require_role("admin"))],
)
async def list_audit_logs(
    tenant: Tenant = Depends(get_current_tenant),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    action: str | None = Query(None, description="Filter by action"),
    resource_type: str | None = Query(None, description="Filter by resource type"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    start_date: date | None = Query(None, description="Start date (inclusive)"),
    end_date: date | None = Query(None, description="End date (inclusive)"),
    sort_order: str = Query("desc", description="Sort by timestamp: asc or desc"),
):
    """Return a paginated, filtered audit log for the current tenant.

    Available filters: action, resource_type, user_id, date range.
    """
    return await audit_service.list_logs(
        tenant_id=tenant.id,
        page=page,
        page_size=page_size,
        action=action,
        resource_type=resource_type,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        sort_order=sort_order,
    )


# ---------------------------------------------------------------------------
# GET /logs/export -- export audit logs as CSV
# ---------------------------------------------------------------------------
@router.get(
    "/logs/export",
    status_code=status.HTTP_200_OK,
    summary="Export audit logs as CSV",
    dependencies=[Depends(require_role("admin"))],
    responses={
        200: {
            "content": {"text/csv": {}},
            "description": "CSV file stream",
        },
    },
)
async def export_audit_logs(
    tenant: Tenant = Depends(get_current_tenant),
    action: str | None = Query(None),
    resource_type: str | None = Query(None),
    user_id: str | None = Query(None),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
):
    """Export audit logs as a streaming CSV download.

    Applies the same filters as the list endpoint.  The response is
    streamed so that very large exports do not exhaust memory.
    """
    csv_stream = audit_service.export_logs_csv(
        tenant_id=tenant.id,
        action=action,
        resource_type=resource_type,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
    )

    return StreamingResponse(
        csv_stream,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=audit_logs_{tenant.slug}.csv",
        },
    )
