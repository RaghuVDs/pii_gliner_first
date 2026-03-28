"""Audit log schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
class AuditLogResponse(BaseModel):
    """A single audit-log entry."""

    id: UUID
    user_id: UUID | None = None
    user_email: str | None = None
    action: str = Field(
        ..., description="Action performed (e.g. 'user.login', 'pii_type.update')"
    )
    resource_type: str = Field(
        ..., description="Resource type (e.g. 'user', 'api_key', 'pii_type')"
    )
    resource_id: str | None = Field(
        None, description="Identifier of the affected resource"
    )
    changes: dict[str, Any] | None = Field(
        None, description="Before/after diff of changed fields"
    )
    ip_address: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Filter (query parameters, used as a schema)
# ---------------------------------------------------------------------------
class AuditLogFilter(BaseModel):
    """Query filters for listing audit logs."""

    action: str | None = Field(None, description="Filter by action name")
    resource_type: str | None = Field(None, description="Filter by resource type")
    user_id: UUID | None = Field(None, description="Filter by acting user")
    date_from: datetime | None = Field(None, description="Start of date range (inclusive)")
    date_to: datetime | None = Field(None, description="End of date range (inclusive)")
