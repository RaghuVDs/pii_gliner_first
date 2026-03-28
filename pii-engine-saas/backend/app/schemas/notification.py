"""In-app notification schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
class NotificationResponse(BaseModel):
    """A single in-app notification."""

    id: UUID
    type: str = Field(
        ...,
        description="Notification type: info, warning, error, success, system",
    )
    title: str
    message: str
    metadata: dict[str, Any] | None = Field(
        None, description="Optional structured payload (links, resource ids, etc.)"
    )
    is_read: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Unread count
# ---------------------------------------------------------------------------
class UnreadCountResponse(BaseModel):
    """Simple unread-notification counter for the nav badge."""

    count: int = Field(..., ge=0)
