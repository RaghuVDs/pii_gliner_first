"""User management schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ---------------------------------------------------------------------------
# Create / invite
# ---------------------------------------------------------------------------
class UserCreate(BaseModel):
    """Admin-initiated user creation (invite) within a tenant."""

    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=255)
    role: str = Field(
        ...,
        pattern=r"^(admin|analyst|viewer)$",
        description="One of: admin, analyst, viewer",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "bob@example.com",
                "full_name": "Bob Jones",
                "role": "analyst",
            }
        },
    )


# ---------------------------------------------------------------------------
# Update (admin)
# ---------------------------------------------------------------------------
class UserUpdate(BaseModel):
    """Admin-level partial update of a user within the same tenant."""

    full_name: str | None = Field(None, min_length=1, max_length=255)
    role: str | None = Field(
        None,
        pattern=r"^(admin|analyst|viewer)$",
        description="One of: admin, analyst, viewer",
    )
    is_active: bool | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "admin",
                "is_active": True,
            }
        },
    )


# ---------------------------------------------------------------------------
# Self-service profile update
# ---------------------------------------------------------------------------
class UserProfileUpdate(BaseModel):
    """Fields the authenticated user can change on their own profile."""

    full_name: str | None = Field(None, min_length=1, max_length=255)
    avatar_url: str | None = Field(None, max_length=2048)
    preferences: dict[str, Any] | None = Field(
        None,
        description="Arbitrary JSON preferences (theme, language, etc.)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "full_name": "Alice S.",
                "preferences": {"theme": "dark", "language": "en"},
            }
        },
    )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
class UserResponse(BaseModel):
    """Public representation of a user record."""

    id: UUID
    tenant_id: UUID
    email: str
    full_name: str
    role: str
    is_active: bool
    last_login_at: datetime | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
