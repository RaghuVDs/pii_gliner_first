"""Authentication and authorization request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    """Payload for new-user / new-tenant registration."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(
        ..., min_length=8, max_length=128, description="Password (8-128 chars)"
    )
    full_name: str = Field(
        ..., min_length=1, max_length=255, description="User full name"
    )
    org_name: str = Field(
        ..., min_length=1, max_length=255, description="Organisation / tenant name"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "alice@example.com",
                "password": "Str0ngP@ss!",
                "full_name": "Alice Smith",
                "org_name": "Acme Corp",
            }
        },
    )

    @field_validator("password")
    @classmethod
    def _password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    """Payload for email/password login."""

    email: EmailStr
    password: str = Field(..., min_length=1)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "alice@example.com",
                "password": "Str0ngP@ss!",
            }
        },
    )


# ---------------------------------------------------------------------------
# Token response
# ---------------------------------------------------------------------------
class TokenResponse(BaseModel):
    """JWT token pair returned after successful authentication."""

    access_token: str
    refresh_token: str | None = None
    token_type: str = Field(default="bearer", description="Always 'bearer'")
    expires_in: int = Field(
        ..., gt=0, description="Access-token lifetime in seconds"
    )
    user_id: str | None = None
    tenant_id: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOi...",
                "refresh_token": "dGhpcyBpcyBh...",
                "token_type": "bearer",
                "expires_in": 3600,
            }
        },
    )


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------
class RefreshRequest(BaseModel):
    """Request to exchange a refresh token for a new access token."""

    refresh_token: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Password flows
# ---------------------------------------------------------------------------
class ForgotPasswordRequest(BaseModel):
    """Initiate a password-reset flow."""

    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """Complete a password-reset using the emailed token."""

    token: str = Field(..., min_length=1, description="Password-reset token")
    new_password: str = Field(..., min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def _password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class ChangePasswordRequest(BaseModel):
    """Change password for the currently authenticated user."""

    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def _password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v
