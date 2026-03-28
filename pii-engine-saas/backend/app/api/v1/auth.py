"""
Authentication endpoints.

Handles registration, login, token refresh, logout, password reset,
and email verification.  All routes are **public** (no bearer token
required) except ``POST /logout`` which needs a valid access token.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.auth import (
    ForgotPasswordRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
)
from app.schemas.common import ErrorResponse, SuccessResponse
from app.services import auth_service

router = APIRouter(tags=["Authentication"])


# ---------------------------------------------------------------------------
# POST /register -- create user + tenant (public)
# ---------------------------------------------------------------------------
@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    response_model=TokenResponse,
    responses={
        409: {"model": ErrorResponse, "description": "Email already registered"},
    },
    summary="Register a new user and tenant",
)
async def register(
    body: RegisterRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Create a new tenant and its first admin user, returning JWT tokens."""
    return await auth_service.register(
        db=db,
        email=body.email,
        password=body.password,
        full_name=body.full_name,
        org_name=body.org_name,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )


# ---------------------------------------------------------------------------
# POST /login -- email / password -> JWT tokens
# ---------------------------------------------------------------------------
@router.post(
    "/login",
    status_code=status.HTTP_200_OK,
    response_model=TokenResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
    },
    summary="Authenticate with email and password",
)
async def login(
    body: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Validate credentials and return access + refresh tokens."""
    return await auth_service.login(
        db=db,
        email=body.email,
        password=body.password,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )


# ---------------------------------------------------------------------------
# POST /refresh -- refresh token -> new access token
# ---------------------------------------------------------------------------
@router.post(
    "/refresh",
    status_code=status.HTTP_200_OK,
    response_model=TokenResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid or expired refresh token"},
    },
    summary="Refresh access token",
)
async def refresh_token(
    body: RefreshRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Exchange a valid refresh token for a fresh access + refresh pair."""
    return await auth_service.refresh(
        db=db,
        refresh_token=body.refresh_token,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )


# ---------------------------------------------------------------------------
# POST /logout -- revoke refresh token
# ---------------------------------------------------------------------------
@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    response_model=SuccessResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid token"},
    },
    summary="Revoke refresh token (logout)",
)
async def logout(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> SuccessResponse:
    """Revoke the provided refresh token so it can no longer be used."""
    await auth_service.logout(db=db, refresh_token=body.refresh_token)
    return SuccessResponse(message="Successfully logged out.")


# ---------------------------------------------------------------------------
# POST /forgot-password -- initiate password reset
# ---------------------------------------------------------------------------
@router.post(
    "/forgot-password",
    status_code=status.HTTP_200_OK,
    response_model=SuccessResponse,
    summary="Request a password-reset email",
)
async def forgot_password(
    body: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
) -> SuccessResponse:
    """Send a password-reset link to the user's email address.

    Always returns 200 regardless of whether the email exists to prevent
    user enumeration.
    """
    await auth_service.forgot_password(db=db, email=body.email)
    return SuccessResponse(
        message="If the email exists, a password-reset link has been sent."
    )


# ---------------------------------------------------------------------------
# POST /reset-password -- complete password reset
# ---------------------------------------------------------------------------
@router.post(
    "/reset-password",
    status_code=status.HTTP_200_OK,
    response_model=SuccessResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or expired token"},
    },
    summary="Reset password using emailed token",
)
async def reset_password(
    body: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
) -> SuccessResponse:
    """Set a new password using the token received via email."""
    await auth_service.reset_password(
        db=db,
        token=body.token,
        new_password=body.new_password,
    )
    return SuccessResponse(message="Password has been reset successfully.")


# ---------------------------------------------------------------------------
# POST /verify-email -- confirm email token
# ---------------------------------------------------------------------------
@router.post(
    "/verify-email",
    status_code=status.HTTP_200_OK,
    response_model=SuccessResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or expired token"},
    },
    summary="Verify email address",
)
async def verify_email(
    token: str,
    db: AsyncSession = Depends(get_db),
) -> SuccessResponse:
    """Mark the user's email as verified using the token from the
    verification link."""
    await auth_service.verify_email(db=db, token=token)
    return SuccessResponse(message="Email verified successfully.")
