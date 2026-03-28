"""
Authentication service -- registration, login, token management, and
password-reset flows.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from jose import jwt
from passlib.context import CryptContext
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.exceptions import (
    ConflictError,
    NotFoundError,
    UnauthorizedError,
    ValidationError,
)
from app.models.tenant import Tenant
from app.models.user import User, UserSession

logger = logging.getLogger(__name__)

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


def _hash_password(password: str) -> str:
    # bcrypt 4.x requires bytes and enforces 72-byte limit
    import bcrypt as _bc
    salt = _bc.gensalt(rounds=12)
    return _bc.hashpw(password.encode("utf-8")[:72], salt).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    import bcrypt as _bc
    try:
        return _bc.checkpw(plain.encode("utf-8")[:72], hashed.encode("utf-8"))
    except Exception:
        return False


def _hash_token(token: str) -> str:
    """SHA-256 hash used for storing refresh / reset tokens."""
    return hashlib.sha256(token.encode()).hexdigest()


def _slugify(name: str) -> str:
    """Create a URL-safe slug from an organisation name."""
    slug = name.lower().strip()
    slug = "".join(c if c.isalnum() or c == " " else "" for c in slug)
    slug = "-".join(slug.split())
    return slug or f"org-{uuid.uuid4().hex[:8]}"


class AuthService:
    """Handles all authentication and registration business logic."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register(
        self,
        email: str,
        password: str,
        full_name: str,
        org_name: str,
        **_extra,
    ) -> dict:
        """Create a new tenant and its first admin user.

        Args:
            email: User email (must be globally unique).
            password: Plaintext password (will be hashed).
            full_name: Display name.
            org_name: Organisation / tenant name.

        Returns:
            Dict containing ``user_id``, ``tenant_id``, ``access_token``,
            ``refresh_token``, and ``expires_in``.

        Raises:
            ConflictError: If the email is already registered.
        """
        # Check for duplicate email
        existing = await self._db.execute(
            select(User).where(User.email == email)
        )
        if existing.scalar_one_or_none() is not None:
            raise ConflictError(
                "A user with this email already exists.",
                error_code="EMAIL_TAKEN",
            )

        # Create tenant
        slug = _slugify(org_name)
        # Ensure slug uniqueness
        slug_check = await self._db.execute(
            select(Tenant).where(Tenant.slug == slug)
        )
        if slug_check.scalar_one_or_none() is not None:
            slug = f"{slug}-{uuid.uuid4().hex[:6]}"

        tenant = Tenant(
            name=org_name,
            slug=slug,
            plan="free",
        )
        self._db.add(tenant)
        await self._db.flush()

        # Create admin user
        user = User(
            tenant_id=tenant.id,
            email=email,
            password_hash=_hash_password(password),
            full_name=full_name,
            role="admin",
            is_active=True,
        )
        self._db.add(user)
        await self._db.flush()

        # Generate tokens
        tokens = self._create_token_pair(user)
        await self._persist_refresh_session(
            user_id=user.id,
            refresh_token=tokens["refresh_token"],
        )

        logger.info(
            "Registered user %s for tenant %s (slug=%s).",
            user.id,
            tenant.id,
            tenant.slug,
        )

        return {
            "user_id": str(user.id),
            "tenant_id": str(tenant.id),
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_type": "bearer",
            "expires_in": self._settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        }

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    async def login(self, email: str, password: str, **_extra) -> dict:
        """Validate credentials and return a token pair.

        Args:
            email: User email.
            password: Plaintext password.

        Returns:
            Dict with ``access_token``, ``refresh_token``, ``expires_in``.

        Raises:
            UnauthorizedError: If credentials are invalid or user is inactive.
        """
        result = await self._db.execute(
            select(User).where(User.email == email, User.deleted_at.is_(None))
        )
        user = result.scalar_one_or_none()

        if user is None or not _verify_password(password, user.password_hash):
            raise UnauthorizedError(
                "Invalid email or password.",
                error_code="INVALID_CREDENTIALS",
            )

        if not user.is_active:
            raise UnauthorizedError(
                "Account is deactivated. Contact your administrator.",
                error_code="ACCOUNT_DEACTIVATED",
            )

        # Update last login
        user.last_login_at = datetime.now(timezone.utc)
        self._db.add(user)

        tokens = self._create_token_pair(user)
        await self._persist_refresh_session(
            user_id=user.id,
            refresh_token=tokens["refresh_token"],
        )

        logger.info("User %s logged in.", user.id)

        return {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_type": "bearer",
            "expires_in": self._settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        }

    # ------------------------------------------------------------------
    # Refresh token
    # ------------------------------------------------------------------

    async def refresh_token(self, refresh_token: str, **_extra) -> dict:
        """Exchange a valid refresh token for a new access token.

        Args:
            refresh_token: The raw refresh token string.

        Returns:
            Dict with ``access_token``, ``expires_in``.

        Raises:
            UnauthorizedError: If the refresh token is invalid or expired.
        """
        token_hash = _hash_token(refresh_token)

        result = await self._db.execute(
            select(UserSession).where(
                UserSession.refresh_token_hash == token_hash,
                UserSession.revoked_at.is_(None),
                UserSession.expires_at > datetime.now(timezone.utc),
            )
        )
        session = result.scalar_one_or_none()

        if session is None:
            raise UnauthorizedError(
                "Invalid or expired refresh token.",
                error_code="INVALID_REFRESH_TOKEN",
            )

        user_result = await self._db.execute(
            select(User).where(
                User.id == session.user_id,
                User.deleted_at.is_(None),
                User.is_active.is_(True),
            )
        )
        user = user_result.scalar_one_or_none()
        if user is None:
            raise UnauthorizedError(
                "User account no longer active.",
                error_code="ACCOUNT_DEACTIVATED",
            )

        access_token = self._create_access_token(user)

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self._settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        }

    # ------------------------------------------------------------------
    # Logout
    # ------------------------------------------------------------------

    async def logout(self, user_id: uuid.UUID, refresh_token: str, **_extra) -> None:
        """Revoke a refresh token session.

        Args:
            user_id: The authenticated user's ID.
            refresh_token: The raw refresh token to revoke.
        """
        token_hash = _hash_token(refresh_token)
        now = datetime.now(timezone.utc)

        await self._db.execute(
            update(UserSession)
            .where(
                UserSession.user_id == user_id,
                UserSession.refresh_token_hash == token_hash,
                UserSession.revoked_at.is_(None),
            )
            .values(revoked_at=now)
        )
        logger.info("User %s logged out (session revoked).", user_id)

    # ------------------------------------------------------------------
    # Forgot password
    # ------------------------------------------------------------------

    async def forgot_password(self, email: str, **_extra) -> dict:
        """Generate a password-reset token.

        In production this would also trigger an email.  Currently returns
        the reset token directly for testing purposes.

        Args:
            email: User email address.

        Returns:
            Dict with ``message`` and (for dev) ``reset_token``.
        """
        result = await self._db.execute(
            select(User).where(User.email == email, User.deleted_at.is_(None))
        )
        user = result.scalar_one_or_none()

        if user is None:
            # Don't reveal whether the email exists
            return {"message": "If that email is registered, a reset link has been sent."}

        reset_token = secrets.token_urlsafe(48)
        token_hash = _hash_token(reset_token)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        # Store token hash in a session-like record with special purpose
        session = UserSession(
            user_id=user.id,
            refresh_token_hash=f"pwd_reset:{token_hash}",
            expires_at=expires_at,
        )
        self._db.add(session)

        logger.info("Password-reset token generated for user %s.", user.id)

        # TODO: send email via notification service
        return {
            "message": "If that email is registered, a reset link has been sent.",
            "reset_token": reset_token,  # Remove in production
        }

    # ------------------------------------------------------------------
    # Reset password
    # ------------------------------------------------------------------

    async def reset_password(self, token: str, new_password: str, **_extra) -> None:
        """Reset a user's password using a valid reset token.

        Args:
            token: The raw reset token.
            new_password: The new plaintext password.

        Raises:
            UnauthorizedError: If the token is invalid or expired.
        """
        token_hash = _hash_token(token)
        now = datetime.now(timezone.utc)

        result = await self._db.execute(
            select(UserSession).where(
                UserSession.refresh_token_hash == f"pwd_reset:{token_hash}",
                UserSession.revoked_at.is_(None),
                UserSession.expires_at > now,
            )
        )
        session = result.scalar_one_or_none()

        if session is None:
            raise UnauthorizedError(
                "Invalid or expired reset token.",
                error_code="INVALID_RESET_TOKEN",
            )

        # Update password
        user_result = await self._db.execute(
            select(User).where(User.id == session.user_id)
        )
        user = user_result.scalar_one_or_none()
        if user is None:
            raise NotFoundError("User not found.", error_code="USER_NOT_FOUND")

        user.password_hash = _hash_password(new_password)
        self._db.add(user)

        # Revoke the reset token
        session.revoked_at = now
        self._db.add(session)

        logger.info("Password reset completed for user %s.", user.id)

    # ------------------------------------------------------------------
    # Email verification
    # ------------------------------------------------------------------

    async def verify_email(self, token: str, **_extra) -> None:
        """Mark a user's email as verified.

        Args:
            token: Email verification token.

        Raises:
            UnauthorizedError: If the token is invalid or expired.
        """
        token_hash = _hash_token(token)
        now = datetime.now(timezone.utc)

        result = await self._db.execute(
            select(UserSession).where(
                UserSession.refresh_token_hash == f"email_verify:{token_hash}",
                UserSession.revoked_at.is_(None),
                UserSession.expires_at > now,
            )
        )
        session = result.scalar_one_or_none()

        if session is None:
            raise UnauthorizedError(
                "Invalid or expired verification token.",
                error_code="INVALID_VERIFICATION_TOKEN",
            )

        user_result = await self._db.execute(
            select(User).where(User.id == session.user_id)
        )
        user = user_result.scalar_one_or_none()
        if user is None:
            raise NotFoundError("User not found.", error_code="USER_NOT_FOUND")

        user.email_verified_at = now
        self._db.add(user)

        session.revoked_at = now
        self._db.add(session)

        logger.info("Email verified for user %s.", user.id)

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _create_access_token(self, user: User) -> str:
        """Create a signed JWT access token."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user.id),
            "tenant_id": str(user.tenant_id),
            "role": user.role,
            "email": user.email,
            "iat": now,
            "exp": now + timedelta(minutes=self._settings.ACCESS_TOKEN_EXPIRE_MINUTES),
            "type": "access",
        }
        return jwt.encode(
            payload,
            self._settings.JWT_SECRET_KEY,
            algorithm=self._settings.JWT_ALGORITHM,
        )

    def _create_refresh_token(self) -> str:
        """Generate a cryptographically random refresh token."""
        return secrets.token_urlsafe(64)

    def _create_token_pair(self, user: User) -> dict:
        """Generate both access and refresh tokens for a user."""
        return {
            "access_token": self._create_access_token(user),
            "refresh_token": self._create_refresh_token(),
        }

    async def _persist_refresh_session(
        self,
        user_id: uuid.UUID,
        refresh_token: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> UserSession:
        """Store a hashed refresh token in user_sessions."""
        session = UserSession(
            user_id=user_id,
            refresh_token_hash=_hash_token(refresh_token),
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.now(timezone.utc)
            + timedelta(days=self._settings.REFRESH_TOKEN_EXPIRE_DAYS),
        )
        self._db.add(session)
        await self._db.flush()
        return session


# ── Module-level convenience functions ──────────────────────────────────
# Route handlers call ``auth_service.register(db=db, ...)`` which delegates
# to the class methods below.


async def register(db: AsyncSession, *, email: str, password: str,
                   full_name: str, org_name: str, **kw) -> dict:
    return await AuthService(db).register(email=email, password=password,
                                          full_name=full_name, org_name=org_name)


async def login(db: AsyncSession, *, email: str, password: str, **kw) -> dict:
    return await AuthService(db).login(email=email, password=password)


async def refresh(db: AsyncSession, *, refresh_token: str, **kw) -> dict:
    return await AuthService(db).refresh_token(refresh_token=refresh_token)


async def logout(db: AsyncSession, *, refresh_token: str, **kw) -> None:
    # logout needs user_id but the route only passes refresh_token
    # Parse user from the refresh session
    svc = AuthService(db)
    token_hash = _hash_token(refresh_token)
    result = await db.execute(
        select(UserSession).where(UserSession.refresh_token_hash == token_hash)
    )
    session = result.scalar_one_or_none()
    if session:
        await svc.logout(user_id=session.user_id, refresh_token=refresh_token)


async def forgot_password(db: AsyncSession, *, email: str, **kw) -> dict:
    return await AuthService(db).forgot_password(email=email)


async def reset_password(db: AsyncSession, *, token: str, new_password: str, **kw) -> None:
    return await AuthService(db).reset_password(token=token, new_password=new_password)


async def verify_email(db: AsyncSession, *, token: str, **kw) -> None:
    return await AuthService(db).verify_email(token=token)
