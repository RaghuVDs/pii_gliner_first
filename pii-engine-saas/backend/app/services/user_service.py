"""
User management service -- CRUD for tenant users, profile management,
and password changes.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

from passlib.context import CryptContext
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    UnauthorizedError,
    ValidationError,
)
from app.models.user import User

logger = logging.getLogger(__name__)

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserService:
    """Business logic for user management within a tenant."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # List users (paginated)
    # ------------------------------------------------------------------

    async def list_users(
        self,
        tenant_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
        role_filter: str | None = None,
        **_extra,
    ) -> dict:
        """Return a paginated list of users for a tenant.

        Args:
            tenant_id: Owning tenant.
            page: 1-indexed page number.
            page_size: Items per page.
            role_filter: Optional filter by role (admin, analyst, viewer).

        Returns:
            Dict with ``items``, ``total``, ``page``, ``page_size``,
            ``total_pages``.
        """
        base = select(User).where(
            User.tenant_id == tenant_id,
            User.deleted_at.is_(None),
        )
        count_q = select(func.count()).select_from(
            base.subquery()
        )

        if role_filter:
            base = base.where(User.role == role_filter)
            count_q = select(func.count()).select_from(
                base.subquery()
            )

        total_result = await self._db.execute(count_q)
        total = total_result.scalar() or 0

        offset = (page - 1) * page_size
        rows = await self._db.execute(
            base.order_by(User.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        users = list(rows.scalars().all())

        total_pages = (total + page_size - 1) // page_size if page_size else 0

        return {
            "items": users,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        }

    # ------------------------------------------------------------------
    # Get single user
    # ------------------------------------------------------------------

    async def get_user(
        self,
        user_id: uuid.UUID,
        tenant_id: uuid.UUID,
        **_extra,
    ) -> User:
        """Fetch a single user within a tenant.

        Args:
            user_id: Target user ID.
            tenant_id: Tenant the user must belong to.

        Returns:
            The User ORM instance.

        Raises:
            NotFoundError: If user not found or doesn't belong to tenant.
        """
        result = await self._db.execute(
            select(User).where(
                User.id == user_id,
                User.tenant_id == tenant_id,
                User.deleted_at.is_(None),
            )
        )
        user = result.scalar_one_or_none()
        if user is None:
            raise NotFoundError("User not found.", error_code="USER_NOT_FOUND")
        return user

    # ------------------------------------------------------------------
    # Create / invite user
    # ------------------------------------------------------------------

    async def create_user(
        self,
        tenant_id: uuid.UUID,
        email: str,
        full_name: str,
        role: str = "viewer",
        **_extra,
    ) -> dict:
        """Create (invite) a new user within a tenant.

        Generates a temporary password.  In production an invitation email
        would be sent instead.

        Args:
            tenant_id: Owning tenant.
            email: New user email (must be globally unique).
            full_name: Display name.
            role: One of admin, analyst, viewer.

        Returns:
            Dict with the created ``user`` and a ``temp_password``.

        Raises:
            ConflictError: If the email is already taken.
        """
        existing = await self._db.execute(
            select(User).where(User.email == email)
        )
        if existing.scalar_one_or_none() is not None:
            raise ConflictError(
                "A user with this email already exists.",
                error_code="EMAIL_TAKEN",
            )

        temp_password = secrets.token_urlsafe(12)

        user = User(
            tenant_id=tenant_id,
            email=email,
            password_hash=_pwd_ctx.hash(temp_password),
            full_name=full_name,
            role=role,
            is_active=True,
        )
        self._db.add(user)
        await self._db.flush()

        logger.info(
            "Created user %s (%s) in tenant %s.",
            user.id,
            email,
            tenant_id,
        )

        return {
            "user": user,
            "temp_password": temp_password,  # Send via email in production
        }

    # ------------------------------------------------------------------
    # Update user (admin)
    # ------------------------------------------------------------------

    async def update_user(
        self,
        user_id: uuid.UUID,
        tenant_id: uuid.UUID,
        data: dict[str, Any],
        **_extra,
    ) -> User:
        """Update user fields (admin-level).

        Args:
            user_id: Target user.
            tenant_id: Tenant the user must belong to.
            data: Dict of fields to update (full_name, role, is_active).

        Returns:
            The updated User instance.

        Raises:
            NotFoundError: If user doesn't exist in the tenant.
        """
        user = await self.get_user(user_id, tenant_id)

        allowed_fields = {"full_name", "role", "is_active"}
        for key, value in data.items():
            if key in allowed_fields and value is not None:
                setattr(user, key, value)

        self._db.add(user)
        await self._db.flush()

        logger.info("Updated user %s fields: %s.", user_id, list(data.keys()))
        return user

    # ------------------------------------------------------------------
    # Soft-delete user
    # ------------------------------------------------------------------

    async def delete_user(
        self,
        user_id: uuid.UUID,
        tenant_id: uuid.UUID,
        **_extra,
    ) -> None:
        """Soft-delete a user within a tenant.

        Args:
            user_id: Target user.
            tenant_id: Tenant the user belongs to.

        Raises:
            NotFoundError: If user not found.
        """
        user = await self.get_user(user_id, tenant_id)
        user.deleted_at = datetime.now(timezone.utc)
        user.is_active = False
        self._db.add(user)
        await self._db.flush()

        logger.info("Soft-deleted user %s from tenant %s.", user_id, tenant_id)

    # ------------------------------------------------------------------
    # Profile (self-service)
    # ------------------------------------------------------------------

    async def get_profile(self, user_id: uuid.UUID, **_extra) -> User:
        """Fetch the current user's own profile.

        Args:
            user_id: Authenticated user ID.

        Returns:
            User instance.

        Raises:
            NotFoundError: If the user no longer exists.
        """
        result = await self._db.execute(
            select(User).where(
                User.id == user_id,
                User.deleted_at.is_(None),
            )
        )
        user = result.scalar_one_or_none()
        if user is None:
            raise NotFoundError("User not found.", error_code="USER_NOT_FOUND")
        return user

    async def update_profile(
        self,
        user_id: uuid.UUID,
        data: dict[str, Any],
        **_extra,
    ) -> User:
        """Update the authenticated user's own profile.

        Args:
            user_id: Authenticated user ID.
            data: Dict of fields to update (full_name, avatar_url,
                  preferences).

        Returns:
            Updated User instance.
        """
        user = await self.get_profile(user_id)

        allowed_fields = {"full_name", "avatar_url", "preferences"}
        for key, value in data.items():
            if key in allowed_fields and value is not None:
                setattr(user, key, value)

        self._db.add(user)
        await self._db.flush()

        logger.info("User %s updated their profile.", user_id)
        return user

    # ------------------------------------------------------------------
    # Change password
    # ------------------------------------------------------------------

    async def change_password(
        self,
        user_id: uuid.UUID,
        current_password: str,
        new_password: str,
        **_extra,
    ) -> None:
        """Change the authenticated user's password.

        Args:
            user_id: Authenticated user ID.
            current_password: Current plaintext password.
            new_password: New plaintext password.

        Raises:
            UnauthorizedError: If the current password is wrong.
            ValidationError: If the new password is the same as current.
        """
        user = await self.get_profile(user_id)

        if not _pwd_ctx.verify(current_password, user.password_hash):
            raise UnauthorizedError(
                "Current password is incorrect.",
                error_code="INVALID_PASSWORD",
            )

        if current_password == new_password:
            raise ValidationError(
                "New password must differ from the current password.",
                error_code="SAME_PASSWORD",
            )

        user.password_hash = _pwd_ctx.hash(new_password)
        self._db.add(user)
        await self._db.flush()

        logger.info("User %s changed their password.", user_id)


# ---------------------------------------------------------------------------
# Module-level convenience functions (proxy to UserService)
# ---------------------------------------------------------------------------

async def list_users(db=None, **kw):
    db = db or kw.pop("db", None)
    return await UserService(db).list_users(**kw)


async def get_user(db=None, **kw):
    db = db or kw.pop("db", None)
    return await UserService(db).get_user(**kw)


async def create_user(db=None, **kw):
    db = db or kw.pop("db", None)
    return await UserService(db).create_user(**kw)


async def update_user(db=None, **kw):
    db = db or kw.pop("db", None)
    return await UserService(db).update_user(**kw)


async def delete_user(db=None, **kw):
    db = db or kw.pop("db", None)
    return await UserService(db).delete_user(**kw)


async def get_profile(db=None, **kw):
    db = db or kw.pop("db", None)
    return await UserService(db).get_profile(**kw)


async def update_profile(db=None, **kw):
    db = db or kw.pop("db", None)
    return await UserService(db).update_profile(**kw)


async def change_password(db=None, **kw):
    db = db or kw.pop("db", None)
    return await UserService(db).change_password(**kw)
