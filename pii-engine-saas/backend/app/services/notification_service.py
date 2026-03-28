"""
Notification service -- in-app notifications stored in PostgreSQL
with per-user read tracking and admin broadcast.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.models.notification import Notification
from app.models.user import User

logger = logging.getLogger(__name__)


class NotificationService:
    """Business logic for in-app notification management."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # List notifications (paginated, unread first)
    # ------------------------------------------------------------------

    async def list_notifications(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
        **_extra,
    ) -> dict[str, Any]:
        """Return paginated notifications for a user, unread first.

        Args:
            tenant_id: Owning tenant.
            user_id: Target user.
            page: 1-indexed page.
            page_size: Items per page.

        Returns:
            Paginated dict with items, total, page, page_size,
            total_pages, unread_count.
        """
        base = select(Notification).where(
            Notification.tenant_id == tenant_id,
            Notification.user_id == user_id,
        )

        # Count
        count_q = select(func.count()).select_from(base.subquery())
        total_result = await self._db.execute(count_q)
        total = total_result.scalar() or 0

        # Unread count
        unread_q = select(func.count()).where(
            Notification.tenant_id == tenant_id,
            Notification.user_id == user_id,
            Notification.is_read.is_(False),
        )
        unread_result = await self._db.execute(unread_q)
        unread_count = unread_result.scalar() or 0

        # Fetch page (unread first, then by created_at desc)
        offset = (page - 1) * page_size
        rows = await self._db.execute(
            base.order_by(
                Notification.is_read.asc(),  # unread first
                Notification.created_at.desc(),
            )
            .offset(offset)
            .limit(page_size)
        )
        items = list(rows.scalars().all())

        total_pages = (total + page_size - 1) // page_size if page_size else 0

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "unread_count": unread_count,
        }

    # ------------------------------------------------------------------
    # Mark single notification as read
    # ------------------------------------------------------------------

    async def mark_read(
        self,
        notification_id: int,
        user_id: uuid.UUID,
        **_extra,
    ) -> Notification:
        """Mark a single notification as read.

        Args:
            notification_id: Notification ID.
            user_id: User who owns the notification.

        Returns:
            The updated Notification.

        Raises:
            NotFoundError: If notification not found or doesn't belong to user.
        """
        result = await self._db.execute(
            select(Notification).where(
                Notification.id == notification_id,
                Notification.user_id == user_id,
            )
        )
        notification = result.scalar_one_or_none()
        if notification is None:
            raise NotFoundError(
                "Notification not found.",
                error_code="NOTIFICATION_NOT_FOUND",
            )

        if not notification.is_read:
            notification.is_read = True
            notification.read_at = datetime.now(timezone.utc)
            self._db.add(notification)
            await self._db.flush()

        return notification

    # ------------------------------------------------------------------
    # Mark all as read
    # ------------------------------------------------------------------

    async def mark_all_read(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        **_extra,
    ) -> int:
        """Mark all unread notifications as read for a user.

        Args:
            tenant_id: Owning tenant.
            user_id: Target user.

        Returns:
            Number of notifications updated.
        """
        now = datetime.now(timezone.utc)

        result = await self._db.execute(
            update(Notification)
            .where(
                Notification.tenant_id == tenant_id,
                Notification.user_id == user_id,
                Notification.is_read.is_(False),
            )
            .values(is_read=True, read_at=now)
            .returning(Notification.id)
        )
        updated_ids = list(result.scalars().all())
        await self._db.flush()

        logger.info(
            "Marked %d notifications as read for user %s.",
            len(updated_ids),
            user_id,
        )
        return len(updated_ids)

    # ------------------------------------------------------------------
    # Get unread count
    # ------------------------------------------------------------------

    async def get_unread_count(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        **_extra,
    ) -> int:
        """Return the number of unread notifications for a user.

        Args:
            tenant_id: Owning tenant.
            user_id: Target user.

        Returns:
            Integer unread count.
        """
        result = await self._db.execute(
            select(func.count()).where(
                Notification.tenant_id == tenant_id,
                Notification.user_id == user_id,
                Notification.is_read.is_(False),
            )
        )
        return result.scalar() or 0

    # ------------------------------------------------------------------
    # Create notification
    # ------------------------------------------------------------------

    async def create_notification(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        type: str,
        title: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        **_extra,
    ) -> Notification:
        """Create a notification for a specific user.

        Args:
            tenant_id: Owning tenant.
            user_id: Target user.
            type: Notification type (e.g. 'info', 'warning', 'model_trained').
            title: Short title.
            message: Full message body.
            metadata: Optional arbitrary JSON metadata.

        Returns:
            The created Notification.
        """
        notification = Notification(
            tenant_id=tenant_id,
            user_id=user_id,
            type=type,
            title=title,
            message=message,
            metadata=metadata or {},
        )
        self._db.add(notification)
        await self._db.flush()

        logger.debug(
            "Created notification '%s' for user %s in tenant %s.",
            title,
            user_id,
            tenant_id,
        )
        return notification

    # ------------------------------------------------------------------
    # Notify all admins
    # ------------------------------------------------------------------

    async def notify_admins(
        self,
        tenant_id: uuid.UUID,
        type: str,
        title: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        **_extra,
    ) -> list[Notification]:
        """Create a notification for every admin user in the tenant.

        Args:
            tenant_id: Owning tenant.
            type: Notification type.
            title: Short title.
            message: Full message body.
            metadata: Optional metadata.

        Returns:
            List of created Notification instances.
        """
        # Fetch admin users
        result = await self._db.execute(
            select(User).where(
                User.tenant_id == tenant_id,
                User.role == "admin",
                User.is_active.is_(True),
                User.deleted_at.is_(None),
            )
        )
        admins = list(result.scalars().all())

        notifications = []
        for admin in admins:
            notification = await self.create_notification(
                tenant_id=tenant_id,
                user_id=admin.id,
                type=type,
                title=title,
                message=message,
                metadata=metadata,
            )
            notifications.append(notification)

        logger.info(
            "Notified %d admins in tenant %s: '%s'.",
            len(notifications),
            tenant_id,
            title,
        )
        return notifications


# ---------------------------------------------------------------------------
# Module-level convenience functions (proxy to NotificationService)
# ---------------------------------------------------------------------------

async def list_notifications(db=None, **kw):
    db = db or kw.pop("db", None)
    return await NotificationService(db).list_notifications(**kw)


async def mark_read(db=None, **kw):
    db = db or kw.pop("db", None)
    return await NotificationService(db).mark_read(**kw)


async def mark_as_read(db=None, **kw):
    db = db or kw.pop("db", None)
    return await NotificationService(db).mark_read(**kw)


async def mark_all_read(db=None, **kw):
    db = db or kw.pop("db", None)
    return await NotificationService(db).mark_all_read(**kw)


async def get_unread_count(db=None, **kw):
    db = db or kw.pop("db", None)
    return await NotificationService(db).get_unread_count(**kw)


async def unread_count(db=None, **kw):
    db = db or kw.pop("db", None)
    return await NotificationService(db).get_unread_count(**kw)


async def create_notification(db=None, **kw):
    db = db or kw.pop("db", None)
    return await NotificationService(db).create_notification(**kw)


async def notify_admins(db=None, **kw):
    db = db or kw.pop("db", None)
    return await NotificationService(db).notify_admins(**kw)
