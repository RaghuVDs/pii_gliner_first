"""
Audit service -- immutable audit logging to MongoDB with querying,
filtering, and CSV export.
"""

from __future__ import annotations

import csv
import io
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.mongodb import mongodb_client

logger = logging.getLogger(__name__)


class AuditService:
    """Business logic for audit trail management."""

    def __init__(
        self,
        mongo: AsyncIOMotorDatabase | None = None,
    ) -> None:
        self._mongo = mongo if mongo is not None else mongodb_client.get_database()

    # ------------------------------------------------------------------
    # Log an audit event
    # ------------------------------------------------------------------

    async def log(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        changes: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        **_extra,
    ) -> str:
        """Insert an immutable audit log entry into MongoDB.

        Args:
            tenant_id: Owning tenant.
            user_id: User who performed the action (None for system actions).
            action: Action name (e.g. 'user.create', 'pii_config.update').
            resource_type: Type of affected resource (e.g. 'user', 'tenant').
            resource_id: Optional ID of the affected resource.
            changes: Optional dict describing what changed (before/after).
            ip_address: Client IP address.
            user_agent: Client User-Agent string.

        Returns:
            The inserted document's ``_id`` as a string.
        """
        doc = {
            "tenant_id": str(tenant_id),
            "user_id": str(user_id) if user_id else None,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "changes": changes or {},
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": datetime.now(timezone.utc),
        }

        result = await self._mongo["audit_logs"].insert_one(doc)

        logger.debug(
            "Audit log: tenant=%s user=%s action=%s resource=%s/%s",
            tenant_id,
            user_id,
            action,
            resource_type,
            resource_id,
        )

        return str(result.inserted_id)

    # ------------------------------------------------------------------
    # List / search audit logs
    # ------------------------------------------------------------------

    async def list_logs(
        self,
        tenant_id: uuid.UUID,
        filters: dict[str, Any] | None = None,
        page: int = 1,
        page_size: int = 50,
        **_extra,
    ) -> dict[str, Any]:
        """Return paginated audit log entries for a tenant.

        Args:
            tenant_id: Owning tenant.
            filters: Optional filter dict supporting:
                - ``user_id``: filter by user
                - ``action``: exact match or prefix with wildcard
                - ``resource_type``: exact match
                - ``date_from``: datetime lower bound
                - ``date_to``: datetime upper bound
            page: 1-indexed page.
            page_size: Items per page.

        Returns:
            Paginated dict with items, total, page, page_size, total_pages.
        """
        query: dict[str, Any] = {"tenant_id": str(tenant_id)}
        filters = filters or {}

        if "user_id" in filters and filters["user_id"]:
            query["user_id"] = str(filters["user_id"])

        if "action" in filters and filters["action"]:
            action = filters["action"]
            if action.endswith("*"):
                # Prefix match
                query["action"] = {"$regex": f"^{action.rstrip('*')}"}
            else:
                query["action"] = action

        if "resource_type" in filters and filters["resource_type"]:
            query["resource_type"] = filters["resource_type"]

        date_range: dict[str, Any] = {}
        if "date_from" in filters and filters["date_from"]:
            date_range["$gte"] = filters["date_from"]
        if "date_to" in filters and filters["date_to"]:
            date_range["$lte"] = filters["date_to"]
        if date_range:
            query["timestamp"] = date_range

        total = await self._mongo["audit_logs"].count_documents(query)

        skip = (page - 1) * page_size
        cursor = (
            self._mongo["audit_logs"]
            .find(query)
            .sort("timestamp", -1)
            .skip(skip)
            .limit(page_size)
        )

        items = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            items.append(doc)

        total_pages = (total + page_size - 1) // page_size if page_size else 0

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        }

    # ------------------------------------------------------------------
    # Export logs as CSV
    # ------------------------------------------------------------------

    async def export_logs(
        self,
        tenant_id: uuid.UUID,
        filters: dict[str, Any] | None = None,
        **_extra,
    ) -> AsyncGenerator[str, None]:
        """Yield CSV rows for audit log export.

        Streams data in chunks to handle large exports without loading
        everything into memory.

        Args:
            tenant_id: Owning tenant.
            filters: Same filter dict as ``list_logs``.

        Yields:
            CSV-formatted strings (header row first, then data rows).
        """
        query: dict[str, Any] = {"tenant_id": str(tenant_id)}
        filters = filters or {}

        if "user_id" in filters and filters["user_id"]:
            query["user_id"] = str(filters["user_id"])
        if "action" in filters and filters["action"]:
            action = filters["action"]
            if action.endswith("*"):
                query["action"] = {"$regex": f"^{action.rstrip('*')}"}
            else:
                query["action"] = action
        if "resource_type" in filters and filters["resource_type"]:
            query["resource_type"] = filters["resource_type"]

        date_range: dict[str, Any] = {}
        if "date_from" in filters and filters["date_from"]:
            date_range["$gte"] = filters["date_from"]
        if "date_to" in filters and filters["date_to"]:
            date_range["$lte"] = filters["date_to"]
        if date_range:
            query["timestamp"] = date_range

        # CSV header
        header_fields = [
            "timestamp",
            "user_id",
            "action",
            "resource_type",
            "resource_id",
            "changes",
            "ip_address",
        ]

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(header_fields)
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate()

        # Stream data
        cursor = (
            self._mongo["audit_logs"]
            .find(query)
            .sort("timestamp", -1)
        )

        batch_size = 100
        batch: list[str] = []

        async for doc in cursor:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow([
                doc.get("timestamp", ""),
                doc.get("user_id", ""),
                doc.get("action", ""),
                doc.get("resource_type", ""),
                doc.get("resource_id", ""),
                str(doc.get("changes", {})),
                doc.get("ip_address", ""),
            ])
            batch.append(buf.getvalue())

            if len(batch) >= batch_size:
                yield "".join(batch)
                batch.clear()

        if batch:
            yield "".join(batch)


# ---------------------------------------------------------------------------
# Module-level convenience functions (proxy to AuditService)
# ---------------------------------------------------------------------------

async def log(**kw):
    return await AuditService().log(**kw)


async def list_logs(**kw):
    return await AuditService().list_logs(**kw)


async def export_logs(**kw):
    return AuditService().export_logs(**kw)


def export_logs_csv(**kw):
    return AuditService().export_logs(**kw)
