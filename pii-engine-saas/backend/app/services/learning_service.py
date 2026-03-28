"""
Learning service -- manages the adaptive learning pipeline: pending rule
review, promotion/rejection, and training data lifecycle.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError, ValidationError
from app.core.mongodb import mongodb_client
from app.models.context_rule import TenantContextRule

logger = logging.getLogger(__name__)


class LearningService:
    """Business logic for the adaptive learning feedback loop."""

    def __init__(
        self,
        db: AsyncSession,
        mongo: AsyncIOMotorDatabase | None = None,
    ) -> None:
        self._db = db
        self._mongo = mongo if mongo is not None else mongodb_client.get_database()

    # ------------------------------------------------------------------
    # Pending rules
    # ------------------------------------------------------------------

    async def list_pending_rules(
        self,
        tenant_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
        **_extra,
    ) -> dict[str, Any]:
        """List pending rules awaiting review for a tenant.

        Args:
            tenant_id: Owning tenant.
            page: 1-indexed page.
            page_size: Items per page.

        Returns:
            Paginated dict with items, total, page, page_size, total_pages.
        """
        query = {
            "tenant_id": str(tenant_id),
            "status": "pending",
        }
        total = await self._mongo["pending_rules"].count_documents(query)

        skip = (page - 1) * page_size
        cursor = (
            self._mongo["pending_rules"]
            .find(query)
            .sort("created_at", -1)
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
    # Pending stats
    # ------------------------------------------------------------------

    async def get_pending_stats(
        self,
        tenant_id: uuid.UUID,
        **_extra,
    ) -> dict[str, Any]:
        """Return summary counts of pending rules for a tenant.

        Args:
            tenant_id: Owning tenant.

        Returns:
            Dict with total_pending, by_entity_type counts,
            oldest_pending_at.
        """
        pipeline = [
            {"$match": {"tenant_id": str(tenant_id), "status": "pending"}},
            {
                "$group": {
                    "_id": "$entity_type",
                    "count": {"$sum": 1},
                    "oldest": {"$min": "$created_at"},
                }
            },
        ]

        by_type: dict[str, int] = {}
        total = 0
        oldest: datetime | None = None

        async for doc in self._mongo["pending_rules"].aggregate(pipeline):
            entity_type = doc["_id"] or "unknown"
            by_type[entity_type] = doc["count"]
            total += doc["count"]
            if oldest is None or (doc["oldest"] and doc["oldest"] < oldest):
                oldest = doc["oldest"]

        return {
            "total_pending": total,
            "by_entity_type": by_type,
            "oldest_pending_at": oldest,
        }

    # ------------------------------------------------------------------
    # Promote a pending rule
    # ------------------------------------------------------------------

    async def promote_rule(
        self,
        tenant_id: uuid.UUID,
        rule_id: str,
        label: str,
        keywords: list[str],
        promoted_by: uuid.UUID,
        **_extra,
    ) -> dict[str, Any]:
        """Promote a pending rule to a tenant context rule in PostgreSQL.

        Updates the MongoDB document status and creates a corresponding
        ``TenantContextRule`` in the relational database.

        Args:
            tenant_id: Owning tenant.
            rule_id: MongoDB document ``_id`` as string.
            label: The PII type name to assign.
            keywords: Context keywords for the new rule.
            promoted_by: User performing the promotion.

        Returns:
            Dict with ``mongo_rule_id`` and ``context_rule_id``.

        Raises:
            NotFoundError: If the pending rule doesn't exist.
        """
        try:
            oid = ObjectId(rule_id)
        except Exception:
            raise ValidationError(
                "Invalid rule_id format.", error_code="INVALID_RULE_ID"
            )

        now = datetime.now(timezone.utc)

        result = await self._mongo["pending_rules"].find_one_and_update(
            {
                "_id": oid,
                "tenant_id": str(tenant_id),
                "status": "pending",
            },
            {
                "$set": {
                    "status": "promoted",
                    "promoted_by": str(promoted_by),
                    "promoted_at": now,
                    "promoted_label": label,
                    "promoted_keywords": keywords,
                }
            },
        )

        if result is None:
            raise NotFoundError(
                "Pending rule not found or already reviewed.",
                error_code="PENDING_RULE_NOT_FOUND",
            )

        # Create context rules in PostgreSQL for each keyword
        created_ids: list[int] = []
        for keyword in keywords:
            ctx_rule = TenantContextRule(
                tenant_id=tenant_id,
                pii_type_name=label,
                keyword_pattern=keyword,
                is_negative=False,
                is_enabled=True,
                source="learning",
                promoted_from_id=None,  # Could store Mongo ObjectId hash
                created_by=promoted_by,
            )
            self._db.add(ctx_rule)
            await self._db.flush()
            created_ids.append(ctx_rule.id)

        logger.info(
            "Promoted pending rule %s -> %s with %d keywords for tenant %s.",
            rule_id,
            label,
            len(keywords),
            tenant_id,
        )

        return {
            "mongo_rule_id": rule_id,
            "context_rule_ids": created_ids,
        }

    # ------------------------------------------------------------------
    # Reject a pending rule
    # ------------------------------------------------------------------

    async def reject_rule(
        self,
        tenant_id: uuid.UUID,
        rule_id: str,
        rejected_by: uuid.UUID,
        **_extra,
    ) -> None:
        """Reject (dismiss) a pending rule.

        Args:
            tenant_id: Owning tenant.
            rule_id: MongoDB document ``_id`` as string.
            rejected_by: User performing the rejection.

        Raises:
            NotFoundError: If the pending rule doesn't exist.
        """
        try:
            oid = ObjectId(rule_id)
        except Exception:
            raise ValidationError(
                "Invalid rule_id format.", error_code="INVALID_RULE_ID"
            )

        result = await self._mongo["pending_rules"].find_one_and_update(
            {
                "_id": oid,
                "tenant_id": str(tenant_id),
                "status": "pending",
            },
            {
                "$set": {
                    "status": "rejected",
                    "rejected_by": str(rejected_by),
                    "rejected_at": datetime.now(timezone.utc),
                }
            },
        )

        if result is None:
            raise NotFoundError(
                "Pending rule not found or already reviewed.",
                error_code="PENDING_RULE_NOT_FOUND",
            )

        logger.info(
            "Rejected pending rule %s for tenant %s.", rule_id, tenant_id
        )

    # ------------------------------------------------------------------
    # Auto-promote
    # ------------------------------------------------------------------

    async def auto_promote(
        self,
        tenant_id: uuid.UUID,
        threshold: int,
        promoted_by: uuid.UUID,
        **_extra,
    ) -> list[dict[str, Any]]:
        """Batch-promote pending rules whose seen_count exceeds a threshold.

        Args:
            tenant_id: Owning tenant.
            threshold: Minimum seen_count to auto-promote.
            promoted_by: User triggering auto-promotion.

        Returns:
            List of promotion result dicts.
        """
        cursor = self._mongo["pending_rules"].find({
            "tenant_id": str(tenant_id),
            "status": "pending",
            "seen_count": {"$gte": threshold},
        })

        promoted = []
        async for doc in cursor:
            rule_id = str(doc["_id"])
            label = doc.get("entity_type", "UNKNOWN_PII")
            keywords = doc.get("keywords", [])
            if not keywords:
                keywords = doc.get("context_words", [])

            try:
                result = await self.promote_rule(
                    tenant_id=tenant_id,
                    rule_id=rule_id,
                    label=label,
                    keywords=keywords if keywords else [label.lower()],
                    promoted_by=promoted_by,
                )
                promoted.append(result)
            except Exception:
                logger.warning(
                    "Failed to auto-promote rule %s for tenant %s.",
                    rule_id,
                    tenant_id,
                    exc_info=True,
                )

        logger.info(
            "Auto-promoted %d rules (threshold=%d) for tenant %s.",
            len(promoted),
            threshold,
            tenant_id,
        )
        return promoted

    # ------------------------------------------------------------------
    # Training data
    # ------------------------------------------------------------------

    async def list_training_data(
        self,
        tenant_id: uuid.UUID,
        page: int = 1,
        page_size: int = 50,
        label_filter: str | None = None,
        entity_type: str | None = None,
        **_extra,
    ) -> dict[str, Any]:
        """List training examples for a tenant.

        Args:
            tenant_id: Owning tenant.
            page: 1-indexed page.
            page_size: Items per page.
            label_filter: Optional filter by entity_type / label.
            entity_type: Alias for label_filter (from API query param).

        Returns:
            Paginated dict.
        """
        effective_filter = label_filter or entity_type
        query: dict[str, Any] = {"tenant_id": str(tenant_id)}
        if effective_filter:
            # Support both 'label' and legacy 'entity_type' field names
            query["$or"] = [
                {"label": effective_filter},
                {"entity_type": effective_filter},
            ]

        total = await self._mongo["training_examples"].count_documents(query)

        skip = (page - 1) * page_size
        cursor = (
            self._mongo["training_examples"]
            .find(query)
            .sort("created_at", -1)
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
    # Training stats
    # ------------------------------------------------------------------

    async def get_training_stats(
        self,
        tenant_id: uuid.UUID,
        **_extra,
    ) -> dict[str, Any]:
        """Return label and source distribution for training examples.

        Args:
            tenant_id: Owning tenant.

        Returns:
            Dict with total_examples, by_label, by_source.
        """
        base_query = {"tenant_id": str(tenant_id)}

        total = await self._mongo["training_examples"].count_documents(base_query)

        # By label -- support both 'label' and legacy 'entity_type' field names
        label_pipeline = [
            {"$match": base_query},
            {
                "$group": {
                    "_id": {"$ifNull": ["$label", {"$ifNull": ["$entity_type", "unknown"]}]},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"count": -1}},
        ]
        by_label: dict[str, int] = {}
        async for doc in self._mongo["training_examples"].aggregate(label_pipeline):
            by_label[doc["_id"] or "unknown"] = doc["count"]

        # By source
        source_pipeline = [
            {"$match": base_query},
            {"$group": {"_id": "$source", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        by_source: dict[str, int] = {}
        async for doc in self._mongo["training_examples"].aggregate(source_pipeline):
            by_source[doc["_id"] or "unknown"] = doc["count"]

        return {
            "total_examples": total,
            "by_label": by_label,
            "by_source": by_source,
        }

    # ------------------------------------------------------------------
    # Delete training example
    # ------------------------------------------------------------------

    async def delete_training_example(
        self,
        tenant_id: uuid.UUID,
        example_id: str,
        **_extra,
    ) -> None:
        """Delete a single training example from MongoDB.

        Args:
            tenant_id: Owning tenant.
            example_id: MongoDB document ``_id`` as string.

        Raises:
            NotFoundError: If example not found.
        """
        try:
            oid = ObjectId(example_id)
        except Exception:
            raise ValidationError(
                "Invalid example_id format.", error_code="INVALID_EXAMPLE_ID"
            )

        result = await self._mongo["training_examples"].delete_one(
            {"_id": oid, "tenant_id": str(tenant_id)}
        )

        if result.deleted_count == 0:
            raise NotFoundError(
                "Training example not found.",
                error_code="TRAINING_EXAMPLE_NOT_FOUND",
            )

        logger.info(
            "Deleted training example %s for tenant %s.",
            example_id,
            tenant_id,
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions (proxy to LearningService)
# ---------------------------------------------------------------------------

async def list_pending_rules(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await LearningService(db, mongo=mongo).list_pending_rules(**kw)


async def get_pending_stats(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await LearningService(db, mongo=mongo).get_pending_stats(**kw)


async def pending_rules_stats(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await LearningService(db, mongo=mongo).get_pending_stats(**kw)


async def promote_rule(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await LearningService(db, mongo=mongo).promote_rule(**kw)


async def reject_rule(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await LearningService(db, mongo=mongo).reject_rule(**kw)


async def auto_promote(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await LearningService(db, mongo=mongo).auto_promote(**kw)


async def list_training_data(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await LearningService(db, mongo=mongo).list_training_data(**kw)


async def get_training_stats(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await LearningService(db, mongo=mongo).get_training_stats(**kw)


async def training_data_stats(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await LearningService(db, mongo=mongo).get_training_stats(**kw)


async def delete_training_example(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await LearningService(db, mongo=mongo).delete_training_example(**kw)
