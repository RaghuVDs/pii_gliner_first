"""
Async MongoDB client powered by Motor.

Provides connection lifecycle management, typed collection accessors,
and automatic index creation for all PII Engine collections.
"""

from __future__ import annotations

import logging
from typing import Optional

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)
from pymongo import ASCENDING, DESCENDING, IndexModel

from app.config import get_settings

logger = logging.getLogger(__name__)


class MongoDBClient:
    """Manages the Motor async MongoDB connection."""

    def __init__(self) -> None:
        self._client: Optional[AsyncIOMotorClient] = None
        self._database: Optional[AsyncIOMotorDatabase] = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the MongoDB connection and create required indexes."""
        settings = get_settings()
        logger.info("Connecting to MongoDB at %s ...", settings.MONGODB_URL[:40])

        self._client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            maxPoolSize=50,
            minPoolSize=5,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        self._database = self._client[settings.MONGODB_DATABASE]

        # Verify connectivity
        await self._client.admin.command("ping")
        logger.info("MongoDB connection established (db=%s).", settings.MONGODB_DATABASE)

        await self._create_indexes()

    async def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client is not None:
            logger.info("Closing MongoDB connection...")
            self._client.close()
            self._client = None
            self._database = None
            logger.info("MongoDB connection closed.")

    def get_database(self) -> AsyncIOMotorDatabase:
        """Return the active database handle.

        Raises:
            RuntimeError: If the client has not connected yet.
        """
        if self._database is None:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return self._database

    # ── Collection Accessors ─────────────────────────────────────────────

    @property
    def pending_rules(self) -> AsyncIOMotorCollection:
        """Collection for PII detection rules awaiting review."""
        return self.get_database()["pending_rules"]

    @property
    def training_examples(self) -> AsyncIOMotorCollection:
        """Collection for ML training examples (feedback loop)."""
        return self.get_database()["training_examples"]

    @property
    def detection_jobs(self) -> AsyncIOMotorCollection:
        """Collection tracking asynchronous detection jobs."""
        return self.get_database()["detection_jobs"]

    @property
    def detection_stats(self) -> AsyncIOMotorCollection:
        """Collection for aggregated detection statistics."""
        return self.get_database()["detection_stats"]

    @property
    def audit_logs(self) -> AsyncIOMotorCollection:
        """Collection for immutable audit log entries."""
        return self.get_database()["audit_logs"]

    # ── Index Management ─────────────────────────────────────────────────

    async def _create_indexes(self) -> None:
        """Create indexes on all collections to ensure query performance."""
        logger.info("Creating MongoDB indexes...")

        # pending_rules
        await self.pending_rules.create_indexes([
            IndexModel([("tenant_id", ASCENDING), ("status", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("entity_type", ASCENDING)]),
        ])

        # training_examples
        await self.training_examples.create_indexes([
            IndexModel([("tenant_id", ASCENDING), ("entity_type", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("is_validated", ASCENDING)]),
        ])

        # detection_jobs
        await self.detection_jobs.create_indexes([
            IndexModel([("tenant_id", ASCENDING), ("status", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("job_id", ASCENDING)], unique=True),
        ])

        # detection_stats
        await self.detection_stats.create_indexes([
            IndexModel([("tenant_id", ASCENDING), ("period", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
        ])

        # audit_logs
        await self.audit_logs.create_indexes([
            IndexModel([("tenant_id", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("action", ASCENDING)]),
            IndexModel(
                [("timestamp", ASCENDING)],
                expireAfterSeconds=365 * 24 * 3600,  # 1 year TTL
            ),
        ])

        logger.info("MongoDB indexes created successfully.")


# Module-level singleton
mongodb_client = MongoDBClient()
