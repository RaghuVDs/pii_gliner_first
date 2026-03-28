"""
Detection service -- run PII detection / redaction using the tenant's
configured engine, and manage asynchronous detection jobs.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError, TenantQuotaExceeded, ValidationError
from app.core.mongodb import mongodb_client
from app.services.tenant_engine_manager import tenant_engine_manager

logger = logging.getLogger(__name__)


class DetectionService:
    """Business logic for PII detection and redaction operations."""

    def __init__(
        self,
        db: AsyncSession,
        mongo: AsyncIOMotorDatabase | None = None,
    ) -> None:
        self._db = db
        self._mongo = mongo if mongo is not None else mongodb_client.get_database()

    # ------------------------------------------------------------------
    # Synchronous detection
    # ------------------------------------------------------------------

    async def detect(
        self,
        tenant_id: uuid.UUID,
        text: str,
        pii_types_filter: list[str] | None = None,
        **_extra,
    ) -> dict[str, Any]:
        """Run PII detection on the provided text.

        Args:
            tenant_id: Owning tenant.
            text: Input text to scan.
            pii_types_filter: Optional list of PII type names to limit
                              detection to.

        Returns:
            Dict with ``detections`` (list of detection dicts),
            ``stats`` (summary counts).
        """
        engine = await tenant_engine_manager.get_engine(tenant_id, self._db)

        # Run the CPU-bound detection in a thread pool
        loop = asyncio.get_running_loop()
        detections = await loop.run_in_executor(None, engine.detect, text)

        # Apply type filter if provided
        if pii_types_filter:
            filter_set = set(pii_types_filter)
            detections = [d for d in detections if d.label in filter_set]

        # Convert to serialisable dicts
        results = [
            {
                "label": d.label,
                "text": d.text,
                "start": d.start,
                "end": d.end,
                "score": round(float(d.score), 4),
                "source": d.source,
                "instance_id": getattr(d, "instance_id", None),
            }
            for d in detections
        ]

        # Record detection stats and training data asynchronously
        asyncio.create_task(
            self._record_stats(tenant_id, len(text), results)
        )
        asyncio.create_task(
            self._collect_training_data(tenant_id, detections, text)
        )

        stats = self._compute_stats(results)

        return {
            "detections": results,
            "stats": stats,
        }

    # ------------------------------------------------------------------
    # Synchronous redaction
    # ------------------------------------------------------------------

    async def redact(
        self,
        tenant_id: uuid.UUID,
        text: str,
        pii_types_filter: list[str] | None = None,
        **_extra,
    ) -> dict[str, Any]:
        """Run PII detection and redaction on the provided text.

        Args:
            tenant_id: Owning tenant.
            text: Input text to scan and redact.
            pii_types_filter: Optional PII type name filter.

        Returns:
            Dict with ``redacted_text``, ``detections``, ``stats``.
        """
        engine = await tenant_engine_manager.get_engine(tenant_id, self._db)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, engine.redact, text)

        detections = result.detections
        if pii_types_filter:
            filter_set = set(pii_types_filter)
            detections = [d for d in detections if d.label in filter_set]

        det_dicts = [
            {
                "label": d.label,
                "text": d.text,
                "start": d.start,
                "end": d.end,
                "score": round(float(d.score), 4),
                "source": d.source,
                "instance_id": getattr(d, "instance_id", None),
            }
            for d in detections
        ]

        asyncio.create_task(
            self._record_stats(tenant_id, len(text), det_dicts)
        )
        asyncio.create_task(
            self._collect_training_data(tenant_id, detections, text)
        )

        stats = self._compute_stats(det_dicts)

        return {
            "redacted_text": result.redacted_text,
            "detections": det_dicts,
            "stats": stats,
        }

    # ------------------------------------------------------------------
    # Async job management (MongoDB)
    # ------------------------------------------------------------------

    async def create_async_job(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        job_type: str,
        text: str,
        pii_types: list[str] | None = None,
        priority: int = 0,
        **_extra,
    ) -> dict[str, Any]:
        """Create an asynchronous detection job.

        The job document is stored in MongoDB.  A Celery task (or
        similar worker) picks it up for processing.

        Args:
            tenant_id: Owning tenant.
            user_id: Requesting user.
            job_type: 'detect' or 'redact'.
            text: Input text.
            pii_types: Optional PII type filter.
            priority: Job priority (higher = more urgent).

        Returns:
            Dict with ``job_id`` and ``status``.
        """
        if job_type not in ("detect", "redact"):
            raise ValidationError(
                "job_type must be 'detect' or 'redact'.",
                error_code="INVALID_JOB_TYPE",
            )

        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        job_doc = {
            "job_id": job_id,
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
            "job_type": job_type,
            "status": "pending",
            "priority": priority,
            "input_text": text,
            "input_char_count": len(text),
            "pii_types_filter": pii_types,
            "result": None,
            "error": None,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
        }

        await self._mongo["detection_jobs"].insert_one(job_doc)

        logger.info(
            "Created async %s job %s for tenant %s.",
            job_type,
            job_id,
            tenant_id,
        )

        return {"job_id": job_id, "status": "pending"}

    async def get_jobs(
        self,
        tenant_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
        status_filter: str | None = None,
        **_extra,
    ) -> dict[str, Any]:
        """List async detection jobs for a tenant.

        Args:
            tenant_id: Owning tenant.
            page: 1-indexed page.
            page_size: Items per page.
            status_filter: Optional filter by status.

        Returns:
            Paginated dict with items, total, page, page_size, total_pages.
        """
        query: dict[str, Any] = {"tenant_id": str(tenant_id)}
        if status_filter:
            query["status"] = status_filter

        total = await self._mongo["detection_jobs"].count_documents(query)

        skip = (page - 1) * page_size
        cursor = (
            self._mongo["detection_jobs"]
            .find(query, {"input_text": 0})  # Exclude large text field
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

    async def get_job(
        self,
        tenant_id: uuid.UUID,
        job_id: str,
        **_extra,
    ) -> dict[str, Any]:
        """Fetch a single async job.

        Args:
            tenant_id: Owning tenant.
            job_id: The job UUID string.

        Returns:
            The job document dict.

        Raises:
            NotFoundError: If job not found.
        """
        doc = await self._mongo["detection_jobs"].find_one(
            {"job_id": job_id, "tenant_id": str(tenant_id)}
        )
        if doc is None:
            raise NotFoundError(
                "Detection job not found.", error_code="JOB_NOT_FOUND"
            )
        doc["_id"] = str(doc["_id"])
        return doc

    async def cancel_job(
        self,
        tenant_id: uuid.UUID,
        job_id: str,
        **_extra,
    ) -> dict[str, Any]:
        """Cancel a pending async job.

        Only jobs with status 'pending' can be cancelled.

        Args:
            tenant_id: Owning tenant.
            job_id: The job UUID string.

        Returns:
            Dict with ``job_id`` and ``status``.

        Raises:
            NotFoundError: If job not found.
            ValidationError: If job is not in 'pending' status.
        """
        result = await self._mongo["detection_jobs"].find_one_and_update(
            {
                "job_id": job_id,
                "tenant_id": str(tenant_id),
                "status": "pending",
            },
            {
                "$set": {
                    "status": "cancelled",
                    "completed_at": datetime.now(timezone.utc),
                }
            },
        )
        if result is None:
            # Check if it exists but isn't pending
            exists = await self._mongo["detection_jobs"].find_one(
                {"job_id": job_id, "tenant_id": str(tenant_id)}
            )
            if exists is None:
                raise NotFoundError(
                    "Detection job not found.", error_code="JOB_NOT_FOUND"
                )
            raise ValidationError(
                f"Cannot cancel job with status '{exists.get('status')}'.",
                error_code="JOB_NOT_CANCELLABLE",
            )

        logger.info("Cancelled job %s for tenant %s.", job_id, tenant_id)
        return {"job_id": job_id, "status": "cancelled"}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _record_stats(
        self,
        tenant_id: uuid.UUID,
        char_count: int,
        detections: list[dict],
    ) -> None:
        """Record detection statistics in MongoDB.

        Args:
            tenant_id: Owning tenant.
            char_count: Length of input text.
            detections: List of detection result dicts.
        """
        try:
            label_counts: dict[str, int] = {}
            for d in detections:
                label_counts[d["label"]] = label_counts.get(d["label"], 0) + 1

            doc = {
                "tenant_id": str(tenant_id),
                "timestamp": datetime.now(timezone.utc),
                "char_count": char_count,
                "total_detections": len(detections),
                "label_counts": label_counts,
                "period": datetime.now(timezone.utc).strftime("%Y-%m"),
            }
            await self._mongo["detection_stats"].insert_one(doc)
        except Exception:
            logger.warning(
                "Failed to record detection stats for tenant %s.",
                tenant_id,
                exc_info=True,
            )

    async def _collect_training_data(
        self,
        tenant_id: uuid.UUID,
        detections: list,
        text: str,
    ) -> None:
        """Collect high-confidence detections as LSTM training examples.

        Stores PII-safe structure patterns (not actual values) in MongoDB.
        Auto-triggers retrain when enough examples accumulate.
        """
        try:
            import hashlib
            import re
            from datetime import datetime, timezone

            COLLECTION_THRESHOLD = 0.40  # Min score to collect
            MIN_EXAMPLES_TO_TRAIN = 30
            RETRAIN_INTERVAL = 100

            def _value_to_structure(value: str) -> str:
                """Convert value to structure pattern (PII-safe)."""
                out = []
                for ch in value:
                    if ch.isdigit():
                        out.append("N")
                    elif ch.isupper():
                        out.append("A")
                    elif ch.islower():
                        out.append("a")
                    else:
                        out.append(ch)
                return "".join(out)

            def _extract_keywords(text: str, start: int, end: int, pad: int = 100) -> list[str]:
                """Extract context keywords near the detection."""
                win_start = max(0, start - pad)
                win_end = min(len(text), end + pad)
                window = text[win_start:win_end].lower()
                stopwords = {"the", "a", "an", "is", "are", "was", "and", "or", "of", "in", "to", "for", "my", "i"}
                tokens = re.findall(r"\b[a-z]{3,}\b", window)
                return [t for t in tokens if t not in stopwords][:12]

            examples = []
            for d in detections:
                if d.score < COLLECTION_THRESHOLD:
                    continue

                structure = _value_to_structure(d.text)
                keywords = _extract_keywords(text, d.start, d.end)
                co_labels = sorted(set(
                    other.label for other in detections
                    if abs(other.start - d.start) < 500 and other.label != d.label
                ))[:8]

                dedup_hash = hashlib.md5(
                    f"{structure}|{d.label}|{'|'.join(keywords[:3])}".encode()
                ).hexdigest()

                examples.append({
                    "tenant_id": str(tenant_id),
                    "structure": structure,
                    "label": d.label,
                    "length": len(d.text),
                    "keywords": keywords,
                    "co_labels": co_labels,
                    "source": d.source,
                    "score": round(float(d.score), 4),
                    "confidence_type": "supervised" if d.score >= 0.8 else "semi_supervised",
                    "dedup_hash": dedup_hash,
                    "created_at": datetime.now(timezone.utc),
                })

            if not examples:
                return

            # Upsert to avoid duplicates (by dedup_hash per tenant)
            inserted = 0
            for ex in examples:
                try:
                    result = await self._mongo["training_examples"].update_one(
                        {"tenant_id": ex["tenant_id"], "dedup_hash": ex["dedup_hash"]},
                        {"$setOnInsert": ex},
                        upsert=True,
                    )
                    if result.upserted_id:
                        inserted += 1
                except Exception:
                    pass  # Duplicate, skip

            if inserted > 0:
                logger.info(
                    "Collected %d new training examples for tenant %s (from %d detections)",
                    inserted, tenant_id, len(detections),
                )

            # Check if we should auto-retrain
            total = await self._mongo["training_examples"].count_documents(
                {"tenant_id": str(tenant_id)}
            )
            if total >= MIN_EXAMPLES_TO_TRAIN and total % RETRAIN_INTERVAL < len(examples):
                logger.info(
                    "Auto-retrain threshold reached (%d examples) for tenant %s",
                    total, tenant_id,
                )
                # Trigger inline retrain (runs in thread pool to not block)
                asyncio.create_task(
                    self._auto_retrain(str(tenant_id), total)
                )

        except Exception:
            logger.warning(
                "Failed to collect training data for tenant %s",
                tenant_id,
                exc_info=True,
            )

    async def _auto_retrain(self, tenant_id: str, num_examples: int) -> None:
        """Auto-retrain LSTM model when enough training data accumulates."""
        try:
            import os
            import yaml
            from datetime import datetime, timezone

            logger.info("[AUTO-RETRAIN] Starting for tenant %s with %d examples...", tenant_id, num_examples)

            # Fetch training examples from MongoDB
            examples = []
            async for doc in self._mongo["training_examples"].find({"tenant_id": tenant_id}):
                examples.append({
                    "structure": doc["structure"],
                    "label": doc["label"],
                    "length": doc["length"],
                    "keywords": doc["keywords"],
                    "co_labels": doc["co_labels"],
                    "source": doc["source"],
                    "score": doc["score"],
                    "confidence_type": doc.get("confidence_type", "supervised"),
                })

            if len(examples) < 30:
                return

            # Write to temp file for trainer
            train_dir = os.path.join("data", "ml_training")
            os.makedirs(train_dir, exist_ok=True)
            training_path = os.path.join(train_dir, f"training_data_{tenant_id[:8]}.yaml")
            model_path = os.path.join(train_dir, f"model_{tenant_id[:8]}.pt")
            vocab_path = os.path.join(train_dir, f"vocab_{tenant_id[:8]}.yaml")

            with open(training_path, "w") as f:
                yaml.dump({"examples": examples}, f, default_flow_style=False)

            # Train in thread pool
            loop = asyncio.get_running_loop()

            def _train():
                from app.engine.ml.trainer import SelfTrainer
                trainer = SelfTrainer(
                    training_data_path=training_path,
                    model_path=model_path,
                    vocab_path=vocab_path,
                )
                return trainer.retrain(epochs=80)

            result = await loop.run_in_executor(None, _train)

            if result.get("status") == "trained":
                logger.info(
                    "[AUTO-RETRAIN] Success for tenant %s: %d examples, %d labels, "
                    "val_acc=%.3f, val_f1=%.3f",
                    tenant_id, result.get("num_examples", 0),
                    result.get("num_labels", 0),
                    result.get("val_accuracy", 0),
                    result.get("val_weighted_f1", 0),
                )

                # Save model version to PostgreSQL
                try:
                    from app.core.database import _async_session_factory
                    from app.models.ml_model_version import MLModelVersion
                    from sqlalchemy import select, func

                    if _async_session_factory:
                        async with _async_session_factory() as db:
                            # Get next version number
                            max_ver = await db.execute(
                                select(func.max(MLModelVersion.version)).where(
                                    MLModelVersion.tenant_id == uuid.UUID(tenant_id)
                                )
                            )
                            next_ver = (max_ver.scalar() or 0) + 1

                            # Deactivate old versions
                            from sqlalchemy import update
                            await db.execute(
                                update(MLModelVersion)
                                .where(MLModelVersion.tenant_id == uuid.UUID(tenant_id))
                                .values(is_active=False)
                            )

                            version = MLModelVersion(
                                tenant_id=uuid.UUID(tenant_id),
                                version=next_ver,
                                model_bucket_path=model_path,
                                vocab_bucket_path=vocab_path,
                                num_labels=result.get("num_labels", 0),
                                num_examples=result.get("num_examples", 0),
                                train_size=result.get("train_size", 0),
                                val_size=result.get("val_size", 0),
                                val_accuracy=result.get("val_accuracy"),
                                val_weighted_f1=result.get("val_weighted_f1"),
                                metrics_detail=result,
                                is_active=True,
                                training_trigger="auto",
                                trained_at=datetime.now(timezone.utc),
                            )
                            db.add(version)
                            await db.commit()
                            logger.info("[AUTO-RETRAIN] Model v%d saved for tenant %s", next_ver, tenant_id)
                except Exception:
                    logger.warning("[AUTO-RETRAIN] Failed to save model version to DB", exc_info=True)
            else:
                logger.warning("[AUTO-RETRAIN] Training did not complete: %s", result.get("status"))

        except Exception:
            logger.warning("[AUTO-RETRAIN] Failed for tenant %s", tenant_id, exc_info=True)

    @staticmethod
    def _compute_stats(detections: list[dict]) -> dict[str, Any]:
        """Compute summary statistics from a list of detection dicts.

        Args:
            detections: List of detection dicts.

        Returns:
            Dict with total, by_type, by_source counts.
        """
        by_type: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for d in detections:
            by_type[d["label"]] = by_type.get(d["label"], 0) + 1
            src = d.get("source", "unknown")
            by_source[src] = by_source.get(src, 0) + 1

        return {
            "total": len(detections),
            "by_type": by_type,
            "by_source": by_source,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions (proxy to DetectionService)
# ---------------------------------------------------------------------------

async def detect(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await DetectionService(db, mongo=mongo).detect(**kw)


async def redact(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await DetectionService(db, mongo=mongo).redact(**kw)


async def detect_async(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await DetectionService(db, mongo=mongo).create_async_job(**kw)


async def redact_async(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await DetectionService(db, mongo=mongo).create_async_job(**kw)


async def batch_detect(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await DetectionService(db, mongo=mongo).detect(**kw)


async def list_jobs(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await DetectionService(db, mongo=mongo).get_jobs(**kw)


async def get_job(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await DetectionService(db, mongo=mongo).get_job(**kw)


async def cancel_job(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await DetectionService(db, mongo=mongo).cancel_job(**kw)


async def create_async_job(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await DetectionService(db, mongo=mongo).create_async_job(**kw)


async def get_jobs(db=None, **kw):
    db = db or kw.pop("db", None)
    from app.core.mongodb import mongodb_client
    mongo = mongodb_client.get_database()
    return await DetectionService(db, mongo=mongo).get_jobs(**kw)
