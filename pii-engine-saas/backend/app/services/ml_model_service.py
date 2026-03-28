"""
ML model service -- manages LSTM model versions, retraining triggers,
activation, metrics, and MinIO artifact storage.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any

import yaml
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.exceptions import NotFoundError, ValidationError
from app.models.ml_model_version import MLModelVersion

logger = logging.getLogger(__name__)


class MLModelService:
    """Business logic for ML model version management."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # List versions
    # ------------------------------------------------------------------

    async def list_versions(
        self,
        tenant_id: uuid.UUID,
        **_extra,
    ) -> list[MLModelVersion]:
        """Return all model versions for a tenant, ordered newest first.

        Args:
            tenant_id: Owning tenant.

        Returns:
            List of MLModelVersion instances.
        """
        result = await self._db.execute(
            select(MLModelVersion)
            .where(MLModelVersion.tenant_id == tenant_id)
            .order_by(MLModelVersion.version.desc())
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Get active model
    # ------------------------------------------------------------------

    async def get_active_model(
        self,
        tenant_id: uuid.UUID,
        **_extra,
    ) -> MLModelVersion:
        """Return the currently active model version for a tenant.

        Args:
            tenant_id: Owning tenant.

        Returns:
            The active MLModelVersion.

        Raises:
            NotFoundError: If no active model exists.
        """
        result = await self._db.execute(
            select(MLModelVersion).where(
                MLModelVersion.tenant_id == tenant_id,
                MLModelVersion.is_active.is_(True),
            )
        )
        version = result.scalar_one_or_none()
        if version is None:
            raise NotFoundError(
                "No active model version found for this tenant.",
                error_code="NO_ACTIVE_MODEL",
            )
        return version

    # ------------------------------------------------------------------
    # Trigger retrain
    # ------------------------------------------------------------------

    async def trigger_retrain(
        self,
        tenant_id: uuid.UUID,
        epochs: int = 50,
        **_extra,
    ) -> dict[str, Any]:
        """Submit a model retraining task via Celery.

        Args:
            tenant_id: Owning tenant.
            epochs: Number of training epochs.

        Returns:
            Dict with ``task_id`` and ``status``.
        """
        if epochs < 1 or epochs > 500:
            raise ValidationError(
                "epochs must be between 1 and 500.",
                error_code="INVALID_EPOCHS",
            )

        # Lazy import to avoid loading Celery at module level
        try:
            from app.tasks.ml_tasks import retrain_model  # type: ignore[import-untyped]

            task = retrain_model.delay(
                str(tenant_id),
                epochs=epochs,
            )
            task_id = task.id
        except ImportError:
            # Celery not configured -- return a placeholder
            task_id = str(uuid.uuid4())
            logger.warning(
                "Celery tasks module not available; returning placeholder task_id."
            )

        logger.info(
            "Triggered retrain for tenant %s (epochs=%d, task_id=%s).",
            tenant_id,
            epochs,
            task_id,
        )

        return {
            "task_id": task_id,
            "status": "submitted",
            "tenant_id": str(tenant_id),
            "epochs": epochs,
        }

    # ------------------------------------------------------------------
    # Inline retrain (no Celery required)
    # ------------------------------------------------------------------

    async def inline_retrain(
        self,
        tenant_id: uuid.UUID,
        epochs: int | None = None,
        triggered_by: uuid.UUID | None = None,
        **_extra,
    ) -> dict[str, Any]:
        """Run model retraining inline without Celery.

        1. Fetches training examples from MongoDB
        2. Writes them to a temporary YAML file
        3. Runs SelfTrainer.retrain() in a thread pool
        4. Saves the result as a new MLModelVersion in PostgreSQL
        5. Returns the training results

        Args:
            tenant_id: Owning tenant.
            epochs: Number of training epochs. If None, auto-determined.
            triggered_by: User who triggered the retrain.

        Returns:
            Training results dict.
        """
        from app.core.mongodb import mongodb_client

        mongo = mongodb_client.get_database()
        collection = mongo["training_examples"]

        # 1. Fetch all training examples for this tenant from MongoDB
        query = {"tenant_id": str(tenant_id)}
        cursor = collection.find(query)
        examples = []
        async for doc in cursor:
            examples.append({
                "structure": doc.get("structure", ""),
                "label": doc.get("label") or doc.get("entity_type", "UNKNOWN_PII"),
                "length": doc.get("length", len(doc.get("structure", ""))),
                "keywords": doc.get("keywords", []),
                "co_labels": doc.get("co_labels", []),
                "source": doc.get("source", "unknown"),
                "score": doc.get("score", 0.5),
            })

        if not examples:
            return {
                "status": "skipped",
                "reason": "No training examples found in MongoDB for this tenant.",
                "num_examples": 0,
            }

        # 2. Write to a temporary YAML file
        tmp_dir = tempfile.mkdtemp(prefix="pii_train_")
        training_data_path = os.path.join(tmp_dir, "training_data.yaml")
        model_path = os.path.join(tmp_dir, "model.pt")
        vocab_path = os.path.join(tmp_dir, "vocab.yaml")

        training_data = {
            "examples": examples,
            "total_count": len(examples),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(training_data_path, "w", encoding="utf-8") as f:
            yaml.dump(training_data, f, default_flow_style=False, allow_unicode=True)

        # 3. Run SelfTrainer.retrain() in a thread pool
        def _run_training():
            from app.engine.ml.trainer import SelfTrainer
            trainer = SelfTrainer(
                training_data_path=training_data_path,
                model_path=model_path,
                vocab_path=vocab_path,
            )
            return trainer.retrain(epochs=epochs)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _run_training)

        if result.get("status") != "trained":
            # Clean up temp files
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return result

        # 4. Save the result as a new MLModelVersion in PostgreSQL
        model_bytes = b""
        vocab_bytes = b""
        try:
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    model_bytes = f.read()
            if os.path.exists(vocab_path):
                with open(vocab_path, "rb") as f:
                    vocab_bytes = f.read()
        except Exception:
            logger.warning("Failed to read model artifacts from temp dir.", exc_info=True)

        metrics = {
            "num_labels": result.get("num_labels", 0),
            "num_examples": result.get("num_examples", 0),
            "train_size": result.get("train_size"),
            "val_size": result.get("val_size"),
            "val_accuracy": result.get("val_accuracy"),
            "val_weighted_f1": result.get("val_weighted_f1"),
            "metrics_detail": {
                "val_per_label": result.get("val_per_label", {}),
                "val_macro_f1": result.get("val_macro_f1"),
                "val_macro_precision": result.get("val_macro_precision"),
                "val_macro_recall": result.get("val_macro_recall"),
                "train_accuracy": result.get("train_accuracy"),
                "train_weighted_f1": result.get("train_weighted_f1"),
                "best_epoch": result.get("best_epoch"),
                "stopped_epoch": result.get("stopped_epoch"),
                "early_stopped": result.get("early_stopped"),
                "best_val_loss": result.get("best_val_loss"),
                "history_sample": result.get("history_sample", {}),
                "val_confusion_matrix": result.get("val_confusion_matrix", {}),
            },
            "training_trigger": "manual",
        }

        version_record = None
        if model_bytes and vocab_bytes:
            try:
                version_record = await self.save_model_version(
                    tenant_id=tenant_id,
                    metrics=metrics,
                    model_bytes=model_bytes,
                    vocab_bytes=vocab_bytes,
                )
            except Exception:
                logger.error(
                    "Failed to save model version for tenant %s.",
                    tenant_id,
                    exc_info=True,
                )

        # Clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # 5. Return the training results
        result["version_id"] = version_record.id if version_record else None
        result["version_number"] = version_record.version if version_record else None

        logger.info(
            "Inline retrain completed for tenant %s: %d examples, %d labels, "
            "val_f1=%.3f, version=%s",
            tenant_id,
            result.get("num_examples", 0),
            result.get("num_labels", 0),
            result.get("val_weighted_f1", 0),
            version_record.version if version_record else "N/A",
        )

        return result

    # ------------------------------------------------------------------
    # Activate version
    # ------------------------------------------------------------------

    async def activate_version(
        self,
        tenant_id: uuid.UUID,
        version_id: int,
        **_extra,
    ) -> MLModelVersion:
        """Set a specific model version as active, deactivating all others.

        Args:
            tenant_id: Owning tenant.
            version_id: The model version record ID to activate.

        Returns:
            The newly activated MLModelVersion.

        Raises:
            NotFoundError: If the version doesn't exist for this tenant.
        """
        # Verify version exists
        result = await self._db.execute(
            select(MLModelVersion).where(
                MLModelVersion.id == version_id,
                MLModelVersion.tenant_id == tenant_id,
            )
        )
        version = result.scalar_one_or_none()
        if version is None:
            raise NotFoundError(
                "Model version not found.",
                error_code="MODEL_VERSION_NOT_FOUND",
            )

        # Deactivate all versions for this tenant
        await self._db.execute(
            update(MLModelVersion)
            .where(
                MLModelVersion.tenant_id == tenant_id,
                MLModelVersion.is_active.is_(True),
            )
            .values(is_active=False)
        )

        # Activate the target version
        version.is_active = True
        self._db.add(version)
        await self._db.flush()

        logger.info(
            "Activated model version %d (v%d) for tenant %s.",
            version.id,
            version.version,
            tenant_id,
        )
        return version

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def get_metrics(
        self,
        tenant_id: uuid.UUID,
        **_extra,
    ) -> list[dict[str, Any]]:
        """Return metrics summary for all model versions.

        Args:
            tenant_id: Owning tenant.

        Returns:
            List of dicts with version info and metrics.
        """
        versions = await self.list_versions(tenant_id)

        return [
            {
                "id": v.id,
                "version": v.version,
                "is_active": v.is_active,
                "num_labels": v.num_labels,
                "num_examples": v.num_examples,
                "train_size": v.train_size,
                "val_size": v.val_size,
                "val_accuracy": float(v.val_accuracy) if v.val_accuracy else None,
                "val_weighted_f1": float(v.val_weighted_f1) if v.val_weighted_f1 else None,
                "training_trigger": v.training_trigger,
                "trained_at": v.trained_at,
            }
            for v in versions
        ]

    async def get_version_metrics(
        self,
        tenant_id: uuid.UUID,
        version_id: int,
        **_extra,
    ) -> dict[str, Any]:
        """Return detailed metrics for a single model version.

        Args:
            tenant_id: Owning tenant.
            version_id: Model version record ID.

        Returns:
            Dict with full metrics including per-label breakdown.

        Raises:
            NotFoundError: If version not found.
        """
        result = await self._db.execute(
            select(MLModelVersion).where(
                MLModelVersion.id == version_id,
                MLModelVersion.tenant_id == tenant_id,
            )
        )
        version = result.scalar_one_or_none()
        if version is None:
            raise NotFoundError(
                "Model version not found.",
                error_code="MODEL_VERSION_NOT_FOUND",
            )

        return {
            "id": version.id,
            "version": version.version,
            "is_active": version.is_active,
            "num_labels": version.num_labels,
            "num_examples": version.num_examples,
            "train_size": version.train_size,
            "val_size": version.val_size,
            "val_accuracy": float(version.val_accuracy) if version.val_accuracy else None,
            "val_weighted_f1": float(version.val_weighted_f1) if version.val_weighted_f1 else None,
            "metrics_detail": version.metrics_detail or {},
            "model_bucket_path": version.model_bucket_path,
            "vocab_bucket_path": version.vocab_bucket_path,
            "training_trigger": version.training_trigger,
            "trained_at": version.trained_at,
            "created_at": version.created_at,
        }

    # ------------------------------------------------------------------
    # Save model version (upload to MinIO + create PG record)
    # ------------------------------------------------------------------

    async def save_model_version(
        self,
        tenant_id: uuid.UUID,
        metrics: dict[str, Any],
        model_bytes: bytes,
        vocab_bytes: bytes,
        **_extra,
    ) -> MLModelVersion:
        """Upload model artifacts to MinIO and create a version record.

        Args:
            tenant_id: Owning tenant.
            metrics: Training metrics dict (num_labels, num_examples,
                     val_accuracy, val_weighted_f1, metrics_detail, etc.).
            model_bytes: Serialised model weights.
            vocab_bytes: Serialised vocabulary / label mapping.

        Returns:
            The created MLModelVersion record.
        """
        # Determine next version number
        max_version_result = await self._db.execute(
            select(func.max(MLModelVersion.version)).where(
                MLModelVersion.tenant_id == tenant_id
            )
        )
        current_max = max_version_result.scalar() or 0
        next_version = current_max + 1

        # Upload to MinIO
        settings = get_settings()
        bucket = "pii-models"
        model_path = f"tenants/{tenant_id}/models/v{next_version}/model.pt"
        vocab_path = f"tenants/{tenant_id}/models/v{next_version}/vocab.json"

        try:
            from minio import Minio  # type: ignore[import-untyped]

            client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
            )

            # Ensure bucket exists
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)

            # Upload model
            client.put_object(
                bucket,
                model_path,
                io.BytesIO(model_bytes),
                len(model_bytes),
                content_type="application/octet-stream",
            )

            # Upload vocab
            client.put_object(
                bucket,
                vocab_path,
                io.BytesIO(vocab_bytes),
                len(vocab_bytes),
                content_type="application/json",
            )

            logger.info(
                "Uploaded model artifacts for tenant %s v%d to MinIO.",
                tenant_id,
                next_version,
            )
        except ImportError:
            logger.warning("MinIO client not available; skipping artifact upload.")
        except Exception:
            logger.error(
                "Failed to upload model artifacts for tenant %s.",
                tenant_id,
                exc_info=True,
            )

        # Create PG record
        version = MLModelVersion(
            tenant_id=tenant_id,
            version=next_version,
            model_bucket_path=f"{bucket}/{model_path}",
            vocab_bucket_path=f"{bucket}/{vocab_path}",
            num_labels=metrics.get("num_labels", 0),
            num_examples=metrics.get("num_examples", 0),
            train_size=metrics.get("train_size"),
            val_size=metrics.get("val_size"),
            val_accuracy=metrics.get("val_accuracy"),
            val_weighted_f1=metrics.get("val_weighted_f1"),
            metrics_detail=metrics.get("metrics_detail"),
            is_active=False,
            training_trigger=metrics.get("training_trigger", "manual"),
            trained_at=datetime.now(timezone.utc),
        )
        self._db.add(version)
        await self._db.flush()

        logger.info(
            "Created model version %d (v%d) for tenant %s.",
            version.id,
            version.version,
            tenant_id,
        )
        return version


# ---------------------------------------------------------------------------
# Module-level convenience functions (proxy to MLModelService)
# ---------------------------------------------------------------------------

async def list_versions(db=None, **kw):
    db = db or kw.pop("db", None)
    return await MLModelService(db).list_versions(**kw)


async def get_active_model(db=None, **kw):
    db = db or kw.pop("db", None)
    return await MLModelService(db).get_active_model(**kw)


async def get_active(db=None, **kw):
    db = db or kw.pop("db", None)
    return await MLModelService(db).get_active_model(**kw)


async def trigger_retrain(db=None, **kw):
    db = db or kw.pop("db", None)
    return await MLModelService(db).trigger_retrain(**kw)


async def inline_retrain(db=None, **kw):
    db = db or kw.pop("db", None)
    return await MLModelService(db).inline_retrain(**kw)


async def get_retrain_status(db=None, **kw):
    db = db or kw.pop("db", None)
    # Route calls this but class may not have it; delegate as-is
    svc = MLModelService(db)
    if hasattr(svc, "get_retrain_status"):
        return await svc.get_retrain_status(**kw)
    raise NotImplementedError("get_retrain_status not yet implemented")


async def activate_version(db=None, **kw):
    db = db or kw.pop("db", None)
    return await MLModelService(db).activate_version(**kw)


async def get_metrics(db=None, **kw):
    db = db or kw.pop("db", None)
    return await MLModelService(db).get_metrics(**kw)


async def metrics_history(db=None, **kw):
    db = db or kw.pop("db", None)
    return await MLModelService(db).get_metrics(**kw)


async def get_version_metrics(db=None, **kw):
    db = db or kw.pop("db", None)
    return await MLModelService(db).get_version_metrics(**kw)


async def save_model_version(db=None, **kw):
    db = db or kw.pop("db", None)
    return await MLModelService(db).save_model_version(**kw)
