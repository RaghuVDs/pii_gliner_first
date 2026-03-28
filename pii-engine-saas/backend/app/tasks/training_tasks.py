"""
Celery tasks for ML model retraining.

Fetches training data from MongoDB, runs the SelfTrainer, uploads the
resulting model artifacts to MinIO, and records the new model version
in PostgreSQL.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pymongo import MongoClient

from app.config import get_settings
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

settings = get_settings()


def _get_mongo_db():
    """Return a synchronous MongoDB database handle."""
    client = MongoClient(settings.MONGODB_URL)
    return client[settings.MONGODB_DATABASE]


@celery_app.task(
    name="app.tasks.training_tasks.retrain_model",
    bind=True,
    max_retries=1,
    time_limit=3600,  # 1 hour hard limit
    soft_time_limit=3300,  # 55 min soft limit
)
def retrain_model(
    self,
    tenant_id: str,
    epochs: int = 5,
    triggered_by: str = "auto",
) -> Dict[str, Any]:
    """Retrain the tenant-specific LSTM classifier model.

    Steps:
        1. Fetch training data from MongoDB (training_samples collection).
        2. Run the SelfTrainer for the requested number of epochs.
        3. Upload model + vocab artifacts to MinIO.
        4. Record MLModelVersion in PostgreSQL.
        5. Optionally mark the new version as active.

    Args:
        tenant_id: UUID of the tenant.
        epochs: Number of training epochs.
        triggered_by: Who/what triggered the retrain (auto / manual / api).

    Returns:
        Dict with model version info and metrics.
    """
    db = _get_mongo_db()

    # Create a training job document for tracking
    training_job = {
        "tenant_id": tenant_id,
        "status": "processing",
        "epochs": epochs,
        "triggered_by": triggered_by,
        "started_at": datetime.now(timezone.utc),
    }
    job_doc = db.training_jobs.insert_one(training_job)
    job_id = str(job_doc.inserted_id)

    try:
        # 1. Fetch training data
        samples_cursor = db.training_samples.find({"tenant_id": tenant_id})
        samples = list(samples_cursor)

        if not samples:
            msg = f"No training data found for tenant {tenant_id}"
            logger.warning(msg)
            db.training_jobs.update_one(
                {"_id": job_doc.inserted_id},
                {"$set": {"status": "failed", "error_message": msg}},
            )
            return {"status": "failed", "error": msg}

        # 2. Prepare data for SelfTrainer
        texts = [s["text"] for s in samples]
        labels = [s["labels"] for s in samples]

        logger.info(
            "Starting model retrain for tenant %s: %d samples, %d epochs",
            tenant_id,
            len(texts),
            epochs,
        )

        # Import the trainer lazily (heavy ML deps)
        from app.services.self_trainer import SelfTrainer

        trainer = SelfTrainer(tenant_id=tenant_id)
        metrics = trainer.train(texts, labels, epochs=epochs)

        # 3. Upload model artifacts to MinIO
        from app.core.minio_client import get_minio_client

        minio = get_minio_client()
        bucket = f"models-{tenant_id}"

        # Ensure bucket exists
        if not minio.bucket_exists(bucket):
            minio.make_bucket(bucket)

        version = metrics.get("version", 1)
        model_path = f"v{version}/model.pt"
        vocab_path = f"v{version}/vocab.json"

        minio.fput_object(bucket, model_path, trainer.model_file_path)
        minio.fput_object(bucket, vocab_path, trainer.vocab_file_path)

        model_bucket_path = f"{bucket}/{model_path}"
        vocab_bucket_path = f"{bucket}/{vocab_path}"

        # 4. Record in PostgreSQL (sync via psycopg2 for Celery context)
        import psycopg2

        pg_url = settings.DATABASE_URL.replace("+asyncpg", "").replace("postgresql+asyncpg", "postgresql")
        conn = psycopg2.connect(pg_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ml_model_versions
                        (tenant_id, version, model_bucket_path, vocab_bucket_path,
                         num_labels, num_examples, train_size, val_size,
                         val_accuracy, val_weighted_f1, metrics_detail,
                         is_active, training_trigger, trained_at)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb,
                         %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        tenant_id,
                        version,
                        model_bucket_path,
                        vocab_bucket_path,
                        metrics.get("num_labels", 0),
                        metrics.get("num_examples", len(texts)),
                        metrics.get("train_size"),
                        metrics.get("val_size"),
                        metrics.get("val_accuracy"),
                        metrics.get("val_weighted_f1"),
                        __import__("json").dumps(metrics.get("detail", {})),
                        False,  # not auto-activated
                        triggered_by,
                        datetime.now(timezone.utc),
                    ),
                )
                row_id = cur.fetchone()[0]
            conn.commit()
        finally:
            conn.close()

        # 5. Update training job in MongoDB
        db.training_jobs.update_one(
            {"_id": job_doc.inserted_id},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc),
                    "model_version_id": row_id,
                    "metrics": metrics,
                }
            },
        )

        logger.info(
            "Model retrain completed for tenant %s: version=%d, accuracy=%.4f",
            tenant_id,
            version,
            metrics.get("val_accuracy", 0.0),
        )

        return {
            "status": "completed",
            "tenant_id": tenant_id,
            "version": version,
            "model_version_id": row_id,
            "metrics": metrics,
        }

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error("retrain_model failed for tenant %s: %s", tenant_id, error_msg)
        db.training_jobs.update_one(
            {"_id": job_doc.inserted_id},
            {
                "$set": {
                    "status": "failed",
                    "error_message": str(exc),
                    "completed_at": datetime.now(timezone.utc),
                }
            },
        )
        raise self.retry(exc=exc, countdown=60)
