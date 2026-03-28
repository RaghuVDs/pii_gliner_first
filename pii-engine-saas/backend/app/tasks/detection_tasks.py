"""
Celery tasks for asynchronous PII detection and redaction.

Each task updates the corresponding job document in MongoDB through its
lifecycle (pending -> processing -> completed / failed), publishes a
completion event via Redis pub/sub, and indexes results in Elasticsearch.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis
from bson import ObjectId
from pymongo import MongoClient

from app.config import get_settings
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

settings = get_settings()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_mongo_db():
    """Return a synchronous MongoDB database handle (for use inside Celery workers)."""
    client = MongoClient(settings.MONGODB_URL)
    return client[settings.MONGODB_DATABASE]


def _get_redis():
    """Return a synchronous Redis client."""
    return redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


def _update_job_status(
    db,
    job_id: str,
    status: str,
    result: Any = None,
    error_message: str | None = None,
) -> None:
    """Update a detection job document in MongoDB."""
    update_fields: Dict[str, Any] = {
        "status": status,
        "updated_at": datetime.now(timezone.utc),
    }
    if status == "completed":
        update_fields["completed_at"] = datetime.now(timezone.utc)
        update_fields["result"] = result
    if error_message:
        update_fields["error_message"] = error_message

    db.detection_jobs.update_one(
        {"_id": ObjectId(job_id)},
        {"$set": update_fields},
    )


def _publish_completion(job_id: str, tenant_id: str, status: str) -> None:
    """Publish a job_completed event on the Redis pub/sub channel."""
    r = _get_redis()
    r.publish(
        f"tenant:{tenant_id}:jobs",
        json.dumps({"event": "job_completed", "job_id": job_id, "status": status}),
    )


def _index_in_elasticsearch(
    tenant_id: str,
    job_id: str,
    result: Any,
) -> None:
    """Index detection results in Elasticsearch for analytics/search."""
    try:
        from elasticsearch import Elasticsearch

        es = Elasticsearch(settings.ELASTICSEARCH_URL)
        doc = {
            "tenant_id": tenant_id,
            "job_id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }
        es.index(index=f"pii-detections-{tenant_id}", document=doc)
    except Exception:
        logger.warning("Failed to index detection result in Elasticsearch", exc_info=True)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@celery_app.task(name="app.tasks.detection_tasks.detect_async", bind=True, max_retries=2)
def detect_async(
    self,
    tenant_id: str,
    job_id: str,
    text: str,
    pii_types_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run PII detection on a single text and persist results.

    Args:
        tenant_id: UUID of the tenant.
        job_id: MongoDB ObjectId string for the detection job.
        text: The input text to scan.
        pii_types_filter: Optional list of PII type names to restrict detection.

    Returns:
        Dict containing detected entities.
    """
    db = _get_mongo_db()

    try:
        # Mark as processing
        _update_job_status(db, job_id, "processing")

        # Run the detection engine (imported lazily to avoid heavy imports at module level)
        from app.services.tenant_engine_manager import tenant_engine_manager

        engine = tenant_engine_manager.get_engine_sync(tenant_id)
        detections = engine.detect(text)
        if pii_types_filter:
            detections = [d for d in detections if d.label in pii_types_filter]
        result = {
            "detections": [d.to_dict() for d in detections],
            "detection_count": len(detections),
            "pii_types_found": list(set(d.label for d in detections)),
        }

        # Persist result
        _update_job_status(db, job_id, "completed", result=result)
        _publish_completion(job_id, tenant_id, "completed")
        _index_in_elasticsearch(tenant_id, job_id, result)

        return result

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error("detect_async failed for job %s: %s", job_id, error_msg)
        _update_job_status(db, job_id, "failed", error_message=str(exc))
        _publish_completion(job_id, tenant_id, "failed")
        raise self.retry(exc=exc, countdown=10)


@celery_app.task(name="app.tasks.detection_tasks.redact_async", bind=True, max_retries=2)
def redact_async(
    self,
    tenant_id: str,
    job_id: str,
    text: str,
    pii_types_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run PII redaction on a single text and persist results.

    Args:
        tenant_id: UUID of the tenant.
        job_id: MongoDB ObjectId string for the redaction job.
        text: The input text to redact.
        pii_types_filter: Optional list of PII type names to restrict redaction.

    Returns:
        Dict containing the redacted text and entity metadata.
    """
    db = _get_mongo_db()

    try:
        _update_job_status(db, job_id, "processing")

        from app.services.tenant_engine_manager import tenant_engine_manager

        engine = tenant_engine_manager.get_engine_sync(tenant_id)
        redaction_result = engine.redact(text)
        detections = redaction_result.detections
        if pii_types_filter:
            detections = [d for d in detections if d.label in pii_types_filter]
        result = {
            "detections": [d.to_dict() for d in detections],
            "redacted_text": redaction_result.redacted_text,
            "detection_count": len(detections),
            "pii_types_found": list(set(d.label for d in detections)),
        }

        _update_job_status(db, job_id, "completed", result=result)
        _publish_completion(job_id, tenant_id, "completed")
        _index_in_elasticsearch(tenant_id, job_id, result)

        return result

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error("redact_async failed for job %s: %s", job_id, error_msg)
        _update_job_status(db, job_id, "failed", error_message=str(exc))
        _publish_completion(job_id, tenant_id, "failed")
        raise self.retry(exc=exc, countdown=10)


@celery_app.task(name="app.tasks.detection_tasks.batch_detect", bind=True, max_retries=2)
def batch_detect(
    self,
    tenant_id: str,
    job_id: str,
    texts: List[str],
    pii_types_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run PII detection on multiple texts and persist aggregated results.

    Args:
        tenant_id: UUID of the tenant.
        job_id: MongoDB ObjectId string for the batch job.
        texts: List of input texts to scan.
        pii_types_filter: Optional list of PII type names to restrict detection.

    Returns:
        Dict containing a list of per-text detection results.
    """
    db = _get_mongo_db()

    try:
        _update_job_status(db, job_id, "processing")

        from app.services.tenant_engine_manager import tenant_engine_manager

        engine = tenant_engine_manager.get_engine_sync(tenant_id)

        results = []
        for idx, text in enumerate(texts):
            try:
                det = engine.detect(text, pii_types_filter=pii_types_filter)
                results.append({"index": idx, "status": "ok", "result": det})
            except Exception as text_exc:
                results.append({"index": idx, "status": "error", "error": str(text_exc)})

        batch_result = {
            "total": len(texts),
            "succeeded": sum(1 for r in results if r["status"] == "ok"),
            "failed": sum(1 for r in results if r["status"] == "error"),
            "results": results,
        }

        _update_job_status(db, job_id, "completed", result=batch_result)
        _publish_completion(job_id, tenant_id, "completed")
        _index_in_elasticsearch(tenant_id, job_id, batch_result)

        return batch_result

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error("batch_detect failed for job %s: %s", job_id, error_msg)
        _update_job_status(db, job_id, "failed", error_message=str(exc))
        _publish_completion(job_id, tenant_id, "failed")
        raise self.retry(exc=exc, countdown=10)
