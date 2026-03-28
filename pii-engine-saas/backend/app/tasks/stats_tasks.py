"""
Celery tasks for detection statistics aggregation and Elasticsearch sync.

These run on a periodic beat schedule (every 5 minutes) to keep analytics
dashboards up to date.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from pymongo import MongoClient

from app.config import get_settings
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

settings = get_settings()


def _get_mongo_db():
    """Return a synchronous MongoDB database handle."""
    client = MongoClient(settings.MONGODB_URL)
    return client[settings.MONGODB_DATABASE]


@celery_app.task(name="app.tasks.stats_tasks.aggregate_detection_stats")
def aggregate_detection_stats() -> Dict[str, Any]:
    """Aggregate recent detection jobs into the detection_stats collection.

    Reads completed detection_jobs from the last 10 minutes and computes
    per-tenant aggregated stats (total scans, entity counts, etc.).

    Returns:
        Dict summarising how many tenant stats were updated.
    """
    db = _get_mongo_db()

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)

    pipeline = [
        {
            "$match": {
                "status": "completed",
                "completed_at": {"$gte": cutoff},
            }
        },
        {
            "$group": {
                "_id": "$tenant_id",
                "total_scans": {"$sum": 1},
                "total_entities_found": {
                    "$sum": {
                        "$cond": [
                            {"$isArray": "$result.entities"},
                            {"$size": "$result.entities"},
                            0,
                        ]
                    }
                },
                "avg_response_ms": {"$avg": "$response_time_ms"},
                "last_scan_at": {"$max": "$completed_at"},
            }
        },
    ]

    results = list(db.detection_jobs.aggregate(pipeline))
    updated_count = 0

    for tenant_stats in results:
        tenant_id = tenant_stats["_id"]
        now = datetime.now(timezone.utc)

        db.detection_stats.update_one(
            {"tenant_id": tenant_id},
            {
                "$inc": {
                    "total_scans": tenant_stats["total_scans"],
                    "total_entities_found": tenant_stats["total_entities_found"],
                },
                "$set": {
                    "avg_response_ms": tenant_stats.get("avg_response_ms"),
                    "last_scan_at": tenant_stats.get("last_scan_at"),
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "tenant_id": tenant_id,
                    "created_at": now,
                },
            },
            upsert=True,
        )
        updated_count += 1

    logger.info("Aggregated detection stats for %d tenants", updated_count)
    return {"tenants_updated": updated_count}


@celery_app.task(name="app.tasks.stats_tasks.sync_to_elasticsearch")
def sync_to_elasticsearch() -> Dict[str, Any]:
    """Sync recent detection records from MongoDB to Elasticsearch.

    Finds detection_jobs completed in the last 10 minutes that have not
    yet been indexed and pushes them to the appropriate ES index.

    Returns:
        Dict summarising how many documents were synced.
    """
    db = _get_mongo_db()

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)

    jobs = list(
        db.detection_jobs.find(
            {
                "status": "completed",
                "completed_at": {"$gte": cutoff},
                "es_indexed": {"$ne": True},
            }
        ).limit(1000)
    )

    if not jobs:
        logger.debug("No new detection jobs to sync to Elasticsearch")
        return {"synced": 0}

    try:
        from elasticsearch import Elasticsearch, helpers

        es = Elasticsearch(settings.ELASTICSEARCH_URL)

        actions = []
        for job in jobs:
            tenant_id = job.get("tenant_id", "unknown")
            actions.append(
                {
                    "_index": f"pii-detections-{tenant_id}",
                    "_source": {
                        "tenant_id": tenant_id,
                        "job_id": str(job["_id"]),
                        "timestamp": job.get("completed_at", datetime.now(timezone.utc)).isoformat(),
                        "result": job.get("result"),
                        "input_char_count": job.get("input_char_count"),
                    },
                }
            )

        if actions:
            helpers.bulk(es, actions, raise_on_error=False)

        # Mark as indexed
        job_ids = [job["_id"] for job in jobs]
        db.detection_jobs.update_many(
            {"_id": {"$in": job_ids}},
            {"$set": {"es_indexed": True}},
        )

        logger.info("Synced %d detection jobs to Elasticsearch", len(actions))
        return {"synced": len(actions)}

    except Exception:
        logger.warning("Failed to sync to Elasticsearch", exc_info=True)
        return {"synced": 0, "error": "elasticsearch_unavailable"}
