"""
Celery tasks for periodic cleanup of stale data.

- Removes completed detection jobs older than 30 days from MongoDB.
- Removes expired user sessions from PostgreSQL.
- Deactivates expired API keys in PostgreSQL.
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


@celery_app.task(name="app.tasks.cleanup_tasks.cleanup_old_jobs")
def cleanup_old_jobs(retention_days: int = 30) -> Dict[str, Any]:
    """Remove completed detection jobs older than *retention_days* from MongoDB.

    Only jobs with status 'completed' or 'failed' are removed; 'processing'
    jobs are left untouched to avoid data loss.

    Args:
        retention_days: Number of days to retain completed jobs (default 30).

    Returns:
        Dict with the count of deleted documents.
    """
    db = _get_mongo_db()
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    result = db.detection_jobs.delete_many(
        {
            "status": {"$in": ["completed", "failed"]},
            "completed_at": {"$lt": cutoff},
        }
    )

    deleted = result.deleted_count
    logger.info("Cleaned up %d old detection jobs (older than %d days)", deleted, retention_days)
    return {"deleted_jobs": deleted}


@celery_app.task(name="app.tasks.cleanup_tasks.cleanup_expired_sessions")
def cleanup_expired_sessions() -> Dict[str, Any]:
    """Remove expired user sessions from PostgreSQL.

    Deletes rows in ``user_sessions`` where ``expires_at`` is in the past
    or ``revoked_at`` is set.

    Returns:
        Dict with the count of deleted sessions.
    """
    import psycopg2

    pg_url = settings.DATABASE_URL.replace("+asyncpg", "").replace("postgresql+asyncpg", "postgresql")
    conn = psycopg2.connect(pg_url)

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM user_sessions
                WHERE expires_at < NOW()
                   OR revoked_at IS NOT NULL
                """
            )
            deleted = cur.rowcount
        conn.commit()
    finally:
        conn.close()

    logger.info("Cleaned up %d expired user sessions", deleted)
    return {"deleted_sessions": deleted}


@celery_app.task(name="app.tasks.cleanup_tasks.cleanup_expired_api_keys")
def cleanup_expired_api_keys() -> Dict[str, Any]:
    """Deactivate API keys that have passed their expiration date.

    Sets ``is_active = false`` and ``revoked_at = NOW()`` for keys where
    ``expires_at`` is in the past and the key is still active.

    Returns:
        Dict with the count of deactivated keys.
    """
    import psycopg2

    pg_url = settings.DATABASE_URL.replace("+asyncpg", "").replace("postgresql+asyncpg", "postgresql")
    conn = psycopg2.connect(pg_url)

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE api_keys
                SET is_active = false,
                    revoked_at = NOW(),
                    updated_at = NOW()
                WHERE expires_at < NOW()
                  AND is_active = true
                  AND revoked_at IS NULL
                """
            )
            deactivated = cur.rowcount
        conn.commit()
    finally:
        conn.close()

    logger.info("Deactivated %d expired API keys", deactivated)
    return {"deactivated_keys": deactivated}
