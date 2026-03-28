"""
Celery tasks for automatic promotion of pending adaptive-learning rules.

Checks the ``pending_rules`` collection in MongoDB and promotes rules
that exceed the confidence threshold into the PostgreSQL context_rules
(or tenant_context_rules) table.
"""

from __future__ import annotations

import logging
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
    name="app.tasks.promotion_tasks.auto_promote_pending_rules",
    bind=True,
    max_retries=1,
)
def auto_promote_pending_rules(
    self,
    tenant_id: Optional[str] = None,
    threshold: float = 0.85,
) -> Dict[str, Any]:
    """Check pending adaptive-learning rules and promote those above threshold.

    If ``tenant_id`` is None, checks rules for *all* tenants (useful when
    called from the beat schedule).

    Args:
        tenant_id: Optional UUID of a specific tenant to process.
        threshold: Minimum confidence score to promote a rule.

    Returns:
        Dict summarising how many rules were promoted / skipped.
    """
    db = _get_mongo_db()

    query: Dict[str, Any] = {"status": "pending"}
    if tenant_id:
        query["tenant_id"] = tenant_id

    pending_rules = list(db.pending_rules.find(query))

    if not pending_rules:
        logger.info("No pending rules to promote (tenant_id=%s)", tenant_id)
        return {"promoted": 0, "skipped": 0, "total": 0}

    promoted = 0
    skipped = 0

    import psycopg2

    pg_url = settings.DATABASE_URL.replace("+asyncpg", "").replace("postgresql+asyncpg", "postgresql")
    conn = psycopg2.connect(pg_url)

    try:
        with conn.cursor() as cur:
            for rule in pending_rules:
                confidence = rule.get("confidence", 0.0)
                if confidence < threshold:
                    skipped += 1
                    continue

                pii_type_name = rule.get("pii_type_name", "")
                keyword_pattern = rule.get("keyword_pattern", "")
                rule_tenant_id = rule.get("tenant_id")
                is_negative = rule.get("is_negative", False)

                if not pii_type_name or not keyword_pattern:
                    skipped += 1
                    continue

                if rule_tenant_id:
                    # Promote to tenant_context_rules
                    cur.execute(
                        """
                        INSERT INTO tenant_context_rules
                            (tenant_id, pii_type_name, keyword_pattern,
                             is_negative, is_enabled, source, promoted_from_id)
                        VALUES (%s, %s, %s, %s, true, 'auto_promoted', %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (
                            rule_tenant_id,
                            pii_type_name,
                            keyword_pattern,
                            is_negative,
                            rule.get("_id_int"),
                        ),
                    )
                else:
                    # Promote to system context_rules
                    cur.execute(
                        """
                        INSERT INTO context_rules
                            (pii_type_name, keyword_pattern, is_negative,
                             is_system, source)
                        VALUES (%s, %s, %s, false, 'auto_promoted')
                        ON CONFLICT DO NOTHING
                        """,
                        (pii_type_name, keyword_pattern, is_negative),
                    )

                # Mark as promoted in MongoDB
                db.pending_rules.update_one(
                    {"_id": rule["_id"]},
                    {
                        "$set": {
                            "status": "promoted",
                            "promoted_at": datetime.now(timezone.utc),
                        }
                    },
                )
                promoted += 1

        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("auto_promote_pending_rules failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc, countdown=30)
    finally:
        conn.close()

    logger.info(
        "Rule promotion complete: promoted=%d, skipped=%d, total=%d",
        promoted,
        skipped,
        len(pending_rules),
    )

    return {
        "promoted": promoted,
        "skipped": skipped,
        "total": len(pending_rules),
    }
