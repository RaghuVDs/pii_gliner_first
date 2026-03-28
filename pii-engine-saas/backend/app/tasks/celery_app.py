"""
Celery application configuration for PII Engine SaaS.

Provides the shared Celery app instance used by all task modules.
Configures broker, result backend, serialisation, task routing, and
periodic beat schedules.
"""

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from app.config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# Celery app instance
# ---------------------------------------------------------------------------

celery_app = Celery(
    "pii_engine",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# ---------------------------------------------------------------------------
# Serialisation & general settings
# ---------------------------------------------------------------------------

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Task behaviour
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Result expiry (24 hours)
    result_expires=86400,
)

# ---------------------------------------------------------------------------
# Task routing -- direct tasks to specialised queues
# ---------------------------------------------------------------------------

celery_app.conf.task_routes = {
    # Detection / redaction tasks -> high-priority detection queue
    "app.tasks.detection_tasks.detect_async": {"queue": "detection"},
    "app.tasks.detection_tasks.redact_async": {"queue": "detection"},
    "app.tasks.detection_tasks.batch_detect": {"queue": "detection"},
    # Model training -> dedicated training queue (GPU workers)
    "app.tasks.training_tasks.retrain_model": {"queue": "training"},
    # Everything else -> default queue
    "app.tasks.stats_tasks.*": {"queue": "default"},
    "app.tasks.cleanup_tasks.*": {"queue": "default"},
    "app.tasks.promotion_tasks.*": {"queue": "default"},
}

# ---------------------------------------------------------------------------
# Beat schedule -- periodic tasks
# ---------------------------------------------------------------------------

celery_app.conf.beat_schedule = {
    "aggregate-stats-every-5-min": {
        "task": "app.tasks.stats_tasks.aggregate_detection_stats",
        "schedule": 300.0,  # every 5 minutes
        "options": {"queue": "default"},
    },
    "cleanup-old-jobs-daily-2am": {
        "task": "app.tasks.cleanup_tasks.cleanup_old_jobs",
        "schedule": crontab(hour=2, minute=0),
        "options": {"queue": "default"},
    },
    "auto-promote-check-hourly": {
        "task": "app.tasks.promotion_tasks.auto_promote_pending_rules",
        "schedule": crontab(minute=0),  # every hour on the hour
        "options": {"queue": "default"},
    },
}

# ---------------------------------------------------------------------------
# Auto-discover tasks from all task modules
# ---------------------------------------------------------------------------

celery_app.autodiscover_tasks(
    [
        "app.tasks.detection_tasks",
        "app.tasks.training_tasks",
        "app.tasks.promotion_tasks",
        "app.tasks.stats_tasks",
        "app.tasks.cleanup_tasks",
    ]
)
