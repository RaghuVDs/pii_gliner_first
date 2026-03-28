"""
Analytics service -- dashboard overview, PII type/source breakdowns,
timeline aggregations, and heatmaps powered by Elasticsearch and MongoDB.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.elasticsearch_client import elastic_client
from app.core.exceptions import ValidationError
from app.core.mongodb import mongodb_client

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Business logic for detection analytics and dashboards."""

    def __init__(
        self,
        mongo: AsyncIOMotorDatabase | None = None,
    ) -> None:
        self._mongo = mongo if mongo is not None else mongodb_client.get_database()
        self._es = elastic_client

    # ------------------------------------------------------------------
    # Dashboard overview
    # ------------------------------------------------------------------

    async def get_overview(
        self,
        tenant_id: uuid.UUID,
        **_extra,
    ) -> dict[str, Any]:
        """Return a dashboard overview combining MongoDB stats and ES data.

        Args:
            tenant_id: Owning tenant.

        Returns:
            Dict with total_scans, total_detections, unique_pii_types,
            avg_detections_per_scan, recent_trend.
        """
        tid = str(tenant_id)

        # Aggregate from MongoDB detection_stats
        pipeline = [
            {"$match": {"tenant_id": tid}},
            {
                "$group": {
                    "_id": None,
                    "total_scans": {"$sum": 1},
                    "total_detections": {"$sum": "$total_detections"},
                    "total_chars": {"$sum": "$char_count"},
                }
            },
        ]

        stats = {"total_scans": 0, "total_detections": 0, "total_chars": 0}
        async for doc in self._mongo["detection_stats"].aggregate(pipeline):
            stats["total_scans"] = doc.get("total_scans", 0)
            stats["total_detections"] = doc.get("total_detections", 0)
            stats["total_chars"] = doc.get("total_chars", 0)

        # Unique PII types from ES
        unique_types = 0
        try:
            es = self._es.get_client()
            if es is None:
                raise RuntimeError("ES not connected")
            es_result = await es.search(
                index="detections-*",
                body={
                    "size": 0,
                    "query": {"term": {"tenant_id": tid}},
                    "aggs": {
                        "unique_types": {"cardinality": {"field": "entity_type"}},
                    },
                },
            )
            unique_types = (
                es_result.get("aggregations", {})
                .get("unique_types", {})
                .get("value", 0)
            )
        except Exception:
            logger.debug(
                "ES unavailable for overview; falling back to zero.",
                exc_info=True,
            )

        avg_per_scan = (
            round(stats["total_detections"] / stats["total_scans"], 2)
            if stats["total_scans"] > 0
            else 0.0
        )

        return {
            "total_scans": stats["total_scans"],
            "total_detections": stats["total_detections"],
            "total_chars_processed": stats["total_chars"],
            "unique_pii_types": unique_types,
            "avg_detections_per_scan": avg_per_scan,
        }

    # ------------------------------------------------------------------
    # By PII type
    # ------------------------------------------------------------------

    async def get_by_type(
        self,
        tenant_id: uuid.UUID,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        **_extra,
    ) -> list[dict[str, Any]]:
        """Return detection counts grouped by PII type.

        Uses Elasticsearch terms aggregation on the ``entity_type`` field.

        Args:
            tenant_id: Owning tenant.
            date_from: Optional start date filter.
            date_to: Optional end date filter.

        Returns:
            List of dicts with ``type`` and ``count``.
        """
        filters = self._build_es_filters(str(tenant_id), date_from, date_to)

        try:
            es = self._es.get_client()
            if es is None:
                return []
            result = await es.search(
                index="detections-*",
                body={
                    "size": 0,
                    "query": {"bool": {"filter": filters}},
                    "aggs": {
                        "by_type": {
                            "terms": {"field": "entity_type", "size": 100},
                        }
                    },
                },
            )
            buckets = (
                result.get("aggregations", {})
                .get("by_type", {})
                .get("buckets", [])
            )
            return [
                {"type": b["key"], "count": b["doc_count"]} for b in buckets
            ]
        except Exception:
            logger.warning(
                "ES by_type query failed for tenant %s.", tenant_id, exc_info=True
            )
            return []

    # ------------------------------------------------------------------
    # By source
    # ------------------------------------------------------------------

    async def get_by_source(
        self,
        tenant_id: uuid.UUID,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        **_extra,
    ) -> list[dict[str, Any]]:
        """Return detection counts grouped by detection source.

        Args:
            tenant_id: Owning tenant.
            date_from: Optional start date.
            date_to: Optional end date.

        Returns:
            List of dicts with ``source`` and ``count``.
        """
        filters = self._build_es_filters(str(tenant_id), date_from, date_to)

        try:
            es = self._es.get_client()
            if es is None:
                return []
            result = await es.search(
                index="detections-*",
                body={
                    "size": 0,
                    "query": {"bool": {"filter": filters}},
                    "aggs": {
                        "by_source": {
                            "terms": {"field": "source_type", "size": 20},
                        }
                    },
                },
            )
            buckets = (
                result.get("aggregations", {})
                .get("by_source", {})
                .get("buckets", [])
            )
            return [
                {"source": b["key"], "count": b["doc_count"]} for b in buckets
            ]
        except Exception:
            logger.warning(
                "ES by_source query failed for tenant %s.",
                tenant_id,
                exc_info=True,
            )
            return []

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    async def get_timeline(
        self,
        tenant_id: uuid.UUID,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        interval: str = "1d",
        **_extra,
    ) -> list[dict[str, Any]]:
        """Return detection count over time using a date_histogram.

        Args:
            tenant_id: Owning tenant.
            date_from: Start date.
            date_to: End date.
            interval: Calendar interval (e.g. '1d', '1w', '1M').

        Returns:
            List of dicts with ``date`` and ``count``.
        """
        allowed_intervals = {"1h", "6h", "1d", "1w", "1M"}
        if interval not in allowed_intervals:
            raise ValidationError(
                f"interval must be one of {sorted(allowed_intervals)}.",
                error_code="INVALID_INTERVAL",
            )

        filters = self._build_es_filters(str(tenant_id), date_from, date_to)

        try:
            es = self._es.get_client()
            if es is None:
                return []
            result = await es.search(
                index="detections-*",
                body={
                    "size": 0,
                    "query": {"bool": {"filter": filters}},
                    "aggs": {
                        "timeline": {
                            "date_histogram": {
                                "field": "detected_at",
                                "calendar_interval": interval,
                                "min_doc_count": 0,
                            }
                        }
                    },
                },
            )
            buckets = (
                result.get("aggregations", {})
                .get("timeline", {})
                .get("buckets", [])
            )
            return [
                {
                    "date": b["key_as_string"],
                    "count": b["doc_count"],
                }
                for b in buckets
            ]
        except Exception:
            logger.warning(
                "ES timeline query failed for tenant %s.",
                tenant_id,
                exc_info=True,
            )
            return []

    # ------------------------------------------------------------------
    # Heatmap (label x hour-of-day)
    # ------------------------------------------------------------------

    async def get_heatmap(
        self,
        tenant_id: uuid.UUID,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        **_extra,
    ) -> list[dict[str, Any]]:
        """Return a cross-tab of PII type vs hour-of-day.

        Args:
            tenant_id: Owning tenant.
            date_from: Optional start date.
            date_to: Optional end date.

        Returns:
            List of dicts with ``type``, ``hour``, ``count``.
        """
        filters = self._build_es_filters(str(tenant_id), date_from, date_to)

        try:
            es = self._es.get_client()
            if es is None:
                return []
            result = await es.search(
                index="detections-*",
                body={
                    "size": 0,
                    "query": {"bool": {"filter": filters}},
                    "aggs": {
                        "by_type": {
                            "terms": {"field": "entity_type", "size": 50},
                            "aggs": {
                                "by_hour": {
                                    "date_histogram": {
                                        "field": "detected_at",
                                        "calendar_interval": "1h",
                                        "min_doc_count": 0,
                                    },
                                    "aggs": {
                                        "hour_of_day": {
                                            "bucket_script": {
                                                "buckets_path": {"_value": "_count"},
                                                "script": "params._value",
                                            }
                                        }
                                    },
                                }
                            },
                        }
                    },
                },
            )

            heatmap = []
            type_buckets = (
                result.get("aggregations", {})
                .get("by_type", {})
                .get("buckets", [])
            )
            for tb in type_buckets:
                entity_type = tb["key"]
                for hb in tb.get("by_hour", {}).get("buckets", []):
                    # Extract hour from the bucket key string
                    try:
                        dt = datetime.fromisoformat(
                            hb["key_as_string"].replace("Z", "+00:00")
                        )
                        hour = dt.hour
                    except (ValueError, KeyError):
                        hour = 0
                    if hb["doc_count"] > 0:
                        heatmap.append({
                            "type": entity_type,
                            "hour": hour,
                            "count": hb["doc_count"],
                        })

            return heatmap
        except Exception:
            logger.warning(
                "ES heatmap query failed for tenant %s.",
                tenant_id,
                exc_info=True,
            )
            return []

    # ------------------------------------------------------------------
    # Top N PII types
    # ------------------------------------------------------------------

    async def get_top_types(
        self,
        tenant_id: uuid.UUID,
        n: int = 10,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        **_extra,
    ) -> list[dict[str, Any]]:
        """Return the top-N most detected PII types.

        Args:
            tenant_id: Owning tenant.
            n: Number of top types to return.
            date_from: Optional start date.
            date_to: Optional end date.

        Returns:
            List of dicts with ``type``, ``count``, ``percentage``.
        """
        filters = self._build_es_filters(str(tenant_id), date_from, date_to)

        try:
            es = self._es.get_client()
            if es is None:
                return []
            result = await es.search(
                index="detections-*",
                body={
                    "size": 0,
                    "query": {"bool": {"filter": filters}},
                    "aggs": {
                        "top_types": {
                            "terms": {"field": "entity_type", "size": n},
                        },
                        "total": {"value_count": {"field": "entity_type"}},
                    },
                },
            )

            total = (
                result.get("aggregations", {})
                .get("total", {})
                .get("value", 1)
            ) or 1
            buckets = (
                result.get("aggregations", {})
                .get("top_types", {})
                .get("buckets", [])
            )

            return [
                {
                    "type": b["key"],
                    "count": b["doc_count"],
                    "percentage": round(b["doc_count"] / total * 100, 2),
                }
                for b in buckets
            ]
        except Exception:
            logger.warning(
                "ES top_types query failed for tenant %s.",
                tenant_id,
                exc_info=True,
            )
            return []

    # ------------------------------------------------------------------
    # Source effectiveness
    # ------------------------------------------------------------------

    async def get_source_effectiveness(
        self,
        tenant_id: uuid.UUID,
        **_extra,
    ) -> list[dict[str, Any]]:
        """Return source comparison stats (detection count, avg confidence).

        Args:
            tenant_id: Owning tenant.

        Returns:
            List of dicts per source with ``source``, ``count``,
            ``avg_confidence``.
        """
        try:
            es = self._es.get_client()
            if es is None:
                return []
            result = await es.search(
                index="detections-*",
                body={
                    "size": 0,
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"tenant_id": str(tenant_id)}},
                            ]
                        }
                    },
                    "aggs": {
                        "by_source": {
                            "terms": {"field": "source_type", "size": 20},
                            "aggs": {
                                "avg_conf": {
                                    "avg": {"field": "confidence_score"},
                                }
                            },
                        }
                    },
                },
            )
            buckets = (
                result.get("aggregations", {})
                .get("by_source", {})
                .get("buckets", [])
            )
            return [
                {
                    "source": b["key"],
                    "count": b["doc_count"],
                    "avg_confidence": round(
                        b.get("avg_conf", {}).get("value", 0) or 0, 4
                    ),
                }
                for b in buckets
            ]
        except Exception:
            logger.warning(
                "ES source_effectiveness query failed for tenant %s.",
                tenant_id,
                exc_info=True,
            )
            return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_es_filters(
        tenant_id: str,
        date_from: datetime | None,
        date_to: datetime | None,
    ) -> list[dict[str, Any]]:
        """Build Elasticsearch filter clauses.

        Args:
            tenant_id: Tenant ID string.
            date_from: Optional start date.
            date_to: Optional end date.

        Returns:
            List of filter dicts for an ES bool query.
        """
        filters: list[dict[str, Any]] = [
            {"term": {"tenant_id": tenant_id}},
        ]
        if date_from or date_to:
            range_filter: dict[str, Any] = {}
            if date_from:
                range_filter["gte"] = date_from.isoformat()
            if date_to:
                range_filter["lte"] = date_to.isoformat()
            filters.append({"range": {"detected_at": range_filter}})
        return filters


# ---------------------------------------------------------------------------
# Module-level convenience functions (proxy to AnalyticsService)
# ---------------------------------------------------------------------------

async def get_overview(**kw):
    return await AnalyticsService().get_overview(**kw)


async def get_by_type(**kw):
    return await AnalyticsService().get_by_type(**kw)


async def detections_by_type(**kw):
    return await AnalyticsService().get_by_type(**kw)


async def get_by_source(**kw):
    return await AnalyticsService().get_by_source(**kw)


async def detections_by_source(**kw):
    return await AnalyticsService().get_by_source(**kw)


async def detections_by_type_source(**kw):
    # Route calls this; delegate to get_by_type as cross-tab fallback
    svc = AnalyticsService()
    if hasattr(svc, "detections_by_type_source"):
        return await svc.detections_by_type_source(**kw)
    if hasattr(svc, "get_by_type_source"):
        return await svc.get_by_type_source(**kw)
    return await svc.get_by_type(**kw)


async def get_timeline(**kw):
    return await AnalyticsService().get_timeline(**kw)


async def detections_timeline(**kw):
    return await AnalyticsService().get_timeline(**kw)


async def get_heatmap(**kw):
    return await AnalyticsService().get_heatmap(**kw)


async def detections_heatmap(**kw):
    return await AnalyticsService().get_heatmap(**kw)


async def get_top_types(**kw):
    return await AnalyticsService().get_top_types(**kw)


async def top_pii_types(**kw):
    return await AnalyticsService().get_top_types(**kw)


async def get_source_effectiveness(**kw):
    return await AnalyticsService().get_source_effectiveness(**kw)


async def source_effectiveness(**kw):
    return await AnalyticsService().get_source_effectiveness(**kw)
