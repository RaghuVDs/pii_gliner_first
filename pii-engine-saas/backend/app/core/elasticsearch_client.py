"""
Async Elasticsearch client for the PII Engine SaaS platform.

Manages connection lifecycle, index templates, and index mappings
for detection results and analytics events.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from elasticsearch import AsyncElasticsearch

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Index Mappings ───────────────────────────────────────────────────────

DETECTIONS_INDEX_TEMPLATE: Dict[str, Any] = {
    "index_patterns": ["detections-*"],
    "settings": {
        "number_of_shards": 2,
        "number_of_replicas": 1,
        "refresh_interval": "5s",
        "index.lifecycle.name": "detections-policy",
    },
    "mappings": {
        "properties": {
            "detection_id": {"type": "keyword"},
            "tenant_id": {"type": "keyword"},
            "job_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "entity_type": {"type": "keyword"},
            "entity_value": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
            "confidence_score": {"type": "float"},
            "start_offset": {"type": "integer"},
            "end_offset": {"type": "integer"},
            "context_snippet": {"type": "text"},
            "model_version": {"type": "keyword"},
            "detected_at": {"type": "date"},
            "source_type": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": False},
        }
    },
}

ANALYTICS_INDEX_TEMPLATE: Dict[str, Any] = {
    "index_patterns": ["analytics-events-*"],
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1,
        "refresh_interval": "30s",
    },
    "mappings": {
        "properties": {
            "event_id": {"type": "keyword"},
            "tenant_id": {"type": "keyword"},
            "event_type": {"type": "keyword"},
            "user_id": {"type": "keyword"},
            "timestamp": {"type": "date"},
            "details": {"type": "object", "enabled": False},
            "entity_type": {"type": "keyword"},
            "document_count": {"type": "integer"},
            "detection_count": {"type": "integer"},
            "processing_time_ms": {"type": "long"},
            "source": {"type": "keyword"},
        }
    },
}


class ElasticClient:
    """Manages the async Elasticsearch connection and index lifecycle."""

    def __init__(self) -> None:
        self._client: Optional[AsyncElasticsearch] = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the Elasticsearch async client and set up index templates."""
        settings = get_settings()
        logger.info("Connecting to Elasticsearch at %s ...", settings.ELASTICSEARCH_URL)

        try:
            from elasticsearch import VERSION as ES_CLIENT_VERSION

            # Use basic transport headers to avoid version negotiation issues
            self._client = AsyncElasticsearch(
                hosts=[settings.ELASTICSEARCH_URL],
                request_timeout=30,
                max_retries=3,
                retry_on_timeout=True,
                meta_header=False,
                headers={
                    "accept": "application/json",
                    "content-type": "application/json",
                },
            )

            # Verify connectivity with a raw request to avoid versioned headers
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{settings.ELASTICSEARCH_URL}/_cluster/health") as resp:
                    if resp.status == 200:
                        health = await resp.json()
                        logger.info(
                            "Elasticsearch connected (cluster=%s, status=%s).",
                            health.get("cluster_name", "unknown"),
                            health.get("status", "unknown"),
                        )
                    else:
                        logger.warning("Elasticsearch health check returned status %s", resp.status)

            await self.create_index_templates()
        except Exception as e:
            logger.warning("Elasticsearch connection failed (non-fatal): %s", e)
            logger.warning("Elasticsearch features (search, analytics) will be unavailable.")
            self._client = None

    async def close(self) -> None:
        """Close the Elasticsearch client and release resources."""
        if self._client is not None:
            logger.info("Closing Elasticsearch client...")
            await self._client.close()
            self._client = None
            logger.info("Elasticsearch client closed.")

    def get_client(self) -> Optional[AsyncElasticsearch]:
        """Return the active Elasticsearch client, or None if not connected.

        Returns:
            The active AsyncElasticsearch client, or None if Elasticsearch
            is unavailable or connection failed.
        """
        return self._client

    # ── Index Templates ──────────────────────────────────────────────────

    async def create_index_templates(self) -> None:
        """Create or update index templates for detections and analytics."""
        if self._client is None:
            logger.warning("Skipping index template creation — Elasticsearch not connected.")
            return
        client = self._client

        logger.info("Creating Elasticsearch index templates...")

        await client.indices.put_index_template(
            name="detections-template",
            body=DETECTIONS_INDEX_TEMPLATE,
        )
        logger.info("Index template 'detections-template' applied.")

        await client.indices.put_index_template(
            name="analytics-events-template",
            body=ANALYTICS_INDEX_TEMPLATE,
        )
        logger.info("Index template 'analytics-events-template' applied.")

    # ── Convenience Methods ──────────────────────────────────────────────

    async def index_detection(self, index: str, document: Dict[str, Any]) -> Optional[str]:
        """Index a single detection document.

        Args:
            index: Target index name (e.g. ``detections-2026.03``).
            document: The detection document body.

        Returns:
            The Elasticsearch document ``_id``, or None if ES is unavailable.
        """
        client = self.get_client()
        if client is None:
            logger.warning("Cannot index detection — Elasticsearch not connected.")
            return None
        result = await client.index(index=index, body=document)
        return result["_id"]

    async def bulk_index(self, actions: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a bulk indexing operation.

        Args:
            actions: List of Elasticsearch bulk action dicts.

        Returns:
            The Elasticsearch bulk response, or empty result if ES is unavailable.
        """
        client = self.get_client()
        if client is None:
            logger.warning("Cannot bulk index — Elasticsearch not connected.")
            return {"success": 0, "errors": []}

        from elasticsearch.helpers import async_bulk

        success, errors = await async_bulk(client, actions, raise_on_error=False)
        return {"success": success, "errors": errors}

    async def search_detections(
        self,
        tenant_id: str,
        query: Dict[str, Any],
        *,
        index: str = "detections-*",
        size: int = 50,
        from_: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Search detections scoped to a tenant.

        Args:
            tenant_id: Tenant to scope results to.
            query: Elasticsearch query DSL body.
            index: Index pattern to search.
            size: Maximum number of results.
            from_: Offset for pagination.

        Returns:
            Raw Elasticsearch search response, or None if ES is unavailable.
        """
        client = self.get_client()
        if client is None:
            logger.warning("Cannot search detections — Elasticsearch not connected.")
            return None

        # Inject tenant filter
        body: Dict[str, Any] = {
            "query": {
                "bool": {
                    "must": [query],
                    "filter": [{"term": {"tenant_id": tenant_id}}],
                }
            },
            "size": size,
            "from": from_,
            "sort": [{"detected_at": {"order": "desc"}}],
        }

        return await client.search(index=index, body=body)


# Module-level singleton
elastic_client = ElasticClient()
