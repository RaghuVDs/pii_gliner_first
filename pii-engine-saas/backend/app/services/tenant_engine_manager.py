"""
Tenant engine manager -- maintains an LRU cache of HybridPIIEngine
instances keyed by (tenant_id, config_version).

Listens for config-change signals over Redis pub/sub so that stale
engines are evicted promptly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from typing import Any, Optional

from cachetools import LRUCache
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.redis_client import redis_client
from app.services.config_assembler import ConfigAssembler

logger = logging.getLogger(__name__)

# Maximum number of tenant engine instances to keep cached.
_DEFAULT_MAX_CACHE_SIZE = 50

# ── Global GLiNER model cache (loaded once, shared across all tenants) ──
_gliner_model = None
_gliner_model_lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None


def _get_cached_gliner_model():
    """Load GLiNER model once and cache globally. Uses CUDA if available."""
    global _gliner_model
    if _gliner_model is not None:
        return _gliner_model

    try:
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

        import torch
        from gliner import GLiNER

        settings = get_settings()
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("CUDA available - using GPU for GLiNER (%s)",
                         torch.cuda.get_device_name(0))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("MPS available - using Apple Silicon for GLiNER")
        else:
            cpu_cores = os.cpu_count() or 4
            torch.set_num_threads(cpu_cores)
            logger.info("Using CPU with %d threads for GLiNER", cpu_cores)

        model_name = settings.GLINER_MODEL_NAME
        logger.info("Loading GLiNER model '%s' on %s (first time only)...",
                     model_name, device.upper())

        # Try loading - first offline (cached), then online as fallback
        try:
            _gliner_model = GLiNER.from_pretrained(model_name).to(device)
        except Exception as offline_err:
            logger.warning("Offline load failed (%s), trying online...", offline_err)
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
            _gliner_model = GLiNER.from_pretrained(model_name).to(device)

        logger.info("GLiNER model loaded and cached globally on %s.", device.upper())
        return _gliner_model
    except Exception as e:
        logger.error("Failed to load GLiNER model: %s", e, exc_info=True)
        return None

# Redis channel for config-change notifications.
_CONFIG_CHANGED_CHANNEL = "pii:config_changed"


class TenantEngineManager:
    """Manages per-tenant ``HybridPIIEngine`` instances with LRU eviction.

    The manager uses ``ConfigAssembler`` to build the full config from
    the database and then constructs a ``HybridPIIEngine`` with those
    settings.  Built engines are cached so that subsequent requests for
    the same (tenant_id, config_version) hit the cache.

    Config versions are derived by hashing the assembled config, so any
    change to a tenant's PII types, rules, or masking settings results
    in a new version and a fresh engine build on next access.
    """

    def __init__(self, max_size: int = _DEFAULT_MAX_CACHE_SIZE) -> None:
        self._cache: LRUCache[str, Any] = LRUCache(maxsize=max_size)
        self._config_versions: dict[str, str] = {}  # tenant_id -> version_hash
        self._listener_task: Optional[asyncio.Task[None]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_engine(
        self,
        tenant_id: uuid.UUID,
        db: AsyncSession,
    ) -> Any:
        """Return a cached or freshly-built ``HybridPIIEngine`` for a tenant.

        Args:
            tenant_id: Target tenant.
            db: Active database session used to assemble config.

        Returns:
            A ``HybridPIIEngine`` instance configured for the tenant.
        """
        tid = str(tenant_id)

        # Assemble current config and compute version hash
        config = await ConfigAssembler.assemble_all(tenant_id, db)
        version = self._compute_version(config)

        cache_key = f"{tid}:{version}"

        # Check cache
        engine = self._cache.get(cache_key)
        if engine is not None:
            logger.debug("Cache hit for tenant %s (version=%s).", tid, version[:8])
            return engine

        # Build new engine
        engine = await self._build_engine(config)
        self._cache[cache_key] = engine
        self._config_versions[tid] = version

        logger.info(
            "Built new engine for tenant %s (version=%s, cache_size=%d).",
            tid,
            version[:8],
            len(self._cache),
        )
        return engine

    def invalidate(self, tenant_id: uuid.UUID) -> None:
        """Remove all cached engines for a tenant.

        Called when a tenant's configuration changes (e.g. after updating
        PII types, rules, or masking settings).

        Args:
            tenant_id: Tenant whose engine should be evicted.
        """
        tid = str(tenant_id)

        # Remove from version tracking
        version = self._config_versions.pop(tid, None)
        if version:
            cache_key = f"{tid}:{version}"
            try:
                del self._cache[cache_key]
                logger.info("Invalidated engine cache for tenant %s.", tid)
            except KeyError:
                pass

    async def publish_invalidation(self, tenant_id: uuid.UUID) -> None:
        """Publish a config-change signal over Redis pub/sub.

        Other instances of the service will pick this up and evict their
        local cache entries.

        Args:
            tenant_id: Tenant whose config changed.
        """
        try:
            await redis_client.publish(
                _CONFIG_CHANGED_CHANNEL,
                {"tenant_id": str(tenant_id)},
            )
            logger.debug(
                "Published config_changed for tenant %s.", tenant_id
            )
        except Exception:
            logger.warning(
                "Failed to publish config_changed for tenant %s.",
                tenant_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Redis pub/sub listener
    # ------------------------------------------------------------------

    async def start_listener(self) -> None:
        """Start background task listening for config_changed events."""
        if self._listener_task is not None:
            return
        self._listener_task = asyncio.create_task(
            self._listen_for_changes(), name="engine-invalidation-listener"
        )
        logger.info("Started config-change listener on channel '%s'.", _CONFIG_CHANGED_CHANNEL)

    async def stop_listener(self) -> None:
        """Cancel the background listener task."""
        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
            logger.info("Stopped config-change listener.")

    async def _listen_for_changes(self) -> None:
        """Subscribe to Redis channel and invalidate engines on signal."""
        try:
            client = redis_client.get_client()
            pubsub = client.pubsub()
            await pubsub.subscribe(_CONFIG_CHANGED_CHANNEL)

            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                try:
                    data = json.loads(message["data"])
                    tenant_id_str = data.get("tenant_id")
                    if tenant_id_str:
                        self.invalidate(uuid.UUID(tenant_id_str))
                        logger.info(
                            "Received config_changed for tenant %s -- cache invalidated.",
                            tenant_id_str,
                        )
                except (json.JSONDecodeError, ValueError, KeyError):
                    logger.warning(
                        "Malformed config_changed message: %s",
                        message.get("data"),
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error(
                "Config-change listener crashed.", exc_info=True
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_version(config: dict[str, Any]) -> str:
        """Compute a SHA-256 hash of the serialised config dict.

        Args:
            config: The assembled engine configuration.

        Returns:
            Hex-encoded hash string.
        """
        raw = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    async def _build_engine(config: dict[str, Any]) -> Any:
        """Construct a ``HybridPIIEngine`` from an assembled config.

        This runs the (potentially CPU-heavy) engine construction in a
        thread pool to avoid blocking the event loop.

        Args:
            config: The full assembled config dict.

        Returns:
            A configured ``HybridPIIEngine`` instance.
        """
        loop = asyncio.get_running_loop()

        def _construct() -> Any:
            from app.engine import HybridPIIEngine

            settings = get_settings()

            # Get cached GLiNER model (loaded once globally)
            gliner_model = _get_cached_gliner_model()

            engine = HybridPIIEngine(
                taxonomy_config=config.get("taxonomy", {}),
                regex_rules=config.get("regex_rules", {}),
                masking_rules=config.get("masking_rules", {}),
                field_patterns=config.get("field_patterns", {}),
                context_rules=config.get("context_rules", {}),
                use_gliner=gliner_model is not None,
                gliner_model_name=settings.GLINER_MODEL_NAME,
                gliner_threshold=0.35,
                gliner_model=gliner_model,
            )

            return engine

        return await loop.run_in_executor(None, _construct)

    def get_engine_sync(self, tenant_id: str) -> Any:
        """Synchronous engine getter for Celery workers.

        Builds a fresh engine each time (no async DB access in sync context).
        Uses a simple in-memory cache keyed by tenant_id.
        """
        cached = self._cache.get(tenant_id)
        if cached is not None:
            return cached

        # In sync context (Celery), build engine with default configs
        # This is a simplified path - full DB assembly requires async
        from app.engine import HybridPIIEngine

        settings = get_settings()
        engine = HybridPIIEngine(
            taxonomy_config={},
            regex_rules={},
            masking_rules={},
            field_patterns={},
            context_rules={},
            use_gliner=True,
            gliner_model_name=settings.GLINER_MODEL_NAME,
            tenant_id=tenant_id,
        )
        self._cache[tenant_id] = engine
        return engine


# Module-level singleton
tenant_engine_manager = TenantEngineManager()
