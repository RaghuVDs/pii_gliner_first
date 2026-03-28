"""
Async Redis client for the PII Engine SaaS platform.

Provides connection management and helper methods for caching,
rate limiting, pub/sub, and cache invalidation.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Manages an async Redis connection pool and common operations."""

    def __init__(self) -> None:
        self._client: Optional[aioredis.Redis] = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the Redis connection pool."""
        settings = get_settings()
        logger.info("Connecting to Redis at %s ...", settings.REDIS_URL)

        self._client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )

        # Verify connectivity
        await self._client.ping()
        logger.info("Redis connection established.")

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self._client is not None:
            logger.info("Closing Redis connection...")
            await self._client.close()
            self._client = None
            logger.info("Redis connection closed.")

    def get_client(self) -> aioredis.Redis:
        """Return the active Redis client.

        Raises:
            RuntimeError: If the client has not been connected yet.
        """
        if self._client is None:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._client

    # ── Rate Limiting (Sliding Window) ───────────────────────────────────

    async def rate_limit_check(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> bool:
        """Check whether a request should be allowed under a sliding window rate limit.

        Args:
            key: Unique identifier (e.g. ``rate:tenant:<id>``).
            max_requests: Maximum allowed requests within the window.
            window_seconds: Length of the sliding window in seconds.

        Returns:
            ``True`` if the request is within the limit, ``False`` otherwise.
        """
        client = self.get_client()
        now = time.time()
        window_start = now - window_seconds

        pipe = client.pipeline(transaction=True)
        # Remove expired entries
        pipe.zremrangebyscore(key, "-inf", window_start)
        # Count current entries
        pipe.zcard(key)
        # Add the current request
        pipe.zadd(key, {f"{now}": now})
        # Set expiry on the key
        pipe.expire(key, window_seconds)
        results = await pipe.execute()

        current_count: int = results[1]
        return current_count < max_requests

    # ── Caching ──────────────────────────────────────────────────────────

    async def cache_get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value, deserialising from JSON.

        Args:
            key: Cache key.

        Returns:
            Deserialised value or ``None`` if not present.
        """
        client = self.get_client()
        raw = await client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    async def cache_set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 300,
    ) -> None:
        """Store a value in cache, serialised as JSON.

        Args:
            key: Cache key.
            value: Value to cache (must be JSON-serialisable).
            ttl_seconds: Time-to-live in seconds (default 5 minutes).
        """
        client = self.get_client()
        serialised = json.dumps(value, default=str)
        await client.setex(key, ttl_seconds, serialised)

    async def cache_invalidate(self, pattern: str) -> int:
        """Delete all keys matching a glob pattern.

        Args:
            pattern: Redis key pattern (e.g. ``cache:tenant:abc:*``).

        Returns:
            Number of keys deleted.
        """
        client = self.get_client()
        deleted = 0
        async for key in client.scan_iter(match=pattern, count=200):
            await client.delete(key)
            deleted += 1
        return deleted

    # ── Pub/Sub ──────────────────────────────────────────────────────────

    async def publish(self, channel: str, message: Any) -> int:
        """Publish a message to a Redis channel.

        Args:
            channel: Channel name.
            message: Message payload (will be JSON-encoded).

        Returns:
            Number of subscribers that received the message.
        """
        client = self.get_client()
        serialised = json.dumps(message, default=str)
        return await client.publish(channel, serialised)


# Module-level singleton
redis_client = RedisClient()
