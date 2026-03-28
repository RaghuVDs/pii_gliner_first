"""
Shared FastAPI dependencies for the PII Engine SaaS API.

These are injected into route handlers via ``Depends(...)`` to provide
database sessions, service clients, and the authenticated user/tenant.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, Optional

from elasticsearch import AsyncElasticsearch
from fastapi import Depends, Header, Request
from motor.motor_asyncio import AsyncIOMotorDatabase
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db as _pg_get_db
from app.core.elasticsearch_client import elastic_client
from app.core.exceptions import NotFoundError, UnauthorizedError
from app.core.mongodb import mongodb_client
from app.core.redis_client import redis_client
from app.core.security import decode_token, hash_api_key

logger = logging.getLogger(__name__)


# ── Database Dependencies ────────────────────────────────────────────────


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async PostgreSQL session (delegates to core.database)."""
    async for session in _pg_get_db():
        yield session


async def get_mongodb() -> AsyncIOMotorDatabase:
    """Return the active MongoDB database handle."""
    return mongodb_client.get_database()


async def get_elasticsearch() -> Optional[AsyncElasticsearch]:
    """Return the active Elasticsearch client, or None if not connected."""
    return elastic_client.get_client()


async def get_redis() -> Redis:
    """Return the active Redis client."""
    return redis_client.get_client()


# ── Authentication Dependencies ──────────────────────────────────────────


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Extract and validate the current user from the JWT bearer token.

    The token is expected in the ``Authorization: Bearer <token>`` header.
    The decoded ``sub`` claim is used to look up the user in PostgreSQL.

    Returns:
        The user ORM object.

    Raises:
        UnauthorizedError: If the token is missing, invalid, or the user
            does not exist.
    """
    # Check if already resolved by middleware
    user = getattr(request.state, "current_user", None)
    if user is not None:
        return user

    auth_header: Optional[str] = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise UnauthorizedError("Missing or malformed Authorization header.")

    token = auth_header.removeprefix("Bearer ").strip()
    payload: Dict[str, Any] = decode_token(token)

    user_id: Optional[str] = payload.get("sub")
    if user_id is None:
        raise UnauthorizedError("Token missing 'sub' claim.")

    # Lazy import to avoid circular dependency with models
    from app.models.user import User  # noqa: WPS433

    result = await db.get(User, user_id)
    if result is None:
        raise UnauthorizedError("User not found.")

    if not getattr(result, "is_active", True):
        raise UnauthorizedError("User account is deactivated.")

    # Stash on request.state for downstream middleware/deps
    request.state.current_user = result
    return result


async def get_current_tenant(
    request: Request,
    current_user: Any = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Resolve the tenant record for the authenticated user.

    Returns:
        The Tenant ORM object.

    Raises:
        UnauthorizedError: If tenant cannot be determined.
    """
    tenant_id = getattr(request.state, "tenant_id", None) or getattr(
        current_user, "tenant_id", None
    )
    if tenant_id is None:
        raise UnauthorizedError("Cannot determine tenant context.")

    from app.models.tenant import Tenant
    from sqlalchemy import select

    result = await db.execute(
        select(Tenant).where(Tenant.id == tenant_id)
    )
    tenant = result.scalar_one_or_none()
    if tenant is None:
        raise UnauthorizedError("Tenant not found.")
    return tenant


async def get_api_key_user(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Authenticate via the ``X-API-Key`` header.

    Looks up the hashed key in PostgreSQL and returns the associated
    user or service account.

    Returns:
        The API key ORM object (which references the owning user/tenant).

    Raises:
        UnauthorizedError: If the header is missing or the key is invalid.
    """
    if x_api_key is None:
        raise UnauthorizedError("X-API-Key header is required.")

    key_hash = hash_api_key(x_api_key)

    # Lazy import to avoid circular dependency
    from sqlalchemy import select

    from app.models.api_key import ApiKey  # noqa: WPS433

    stmt = select(ApiKey).where(
        ApiKey.key_hash == key_hash,
        ApiKey.is_active.is_(True),
    )
    result = await db.execute(stmt)
    api_key = result.scalar_one_or_none()

    if api_key is None:
        raise UnauthorizedError("Invalid or revoked API key.")

    # Stash scopes on request.state for scope checks
    request.state.api_key_scopes = getattr(api_key, "scopes", [])
    request.state.tenant_id = str(api_key.tenant_id)

    return api_key
