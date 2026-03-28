"""
API key service -- key lifecycle management, access request workflows,
usage recording, and usage analytics.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    UnauthorizedError,
    ValidationError,
)
from app.models.api_key import APIAccessRequest, APIKey, APIUsageLog

logger = logging.getLogger(__name__)

# API key prefix format: "pii_" + 8 random chars
_KEY_PREFIX_LEN = 8


def _generate_api_key() -> tuple[str, str, str]:
    """Generate a new API key.

    Returns:
        Tuple of (full_key, key_prefix, key_hash).
    """
    prefix = secrets.token_hex(4)  # 8 hex chars
    body = secrets.token_urlsafe(32)
    full_key = f"pii_{prefix}_{body}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    return full_key, prefix, key_hash


class APIKeyService:
    """Business logic for API key management and access request workflows."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ==================================================================
    # API Key CRUD
    # ==================================================================

    async def list_keys(self, tenant_id: uuid.UUID, **_extra) -> list[APIKey]:
        """Return all API keys for a tenant.

        Args:
            tenant_id: Owning tenant.

        Returns:
            List of APIKey instances.
        """
        result = await self._db.execute(
            select(APIKey)
            .where(APIKey.tenant_id == tenant_id)
            .order_by(APIKey.created_at.desc())
        )
        return list(result.scalars().all())

    async def create_key(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        name: str,
        scopes: list[str] | None = None,
        rate_limit: int = 60,
        expires_in_days: int | None = None,
        access_request_id: uuid.UUID | None = None,
        **_extra,
    ) -> dict[str, Any]:
        """Generate a new API key.

        The full plaintext key is returned **only once** in the response.

        Args:
            tenant_id: Owning tenant.
            user_id: User creating the key.
            name: Human-readable key name.
            scopes: Permission scopes (default: detect, redact).
            rate_limit: Requests per minute (default 60).
            expires_in_days: Optional expiry in days.
            access_request_id: Optional linked access request.

        Returns:
            Dict with the created APIKey and the plaintext ``key``.
        """
        full_key, prefix, key_hash = _generate_api_key()

        expires_at = None
        if expires_in_days and expires_in_days > 0:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        api_key = APIKey(
            tenant_id=tenant_id,
            user_id=user_id,
            access_request_id=access_request_id,
            name=name,
            key_prefix=prefix,
            key_hash=key_hash,
            scopes=scopes or ["detect", "redact"],
            rate_limit_per_min=rate_limit,
            is_active=True,
            expires_at=expires_at,
        )
        self._db.add(api_key)
        await self._db.flush()

        logger.info(
            "Created API key '%s' (prefix=%s) for tenant %s.",
            name,
            prefix,
            tenant_id,
        )

        return {
            "api_key": api_key,
            "key": full_key,  # Only time the plaintext key is returned
        }

    async def revoke_key(
        self,
        tenant_id: uuid.UUID,
        key_id: uuid.UUID,
        **_extra,
    ) -> APIKey:
        """Revoke an API key.

        Args:
            tenant_id: Owning tenant.
            key_id: API key ID to revoke.

        Returns:
            The revoked APIKey.

        Raises:
            NotFoundError: If key not found.
        """
        result = await self._db.execute(
            select(APIKey).where(
                APIKey.id == key_id,
                APIKey.tenant_id == tenant_id,
            )
        )
        api_key = result.scalar_one_or_none()
        if api_key is None:
            raise NotFoundError(
                "API key not found.", error_code="API_KEY_NOT_FOUND"
            )

        api_key.revoked_at = datetime.now(timezone.utc)
        api_key.is_active = False
        self._db.add(api_key)
        await self._db.flush()

        logger.info("Revoked API key %s for tenant %s.", key_id, tenant_id)
        return api_key

    async def rotate_key(
        self,
        tenant_id: uuid.UUID,
        key_id: uuid.UUID,
        **_extra,
    ) -> dict[str, Any]:
        """Rotate an API key: revoke the old one and create a new one.

        The new key inherits the name, scopes, and rate limit of the old.

        Args:
            tenant_id: Owning tenant.
            key_id: Old key ID to rotate.

        Returns:
            Dict with the new ``api_key`` and plaintext ``key``.

        Raises:
            NotFoundError: If old key not found.
        """
        result = await self._db.execute(
            select(APIKey).where(
                APIKey.id == key_id,
                APIKey.tenant_id == tenant_id,
            )
        )
        old_key = result.scalar_one_or_none()
        if old_key is None:
            raise NotFoundError(
                "API key not found.", error_code="API_KEY_NOT_FOUND"
            )

        # Revoke old
        old_key.revoked_at = datetime.now(timezone.utc)
        old_key.is_active = False
        self._db.add(old_key)

        # Create new with same properties
        new_result = await self.create_key(
            tenant_id=tenant_id,
            user_id=old_key.user_id,
            name=old_key.name,
            scopes=old_key.scopes,
            rate_limit=old_key.rate_limit_per_min,
            access_request_id=old_key.access_request_id,
        )

        logger.info(
            "Rotated API key %s -> %s for tenant %s.",
            key_id,
            new_result["api_key"].id,
            tenant_id,
        )
        return new_result

    async def get_key_usage(
        self,
        tenant_id: uuid.UUID,
        key_id: uuid.UUID,
        **_extra,
    ) -> dict[str, Any]:
        """Get usage statistics for a specific API key.

        Args:
            tenant_id: Owning tenant.
            key_id: API key ID.

        Returns:
            Dict with total_requests, last_used_at, breakdown by endpoint.

        Raises:
            NotFoundError: If key not found.
        """
        # Verify key exists
        key_result = await self._db.execute(
            select(APIKey).where(
                APIKey.id == key_id,
                APIKey.tenant_id == tenant_id,
            )
        )
        api_key = key_result.scalar_one_or_none()
        if api_key is None:
            raise NotFoundError(
                "API key not found.", error_code="API_KEY_NOT_FOUND"
            )

        # Aggregate usage
        endpoint_stats_result = await self._db.execute(
            select(
                APIUsageLog.endpoint,
                func.count().label("count"),
                func.avg(APIUsageLog.response_time_ms).label("avg_response_ms"),
            )
            .where(APIUsageLog.api_key_id == key_id)
            .group_by(APIUsageLog.endpoint)
        )
        by_endpoint = [
            {
                "endpoint": row.endpoint,
                "count": row.count,
                "avg_response_ms": round(float(row.avg_response_ms or 0), 2),
            }
            for row in endpoint_stats_result.all()
        ]

        return {
            "key_id": str(key_id),
            "total_requests": api_key.total_requests,
            "last_used_at": api_key.last_used_at,
            "by_endpoint": by_endpoint,
        }

    # ------------------------------------------------------------------
    # Validate key (for authentication middleware)
    # ------------------------------------------------------------------

    async def validate_key(self, key_string: str, **_extra) -> tuple[APIKey, uuid.UUID]:
        """Validate an API key and return the key record and tenant_id.

        Args:
            key_string: The full plaintext API key.

        Returns:
            Tuple of (APIKey record, tenant_id).

        Raises:
            UnauthorizedError: If the key is invalid, revoked, or expired.
        """
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()

        result = await self._db.execute(
            select(APIKey).where(APIKey.key_hash == key_hash)
        )
        api_key = result.scalar_one_or_none()

        if api_key is None:
            raise UnauthorizedError(
                "Invalid API key.", error_code="INVALID_API_KEY"
            )

        if not api_key.is_active or api_key.revoked_at is not None:
            raise UnauthorizedError(
                "API key has been revoked.", error_code="REVOKED_API_KEY"
            )

        if api_key.expires_at and api_key.expires_at < datetime.now(timezone.utc):
            raise UnauthorizedError(
                "API key has expired.", error_code="EXPIRED_API_KEY"
            )

        # Update last used
        api_key.last_used_at = datetime.now(timezone.utc)
        api_key.total_requests += 1
        self._db.add(api_key)

        return api_key, api_key.tenant_id

    # ------------------------------------------------------------------
    # Record usage
    # ------------------------------------------------------------------

    async def record_usage(
        self,
        api_key_id: uuid.UUID,
        tenant_id: uuid.UUID,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: int | None = None,
        pii_found: int = 0,
        char_count: int | None = None,
        ip: str | None = None,
        **_extra,
    ) -> None:
        """Record an API usage log entry.

        Args:
            api_key_id: Key that was used.
            tenant_id: Owning tenant.
            endpoint: API endpoint path.
            method: HTTP method.
            status_code: Response status code.
            response_time_ms: Request duration.
            pii_found: Number of PII types detected.
            char_count: Input character count.
            ip: Client IP address.
        """
        log = APIUsageLog(
            api_key_id=api_key_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            pii_types_found=pii_found,
            input_char_count=char_count,
            ip_address=ip,
        )
        self._db.add(log)
        # Don't flush -- let the request lifecycle handle commit

    # ==================================================================
    # Access Request Workflows
    # ==================================================================

    async def list_access_requests(
        self,
        tenant_id: uuid.UUID,
        status_filter: str | None = None,
        **_extra,
    ) -> list[APIAccessRequest]:
        """List all API access requests for a tenant.

        Args:
            tenant_id: Owning tenant.
            status_filter: Optional filter (pending, approved, rejected).

        Returns:
            List of APIAccessRequest instances.
        """
        query = select(APIAccessRequest).where(
            APIAccessRequest.tenant_id == tenant_id
        )
        if status_filter:
            query = query.where(APIAccessRequest.status == status_filter)

        result = await self._db.execute(
            query.order_by(APIAccessRequest.created_at.desc())
        )
        return list(result.scalars().all())

    async def create_access_request(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        use_case: str,
        scopes: list[str] | None = None,
        rate_limit: int | None = None,
        **_extra,
    ) -> APIAccessRequest:
        """Create a new API access request.

        Args:
            tenant_id: Owning tenant.
            user_id: Requesting user.
            use_case: Business justification.
            scopes: Requested permission scopes.
            rate_limit: Requested rate limit per minute.

        Returns:
            The created APIAccessRequest.
        """
        # Check for duplicate pending request
        existing = await self._db.execute(
            select(APIAccessRequest).where(
                APIAccessRequest.tenant_id == tenant_id,
                APIAccessRequest.requested_by == user_id,
                APIAccessRequest.status == "pending",
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise ConflictError(
                "You already have a pending access request.",
                error_code="DUPLICATE_ACCESS_REQUEST",
            )

        request = APIAccessRequest(
            tenant_id=tenant_id,
            requested_by=user_id,
            use_case=use_case,
            scopes_requested=scopes or ["detect", "redact"],
            rate_limit_requested=rate_limit,
            status="pending",
        )
        self._db.add(request)
        await self._db.flush()

        logger.info(
            "Created API access request %s for user %s in tenant %s.",
            request.id,
            user_id,
            tenant_id,
        )
        return request

    async def review_access_request(
        self,
        tenant_id: uuid.UUID,
        request_id: uuid.UUID,
        reviewer_id: uuid.UUID,
        status: str,
        notes: str | None = None,
        rate_limit_override: int | None = None,
        scopes_override: list[str] | None = None,
        **_extra,
    ) -> dict[str, Any]:
        """Review (approve/reject) an API access request.

        If approved, automatically creates an API key for the requester.

        Args:
            tenant_id: Owning tenant.
            request_id: Access request ID.
            reviewer_id: User performing the review.
            status: 'approved' or 'rejected'.
            notes: Optional review notes.
            rate_limit_override: Override the requested rate limit.
            scopes_override: Override the requested scopes.

        Returns:
            Dict with ``request`` and optionally ``api_key`` + ``key``.

        Raises:
            NotFoundError: If request not found.
            ValidationError: If status is invalid or request is not pending.
        """
        if status not in ("approved", "rejected"):
            raise ValidationError(
                "status must be 'approved' or 'rejected'.",
                error_code="INVALID_STATUS",
            )

        result = await self._db.execute(
            select(APIAccessRequest).where(
                APIAccessRequest.id == request_id,
                APIAccessRequest.tenant_id == tenant_id,
            )
        )
        request = result.scalar_one_or_none()
        if request is None:
            raise NotFoundError(
                "Access request not found.",
                error_code="ACCESS_REQUEST_NOT_FOUND",
            )

        if request.status != "pending":
            raise ValidationError(
                f"Request has already been {request.status}.",
                error_code="REQUEST_ALREADY_REVIEWED",
            )

        now = datetime.now(timezone.utc)
        request.status = status
        request.reviewed_by = reviewer_id
        request.reviewed_at = now
        request.review_notes = notes
        self._db.add(request)

        response: dict[str, Any] = {"request": request}

        # Auto-create API key on approval
        if status == "approved":
            scopes = scopes_override or request.scopes_requested
            rate_limit = rate_limit_override or request.rate_limit_requested or 60

            key_result = await self.create_key(
                tenant_id=tenant_id,
                user_id=request.requested_by,
                name=f"Key from request {request_id}",
                scopes=scopes,
                rate_limit=rate_limit,
                access_request_id=request.id,
            )
            response["api_key"] = key_result["api_key"]
            response["key"] = key_result["key"]

        await self._db.flush()

        logger.info(
            "Reviewed access request %s -> %s for tenant %s.",
            request_id,
            status,
            tenant_id,
        )
        return response

    # ==================================================================
    # Usage Analytics
    # ==================================================================

    async def get_usage_analytics(
        self,
        tenant_id: uuid.UUID,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        **_extra,
    ) -> dict[str, Any]:
        """Aggregate API usage statistics for a tenant.

        Args:
            tenant_id: Owning tenant.
            date_from: Optional start date.
            date_to: Optional end date.

        Returns:
            Dict with total_requests, total_pii_found, by_endpoint,
            by_status_code, avg_response_time_ms.
        """
        base = select(
            func.count().label("total"),
            func.sum(APIUsageLog.pii_types_found).label("total_pii"),
            func.avg(APIUsageLog.response_time_ms).label("avg_ms"),
        ).where(APIUsageLog.tenant_id == tenant_id)

        if date_from:
            base = base.where(APIUsageLog.created_at >= date_from)
        if date_to:
            base = base.where(APIUsageLog.created_at <= date_to)

        summary_result = await self._db.execute(base)
        summary = summary_result.one()

        # By endpoint
        ep_query = (
            select(
                APIUsageLog.endpoint,
                func.count().label("count"),
            )
            .where(APIUsageLog.tenant_id == tenant_id)
            .group_by(APIUsageLog.endpoint)
        )
        if date_from:
            ep_query = ep_query.where(APIUsageLog.created_at >= date_from)
        if date_to:
            ep_query = ep_query.where(APIUsageLog.created_at <= date_to)

        ep_result = await self._db.execute(ep_query)
        by_endpoint = [
            {"endpoint": row.endpoint, "count": row.count}
            for row in ep_result.all()
        ]

        # By status code
        sc_query = (
            select(
                APIUsageLog.status_code,
                func.count().label("count"),
            )
            .where(APIUsageLog.tenant_id == tenant_id)
            .group_by(APIUsageLog.status_code)
        )
        if date_from:
            sc_query = sc_query.where(APIUsageLog.created_at >= date_from)
        if date_to:
            sc_query = sc_query.where(APIUsageLog.created_at <= date_to)

        sc_result = await self._db.execute(sc_query)
        by_status = [
            {"status_code": row.status_code, "count": row.count}
            for row in sc_result.all()
        ]

        return {
            "total_requests": summary.total or 0,
            "total_pii_found": int(summary.total_pii or 0),
            "avg_response_time_ms": round(float(summary.avg_ms or 0), 2),
            "by_endpoint": by_endpoint,
            "by_status_code": by_status,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions (proxy to APIKeyService)
# ---------------------------------------------------------------------------

async def list_keys(db=None, **kw):
    db = db or kw.pop("db", None)
    return await APIKeyService(db).list_keys(**kw)


async def create_key(db=None, **kw):
    db = db or kw.pop("db", None)
    return await APIKeyService(db).create_key(**kw)


async def revoke_key(db=None, **kw):
    db = db or kw.pop("db", None)
    return await APIKeyService(db).revoke_key(**kw)


async def rotate_key(db=None, **kw):
    db = db or kw.pop("db", None)
    return await APIKeyService(db).rotate_key(**kw)


async def get_key_usage(db=None, **kw):
    db = db or kw.pop("db", None)
    return await APIKeyService(db).get_key_usage(**kw)


async def validate_key(db=None, **kw):
    db = db or kw.pop("db", None)
    return await APIKeyService(db).validate_key(**kw)


async def record_usage(db=None, **kw):
    db = db or kw.pop("db", None)
    return await APIKeyService(db).record_usage(**kw)


async def list_access_requests(db=None, **kw):
    db = db or kw.pop("db", None)
    return await APIKeyService(db).list_access_requests(**kw)


async def create_access_request(db=None, **kw):
    db = db or kw.pop("db", None)
    return await APIKeyService(db).create_access_request(**kw)


async def review_access_request(db=None, **kw):
    db = db or kw.pop("db", None)
    return await APIKeyService(db).review_access_request(**kw)


async def get_usage_analytics(db=None, **kw):
    db = db or kw.pop("db", None)
    return await APIKeyService(db).get_usage_analytics(**kw)


async def aggregate_usage(db=None, **kw):
    db = db or kw.pop("db", None)
    return await APIKeyService(db).get_usage_analytics(**kw)
