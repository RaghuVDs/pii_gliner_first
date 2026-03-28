"""
Tenant management service -- tenant details, settings, usage tracking,
and onboarding (cloning default PII configs).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.models.api_key import APIKey
from app.models.pii_type import PIIType, TenantPIIConfig
from app.models.tenant import Tenant
from app.models.user import User

logger = logging.getLogger(__name__)


class TenantService:
    """Business logic for tenant (organisation) management."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Get tenant
    # ------------------------------------------------------------------

    async def get_tenant(self, tenant_id: uuid.UUID, **_extra) -> Tenant:
        """Fetch tenant details.

        Args:
            tenant_id: Target tenant ID.

        Returns:
            Tenant ORM instance.

        Raises:
            NotFoundError: If tenant does not exist or is soft-deleted.
        """
        result = await self._db.execute(
            select(Tenant).where(
                Tenant.id == tenant_id,
                Tenant.deleted_at.is_(None),
            )
        )
        tenant = result.scalar_one_or_none()
        if tenant is None:
            raise NotFoundError("Tenant not found.", error_code="TENANT_NOT_FOUND")
        return tenant

    # ------------------------------------------------------------------
    # Update tenant
    # ------------------------------------------------------------------

    async def update_tenant(
        self,
        tenant_id: uuid.UUID,
        data: dict[str, Any],
        **_extra,
    ) -> Tenant:
        """Update tenant settings and limits.

        Args:
            tenant_id: Target tenant.
            data: Dict of fields to update (name, settings, max_users,
                  max_api_keys, max_monthly_scans).

        Returns:
            Updated Tenant instance.

        Raises:
            NotFoundError: If tenant does not exist.
        """
        tenant = await self.get_tenant(tenant_id)

        allowed_fields = {"name", "settings", "max_users", "max_api_keys", "max_monthly_scans"}
        for key, value in data.items():
            if key in allowed_fields and value is not None:
                setattr(tenant, key, value)

        self._db.add(tenant)
        await self._db.flush()

        logger.info("Tenant %s updated: %s.", tenant_id, list(data.keys()))
        return tenant

    # ------------------------------------------------------------------
    # Usage stats
    # ------------------------------------------------------------------

    async def get_usage(self, tenant_id: uuid.UUID, **_extra) -> dict:
        """Return current usage statistics for a tenant.

        Args:
            tenant_id: Target tenant.

        Returns:
            Dict with current_month_scans, max_monthly_scans,
            active_users, max_users, active_api_keys, max_api_keys.
        """
        tenant = await self.get_tenant(tenant_id)

        # Active users
        user_count_result = await self._db.execute(
            select(func.count()).where(
                User.tenant_id == tenant_id,
                User.is_active.is_(True),
                User.deleted_at.is_(None),
            )
        )
        active_users = user_count_result.scalar() or 0

        # Active API keys
        api_key_count_result = await self._db.execute(
            select(func.count()).where(
                APIKey.tenant_id == tenant_id,
                APIKey.is_active.is_(True),
                APIKey.revoked_at.is_(None),
            )
        )
        active_api_keys = api_key_count_result.scalar() or 0

        # Scan count for current month -- derived from API usage logs or
        # a dedicated counter.  For now, use the tenant settings as the
        # source of truth for the cap; actual count comes from MongoDB
        # detection_stats in a full implementation.
        current_month_scans = 0  # TODO: aggregate from detection_stats

        return {
            "current_month_scans": current_month_scans,
            "max_monthly_scans": tenant.max_monthly_scans,
            "active_users": active_users,
            "max_users": tenant.max_users,
            "active_api_keys": active_api_keys,
            "max_api_keys": tenant.max_api_keys,
        }

    # ------------------------------------------------------------------
    # Onboarding -- clone system PII types as tenant configs
    # ------------------------------------------------------------------

    async def onboard_tenant(self, tenant_id: uuid.UUID, **_extra) -> int:
        """Clone all system PII type definitions into tenant_pii_configs.

        Creates a TenantPIIConfig row for every PIIType record, giving
        the tenant full control over which types are enabled and at what
        threshold.

        Args:
            tenant_id: Target tenant.

        Returns:
            Number of PII type configs created.
        """
        tenant = await self.get_tenant(tenant_id)

        # Remove any existing configs first (idempotent re-onboarding)
        await self._db.execute(
            delete(TenantPIIConfig).where(
                TenantPIIConfig.tenant_id == tenant_id
            )
        )

        # Fetch all system PII types
        result = await self._db.execute(
            select(PIIType).where(PIIType.is_system.is_(True))
        )
        pii_types = list(result.scalars().all())

        configs_created = 0
        for pii_type in pii_types:
            config = TenantPIIConfig(
                tenant_id=tenant_id,
                pii_type_id=pii_type.id,
                is_enabled=True,
                custom_threshold=None,  # Use system default
                custom_aliases=None,
            )
            self._db.add(config)
            configs_created += 1

        # Mark tenant as onboarded
        tenant.onboarded_at = datetime.now(timezone.utc)
        self._db.add(tenant)
        await self._db.flush()

        logger.info(
            "Onboarded tenant %s -- created %d PII type configs.",
            tenant_id,
            configs_created,
        )
        return configs_created


# ---------------------------------------------------------------------------
# Module-level convenience functions (proxy to TenantService)
# ---------------------------------------------------------------------------

async def get_tenant(db=None, **kw):
    db = db or kw.pop("db", None)
    return await TenantService(db).get_tenant(**kw)


async def update_tenant(db=None, **kw):
    db = db or kw.pop("db", None)
    return await TenantService(db).update_tenant(**kw)


async def get_usage(db=None, **kw):
    db = db or kw.pop("db", None)
    return await TenantService(db).get_usage(**kw)


async def onboard_tenant(db=None, **kw):
    db = db or kw.pop("db", None)
    return await TenantService(db).onboard_tenant(**kw)


async def onboard(db=None, **kw):
    db = db or kw.pop("db", None)
    return await TenantService(db).onboard_tenant(**kw)
