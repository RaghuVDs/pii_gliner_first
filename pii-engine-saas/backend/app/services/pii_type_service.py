"""
PII type management service -- listing, filtering, tenant config overrides,
batch operations, custom type CRUD.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.models.pii_type import (
    CustomPIIType,
    PIICategory,
    PIIType,
    TenantPIIConfig,
)

logger = logging.getLogger(__name__)


class PIITypeService:
    """Business logic for PII type browsing and per-tenant configuration."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Categories
    # ------------------------------------------------------------------

    async def list_categories(self, **_extra) -> list[PIICategory]:
        """Return all PII categories ordered by tier and sort_order.

        Returns:
            List of PIICategory instances.
        """
        result = await self._db.execute(
            select(PIICategory).order_by(
                PIICategory.tier, PIICategory.sort_order
            )
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # List PII types (merged with tenant config)
    # ------------------------------------------------------------------

    async def list_pii_types(
        self,
        tenant_id: uuid.UUID,
        *,
        category_id: int | None = None,
        tier: int | None = None,
        is_enabled: bool | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 50,
        **_extra,
    ) -> dict:
        """Return paginated PII types with tenant-specific config merged.

        Each item contains the system PII type fields plus
        ``tenant_config`` (the TenantPIIConfig row, or None if the
        tenant has not customised it).

        Args:
            tenant_id: Current tenant.
            category_id: Optional filter by category.
            tier: Optional filter by tier.
            is_enabled: Optional filter by enabled status (in tenant config).
            search: Optional text search on name / display_name.
            page: 1-indexed page.
            page_size: Items per page.

        Returns:
            Paginated dict with items, total, page, page_size, total_pages.
        """
        base = (
            select(PIIType)
            .outerjoin(
                TenantPIIConfig,
                (TenantPIIConfig.pii_type_id == PIIType.id)
                & (TenantPIIConfig.tenant_id == tenant_id),
            )
            .options(selectinload(PIIType.category))
        )

        if category_id is not None:
            base = base.where(PIIType.category_id == category_id)

        if tier is not None:
            base = base.join(PIICategory).where(PIICategory.tier == tier)

        if search:
            pattern = f"%{search}%"
            base = base.where(
                or_(
                    PIIType.name.ilike(pattern),
                    PIIType.display_name.ilike(pattern),
                )
            )

        if is_enabled is not None:
            if is_enabled:
                # Enabled = config exists and is_enabled, or no config (default enabled)
                base = base.where(
                    or_(
                        TenantPIIConfig.is_enabled.is_(True),
                        TenantPIIConfig.id.is_(None),
                    )
                )
            else:
                base = base.where(TenantPIIConfig.is_enabled.is_(False))

        # Count
        count_q = select(func.count()).select_from(base.subquery())
        total_result = await self._db.execute(count_q)
        total = total_result.scalar() or 0

        # Fetch page
        offset = (page - 1) * page_size
        rows_result = await self._db.execute(
            base.order_by(PIIType.category_id, PIIType.id)
            .offset(offset)
            .limit(page_size)
        )
        pii_types = list(rows_result.scalars().unique().all())

        # Fetch tenant configs for these types in one query
        type_ids = [pt.id for pt in pii_types]
        configs_result = await self._db.execute(
            select(TenantPIIConfig).where(
                TenantPIIConfig.tenant_id == tenant_id,
                TenantPIIConfig.pii_type_id.in_(type_ids),
            )
        )
        config_map = {
            c.pii_type_id: c for c in configs_result.scalars().all()
        }

        items = []
        for pt in pii_types:
            items.append({
                "pii_type": pt,
                "tenant_config": config_map.get(pt.id),
            })

        total_pages = (total + page_size - 1) // page_size if page_size else 0

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        }

    # ------------------------------------------------------------------
    # Get single PII type with tenant config
    # ------------------------------------------------------------------

    async def get_pii_type(
        self,
        pii_type_id: int,
        tenant_id: uuid.UUID,
        **_extra,
    ) -> dict:
        """Fetch a single PII type with its tenant configuration.

        Args:
            pii_type_id: The system PII type ID.
            tenant_id: Current tenant.

        Returns:
            Dict with ``pii_type`` and ``tenant_config``.

        Raises:
            NotFoundError: If the PII type does not exist.
        """
        result = await self._db.execute(
            select(PIIType)
            .options(selectinload(PIIType.category))
            .where(PIIType.id == pii_type_id)
        )
        pii_type = result.scalar_one_or_none()
        if pii_type is None:
            raise NotFoundError(
                "PII type not found.", error_code="PII_TYPE_NOT_FOUND"
            )

        config_result = await self._db.execute(
            select(TenantPIIConfig).where(
                TenantPIIConfig.tenant_id == tenant_id,
                TenantPIIConfig.pii_type_id == pii_type_id,
            )
        )
        tenant_config = config_result.scalar_one_or_none()

        return {
            "pii_type": pii_type,
            "tenant_config": tenant_config,
        }

    # ------------------------------------------------------------------
    # Update tenant config for a PII type
    # ------------------------------------------------------------------

    async def update_config(
        self,
        tenant_id: uuid.UUID,
        pii_type_id: int,
        data: dict[str, Any],
        **_extra,
    ) -> TenantPIIConfig:
        """Create or update tenant-specific config for a PII type.

        Args:
            tenant_id: Current tenant.
            pii_type_id: System PII type ID.
            data: Fields to set (is_enabled, custom_threshold,
                  custom_aliases, notes).

        Returns:
            The created or updated TenantPIIConfig.

        Raises:
            NotFoundError: If the PII type does not exist.
        """
        # Verify PII type exists
        pt_result = await self._db.execute(
            select(PIIType).where(PIIType.id == pii_type_id)
        )
        if pt_result.scalar_one_or_none() is None:
            raise NotFoundError(
                "PII type not found.", error_code="PII_TYPE_NOT_FOUND"
            )

        # Upsert tenant config
        config_result = await self._db.execute(
            select(TenantPIIConfig).where(
                TenantPIIConfig.tenant_id == tenant_id,
                TenantPIIConfig.pii_type_id == pii_type_id,
            )
        )
        config = config_result.scalar_one_or_none()

        if config is None:
            config = TenantPIIConfig(
                tenant_id=tenant_id,
                pii_type_id=pii_type_id,
            )

        allowed_fields = {"is_enabled", "custom_threshold", "custom_aliases", "notes"}
        for key, value in data.items():
            if key in allowed_fields:
                setattr(config, key, value)

        self._db.add(config)
        await self._db.flush()

        logger.info(
            "Updated PII config for tenant %s, pii_type_id %d.",
            tenant_id,
            pii_type_id,
        )
        return config

    # ------------------------------------------------------------------
    # Batch enable/disable
    # ------------------------------------------------------------------

    async def batch_update_config(
        self,
        tenant_id: uuid.UUID,
        pii_type_ids: list[int],
        is_enabled: bool,
        **_extra,
    ) -> int:
        """Batch enable or disable PII types for a tenant.

        Creates TenantPIIConfig rows where they don't exist, and
        updates existing ones.

        Args:
            tenant_id: Current tenant.
            pii_type_ids: List of PII type IDs.
            is_enabled: Whether to enable or disable.

        Returns:
            Number of configs affected.
        """
        # Verify all IDs are valid
        result = await self._db.execute(
            select(PIIType.id).where(PIIType.id.in_(pii_type_ids))
        )
        valid_ids = set(result.scalars().all())
        invalid_ids = set(pii_type_ids) - valid_ids
        if invalid_ids:
            raise ValidationError(
                f"Invalid PII type IDs: {sorted(invalid_ids)}",
                error_code="INVALID_PII_TYPE_IDS",
            )

        # Fetch existing configs
        existing_result = await self._db.execute(
            select(TenantPIIConfig).where(
                TenantPIIConfig.tenant_id == tenant_id,
                TenantPIIConfig.pii_type_id.in_(pii_type_ids),
            )
        )
        existing_map = {
            c.pii_type_id: c for c in existing_result.scalars().all()
        }

        count = 0
        for pii_type_id in pii_type_ids:
            config = existing_map.get(pii_type_id)
            if config is None:
                config = TenantPIIConfig(
                    tenant_id=tenant_id,
                    pii_type_id=pii_type_id,
                    is_enabled=is_enabled,
                )
            else:
                config.is_enabled = is_enabled
            self._db.add(config)
            count += 1

        await self._db.flush()

        logger.info(
            "Batch %s %d PII types for tenant %s.",
            "enabled" if is_enabled else "disabled",
            count,
            tenant_id,
        )
        return count

    # ------------------------------------------------------------------
    # Reset to defaults
    # ------------------------------------------------------------------

    async def reset_defaults(self, tenant_id: uuid.UUID, **_extra) -> int:
        """Delete all tenant PII configs, reverting to system defaults.

        Args:
            tenant_id: Current tenant.

        Returns:
            Number of config rows deleted.
        """
        result = await self._db.execute(
            delete(TenantPIIConfig)
            .where(TenantPIIConfig.tenant_id == tenant_id)
            .returning(TenantPIIConfig.id)
        )
        deleted = len(list(result.scalars().all()))
        await self._db.flush()

        logger.info(
            "Reset %d PII type configs to defaults for tenant %s.",
            deleted,
            tenant_id,
        )
        return deleted

    # ------------------------------------------------------------------
    # Custom PII types
    # ------------------------------------------------------------------

    async def create_custom_type(
        self,
        tenant_id: uuid.UUID,
        data: dict[str, Any],
        **_extra,
    ) -> CustomPIIType:
        """Create a tenant-specific custom PII type.

        Args:
            tenant_id: Owning tenant.
            data: Required fields: name, display_name.  Optional:
                  description, category_id, gliner_aliases,
                  default_threshold, compliance_tags, created_by.

        Returns:
            The created CustomPIIType.

        Raises:
            ConflictError: If a custom type with that name already exists
                           for this tenant.
        """
        # Check uniqueness within tenant
        existing = await self._db.execute(
            select(CustomPIIType).where(
                CustomPIIType.tenant_id == tenant_id,
                CustomPIIType.name == data["name"],
                CustomPIIType.deleted_at.is_(None),
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise ConflictError(
                f"Custom PII type '{data['name']}' already exists for this tenant.",
                error_code="CUSTOM_PII_TYPE_EXISTS",
            )

        custom_type = CustomPIIType(
            tenant_id=tenant_id,
            name=data["name"],
            display_name=data["display_name"],
            description=data.get("description"),
            category_id=data.get("category_id"),
            gliner_aliases=data.get("gliner_aliases", []),
            default_threshold=data.get("default_threshold", 0.4),
            compliance_tags=data.get("compliance_tags", []),
            created_by=data.get("created_by"),
        )
        self._db.add(custom_type)
        await self._db.flush()

        logger.info(
            "Created custom PII type '%s' for tenant %s.",
            custom_type.name,
            tenant_id,
        )
        return custom_type

    async def update_custom_type(
        self,
        tenant_id: uuid.UUID,
        custom_id: int,
        data: dict[str, Any],
        **_extra,
    ) -> CustomPIIType:
        """Update a tenant's custom PII type.

        Args:
            tenant_id: Owning tenant.
            custom_id: Custom PII type ID.
            data: Fields to update.

        Returns:
            Updated CustomPIIType.

        Raises:
            NotFoundError: If the custom type doesn't exist.
        """
        result = await self._db.execute(
            select(CustomPIIType).where(
                CustomPIIType.id == custom_id,
                CustomPIIType.tenant_id == tenant_id,
                CustomPIIType.deleted_at.is_(None),
            )
        )
        custom_type = result.scalar_one_or_none()
        if custom_type is None:
            raise NotFoundError(
                "Custom PII type not found.",
                error_code="CUSTOM_PII_TYPE_NOT_FOUND",
            )

        allowed_fields = {
            "display_name", "description", "category_id", "gliner_aliases",
            "default_threshold", "is_enabled", "compliance_tags",
        }
        for key, value in data.items():
            if key in allowed_fields and value is not None:
                setattr(custom_type, key, value)

        self._db.add(custom_type)
        await self._db.flush()

        logger.info(
            "Updated custom PII type %d for tenant %s.",
            custom_id,
            tenant_id,
        )
        return custom_type

    async def delete_custom_type(
        self,
        tenant_id: uuid.UUID,
        custom_id: int,
        **_extra,
    ) -> None:
        """Soft-delete a tenant's custom PII type.

        Args:
            tenant_id: Owning tenant.
            custom_id: Custom PII type ID.

        Raises:
            NotFoundError: If the custom type doesn't exist.
        """
        result = await self._db.execute(
            select(CustomPIIType).where(
                CustomPIIType.id == custom_id,
                CustomPIIType.tenant_id == tenant_id,
                CustomPIIType.deleted_at.is_(None),
            )
        )
        custom_type = result.scalar_one_or_none()
        if custom_type is None:
            raise NotFoundError(
                "Custom PII type not found.",
                error_code="CUSTOM_PII_TYPE_NOT_FOUND",
            )

        custom_type.deleted_at = datetime.now(timezone.utc)
        self._db.add(custom_type)
        await self._db.flush()

        logger.info(
            "Soft-deleted custom PII type %d for tenant %s.",
            custom_id,
            tenant_id,
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions (proxy to PIITypeService)
# ---------------------------------------------------------------------------

async def list_categories(db=None, **kw):
    db = db or kw.pop("db", None)
    return await PIITypeService(db).list_categories(**kw)


async def list_pii_types(db=None, **kw):
    db = db or kw.pop("db", None)
    return await PIITypeService(db).list_pii_types(**kw)


async def get_pii_type(db=None, **kw):
    db = db or kw.pop("db", None)
    return await PIITypeService(db).get_pii_type(**kw)


async def update_config(db=None, **kw):
    db = db or kw.pop("db", None)
    return await PIITypeService(db).update_config(**kw)


async def update_tenant_config(db=None, **kw):
    db = db or kw.pop("db", None)
    return await PIITypeService(db).update_config(**kw)


async def batch_update_config(db=None, **kw):
    db = db or kw.pop("db", None)
    return await PIITypeService(db).batch_update_config(**kw)


async def batch_config(db=None, **kw):
    db = db or kw.pop("db", None)
    return await PIITypeService(db).batch_update_config(**kw)


async def reset_defaults(db=None, **kw):
    db = db or kw.pop("db", None)
    return await PIITypeService(db).reset_defaults(**kw)


async def create_custom_type(db=None, **kw):
    db = db or kw.pop("db", None)
    return await PIITypeService(db).create_custom_type(**kw)


async def create_custom(db=None, **kw):
    db = db or kw.pop("db", None)
    return await PIITypeService(db).create_custom_type(**kw)


async def update_custom_type(db=None, **kw):
    db = db or kw.pop("db", None)
    return await PIITypeService(db).update_custom_type(**kw)


async def update_custom(db=None, **kw):
    db = db or kw.pop("db", None)
    return await PIITypeService(db).update_custom_type(**kw)


async def delete_custom_type(db=None, **kw):
    db = db or kw.pop("db", None)
    return await PIITypeService(db).delete_custom_type(**kw)


async def delete_custom(db=None, **kw):
    db = db or kw.pop("db", None)
    return await PIITypeService(db).delete_custom_type(**kw)
