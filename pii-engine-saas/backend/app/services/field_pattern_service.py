"""
Field pattern service -- manage system and tenant-level field label patterns
used by the PatternFieldDetector.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.models.field_pattern import FieldPattern, TenantFieldPattern

logger = logging.getLogger(__name__)


class FieldPatternService:
    """Business logic for field-pattern rule management."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # List patterns (merged system + tenant)
    # ------------------------------------------------------------------

    async def list_rules(
        self,
        tenant_id: uuid.UUID,
        pii_type_name: str | None = None,
        **_extra,
    ) -> list[dict]:
        """Return all field patterns merged with tenant customisations.

        System patterns form the base.  Tenant patterns can extend the
        set for a given PII type.  A disabled tenant pattern suppresses
        the matching system entry.

        Args:
            tenant_id: Current tenant.
            pii_type_name: Optional PII type filter.

        Returns:
            List of merged pattern dicts.
        """
        # System patterns
        sys_query = select(FieldPattern)
        if pii_type_name:
            sys_query = sys_query.where(
                FieldPattern.pii_type_name == pii_type_name
            )
        sys_result = await self._db.execute(
            sys_query.order_by(
                FieldPattern.pii_type_name, FieldPattern.sort_order
            )
        )
        system_patterns = list(sys_result.scalars().all())

        # Tenant patterns
        tenant_query = select(TenantFieldPattern).where(
            TenantFieldPattern.tenant_id == tenant_id,
            TenantFieldPattern.deleted_at.is_(None),
        )
        if pii_type_name:
            tenant_query = tenant_query.where(
                TenantFieldPattern.pii_type_name == pii_type_name
            )
        tenant_result = await self._db.execute(tenant_query)
        tenant_patterns = list(tenant_result.scalars().all())

        # Build a lookup of tenant patterns for override detection
        tenant_lookup: dict[tuple[str, str], TenantFieldPattern] = {
            (tp.pii_type_name, tp.label_pattern): tp for tp in tenant_patterns
        }

        merged: list[dict] = []
        seen_keys: set[tuple[str, str]] = set()

        # System patterns (possibly overridden)
        for sp in system_patterns:
            key = (sp.pii_type_name, sp.label_pattern)
            seen_keys.add(key)
            override = tenant_lookup.get(key)
            if override is not None:
                merged.append({
                    "id": override.id,
                    "source": "tenant_override",
                    "pii_type_name": override.pii_type_name,
                    "label_pattern": override.label_pattern,
                    "is_enabled": override.is_enabled,
                    "sort_order": sp.sort_order,
                })
            else:
                merged.append({
                    "id": sp.id,
                    "source": "system",
                    "pii_type_name": sp.pii_type_name,
                    "label_pattern": sp.label_pattern,
                    "is_enabled": True,
                    "sort_order": sp.sort_order,
                })

        # Tenant-only patterns
        for tp in tenant_patterns:
            key = (tp.pii_type_name, tp.label_pattern)
            if key not in seen_keys:
                merged.append({
                    "id": tp.id,
                    "source": "tenant",
                    "pii_type_name": tp.pii_type_name,
                    "label_pattern": tp.label_pattern,
                    "is_enabled": tp.is_enabled,
                    "sort_order": 0,
                })

        return merged

    # ------------------------------------------------------------------
    # Get patterns for a specific type
    # ------------------------------------------------------------------

    async def get_rules_for_type(
        self,
        tenant_id: uuid.UUID,
        pii_type_name: str,
        **_extra,
    ) -> list[dict]:
        """Return merged field patterns for a specific PII type.

        Args:
            tenant_id: Current tenant.
            pii_type_name: PII type to filter.

        Returns:
            List of merged pattern dicts.
        """
        return await self.list_rules(tenant_id, pii_type_name=pii_type_name)

    # ------------------------------------------------------------------
    # Create tenant pattern
    # ------------------------------------------------------------------

    async def create_rule(
        self,
        tenant_id: uuid.UUID,
        data: dict[str, Any],
        **_extra,
    ) -> TenantFieldPattern:
        """Create a new tenant-specific field pattern.

        Args:
            tenant_id: Owning tenant.
            data: Fields: pii_type_name, label_pattern, is_enabled,
                  created_by.

        Returns:
            The created TenantFieldPattern.

        Raises:
            ConflictError: If a duplicate (tenant, type, pattern)
                           combination exists.
        """
        existing = await self._db.execute(
            select(TenantFieldPattern).where(
                TenantFieldPattern.tenant_id == tenant_id,
                TenantFieldPattern.pii_type_name == data["pii_type_name"],
                TenantFieldPattern.label_pattern == data["label_pattern"],
                TenantFieldPattern.deleted_at.is_(None),
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise ConflictError(
                "A field pattern with this label already exists for the PII type.",
                error_code="FIELD_PATTERN_EXISTS",
            )

        pattern = TenantFieldPattern(
            tenant_id=tenant_id,
            pii_type_name=data["pii_type_name"],
            label_pattern=data["label_pattern"],
            is_enabled=data.get("is_enabled", True),
            created_by=data.get("created_by"),
        )
        self._db.add(pattern)
        await self._db.flush()

        logger.info(
            "Created tenant field pattern %d for '%s' in tenant %s.",
            pattern.id,
            pattern.pii_type_name,
            tenant_id,
        )
        return pattern

    # ------------------------------------------------------------------
    # Update tenant pattern
    # ------------------------------------------------------------------

    async def update_rule(
        self,
        tenant_id: uuid.UUID,
        rule_id: int,
        data: dict[str, Any],
        **_extra,
    ) -> TenantFieldPattern:
        """Update an existing tenant field pattern.

        Args:
            tenant_id: Owning tenant.
            rule_id: Pattern ID.
            data: Fields to update (label_pattern, is_enabled).

        Returns:
            Updated TenantFieldPattern.

        Raises:
            NotFoundError: If pattern not found.
        """
        result = await self._db.execute(
            select(TenantFieldPattern).where(
                TenantFieldPattern.id == rule_id,
                TenantFieldPattern.tenant_id == tenant_id,
                TenantFieldPattern.deleted_at.is_(None),
            )
        )
        pattern = result.scalar_one_or_none()
        if pattern is None:
            raise NotFoundError(
                "Field pattern not found.",
                error_code="FIELD_PATTERN_NOT_FOUND",
            )

        allowed_fields = {"label_pattern", "is_enabled", "pii_type_name"}
        for key, value in data.items():
            if key in allowed_fields and value is not None:
                setattr(pattern, key, value)

        self._db.add(pattern)
        await self._db.flush()

        logger.info(
            "Updated tenant field pattern %d in tenant %s.",
            rule_id,
            tenant_id,
        )
        return pattern

    # ------------------------------------------------------------------
    # Soft-delete tenant pattern
    # ------------------------------------------------------------------

    async def delete_rule(
        self,
        tenant_id: uuid.UUID,
        rule_id: int,
        **_extra,
    ) -> None:
        """Soft-delete a tenant field pattern.

        Args:
            tenant_id: Owning tenant.
            rule_id: Pattern ID.

        Raises:
            NotFoundError: If pattern not found.
        """
        result = await self._db.execute(
            select(TenantFieldPattern).where(
                TenantFieldPattern.id == rule_id,
                TenantFieldPattern.tenant_id == tenant_id,
                TenantFieldPattern.deleted_at.is_(None),
            )
        )
        pattern = result.scalar_one_or_none()
        if pattern is None:
            raise NotFoundError(
                "Field pattern not found.",
                error_code="FIELD_PATTERN_NOT_FOUND",
            )

        pattern.deleted_at = datetime.now(timezone.utc)
        self._db.add(pattern)
        await self._db.flush()

        logger.info(
            "Soft-deleted tenant field pattern %d in tenant %s.",
            rule_id,
            tenant_id,
        )

    # ------------------------------------------------------------------
    # Test pattern (label matching)
    # ------------------------------------------------------------------

    async def test_pattern(
        self,
        pattern: str,
        test_text: str,
        **_extra,
    ) -> dict:
        """Test a field label pattern against sample text.

        Performs a case-insensitive search for the label pattern
        in the test text.

        Args:
            pattern: The label pattern string.
            test_text: Text to test against.

        Returns:
            Dict with ``matches`` (list of match locations).
        """
        import re as re_mod

        matches = []
        try:
            for m in re_mod.finditer(
                re_mod.escape(pattern), test_text, re_mod.IGNORECASE
            ):
                matches.append({
                    "match": m.group(),
                    "start": m.start(),
                    "end": m.end(),
                })
        except re_mod.error:
            pass

        return {
            "pattern": pattern,
            "matches": matches,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions (proxy to FieldPatternService)
# ---------------------------------------------------------------------------

async def list_rules(db=None, **kw):
    db = db or kw.pop("db", None)
    return await FieldPatternService(db).list_rules(**kw)


async def list_patterns(db=None, **kw):
    db = db or kw.pop("db", None)
    return await FieldPatternService(db).list_rules(**kw)


async def get_rules_for_type(db=None, **kw):
    db = db or kw.pop("db", None)
    return await FieldPatternService(db).get_rules_for_type(**kw)


async def get_patterns_by_type(db=None, **kw):
    db = db or kw.pop("db", None)
    return await FieldPatternService(db).get_rules_for_type(**kw)


async def create_rule(db=None, **kw):
    db = db or kw.pop("db", None)
    return await FieldPatternService(db).create_rule(**kw)


async def create_pattern(db=None, **kw):
    db = db or kw.pop("db", None)
    return await FieldPatternService(db).create_rule(**kw)


async def update_rule(db=None, **kw):
    db = db or kw.pop("db", None)
    return await FieldPatternService(db).update_rule(**kw)


async def update_pattern(db=None, **kw):
    db = db or kw.pop("db", None)
    return await FieldPatternService(db).update_rule(**kw)


async def delete_rule(db=None, **kw):
    db = db or kw.pop("db", None)
    return await FieldPatternService(db).delete_rule(**kw)


async def delete_pattern(db=None, **kw):
    db = db or kw.pop("db", None)
    return await FieldPatternService(db).delete_rule(**kw)


async def test_pattern(db=None, **kw):
    db = db or kw.pop("db", None)
    return await FieldPatternService(db).test_pattern(**kw)
