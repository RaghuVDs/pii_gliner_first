"""
Context rule service -- manage system and tenant-level context keyword rules
used by the ContextDetector for label promotion / demotion.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.models.context_rule import ContextRule, TenantContextRule

logger = logging.getLogger(__name__)


class ContextRuleService:
    """Business logic for context-keyword rule management."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # List rules (merged system + tenant)
    # ------------------------------------------------------------------

    async def list_rules(
        self,
        tenant_id: uuid.UUID,
        pii_type_name: str | None = None,
        **_extra,
    ) -> list[dict]:
        """Return all context rules merged with tenant overrides.

        System rules are the base.  Tenant rules extend or supplement
        them.  The merge key is (pii_type_name, keyword_pattern,
        is_negative).

        Args:
            tenant_id: Current tenant.
            pii_type_name: Optional PII type filter.

        Returns:
            List of merged rule dicts.
        """
        # System rules
        sys_query = select(ContextRule)
        if pii_type_name:
            sys_query = sys_query.where(
                ContextRule.pii_type_name == pii_type_name
            )
        sys_result = await self._db.execute(
            sys_query.order_by(ContextRule.pii_type_name, ContextRule.id)
        )
        system_rules = list(sys_result.scalars().all())

        # Tenant rules
        tenant_query = select(TenantContextRule).where(
            TenantContextRule.tenant_id == tenant_id,
            TenantContextRule.deleted_at.is_(None),
        )
        if pii_type_name:
            tenant_query = tenant_query.where(
                TenantContextRule.pii_type_name == pii_type_name
            )
        tenant_result = await self._db.execute(
            tenant_query.order_by(
                TenantContextRule.pii_type_name, TenantContextRule.id
            )
        )
        tenant_rules = list(tenant_result.scalars().all())

        # Build tenant lookup by semantic key
        tenant_lookup: dict[tuple[str, str, bool], TenantContextRule] = {
            (tr.pii_type_name, tr.keyword_pattern, tr.is_negative): tr
            for tr in tenant_rules
        }

        merged: list[dict] = []
        seen_keys: set[tuple[str, str, bool]] = set()

        for sr in system_rules:
            key = (sr.pii_type_name, sr.keyword_pattern, sr.is_negative)
            seen_keys.add(key)
            override = tenant_lookup.get(key)
            if override is not None:
                merged.append({
                    "id": override.id,
                    "source": "tenant_override",
                    "pii_type_name": override.pii_type_name,
                    "keyword_pattern": override.keyword_pattern,
                    "is_negative": override.is_negative,
                    "is_enabled": override.is_enabled,
                    "origin": override.source,
                })
            else:
                merged.append({
                    "id": sr.id,
                    "source": "system",
                    "pii_type_name": sr.pii_type_name,
                    "keyword_pattern": sr.keyword_pattern,
                    "is_negative": sr.is_negative,
                    "is_enabled": True,
                    "origin": sr.source,
                })

        # Tenant-only rules
        for tr in tenant_rules:
            key = (tr.pii_type_name, tr.keyword_pattern, tr.is_negative)
            if key not in seen_keys:
                merged.append({
                    "id": tr.id,
                    "source": "tenant",
                    "pii_type_name": tr.pii_type_name,
                    "keyword_pattern": tr.keyword_pattern,
                    "is_negative": tr.is_negative,
                    "is_enabled": tr.is_enabled,
                    "origin": tr.source,
                })

        return merged

    # ------------------------------------------------------------------
    # Get rules for a specific type
    # ------------------------------------------------------------------

    async def get_rules_for_type(
        self,
        tenant_id: uuid.UUID,
        pii_type_name: str,
        **_extra,
    ) -> list[dict]:
        """Return merged context rules for a specific PII type.

        Args:
            tenant_id: Current tenant.
            pii_type_name: PII type filter.

        Returns:
            List of merged rule dicts.
        """
        return await self.list_rules(tenant_id, pii_type_name=pii_type_name)

    # ------------------------------------------------------------------
    # Create tenant rule
    # ------------------------------------------------------------------

    async def create_rule(
        self,
        tenant_id: uuid.UUID,
        data: dict[str, Any],
        **_extra,
    ) -> TenantContextRule:
        """Create a new tenant-specific context rule.

        Args:
            tenant_id: Owning tenant.
            data: Fields: pii_type_name, keyword_pattern, is_negative,
                  is_enabled, source, promoted_from_id, created_by.

        Returns:
            The created TenantContextRule.
        """
        rule = TenantContextRule(
            tenant_id=tenant_id,
            pii_type_name=data["pii_type_name"],
            keyword_pattern=data["keyword_pattern"],
            is_negative=data.get("is_negative", False),
            is_enabled=data.get("is_enabled", True),
            source=data.get("source", "manual"),
            promoted_from_id=data.get("promoted_from_id"),
            created_by=data.get("created_by"),
        )
        self._db.add(rule)
        await self._db.flush()

        logger.info(
            "Created tenant context rule %d for '%s' in tenant %s.",
            rule.id,
            rule.pii_type_name,
            tenant_id,
        )
        return rule

    # ------------------------------------------------------------------
    # Update tenant rule
    # ------------------------------------------------------------------

    async def update_rule(
        self,
        tenant_id: uuid.UUID,
        rule_id: int,
        data: dict[str, Any],
        **_extra,
    ) -> TenantContextRule:
        """Update an existing tenant context rule.

        Args:
            tenant_id: Owning tenant.
            rule_id: Rule ID.
            data: Fields to update.

        Returns:
            Updated TenantContextRule.

        Raises:
            NotFoundError: If rule not found.
        """
        result = await self._db.execute(
            select(TenantContextRule).where(
                TenantContextRule.id == rule_id,
                TenantContextRule.tenant_id == tenant_id,
                TenantContextRule.deleted_at.is_(None),
            )
        )
        rule = result.scalar_one_or_none()
        if rule is None:
            raise NotFoundError(
                "Context rule not found.",
                error_code="CONTEXT_RULE_NOT_FOUND",
            )

        allowed_fields = {
            "pii_type_name", "keyword_pattern", "is_negative",
            "is_enabled", "source",
        }
        for key, value in data.items():
            if key in allowed_fields and value is not None:
                setattr(rule, key, value)

        self._db.add(rule)
        await self._db.flush()

        logger.info(
            "Updated tenant context rule %d in tenant %s.",
            rule_id,
            tenant_id,
        )
        return rule

    # ------------------------------------------------------------------
    # Soft-delete tenant rule
    # ------------------------------------------------------------------

    async def delete_rule(
        self,
        tenant_id: uuid.UUID,
        rule_id: int,
        **_extra,
    ) -> None:
        """Soft-delete a tenant context rule.

        Args:
            tenant_id: Owning tenant.
            rule_id: Rule ID.

        Raises:
            NotFoundError: If rule not found.
        """
        result = await self._db.execute(
            select(TenantContextRule).where(
                TenantContextRule.id == rule_id,
                TenantContextRule.tenant_id == tenant_id,
                TenantContextRule.deleted_at.is_(None),
            )
        )
        rule = result.scalar_one_or_none()
        if rule is None:
            raise NotFoundError(
                "Context rule not found.",
                error_code="CONTEXT_RULE_NOT_FOUND",
            )

        rule.deleted_at = datetime.now(timezone.utc)
        self._db.add(rule)
        await self._db.flush()

        logger.info(
            "Soft-deleted tenant context rule %d in tenant %s.",
            rule_id,
            tenant_id,
        )

    # ------------------------------------------------------------------
    # Test pattern
    # ------------------------------------------------------------------

    async def test_pattern(
        self,
        pattern: str,
        test_text: str,
        **_extra,
    ) -> dict:
        """Test a context keyword pattern against sample text.

        Args:
            pattern: The keyword pattern to search for.
            test_text: Text to test against.

        Returns:
            Dict with ``found`` (bool) and ``positions`` (list of start indices).
        """
        lower_text = test_text.lower()
        lower_pattern = pattern.lower()

        positions: list[int] = []
        start = 0
        while True:
            idx = lower_text.find(lower_pattern, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + 1

        return {
            "pattern": pattern,
            "found": len(positions) > 0,
            "positions": positions,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions (proxy to ContextRuleService)
# ---------------------------------------------------------------------------

async def list_rules(db=None, **kw):
    db = db or kw.pop("db", None)
    return await ContextRuleService(db).list_rules(**kw)


async def get_rules_for_type(db=None, **kw):
    db = db or kw.pop("db", None)
    return await ContextRuleService(db).get_rules_for_type(**kw)


async def get_rules_by_type(db=None, **kw):
    db = db or kw.pop("db", None)
    return await ContextRuleService(db).get_rules_for_type(**kw)


async def create_rule(db=None, **kw):
    db = db or kw.pop("db", None)
    return await ContextRuleService(db).create_rule(**kw)


async def update_rule(db=None, **kw):
    db = db or kw.pop("db", None)
    return await ContextRuleService(db).update_rule(**kw)


async def delete_rule(db=None, **kw):
    db = db or kw.pop("db", None)
    return await ContextRuleService(db).delete_rule(**kw)


async def test_pattern(db=None, **kw):
    db = db or kw.pop("db", None)
    return await ContextRuleService(db).test_pattern(**kw)
