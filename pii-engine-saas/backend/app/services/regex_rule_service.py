"""
Regex rule service -- manage system and tenant-level regex detection rules.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError, ValidationError
from app.models.regex_rule import RegexRule, TenantRegexRule

logger = logging.getLogger(__name__)


class RegexRuleService:
    """Business logic for regex rule management."""

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
        """Return all regex rules merged with tenant overrides.

        System rules form the base set.  Tenant rules can override or
        extend them.  Disabled tenant overrides suppress the system rule.

        Args:
            tenant_id: Current tenant.
            pii_type_name: Optional filter to a specific PII type.

        Returns:
            List of rule dicts, each with ``source`` ('system' or 'tenant'),
            ``is_enabled``, and the rule fields.
        """
        # System rules
        sys_query = select(RegexRule)
        if pii_type_name:
            sys_query = sys_query.where(RegexRule.pii_type_name == pii_type_name)
        sys_result = await self._db.execute(
            sys_query.order_by(RegexRule.pii_type_name, RegexRule.sort_order)
        )
        system_rules = list(sys_result.scalars().all())

        # Tenant rules / overrides
        tenant_query = select(TenantRegexRule).where(
            TenantRegexRule.tenant_id == tenant_id,
            TenantRegexRule.deleted_at.is_(None),
        )
        if pii_type_name:
            tenant_query = tenant_query.where(
                TenantRegexRule.pii_type_name == pii_type_name
            )
        tenant_result = await self._db.execute(
            tenant_query.order_by(
                TenantRegexRule.pii_type_name, TenantRegexRule.sort_order
            )
        )
        tenant_rules = list(tenant_result.scalars().all())

        # Build override lookup: system_rule_id -> tenant_override
        override_map: dict[int | None, TenantRegexRule] = {}
        tenant_only: list[TenantRegexRule] = []
        for tr in tenant_rules:
            if tr.is_system_override and tr.system_rule_id is not None:
                override_map[tr.system_rule_id] = tr
            else:
                tenant_only.append(tr)

        merged: list[dict] = []

        # System rules with possible tenant overrides
        for sr in system_rules:
            override = override_map.get(sr.id)
            if override is not None:
                merged.append({
                    "id": override.id,
                    "source": "tenant_override",
                    "pii_type_name": override.pii_type_name,
                    "pattern": override.pattern,
                    "description": override.description,
                    "is_enabled": override.is_enabled,
                    "sort_order": override.sort_order,
                    "system_rule_id": sr.id,
                })
            else:
                merged.append({
                    "id": sr.id,
                    "source": "system",
                    "pii_type_name": sr.pii_type_name,
                    "pattern": sr.pattern,
                    "description": sr.description,
                    "is_enabled": True,
                    "sort_order": sr.sort_order,
                    "system_rule_id": None,
                })

        # Tenant-only rules (not system overrides)
        for tr in tenant_only:
            merged.append({
                "id": tr.id,
                "source": "tenant",
                "pii_type_name": tr.pii_type_name,
                "pattern": tr.pattern,
                "description": tr.description,
                "is_enabled": tr.is_enabled,
                "sort_order": tr.sort_order,
                "system_rule_id": None,
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
        """Convenience wrapper returning merged rules for a specific PII type.

        Args:
            tenant_id: Current tenant.
            pii_type_name: The PII type name to filter by.

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
    ) -> TenantRegexRule:
        """Create a new tenant-specific regex rule.

        Args:
            tenant_id: Owning tenant.
            data: Fields: pii_type_name, pattern, description (optional),
                  is_enabled, sort_order, created_by.

        Returns:
            The created TenantRegexRule.

        Raises:
            ValidationError: If the pattern is not valid regex.
        """
        pattern = data.get("pattern", "")
        self._validate_pattern(pattern)

        rule = TenantRegexRule(
            tenant_id=tenant_id,
            pii_type_name=data["pii_type_name"],
            pattern=pattern,
            description=data.get("description"),
            is_enabled=data.get("is_enabled", True),
            is_system_override=data.get("is_system_override", False),
            system_rule_id=data.get("system_rule_id"),
            sort_order=data.get("sort_order", 0),
            created_by=data.get("created_by"),
        )
        self._db.add(rule)
        await self._db.flush()

        logger.info(
            "Created tenant regex rule %d for type '%s' in tenant %s.",
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
    ) -> TenantRegexRule:
        """Update an existing tenant regex rule.

        Args:
            tenant_id: Owning tenant.
            rule_id: The tenant rule ID.
            data: Fields to update.

        Returns:
            Updated TenantRegexRule.

        Raises:
            NotFoundError: If rule not found.
            ValidationError: If the new pattern is invalid regex.
        """
        result = await self._db.execute(
            select(TenantRegexRule).where(
                TenantRegexRule.id == rule_id,
                TenantRegexRule.tenant_id == tenant_id,
                TenantRegexRule.deleted_at.is_(None),
            )
        )
        rule = result.scalar_one_or_none()
        if rule is None:
            raise NotFoundError(
                "Regex rule not found.", error_code="REGEX_RULE_NOT_FOUND"
            )

        if "pattern" in data:
            self._validate_pattern(data["pattern"])

        allowed_fields = {
            "pii_type_name", "pattern", "description", "is_enabled", "sort_order",
        }
        for key, value in data.items():
            if key in allowed_fields and value is not None:
                setattr(rule, key, value)

        self._db.add(rule)
        await self._db.flush()

        logger.info(
            "Updated tenant regex rule %d in tenant %s.", rule_id, tenant_id
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
        """Soft-delete a tenant regex rule.

        Args:
            tenant_id: Owning tenant.
            rule_id: Rule ID.

        Raises:
            NotFoundError: If rule not found.
        """
        result = await self._db.execute(
            select(TenantRegexRule).where(
                TenantRegexRule.id == rule_id,
                TenantRegexRule.tenant_id == tenant_id,
                TenantRegexRule.deleted_at.is_(None),
            )
        )
        rule = result.scalar_one_or_none()
        if rule is None:
            raise NotFoundError(
                "Regex rule not found.", error_code="REGEX_RULE_NOT_FOUND"
            )

        rule.deleted_at = datetime.now(timezone.utc)
        self._db.add(rule)
        await self._db.flush()

        logger.info(
            "Soft-deleted tenant regex rule %d in tenant %s.",
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
        """Test a regex pattern against sample text.

        Args:
            pattern: The regex pattern string.
            test_text: Text to test against.

        Returns:
            Dict with ``is_valid``, ``matches`` (list of match dicts),
            ``error`` (if pattern is invalid).
        """
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return {
                "is_valid": False,
                "matches": [],
                "error": str(exc),
            }

        matches = []
        for m in compiled.finditer(test_text):
            matches.append({
                "match": m.group(),
                "start": m.start(),
                "end": m.end(),
                "groups": list(m.groups()),
            })

        return {
            "is_valid": True,
            "matches": matches,
            "error": None,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_pattern(pattern: str) -> None:
        """Validate that a string is a compilable regex pattern.

        Raises:
            ValidationError: If the pattern cannot be compiled.
        """
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValidationError(
                f"Invalid regex pattern: {exc}",
                error_code="INVALID_REGEX_PATTERN",
            )


# ---------------------------------------------------------------------------
# Module-level convenience functions (proxy to RegexRuleService)
# ---------------------------------------------------------------------------

async def list_rules(db=None, **kw):
    db = db or kw.pop("db", None)
    return await RegexRuleService(db).list_rules(**kw)


async def get_rules_for_type(db=None, **kw):
    db = db or kw.pop("db", None)
    return await RegexRuleService(db).get_rules_for_type(**kw)


async def get_rules_by_type(db=None, **kw):
    db = db or kw.pop("db", None)
    return await RegexRuleService(db).get_rules_for_type(**kw)


async def create_rule(db=None, **kw):
    db = db or kw.pop("db", None)
    return await RegexRuleService(db).create_rule(**kw)


async def update_rule(db=None, **kw):
    db = db or kw.pop("db", None)
    return await RegexRuleService(db).update_rule(**kw)


async def delete_rule(db=None, **kw):
    db = db or kw.pop("db", None)
    return await RegexRuleService(db).delete_rule(**kw)


async def test_pattern(db=None, **kw):
    db = db or kw.pop("db", None)
    return await RegexRuleService(db).test_pattern(**kw)
