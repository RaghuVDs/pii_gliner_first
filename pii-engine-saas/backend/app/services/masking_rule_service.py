"""
Masking rule service -- manage masking strategies and per-tenant masking
overrides for each PII type.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError, ValidationError
from app.models.masking_rule import (
    MaskingRule,
    MaskingStrategy,
    TenantMaskingRule,
)

logger = logging.getLogger(__name__)


class MaskingRuleService:
    """Business logic for masking strategy and rule management."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # List all masking strategies
    # ------------------------------------------------------------------

    async def list_strategies(self, **_extra) -> list[MaskingStrategy]:
        """Return all available masking strategies.

        Returns:
            List of MaskingStrategy instances ordered by name.
        """
        result = await self._db.execute(
            select(MaskingStrategy).order_by(MaskingStrategy.name)
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # List masking rules (merged system + tenant overrides)
    # ------------------------------------------------------------------

    async def list_rules(self, tenant_id: uuid.UUID, **_extra) -> list[dict]:
        """Return masking rules merged with tenant overrides.

        For each PII type name, returns the effective masking strategy:
        the tenant override if one exists, otherwise the system default.

        Args:
            tenant_id: Current tenant.

        Returns:
            List of merged rule dicts.
        """
        # System masking rules
        sys_result = await self._db.execute(
            select(MaskingRule).order_by(MaskingRule.pii_type_name)
        )
        system_rules = list(sys_result.scalars().all())

        # Tenant overrides
        tenant_result = await self._db.execute(
            select(TenantMaskingRule).where(
                TenantMaskingRule.tenant_id == tenant_id,
            )
        )
        tenant_rules = list(tenant_result.scalars().all())
        tenant_map = {tr.pii_type_name: tr for tr in tenant_rules}

        # Strategies lookup
        strats_result = await self._db.execute(select(MaskingStrategy))
        strategy_map = {s.id: s for s in strats_result.scalars().all()}

        merged: list[dict] = []

        for sr in system_rules:
            override = tenant_map.pop(sr.pii_type_name, None)
            if override is not None:
                strategy = strategy_map.get(override.strategy_id)
                merged.append({
                    "pii_type_name": sr.pii_type_name,
                    "source": "tenant_override",
                    "strategy_id": override.strategy_id,
                    "strategy_name": strategy.name if strategy else None,
                    "custom_params": override.custom_params,
                    "is_default": sr.is_default,
                    "override_id": override.id,
                })
            else:
                strategy = strategy_map.get(sr.strategy_id)
                merged.append({
                    "pii_type_name": sr.pii_type_name,
                    "source": "system",
                    "strategy_id": sr.strategy_id,
                    "strategy_name": strategy.name if strategy else None,
                    "custom_params": None,
                    "is_default": sr.is_default,
                    "override_id": None,
                })

        # Tenant rules for PII types not in system defaults
        for pii_type_name, tr in tenant_map.items():
            strategy = strategy_map.get(tr.strategy_id)
            merged.append({
                "pii_type_name": pii_type_name,
                "source": "tenant",
                "strategy_id": tr.strategy_id,
                "strategy_name": strategy.name if strategy else None,
                "custom_params": tr.custom_params,
                "is_default": False,
                "override_id": tr.id,
            })

        return merged

    # ------------------------------------------------------------------
    # Update tenant masking rule
    # ------------------------------------------------------------------

    async def update_rule(
        self,
        tenant_id: uuid.UUID,
        pii_type_name: str,
        strategy_id: int,
        custom_params: dict[str, Any] | None = None,
        **_extra,
    ) -> TenantMaskingRule:
        """Set or update the masking strategy for a PII type within a tenant.

        Creates the TenantMaskingRule if it doesn't exist (upsert).

        Args:
            tenant_id: Owning tenant.
            pii_type_name: The PII type to configure.
            strategy_id: ID of the masking strategy to use.
            custom_params: Optional custom parameters for the strategy.

        Returns:
            The created or updated TenantMaskingRule.

        Raises:
            NotFoundError: If the strategy doesn't exist.
        """
        # Validate strategy
        strat_result = await self._db.execute(
            select(MaskingStrategy).where(MaskingStrategy.id == strategy_id)
        )
        if strat_result.scalar_one_or_none() is None:
            raise NotFoundError(
                "Masking strategy not found.",
                error_code="MASKING_STRATEGY_NOT_FOUND",
            )

        # Upsert tenant rule
        result = await self._db.execute(
            select(TenantMaskingRule).where(
                TenantMaskingRule.tenant_id == tenant_id,
                TenantMaskingRule.pii_type_name == pii_type_name,
            )
        )
        rule = result.scalar_one_or_none()

        if rule is None:
            rule = TenantMaskingRule(
                tenant_id=tenant_id,
                pii_type_name=pii_type_name,
                strategy_id=strategy_id,
                custom_params=custom_params,
            )
        else:
            rule.strategy_id = strategy_id
            rule.custom_params = custom_params

        self._db.add(rule)
        await self._db.flush()

        logger.info(
            "Set masking for '%s' to strategy %d in tenant %s.",
            pii_type_name,
            strategy_id,
            tenant_id,
        )
        return rule

    # ------------------------------------------------------------------
    # Reset tenant masking to system defaults
    # ------------------------------------------------------------------

    async def reset_defaults(self, tenant_id: uuid.UUID, **_extra) -> int:
        """Delete all tenant masking overrides, reverting to system defaults.

        Args:
            tenant_id: Owning tenant.

        Returns:
            Number of overrides deleted.
        """
        result = await self._db.execute(
            delete(TenantMaskingRule)
            .where(TenantMaskingRule.tenant_id == tenant_id)
            .returning(TenantMaskingRule.id)
        )
        deleted = len(list(result.scalars().all()))
        await self._db.flush()

        logger.info(
            "Reset %d masking overrides for tenant %s.", deleted, tenant_id
        )
        return deleted


# ---------------------------------------------------------------------------
# Module-level convenience functions (proxy to MaskingRuleService)
# ---------------------------------------------------------------------------

async def list_strategies(db=None, **kw):
    db = db or kw.pop("db", None)
    return await MaskingRuleService(db).list_strategies(**kw)


async def list_rules(db=None, **kw):
    db = db or kw.pop("db", None)
    return await MaskingRuleService(db).list_rules(**kw)


async def update_rule(db=None, **kw):
    db = db or kw.pop("db", None)
    return await MaskingRuleService(db).update_rule(**kw)


async def reset_defaults(db=None, **kw):
    db = db or kw.pop("db", None)
    return await MaskingRuleService(db).reset_defaults(**kw)
