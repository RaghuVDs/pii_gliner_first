"""
Config assembler -- builds YAML-compatible configuration dicts from the
database by merging system defaults with tenant overrides.

These assembled configs are consumed by ``TenantEngineManager`` to
initialise ``HybridPIIEngine`` instances with the correct per-tenant
settings.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.context_rule import ContextRule, TenantContextRule
from app.models.field_pattern import FieldPattern, TenantFieldPattern
from app.models.masking_rule import MaskingRule, MaskingStrategy, TenantMaskingRule
from app.models.pii_type import (
    CustomPIIType,
    PIICategory,
    PIIType,
    TenantPIIConfig,
)
from app.models.regex_rule import RegexRule, TenantRegexRule

logger = logging.getLogger(__name__)


class ConfigAssembler:
    """Assembles engine configuration dicts from database state.

    Each ``assemble_*`` method produces a dict whose structure mirrors the
    corresponding YAML config file used by the standalone
    ``HybridPIIEngine``.
    """

    # ------------------------------------------------------------------
    # Taxonomy (pii_taxonomy.yaml)
    # ------------------------------------------------------------------

    @staticmethod
    async def assemble_taxonomy(
        tenant_id: uuid.UUID,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Build the PII taxonomy dict for a tenant.

        Merges system PII types with tenant configs and custom types.
        Disabled types are excluded.

        Args:
            tenant_id: Target tenant.
            db: Active async session.

        Returns:
            Dict matching the pii_taxonomy.yaml structure::

                {
                    "categories": [
                        {
                            "name": "...",
                            "display_name": "...",
                            "tier": 1,
                            "types": [
                                {
                                    "name": "...",
                                    "display_name": "...",
                                    "gliner_aliases": [...],
                                    "threshold": 0.4,
                                    ...
                                },
                                ...
                            ]
                        },
                        ...
                    ]
                }
        """
        # Categories
        cat_result = await db.execute(
            select(PIICategory).order_by(PIICategory.tier, PIICategory.sort_order)
        )
        categories = list(cat_result.scalars().all())

        # System types with tenant configs
        types_result = await db.execute(
            select(PIIType).order_by(PIIType.category_id, PIIType.id)
        )
        all_types = list(types_result.scalars().all())

        if tenant_id is not None:
            configs_result = await db.execute(
                select(TenantPIIConfig).where(
                    TenantPIIConfig.tenant_id == tenant_id
                )
            )
            tenant_configs = list(configs_result.scalars().all())
        else:
            tenant_configs = []

        config_map = {
            c.pii_type_id: c for c in tenant_configs
        }

        # Custom types
        if tenant_id is not None:
            custom_result = await db.execute(
                select(CustomPIIType).where(
                    CustomPIIType.tenant_id == tenant_id,
                    CustomPIIType.deleted_at.is_(None),
                    CustomPIIType.is_enabled.is_(True),
                )
            )
            custom_types = list(custom_result.scalars().all())
        else:
            custom_types = []

        # Group types by category
        cat_types: dict[int, list[dict]] = {c.id: [] for c in categories}

        for pt in all_types:
            config = config_map.get(pt.id)
            # If config exists and is disabled, skip
            if config is not None and not config.is_enabled:
                continue

            threshold = float(
                config.custom_threshold
                if config and config.custom_threshold is not None
                else pt.default_threshold
            )
            aliases = (
                config.custom_aliases
                if config and config.custom_aliases
                else pt.gliner_aliases
            )

            type_dict = {
                "name": pt.name,
                "display_name": pt.display_name,
                "description": pt.description,
                "gliner_aliases": aliases or [],
                "threshold": threshold,
                "is_sensitive": pt.is_sensitive,
                "compliance_tags": pt.compliance_tags or [],
            }

            if pt.category_id in cat_types:
                cat_types[pt.category_id].append(type_dict)

        # Add custom types
        for ct in custom_types:
            type_dict = {
                "name": ct.name,
                "display_name": ct.display_name,
                "description": ct.description,
                "gliner_aliases": ct.gliner_aliases or [],
                "threshold": float(ct.default_threshold),
                "is_sensitive": False,
                "compliance_tags": ct.compliance_tags or [],
                "custom": True,
            }
            cat_id = ct.category_id
            if cat_id and cat_id in cat_types:
                cat_types[cat_id].append(type_dict)
            else:
                # Attach to first category if no category assigned
                if categories:
                    cat_types[categories[0].id].append(type_dict)

        # Build engine-compatible format: {category_name: {TYPE_NAME: {gliner_aliases, threshold}}}
        taxonomy = {}
        for c in categories:
            types_in_cat = cat_types.get(c.id, [])
            if not types_in_cat:
                continue
            category_dict = {}
            for t in types_in_cat:
                category_dict[t["name"]] = {
                    "gliner_aliases": t["gliner_aliases"],
                    "threshold": t["threshold"],
                }
            taxonomy[c.name] = category_dict

        return taxonomy

    # ------------------------------------------------------------------
    # Regex rules (regex_rules.yaml)
    # ------------------------------------------------------------------

    @staticmethod
    async def assemble_regex_rules(
        tenant_id: uuid.UUID,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Build the regex rules dict for a tenant.

        Merges system rules with tenant overrides/additions.

        Args:
            tenant_id: Target tenant.
            db: Active async session.

        Returns:
            Dict matching regex_rules.yaml::

                {
                    "PII_TYPE_NAME": [
                        {"pattern": "...", "description": "..."},
                        ...
                    ],
                    ...
                }
        """
        # System rules
        sys_result = await db.execute(
            select(RegexRule).order_by(RegexRule.pii_type_name, RegexRule.sort_order)
        )
        system_rules = list(sys_result.scalars().all())

        # Tenant rules
        tenant_result = await db.execute(
            select(TenantRegexRule).where(
                TenantRegexRule.tenant_id == tenant_id if tenant_id is not None else False,
                TenantRegexRule.deleted_at.is_(None),
            ).order_by(TenantRegexRule.pii_type_name, TenantRegexRule.sort_order)
        )
        tenant_rules = list(tenant_result.scalars().all())

        # Build override map (system_rule_id -> tenant_rule)
        override_map: dict[int, TenantRegexRule] = {}
        tenant_only: list[TenantRegexRule] = []
        for tr in tenant_rules:
            if tr.is_system_override and tr.system_rule_id is not None:
                override_map[tr.system_rule_id] = tr
            else:
                tenant_only.append(tr)

        rules_by_type: dict[str, list[dict]] = {}

        # Process system rules
        for sr in system_rules:
            override = override_map.get(sr.id)
            if override is not None:
                if not override.is_enabled:
                    continue  # Suppressed
                pattern = override.pattern
            else:
                pattern = sr.pattern
            rules_by_type.setdefault(sr.pii_type_name, []).append(pattern)

        # Add tenant-only rules
        for tr in tenant_only:
            if not tr.is_enabled:
                continue
            rules_by_type.setdefault(tr.pii_type_name, []).append(tr.pattern)

        return rules_by_type

    # ------------------------------------------------------------------
    # Field patterns (field_patterns.yaml)
    # ------------------------------------------------------------------

    @staticmethod
    async def assemble_field_patterns(
        tenant_id: uuid.UUID,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Build the field patterns dict for a tenant.

        Args:
            tenant_id: Target tenant.
            db: Active async session.

        Returns:
            Dict matching field_patterns.yaml::

                {
                    "PII_TYPE_NAME": ["pattern1", "pattern2", ...],
                    ...
                }
        """
        # System patterns
        sys_result = await db.execute(
            select(FieldPattern).order_by(
                FieldPattern.pii_type_name, FieldPattern.sort_order
            )
        )
        system_patterns = list(sys_result.scalars().all())

        # Tenant patterns
        tenant_result = await db.execute(
            select(TenantFieldPattern).where(
                TenantFieldPattern.tenant_id == tenant_id if tenant_id is not None else False,
                TenantFieldPattern.deleted_at.is_(None),
            )
        )
        tenant_patterns = list(tenant_result.scalars().all())

        # Tenant lookup for override detection
        tenant_lookup: dict[tuple[str, str], TenantFieldPattern] = {
            (tp.pii_type_name, tp.label_pattern): tp for tp in tenant_patterns
        }

        patterns_by_type: dict[str, list[str]] = {}
        seen: set[tuple[str, str]] = set()

        for sp in system_patterns:
            key = (sp.pii_type_name, sp.label_pattern)
            seen.add(key)
            override = tenant_lookup.get(key)
            if override is not None and not override.is_enabled:
                continue  # Suppressed
            patterns_by_type.setdefault(sp.pii_type_name, []).append(
                sp.label_pattern
            )

        # Add tenant-only patterns
        for tp in tenant_patterns:
            key = (tp.pii_type_name, tp.label_pattern)
            if key not in seen and tp.is_enabled:
                patterns_by_type.setdefault(tp.pii_type_name, []).append(
                    tp.label_pattern
                )

        return patterns_by_type

    # ------------------------------------------------------------------
    # Context rules (context_rules.yaml)
    # ------------------------------------------------------------------

    @staticmethod
    async def assemble_context_rules(
        tenant_id: uuid.UUID,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Build the context rules dict for a tenant.

        Args:
            tenant_id: Target tenant.
            db: Active async session.

        Returns:
            Dict matching context_rules.yaml::

                {
                    "PII_TYPE_NAME": {
                        "positive": ["keyword1", ...],
                        "negative": ["keyword2", ...]
                    },
                    ...
                }
        """
        # System rules
        sys_result = await db.execute(
            select(ContextRule).order_by(ContextRule.pii_type_name)
        )
        system_rules = list(sys_result.scalars().all())

        # Tenant rules
        tenant_result = await db.execute(
            select(TenantContextRule).where(
                TenantContextRule.tenant_id == tenant_id if tenant_id is not None else False,
                TenantContextRule.deleted_at.is_(None),
            )
        )
        tenant_rules = list(tenant_result.scalars().all())

        tenant_lookup: dict[tuple[str, str, bool], TenantContextRule] = {
            (tr.pii_type_name, tr.keyword_pattern, tr.is_negative): tr
            for tr in tenant_rules
        }

        rules_by_type: dict[str, dict[str, list[str]]] = {}
        seen: set[tuple[str, str, bool]] = set()

        for sr in system_rules:
            key = (sr.pii_type_name, sr.keyword_pattern, sr.is_negative)
            seen.add(key)
            override = tenant_lookup.get(key)
            if override is not None and not override.is_enabled:
                continue  # Suppressed

            # Engine expects flat list: {TYPE: [kw1, kw2, ...]}
            if not sr.is_negative:
                rules_by_type.setdefault(sr.pii_type_name, []).append(sr.keyword_pattern)

        # Tenant-only rules
        for tr in tenant_rules:
            key = (tr.pii_type_name, tr.keyword_pattern, tr.is_negative)
            if key not in seen and tr.is_enabled and not tr.is_negative:
                rules_by_type.setdefault(tr.pii_type_name, []).append(tr.keyword_pattern)

        return rules_by_type

    # ------------------------------------------------------------------
    # Masking rules (masking_rules.yaml)
    # ------------------------------------------------------------------

    @staticmethod
    async def assemble_masking_rules(
        tenant_id: uuid.UUID,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Build the masking rules dict for a tenant.

        Args:
            tenant_id: Target tenant.
            db: Active async session.

        Returns:
            Dict matching masking_rules.yaml::

                {
                    "PII_TYPE_NAME": {
                        "strategy": "strategy_name",
                        "params": {...}  # optional
                    },
                    ...
                }
        """
        # System masking rules
        sys_result = await db.execute(
            select(MaskingRule).order_by(MaskingRule.pii_type_name)
        )
        system_rules = list(sys_result.scalars().all())

        # Tenant overrides
        tenant_result = await db.execute(
            select(TenantMaskingRule).where(
                TenantMaskingRule.tenant_id == tenant_id if tenant_id is not None else False,
            )
        )
        tenant_rules = list(tenant_result.scalars().all())
        tenant_map = {tr.pii_type_name: tr for tr in tenant_rules}

        # Strategy lookup
        strat_result = await db.execute(select(MaskingStrategy))
        strat_map = {s.id: s for s in strat_result.scalars().all()}

        masking: dict[str, dict[str, Any]] = {}

        for sr in system_rules:
            override = tenant_map.pop(sr.pii_type_name, None)
            if override is not None:
                strategy = strat_map.get(override.strategy_id)
                entry: dict[str, Any] = {
                    "strategy": strategy.implementation_key if strategy else "redact",
                }
                if override.custom_params:
                    entry["params"] = override.custom_params
                masking[sr.pii_type_name] = entry
            else:
                strategy = strat_map.get(sr.strategy_id)
                masking[sr.pii_type_name] = {
                    "strategy": strategy.implementation_key if strategy else "redact",
                }

        # Tenant-only masking rules
        for pii_type_name, tr in tenant_map.items():
            strategy = strat_map.get(tr.strategy_id)
            entry = {
                "strategy": strategy.implementation_key if strategy else "redact",
            }
            if tr.custom_params:
                entry["params"] = tr.custom_params
            masking[pii_type_name] = entry

        return masking

    # ------------------------------------------------------------------
    # Assemble all configs
    # ------------------------------------------------------------------

    @classmethod
    async def assemble_all(
        cls,
        tenant_id: uuid.UUID,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Assemble the complete engine configuration for a tenant.

        Args:
            tenant_id: Target tenant.
            db: Active async session.

        Returns:
            Dict with keys: taxonomy, regex_rules, field_patterns,
            context_rules, masking_rules.
        """
        taxonomy = await cls.assemble_taxonomy(tenant_id, db)
        regex_rules = await cls.assemble_regex_rules(tenant_id, db)
        field_patterns = await cls.assemble_field_patterns(tenant_id, db)
        context_rules = await cls.assemble_context_rules(tenant_id, db)
        masking_rules = await cls.assemble_masking_rules(tenant_id, db)

        return {
            "taxonomy": taxonomy,
            "regex_rules": regex_rules,
            "field_patterns": field_patterns,
            "context_rules": context_rules,
            "masking_rules": masking_rules,
        }
