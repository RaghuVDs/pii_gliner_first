"""
Seed masking strategies and masking rules from masking_rules.yaml.

Reads the YAML config from the original PII engine project and creates:
1. masking_strategies rows (full_mask, keep_last_4, etc.)
2. masking_rules rows mapping PII types to their strategies
3. A default masking_rules row with is_default=True
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.masking_rule import MaskingRule, MaskingStrategy

logger = logging.getLogger(__name__)

MASKING_YAML = Path(__file__).resolve().parents[4] / "app" / "config" / "masking_rules.yaml"

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------
STRATEGIES = [
    {
        "name": "full_mask",
        "display_name": "Full Mask",
        "description": "Replace entire value with asterisks (e.g. '****')",
        "implementation_key": "full_mask",
    },
    {
        "name": "keep_last_4",
        "display_name": "Keep Last 4",
        "description": "Mask all but the last 4 characters (e.g. '****7890')",
        "implementation_key": "keep_last_4",
    },
    {
        "name": "mask_before_at_keep_first",
        "display_name": "Mask Before @ Keep First",
        "description": "For emails: keep first char + mask + @domain (e.g. 'j****@example.com')",
        "implementation_key": "mask_before_at_keep_first",
    },
    {
        "name": "keep_bin_and_last_4",
        "display_name": "Keep BIN and Last 4",
        "description": "For card numbers: keep first 6 + mask + last 4 (e.g. '377812****1234')",
        "implementation_key": "keep_bin_and_last_4",
    },
    {
        "name": "keep_first_two_chars",
        "display_name": "Keep First Two Chars",
        "description": "Keep first two characters, mask the rest (e.g. 'Jo****')",
        "implementation_key": "keep_first_two_chars",
    },
]


async def seed_masking_rules(session: AsyncSession) -> Dict[str, int]:
    """Seed masking_strategies and masking_rules from the masking_rules YAML.

    Returns:
        Dict with counts of created strategies and rules.
    """
    # Idempotency check
    existing = (await session.execute(select(MaskingStrategy).limit(1))).scalars().first()
    if existing:
        logger.info("Masking strategies already seeded. Skipping.")
        return {"strategies": 0, "masking_rules": 0}

    # Load YAML
    with open(MASKING_YAML, "r", encoding="utf-8") as f:
        masking_data = yaml.safe_load(f)

    # 1. Create masking strategies
    strategy_map: Dict[str, MaskingStrategy] = {}
    for strat_def in STRATEGIES:
        strat = MaskingStrategy(
            name=strat_def["name"],
            display_name=strat_def["display_name"],
            description=strat_def["description"],
            implementation_key=strat_def["implementation_key"],
        )
        session.add(strat)
        await session.flush()
        strategy_map[strat_def["name"]] = strat

    strategies_created = len(STRATEGIES)

    # 2. Create masking rules from the YAML partial_masking section
    rules_created = 0
    partial_masking = masking_data.get("partial_masking", {})

    for pii_type_name, rule_config in partial_masking.items():
        strategy_name = rule_config.get("strategy", "full_mask")
        strategy = strategy_map.get(strategy_name)
        if not strategy:
            logger.warning(
                "Unknown masking strategy '%s' for PII type '%s'. Skipping.",
                strategy_name,
                pii_type_name,
            )
            continue

        rule = MaskingRule(
            pii_type_name=pii_type_name,
            strategy_id=strategy.id,
            is_default=False,
        )
        session.add(rule)
        rules_created += 1

    # 3. Create default masking rule
    default_strategy_name = masking_data.get("default_strategy", "full_mask")
    default_strategy = strategy_map.get(default_strategy_name)
    if default_strategy:
        default_rule = MaskingRule(
            pii_type_name="__DEFAULT__",
            strategy_id=default_strategy.id,
            is_default=True,
        )
        session.add(default_rule)
        rules_created += 1

    await session.flush()
    logger.info("Seeded %d masking strategies and %d masking rules", strategies_created, rules_created)
    return {"strategies": strategies_created, "masking_rules": rules_created}
