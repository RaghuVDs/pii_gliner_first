"""
Seed regex rules from regex_rules.yaml.

Reads the YAML config from the original PII engine project and creates
system regex_rules rows in PostgreSQL.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.regex_rule import RegexRule

logger = logging.getLogger(__name__)

REGEX_YAML = Path(__file__).resolve().parents[4] / "app" / "config" / "regex_rules.yaml"


async def seed_regex_rules(session: AsyncSession) -> Dict[str, int]:
    """Seed regex_rules from the regex_rules YAML.

    Creates one row per PII-type + pattern combination with is_system=True.

    Returns:
        Dict with the count of created rules.
    """
    # Idempotency check
    existing = (await session.execute(select(RegexRule).limit(1))).scalars().first()
    if existing:
        logger.info("Regex rules already seeded. Skipping.")
        return {"regex_rules": 0}

    with open(REGEX_YAML, "r", encoding="utf-8") as f:
        rules_data = yaml.safe_load(f)

    created = 0
    global_sort = 0

    for pii_type_name, patterns in rules_data.items():
        if not isinstance(patterns, list):
            continue

        for idx, pattern in enumerate(patterns):
            # Skip YAML comments that sneak through as None
            if pattern is None:
                continue

            # The pattern might be a plain string or have a comment prefix
            pattern_str = str(pattern).strip()
            if not pattern_str:
                continue

            rule = RegexRule(
                pii_type_name=pii_type_name,
                pattern=pattern_str,
                sort_order=global_sort,
                is_system=True,
            )
            session.add(rule)
            created += 1
            global_sort += 1

    await session.flush()
    logger.info("Seeded %d regex rules", created)
    return {"regex_rules": created}
