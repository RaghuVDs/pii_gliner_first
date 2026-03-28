"""
Seed context rules from context_rules.yaml.

Reads the YAML config from the original PII engine project and creates
system context_rules rows in PostgreSQL.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.context_rule import ContextRule

logger = logging.getLogger(__name__)

CONTEXT_RULES_YAML = Path(__file__).resolve().parents[4] / "app" / "config" / "context_rules.yaml"


async def seed_context_rules(session: AsyncSession) -> Dict[str, int]:
    """Seed context_rules from the context_rules YAML.

    Creates one row per PII-type + keyword pattern combination with
    is_system=True and is_negative=False.

    If a key starts with ``negative_`` the corresponding rules are
    inserted with ``is_negative=True`` (the YAML does not currently
    contain negative rules, but this handles them if added later).

    Returns:
        Dict with the count of created rules.
    """
    # Idempotency check
    existing = (await session.execute(select(ContextRule).limit(1))).scalars().first()
    if existing:
        logger.info("Context rules already seeded. Skipping.")
        return {"context_rules": 0}

    with open(CONTEXT_RULES_YAML, "r", encoding="utf-8") as f:
        rules_data = yaml.safe_load(f)

    created = 0

    for raw_key, keywords in rules_data.items():
        if not isinstance(keywords, list):
            continue

        # Determine if this is a negative rule set
        is_negative = False
        pii_type_name = raw_key

        if raw_key.startswith("negative_"):
            is_negative = True
            pii_type_name = raw_key[len("negative_"):]

        for keyword in keywords:
            if keyword is None:
                continue

            keyword_str = str(keyword).strip()
            if not keyword_str:
                continue

            rule = ContextRule(
                pii_type_name=pii_type_name,
                keyword_pattern=keyword_str,
                is_negative=is_negative,
                is_system=True,
                source="yaml_seed",
            )
            session.add(rule)
            created += 1

    await session.flush()
    logger.info("Seeded %d context rules", created)
    return {"context_rules": created}
