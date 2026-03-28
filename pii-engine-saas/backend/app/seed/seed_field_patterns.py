"""
Seed field patterns from field_patterns.yaml.

Reads the YAML config from the original PII engine project and creates
system field_patterns rows in PostgreSQL.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.field_pattern import FieldPattern

logger = logging.getLogger(__name__)

FIELD_PATTERNS_YAML = Path(__file__).resolve().parents[4] / "app" / "config" / "field_patterns.yaml"


async def seed_field_patterns(session: AsyncSession) -> Dict[str, int]:
    """Seed field_patterns from the field_patterns YAML.

    Creates one row per PII-type + label pattern combination with is_system=True.

    Returns:
        Dict with the count of created patterns.
    """
    # Idempotency check
    existing = (await session.execute(select(FieldPattern).limit(1))).scalars().first()
    if existing:
        logger.info("Field patterns already seeded. Skipping.")
        return {"field_patterns": 0}

    with open(FIELD_PATTERNS_YAML, "r", encoding="utf-8") as f:
        patterns_data = yaml.safe_load(f)

    created = 0
    global_sort = 0

    for pii_type_name, labels in patterns_data.items():
        if not isinstance(labels, list):
            continue

        for label in labels:
            if label is None:
                continue

            label_str = str(label).strip()
            if not label_str:
                continue

            fp = FieldPattern(
                pii_type_name=pii_type_name,
                label_pattern=label_str,
                is_system=True,
                sort_order=global_sort,
            )
            session.add(fp)
            created += 1
            global_sort += 1

    await session.flush()
    logger.info("Seeded %d field patterns", created)
    return {"field_patterns": created}
