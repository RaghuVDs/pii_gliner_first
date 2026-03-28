"""
Run all database seeders in dependency order.

Usage:
    python -m app.seed.run_all_seeds

Connects to the PostgreSQL database using async SQLAlchemy and executes
each seeder sequentially. Includes idempotency checks -- re-running is safe.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Dict

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.seed.seed_pii_taxonomy import seed_pii_taxonomy
from app.seed.seed_regex_rules import seed_regex_rules
from app.seed.seed_field_patterns import seed_field_patterns
from app.seed.seed_context_rules import seed_context_rules
from app.seed.seed_masking_rules import seed_masking_rules

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_all_seeds() -> Dict[str, Dict[str, int]]:
    """Execute all seeders in order and return summary results."""
    settings = get_settings()

    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )
    async_session = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    results: Dict[str, Dict[str, int]] = {}

    async with async_session() as session:
        try:
            # 1. PII Taxonomy (categories + types) -- must be first
            logger.info("=" * 60)
            logger.info("Step 1/5: Seeding PII taxonomy...")
            results["pii_taxonomy"] = await seed_pii_taxonomy(session)

            # 2. Regex rules
            logger.info("=" * 60)
            logger.info("Step 2/5: Seeding regex rules...")
            results["regex_rules"] = await seed_regex_rules(session)

            # 3. Field patterns
            logger.info("=" * 60)
            logger.info("Step 3/5: Seeding field patterns...")
            results["field_patterns"] = await seed_field_patterns(session)

            # 4. Context rules
            logger.info("=" * 60)
            logger.info("Step 4/5: Seeding context rules...")
            results["context_rules"] = await seed_context_rules(session)

            # 5. Masking rules (strategies + rules)
            logger.info("=" * 60)
            logger.info("Step 5/5: Seeding masking rules...")
            results["masking_rules"] = await seed_masking_rules(session)

            await session.commit()
            logger.info("=" * 60)
            logger.info("All seeds completed successfully!")

        except Exception:
            await session.rollback()
            logger.exception("Seeding failed -- transaction rolled back")
            raise
        finally:
            await session.close()

    await engine.dispose()
    return results


def main() -> None:
    """Entry point for ``python -m app.seed.run_all_seeds``."""
    results = asyncio.run(run_all_seeds())

    print("\n" + "=" * 60)
    print("SEED SUMMARY")
    print("=" * 60)
    for seeder_name, counts in results.items():
        parts = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  {seeder_name}: {parts}")
    print("=" * 60)


if __name__ == "__main__":
    main()
