"""
Seed PII taxonomy -- categories and types from pii_taxonomy.yaml.

Reads the YAML config from the original PII engine project, creates
pii_categories and pii_types rows in PostgreSQL, including tier mapping,
compliance tags, and sensitivity flags.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.pii_type import PIICategory, PIIType

logger = logging.getLogger(__name__)

# Path to the original YAML config
TAXONOMY_YAML = Path(__file__).resolve().parents[4] / "app" / "config" / "pii_taxonomy.yaml"

# ---------------------------------------------------------------------------
# Tier mapping (from gliner_detector.py GROUP_TO_TIER)
# ---------------------------------------------------------------------------
GROUP_TO_TIER = {
    "general": 0,
    "country_specific": 0,
    "financial_general": 1,
    "amex_specific": 1,
    "medical": 2,
    "education": 2,
    "employment": 2,
    "digital_security": 2,
    "vehicle_property": 3,
    "legal_criminal": 3,
    "communication_utility": 3,
    "contextual": 3,
    "fallback": 3,
}

# Display names for category groups
GROUP_DISPLAY_NAMES = {
    "general": "General Personal Identifiers",
    "country_specific": "Country-Specific Government IDs",
    "financial_general": "Financial Information",
    "amex_specific": "AMEX-Specific Identifiers",
    "medical": "Medical / HIPAA",
    "education": "Education / FERPA",
    "employment": "Employment & HR",
    "digital_security": "Digital Security",
    "vehicle_property": "Vehicle & Property",
    "legal_criminal": "Legal & Criminal",
    "communication_utility": "Communication & Utility",
    "contextual": "Contextual / Sensitive Attributes",
    "fallback": "Fallback",
}

# Compliance tag mapping -- PII types mapped to compliance frameworks
COMPLIANCE_MAP: Dict[str, List[str]] = {
    # HIPAA
    "MEDICAL_RECORD_NUMBER": ["HIPAA"],
    "MEDICARE_MEDICAID_ID": ["HIPAA"],
    "HEALTH_PLAN_BENEFICIARY_NUMBER": ["HIPAA"],
    "PRESCRIPTION_NUMBER": ["HIPAA"],
    "LAB_ACCESSION_NUMBER": ["HIPAA"],
    "CLINICAL_TRIAL_ID": ["HIPAA"],
    "MEDICAL_DEVICE_SERIAL": ["HIPAA"],
    "HEALTH_INSURANCE_ID": ["HIPAA"],
    "PHI": ["HIPAA"],
    "GENETIC_INFORMATION": ["HIPAA"],
    "BIOMETRIC_INFORMATION": ["HIPAA"],
    # PCI-DSS
    "CREDIT_CARD_NUMBER": ["PCI-DSS"],
    "CARD_EXPIRATION_DATE": ["PCI-DSS"],
    "CARD_SECURITY_CODE": ["PCI-DSS"],
    "ACCOUNT_NUMBER_AMEX": ["PCI-DSS"],
    "TRACK_DATA": ["PCI-DSS"],
    "PIN": ["PCI-DSS"],
    # GDPR (broad personal data)
    "PERSON_FULL_NAME": ["GDPR", "CCPA"],
    "PERSON_FIRST_NAME": ["GDPR", "CCPA"],
    "PERSON_LAST_NAME": ["GDPR", "CCPA"],
    "EMAIL_ADDRESS": ["GDPR", "CCPA"],
    "PHONE_NUMBER": ["GDPR", "CCPA"],
    "STREET_ADDRESS": ["GDPR", "CCPA"],
    "DATE_OF_BIRTH": ["GDPR", "CCPA"],
    "SSN": ["GDPR", "CCPA"],
    "IP_ADDRESS": ["GDPR", "CCPA"],
    "PRECISE_GEOLOCATION": ["GDPR", "CCPA"],
    "ETHNICITY_OR_RACE": ["GDPR"],
    "RELIGION": ["GDPR"],
    "POLITICAL_OPINIONS": ["GDPR"],
    "SEXUAL_ORIENTATION": ["GDPR"],
    "NATIONALITY": ["GDPR"],
    # CCPA
    "DRIVERS_LICENSE_NUMBER": ["CCPA"],
    "PASSPORT_NUMBER": ["CCPA"],
    "BANK_ACCOUNT_NUMBER": ["CCPA"],
    "BIOMETRIC_INFORMATION": ["CCPA", "HIPAA"],
}

# Sensitive PII types
SENSITIVE_TYPES = {
    "PHI",
    "GENETIC_INFORMATION",
    "BIOMETRIC_INFORMATION",
    "SEXUAL_ORIENTATION",
    "ETHNICITY_OR_RACE",
    "RELIGION",
    "POLITICAL_OPINIONS",
    "CRIMINAL_RECORD",
    "DISABILITY_STATUS",
    "CHILDBEARING_STATUS",
    "MEDICAL_RECORD_NUMBER",
    "MEDICARE_MEDICAID_ID",
    "HEALTH_PLAN_BENEFICIARY_NUMBER",
    "SSN",
    "CREDIT_CARD_NUMBER",
    "CARD_SECURITY_CODE",
    "PASSWORD",
    "SECURITY_ANSWER",
    "PIN",
    "ONE_TIME_CODE",
    "API_KEY",
    "SSH_KEY",
    "PGP_KEY",
    "TRACK_DATA",
}


def _make_display_name(name: str) -> str:
    """Convert 'PERSON_FULL_NAME' -> 'Person Full Name'."""
    return name.replace("_", " ").title()


async def seed_pii_taxonomy(session: AsyncSession) -> Dict[str, int]:
    """Seed pii_categories and pii_types from the taxonomy YAML.

    Returns:
        Dict with counts of created categories and types.
    """
    # Idempotency check
    existing = (await session.execute(select(PIICategory))).scalars().all()
    if existing:
        logger.info("PII taxonomy already seeded (%d categories). Skipping.", len(existing))
        return {"categories": 0, "types": 0}

    # Load YAML
    with open(TAXONOMY_YAML, "r", encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)

    categories_created = 0
    types_created = 0

    sort_order = 0
    category_map: Dict[str, PIICategory] = {}

    for group_name, pii_entries in taxonomy.items():
        if not isinstance(pii_entries, dict):
            continue

        tier = GROUP_TO_TIER.get(group_name, 3)
        display_name = GROUP_DISPLAY_NAMES.get(group_name, _make_display_name(group_name))

        category = PIICategory(
            name=group_name,
            display_name=display_name,
            tier=tier,
            sort_order=sort_order,
            is_system=True,
        )
        session.add(category)
        await session.flush()  # get the id

        category_map[group_name] = category
        categories_created += 1
        sort_order += 1

        for pii_name, attrs in pii_entries.items():
            gliner_aliases = attrs.get("gliner_aliases", [])
            threshold = attrs.get("threshold", 0.40)
            compliance_tags = COMPLIANCE_MAP.get(pii_name, [])
            is_sensitive = pii_name in SENSITIVE_TYPES

            pii_type = PIIType(
                category_id=category.id,
                name=pii_name,
                display_name=_make_display_name(pii_name),
                gliner_aliases=gliner_aliases,
                default_threshold=threshold,
                is_system=True,
                is_sensitive=is_sensitive,
                compliance_tags=compliance_tags,
            )
            session.add(pii_type)
            types_created += 1

    await session.flush()
    logger.info("Seeded %d PII categories and %d PII types", categories_created, types_created)
    return {"categories": categories_created, "types": types_created}
