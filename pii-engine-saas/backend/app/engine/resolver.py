from __future__ import annotations
from typing import List
from app.engine.models import Detection
from app.engine.utils import overlap

LABEL_PRIORITY = {
    # Contextual Highest -- Protected/sensitive categories
    "BIOMETRIC_INFORMATION": 100, "PHI": 100, "GENETIC_INFORMATION": 100,
    "CHILDBEARING_STATUS": 99, "CRIMINAL_RECORD": 99, "ETHNICITY_OR_RACE": 99,
    "POLITICAL_OPINIONS": 99, "SEXUAL_ORIENTATION": 99, "RELIGION": 99,
    "SEX_OFFENDER_REGISTRY": 99,
    "LEGAL_JUDGEMENT": 98, "DISABILITY_STATUS": 98, "VETERAN_STATUS": 98,
    "OPINIONS_CAPTURED": 98, "PERFORMANCE_RATING": 98, "PUBLIC_ASSISTANCE_RECEIPT": 98,
    "CHILDREN_INFORMATION": 98, "MEMBERSHIP_OR_TRADE_UNIONS": 98,
    "LEAVE_OF_ABSENCE_TYPE": 98, "PROTECTIVE_ORDER_NUMBER": 98,
    "MARITAL_OR_FAMILIAL_STATUS": 97, "NATIONALITY": 97, "GENDER": 97,
    "ONLINE_UNIQUE_IDENTIFIER": 97, "PHOTOGRAPH_METADATA": 97, "PLACE_OF_BIRTH": 97,

    "CHARITABLE_CONTRIBUTION": 96, "CUSTOMER_STATUS": 96, "PAYMENT_HISTORY": 96,

    # Financial/Identity -- High-value structured data
    "TRACK_DATA": 95, "ACCOUNT_NUMBER_AMEX": 94, "CREDIT_CARD_NUMBER": 94,
    "BANK_ACCOUNT_NUMBER": 93, "IBAN": 93, "CRYPTO_WALLET": 93, "SWIFT_BIC_CODE": 93,
    "ROUTING_NUMBER": 92, "SSN": 91, "TAX_ID": 91, "NATIONAL_ID": 91,
    "PASSPORT_NUMBER": 90, "GREEN_CARD_NUMBER": 90, "DRIVERS_LICENSE_NUMBER": 90,
    "NON_DRIVERS_ID": 90, "PARTNER_ACCOUNT_NUMBER": 90,
    "MORTGAGE_LOAN_INFO": 90, "INSURANCE_POLICY": 90,
    "VISA_NUMBER": 90, "SELECTIVE_SERVICE_NUMBER": 90, "VOTER_REGISTRATION_NUMBER": 90,
    "KNOWN_TRAVELER_NUMBER": 90, "PROFESSIONAL_LICENSE_NUMBER": 90,
    "INVESTMENT_ACCOUNT_NUMBER": 90, "CUSIP_NUMBER": 90, "EBT_CARD_NUMBER": 90,
    "CREDIT_REPORT_ID": 90, "HEALTH_INSURANCE_ID": 90, "WORKERS_COMP_CLAIM_NUMBER": 90,
    "BANKRUPTCY_CASE_NUMBER": 90, "CHILD_SUPPORT_CASE_NUMBER": 90,
    "ESCROW_ACCOUNT_NUMBER": 90, "PROPERTY_TAX_PARCEL": 90, "FINRA_CRD_NUMBER": 90,

    # Medical -- HIPAA PHI
    "MEDICAL_RECORD_NUMBER": 89, "MEDICARE_MEDICAID_ID": 89,
    "HEALTH_PLAN_BENEFICIARY_NUMBER": 89,
    "PRESCRIPTION_NUMBER": 88, "LAB_ACCESSION_NUMBER": 88,
    "CLINICAL_TRIAL_ID": 88, "MEDICAL_DEVICE_SERIAL": 88,

    # Granular Names
    "PERSON_FIRST_NAME": 89, "PERSON_MIDDLE_NAME": 89, "PERSON_LAST_NAME": 89,
    "PERSON_FULL_NAME": 80,

    # Standard SDEs
    "MOTHERS_MAIDEN_NAME": 88, "PASSWORD": 87, "PIN": 86,
    "ONE_TIME_CODE": 85, "CREDIT_FREEZE_PIN": 86,
    "EMAIL_ADDRESS": 84, "PHONE_NUMBER": 83, "STREET_ADDRESS": 82,
    "DIGITAL_PAYMENT_ACCOUNT": 84, "WIRE_TRANSFER_REFERENCE": 84, "GIFT_CARD_NUMBER": 84,
    "DATE_OF_BIRTH": 81, "RETIREMENT_DATE": 81, "SEPARATION_DATE": 81,
    "TAX_RETURN_DATA": 81,
    "INCOME": 80, "FICO_SCORE": 80, "BAND_LEVEL": 80, "ALIAS_NICKNAME": 80,
    "AGE": 79, "OCCUPATION": 78, "EMPLOYER_NAME": 78,
    "SIGNATURE": 79, "PHOTOGRAPH": 79,
    "EMPLOYEE_ID": 77, "USERNAME": 76,
    "EDUCATIONAL_AFFILIATIONS": 76, "EMPLOYMENT_HISTORY": 76,
    "TICKET_NUMBER": 76, "AUTHORIZED_AGENT_INFO": 76, "HIGH_VALUE_INDICATOR": 76,
    "MARKETING_PREFERENCES": 76, "ROC_DATA": 76, "SOLE_TRADER_DATA": 76,

    # Education & Employment
    "STUDENT_ID": 77, "FAFSA_DATA": 77, "TRANSCRIPT": 77,
    "PAYROLL_NUMBER": 77, "BACKGROUND_CHECK_REFERENCE": 77,
    "I9_EMPLOYMENT_DATA": 77, "E_VERIFY_CASE_NUMBER": 77,
    "SECURITY_CLEARANCE": 77, "OCCUPATIONAL_CERTIFICATION": 77,

    # Vehicle, Legal, Location
    "PRECISE_GEOLOCATION": 75, "DEVICE_APP_DATA": 74,
    "VIN": 75, "LICENSE_PLATE_NUMBER": 75, "VEHICLE_REGISTRATION_NUMBER": 75,
    "BOAT_HULL_ID": 75, "AIRCRAFT_REGISTRATION": 75,
    "COURT_CASE_NUMBER": 75, "ARREST_BOOKING_NUMBER": 75,
    "INMATE_OFFENDER_ID": 75, "PROBATION_PAROLE_ID": 75,
    "FBI_NUMBER": 75, "STATE_CRIMINAL_ID": 75,

    # Digital & Communication
    "API_KEY": 74, "SSH_KEY": 74, "PGP_KEY": 74,
    "DIGITAL_CERTIFICATE": 74, "BROWSER_FINGERPRINT": 74,
    "IP_ADDRESS": 73, "URL": 72,
    "UTILITY_ACCOUNT_NUMBER": 72, "CABLE_INTERNET_ACCOUNT": 72,
    "SOCIAL_MEDIA_PROFILE": 70, "SHIPPING_TRACKING_NUMBER": 70,
    "CDR_CALL_DETAIL_RECORD": 70, "CUSTOMER_PURCHASE_DATA": 69,
    "EMERGENCY_CONTACT_INFORMATION": 68, "UNKNOWN_PII": 10,
}

SOURCE_PRIORITY = {
    "gliner": 50,        # PRIMARY — semantic understanding handles diverse transcripts
    "pattern_lstm": 45,  # Learned patterns — trained from GLiNER/regex, trusted near-primary
    "field_label": 40,   # Supplement — explicit label:value patterns
    "context": 35,       # Keyword-confirmed promotions
    "regex": 30,         # Fallback — pattern matching
    "derived": 25,       # Split from trusted sources (GLiNER/regex) — should rank near regex
    "propagated": 15,
}

# Labels where regex/field_label have mathematical/structural format validation
# (Luhn checksum, ABA checksum, SSN format, email RFC, etc.)
# These get a source priority boost to outrank GLiNER for the same span.
FORMAT_VALIDATED_LABELS = {
    "SSN", "EMAIL_ADDRESS", "CREDIT_CARD_NUMBER", "ACCOUNT_NUMBER_AMEX",
    "ROUTING_NUMBER", "IBAN", "IP_ADDRESS", "TRACK_DATA",
    "PHONE_NUMBER", "VIN", "SWIFT_BIC_CODE", "CUSIP_NUMBER",
    "MEDICARE_MEDICAID_ID",
}

def _normalize_score(score: float, source: str) -> float:
    """Normalize confidence scores across sources to a comparable 0-1 scale.

    GLiNER is the PRIMARY detector — its scores are normalized to [0.0, 0.96]
    to preserve discrimination while staying just below regex/field raw scores.
    The source priority (Change 2) ensures GLiNER wins on priority anyway;
    format-validated regex labels override via FORMAT_VALIDATED_LABELS.
    """
    if source == "gliner":
        # Map GLiNER's effective range [0.20, 1.0] → [0.0, 0.96]
        return max(0.0, min((score - 0.20) / 0.80 * 0.96, 0.96))
    if source == "pattern_lstm":
        # LSTM predictions — confidence already calibrated via softmax, cap at 0.94
        return max(0.0, min(score, 0.94))
    if source == "derived":
        # Derived from name splitting — keep proportionally below GLiNER
        return max(0.0, min((score - 0.20) / 0.80 * 0.90, 0.90))
    # regex, field_label, context: keep as-is (they're already well-calibrated)
    return float(score)

def _score(d: Detection):
    source_pri = SOURCE_PRIORITY.get(d.source, 0)

    # Format-validated regex/field_label detections (SSN, Luhn, ABA, email RFC)
    # get a priority boost above GLiNER's 50 — proven format correctness wins.
    if d.source in ("regex", "field_label") and d.label in FORMAT_VALIDATED_LABELS:
        source_pri = max(source_pri, 55)

    return (
        len(d.text),                          # 1. Longer matches win
        source_pri,                           # 2. Source priority (GLiNER > field > regex; format-validated overrides)
        _normalize_score(d.score, d.source),  # 3. Normalized confidence score
        LABEL_PRIORITY.get(d.label, 10)       # 4. Taxonomy Priority tie-breaker
    )

def _labels_compatible(label1: str, label2: str) -> bool:
    """Check if two labels are semantically compatible for cross-validation."""
    if label1 == label2:
        return True
    # Groups of labels that may legitimately cross-validate each other
    _COMPAT_GROUPS = [
        {"PERSON_FULL_NAME", "PERSON_FIRST_NAME", "PERSON_LAST_NAME", "PERSON_MIDDLE_NAME"},
        {"CREDIT_CARD_NUMBER", "ACCOUNT_NUMBER_AMEX"},
        {"SSN", "TAX_ID"},
        {"MEDICAL_RECORD_NUMBER", "HEALTH_PLAN_BENEFICIARY_NUMBER"},
        {"EMPLOYEE_ID", "PAYROLL_NUMBER"},
    ]
    for group in _COMPAT_GROUPS:
        if label1 in group and label2 in group:
            return True
    # Explicitly incompatible: temporal labels must not cross-validate each other
    _TEMPORAL_LABELS = {
        "DATE_OF_BIRTH", "CARD_EXPIRATION_DATE", "RETIREMENT_DATE",
        "SEPARATION_DATE", "TRANSACTION_DATE",
    }
    if label1 in _TEMPORAL_LABELS and label2 in _TEMPORAL_LABELS and label1 != label2:
        return False
    return False


def apply_cross_validation_bonus(detections: List[Detection]) -> List[Detection]:
    """Boost score when GLiNER and regex/field independently agree on the same span.

    Two independent sources agreeing is extremely high confidence.
    The GLiNER detection gets a +0.05 score boost (capped at 0.99).
    Must be called BEFORE resolve_detections() so the bonus affects winner selection.
    """
    boosted = list(detections)

    for i, d1 in enumerate(boosted):
        if d1.source != "gliner":
            continue
        for j, d2 in enumerate(boosted):
            if d2.source not in ("regex", "field_label"):
                continue
            # Check span overlap > 80%
            overlap_start = max(d1.start, d2.start)
            overlap_end = min(d1.end, d2.end)
            overlap_len = max(0, overlap_end - overlap_start)
            shorter_len = min(d1.end - d1.start, d2.end - d2.start)
            if shorter_len <= 0 or overlap_len / shorter_len < 0.80:
                continue
            if not _labels_compatible(d1.label, d2.label):
                continue

            # Apply cross-validation bonus to GLiNER detection
            new_score = min(d1.score + 0.05, 0.99)
            boosted[i] = Detection(
                label=d1.label,
                text=d1.text,
                start=d1.start,
                end=d1.end,
                score=new_score,
                source=d1.source,
                meta={**d1.meta, "cross_validated": True, "cross_source": d2.source},
            )
            break  # One cross-validation per GLiNER detection is enough

    return boosted


def resolve_detections(detections: List[Detection]) -> List[Detection]:
    if not detections: return []
    ordered = sorted(detections, key=lambda d: (d.start, d.end, -len(d.text)))
    kept: List[Detection] = []

    for d in ordered:
        replaced = False
        discard = False
        for i, k in enumerate(kept):
            if overlap(d.start, d.end, k.start, k.end):
                if _score(d) > _score(k):
                    kept[i] = d
                    replaced = True
                else:
                    discard = True
                break
        if not discard and not replaced:
            kept.append(d)

    kept.sort(key=lambda x: (x.start, x.end))
    return kept
