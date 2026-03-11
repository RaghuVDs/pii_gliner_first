from __future__ import annotations
from typing import List
from app.models import Detection
from app.utils import overlap

LABEL_PRIORITY = {
    # Contextual Highest
    "BIOMETRIC_INFORMATION": 100, "PHI": 100, "CHILDBEARING_STATUS": 99, 
    "CRIMINAL_RECORD": 99, "ETHNICITY_OR_RACE": 99, "POLITICAL_OPINIONS": 99, 
    "SEXUAL_ORIENTATION": 99, "GENETIC_INFORMATION": 99, "LEGAL_JUDGEMENT": 98, 
    "OPINIONS_CAPTURED": 98, "PERFORMANCE_RATING": 98, "PUBLIC_ASSISTANCE_RECEIPT": 98,
    "CHILDREN_INFORMATION": 98, "MEMBERSHIP_OR_TRADE_UNIONS": 98,
    "ONLINE_UNIQUE_IDENTIFIER": 98, "PHOTOGRAPH_METADATA": 98, "VETERAN_STATUS": 98,
    "CHARITABLE_CONTRIBUTION": 97, "CUSTOMER_STATUS": 97, "PAYMENT_HISTORY": 96,
    
    # Financial/Identity
    "TRACK_DATA": 95, "ACCOUNT_NUMBER_AMEX": 94, "CREDIT_CARD_NUMBER": 94, 
    "BANK_ACCOUNT_NUMBER": 93, "IBAN": 93, "CRYPTO_WALLET": 93,
    "ROUTING_NUMBER": 92, "SSN": 91, "TAX_ID": 91, "NATIONAL_ID": 91, 
    "PASSPORT_NUMBER": 90, "GREEN_CARD_NUMBER": 90, "DRIVERS_LICENSE_NUMBER": 90,
    "NON_DRIVERS_ID": 90, "PARTNER_ACCOUNT_NUMBER": 90, "MORTGAGE_LOAN_INFO": 90,
    "INSURANCE_POLICY": 90,
    
    # Granular Names
    "PERSON_FIRST_NAME": 89, "PERSON_MIDDLE_NAME": 89, "PERSON_LAST_NAME": 89, "PERSON_FULL_NAME": 80,
    
    # Standard SDEs
    "MOTHERS_MAIDEN_NAME": 88, "PASSWORD": 87, "PIN": 86, "ONE_TIME_CODE": 85, 
    "EMAIL_ADDRESS": 84, "PHONE_NUMBER": 83, "STREET_ADDRESS": 82, 
    "DATE_OF_BIRTH": 81, "RETIREMENT_DATE": 81, "SEPARATION_DATE": 81, "AGE": 79, 
    "EMPLOYER_NAME": 78, "EMPLOYEE_ID": 77, "USERNAME": 76,
    "EDUCATIONAL_AFFILIATIONS": 76, "EMPLOYMENT_HISTORY": 76, "TICKET_NUMBER": 76,
    "CARD_TYPE": 76, "AUTHORIZED_AGENT_INFO": 76, "HIGH_VALUE_INDICATOR": 76,
    "MARKETING_PREFERENCES": 76, "ROC_DATA": 76, "SOLE_TRADER_DATA": 76,
    
    # Lower Priority / Fallbacks
    "PRECISE_GEOLOCATION": 75, "DEVICE_APP_DATA": 74, "IP_ADDRESS": 73, "URL": 72,
    "CUSTOMER_PURCHASE_DATA": 69, "SOCIAL_MEDIA_PROFILE": 68, "UNKNOWN_PII": 10
}

def _score(d: Detection):
    return (
        len(d.text),                     # 1. Longer matches win overlap battles
        float(d.score),                  # 2. Confidence wins (Regex 0.98 kills GLiNER 0.40)
        LABEL_PRIORITY.get(d.label, 10), # 3. Placemat Priority tie-breaker
        1 if d.source == "regex" else 0  # 4. Final tie-breaker
    )

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