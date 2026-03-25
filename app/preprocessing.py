from __future__ import annotations

import re

EMAIL_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.I)
PHONE_RE = re.compile(r"^\+?[\d\(\)][\d\-\(\) xX\.]{6,}$")
SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")
IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
PASSPORT_RE = re.compile(r"^[A-Z]\d{7,8}$|^[A-Z0-9]{6,9}$", re.I)
DL_RE = re.compile(r"^[A-Z]{0,3}[- ]?[A-Z]{0,3}[- ]?\d{5,12}$", re.I)
MR_RE = re.compile(r"^(MR[- ]?)?[A-Z0-9\-]{8,30}$", re.I)
PAR_RE = re.compile(r"^PAR-[A-Z0-9\-]{8,}$", re.I)
AMEX_ACCOUNT_ID_RE = re.compile(r"^AXP-[A-Z0-9\-]{8,}$", re.I)
CUSTOMER_ID_RE = re.compile(r"^(AMEX-)?CUST-[A-Z]{2}-[A-Z0-9\-]{4,}$", re.I)
EMPLOYEE_ID_RE = re.compile(r"^(AXP-EMP-[A-Z]{2}-[A-Z]{3}-\d{3,}|[0-9]{5,10}|[A-Z]{2,5}[\-][A-Z0-9\-]{2,15})$", re.I)
TRACK_DATA_RE = re.compile(r"^[%;]?[A-Z0-9\^\?=/\-\.]{20,}$", re.I)
ROUTING_RE = re.compile(r"^\d{9}$")
BANK_ACCOUNT_RE = re.compile(r"^\d{8,17}$")
DATE_MMDDYYYY_RE = re.compile(r"^(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](19|20)\d{2}$")
DATE_TEXT_RE = re.compile(
    r"^(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}$",
    re.IGNORECASE,
)
DATE_TEXT_ALT_RE = re.compile(
    r"^\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
    r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?),?\s+\d{4}$",
    re.IGNORECASE,
)
DATE_ISO_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")
CARD_EXP_RE = re.compile(r"^(0[1-9]|1[0-2])/\d{2}$")
CSC_RE = re.compile(r"^\d{3,4}$")
GEO_COORD_RE = re.compile(r"^-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+$")
MAC_RE = re.compile(r"^[0-9A-F]{2}(:[0-9A-F]{2}){5}$", re.I)
USERNAME_RE = re.compile(r"^[A-Z0-9][A-Z0-9._\-]{2,31}$", re.I)
PASSWORD_COMPLEX_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[^A-Za-z\d]).{8,}$")

BAD_PERSON_VALUES = {
    "name", "first name", "last name", "full name", "customer",
    "cardmember", "individual", "employee", "agent", "merchant",
    "american express", "amex", "centurion", "platinum", "delta",
    "optima", "representative", "supervisor", "manager", "associate",
    # Expanded: common false-positive person names
    "admin", "administrator", "analyst", "advisor", "consultant",
    "coordinator", "director", "executive", "officer", "operator",
    "specialist", "support", "service", "system", "user", "member",
    "account", "cardholder", "applicant", "beneficiary", "claimant",
    "complainant", "correspondent", "defendant", "plaintiff",
    "primary", "secondary", "authorized", "unknown", "anonymous",
    "test", "demo", "sample", "example", "null", "none", "n/a",
    "company", "organization", "corporation", "enterprise", "business",
    # Function/stop words that regex IGNORECASE can capture as name parts
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "for", "and", "but", "or", "not", "nor", "so", "yet", "at", "by",
    "in", "on", "to", "of", "up", "if", "it", "no", "do", "my", "we",
    "he", "me", "us", "his", "her", "its", "our", "who", "how", "all",
    "had", "has", "did", "got", "can", "may", "will", "shall", "with",
    "from", "that", "this", "than", "then", "also", "very", "just",
    "about", "there", "their", "your", "what", "which", "when", "where",
}

BAD_ORG_VALUES = {
    "american express travel related services", "american express",
    "target", "dallas, tx", "phoenix, arizona", "visa", "mastercard",
    "discover", "jcb"
}

def luhn_checksum(card_number: str) -> bool:
    """Validates credit card numbers, AMEX, and Track Data using the Luhn Algorithm."""
    digits = [int(c) for c in card_number if c.isdigit()]
    if not digits:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0

def aba_routing_checksum(routing_number: str) -> bool:
    """Validates US Bank Routing Numbers using the mathematical Mod-10 checksum."""
    digits = [int(c) for c in routing_number if c.isdigit()]
    if len(digits) != 9:
        return False
    calc = (
        3 * (digits[0] + digits[3] + digits[6]) +
        7 * (digits[1] + digits[4] + digits[7]) +
        1 * (digits[2] + digits[5] + digits[8])
    )
    return calc % 10 == 0

def normalize_candidate(text: str) -> str:
    return text.strip().strip("\"'“”‘’").strip()

def is_valid_email(value: str) -> bool:
    return bool(EMAIL_RE.fullmatch(normalize_candidate(value)))

def is_valid_phone(value: str) -> bool:
    v = normalize_candidate(value)
    if len(re.sub(r"\D", "", v)) < 7:
        return False
    return bool(PHONE_RE.fullmatch(v))

def is_valid_ssn(value: str) -> bool:
    return bool(SSN_RE.fullmatch(normalize_candidate(value)))

def is_valid_ip(value: str) -> bool:
    v = normalize_candidate(value)
    if not IP_RE.fullmatch(v):
        return False
    parts = [int(x) for x in v.split(".")]
    return all(0 <= p <= 255 for p in parts)

def is_valid_passport(value: str) -> bool:
    return bool(PASSPORT_RE.fullmatch(normalize_candidate(value)))

def is_valid_drivers_license(value: str) -> bool:
    v = normalize_candidate(value)
    if v.lower() == "image":
        return False
    return bool(DL_RE.fullmatch(v))

def is_valid_mr_number(value: str) -> bool:
    v = normalize_candidate(value)
    if "membership rewards" in v.lower():
        return False
    return bool(MR_RE.fullmatch(v))

def is_valid_par(value: str) -> bool:
    return bool(PAR_RE.fullmatch(normalize_candidate(value)))

def is_valid_amex_account_id(value: str) -> bool:
    return bool(AMEX_ACCOUNT_ID_RE.fullmatch(normalize_candidate(value)))

def is_valid_customer_id(value: str) -> bool:
    return bool(CUSTOMER_ID_RE.fullmatch(normalize_candidate(value)))

def is_valid_employee_id(value: str) -> bool:
    return bool(EMPLOYEE_ID_RE.fullmatch(normalize_candidate(value)))

def is_valid_track_data(value: str) -> bool:
    v = normalize_candidate(value)
    # Extract the PAN (Primary Account Number) from track data and check Luhn
    pan_match = re.search(r"%[Bb](\d{12,19})\^", v)
    if pan_match and not luhn_checksum(pan_match.group(1)):
        return False
    return bool(TRACK_DATA_RE.fullmatch(v)) and any(ch in v for ch in ["^", "?", "=", "%", ";"])

def is_valid_routing_number(value: str) -> bool:
    v = normalize_candidate(value)
    return bool(ROUTING_RE.fullmatch(v)) and aba_routing_checksum(v)

def is_valid_bank_account_number(value: str) -> bool:
    v = normalize_candidate(value)
    return bool(BANK_ACCOUNT_RE.fullmatch(v)) and not is_valid_routing_number(v)

def is_valid_date_of_birth(value: str) -> bool:
    v = normalize_candidate(value)
    return bool(
        DATE_MMDDYYYY_RE.fullmatch(v)
        or DATE_TEXT_RE.fullmatch(v)
        or DATE_TEXT_ALT_RE.fullmatch(v)
        or DATE_ISO_RE.fullmatch(v)
    )

def is_valid_card_expiration(value: str) -> bool:
    return bool(CARD_EXP_RE.fullmatch(normalize_candidate(value)))

def is_valid_card_security_code(value: str) -> bool:
    return bool(CSC_RE.fullmatch(normalize_candidate(value)))

def is_valid_precise_geo(value: str) -> bool:
    return bool(GEO_COORD_RE.fullmatch(normalize_candidate(value)))

def is_valid_device_app_data(value: str) -> bool:
    v = normalize_candidate(value)
    return bool(
        MAC_RE.search(v)
        or is_valid_ip(v)
        or re.fullmatch(r"[A-Za-z0-9._\- :]{3,40}", v) 
    )

def is_valid_username(value: str) -> bool:
    v = normalize_candidate(value)
    if v.lower() in BAD_PERSON_VALUES:
        return False
    return bool(USERNAME_RE.fullmatch(v))

def is_valid_password(value: str) -> bool:
    return bool(PASSWORD_COMPLEX_RE.fullmatch(normalize_candidate(value)))

def is_valid_person_name(value: str) -> bool:
    v = normalize_candidate(value)
    if not v:
        return False
    if v.lower() in BAD_PERSON_VALUES:
        return False
    if "@" in v:
        return False
    if re.search(r"\d{3,}", v):
        return False
    parts = [p for p in re.split(r"\s+", v) if p]
    if len(parts) < 1 or len(parts) > 5:
        return False
    return bool(re.fullmatch(r"[A-Za-z\u00C0-\u024F][A-Za-z\u00C0-\u024F.\-']*(?:\s+[A-Za-z\u00C0-\u024F][A-Za-z\u00C0-\u024F.\-']*){0,4}", v))

# Minimum character lengths per entity type to reject noise
MIN_LENGTH_RULES = {
    "PERSON_FULL_NAME": 3,
    "PERSON_FIRST_NAME": 2,
    "PERSON_MIDDLE_NAME": 2,
    "PERSON_LAST_NAME": 2,
    "EMAIL_ADDRESS": 6,
    "STREET_ADDRESS": 10,
    "USERNAME": 3,
    "PASSWORD": 8,
    "PHONE_NUMBER": 7,
    "BANK_ACCOUNT_NUMBER": 8,
    "ACCOUNT_NUMBER_AMEX": 8,
    "CREDIT_CARD_NUMBER": 13,
    "IBAN": 15,
    "EMPLOYEE_ID": 4,
    "CUSTOMER_ID": 4,
    "AMEX_ACCOUNT_ID": 4,
    "CARDMEMBER_ID": 4,
    "MR_NUMBER": 4,
    "PARTNER_ACCOUNT_NUMBER": 4,
    "REFERENCE_IDENTIFIER": 4,
    "TRACK_DATA": 20,
    "IP_ADDRESS": 7,
    "URL": 8,
    "CRYPTO_WALLET": 20,
    "VIN": 17,
    "LICENSE_PLATE_NUMBER": 4,
    "SWIFT_BIC_CODE": 8,
    "CUSIP_NUMBER": 9,
    "MEDICAL_RECORD_NUMBER": 4,
    "MEDICARE_MEDICAID_ID": 8,
    "PRESCRIPTION_NUMBER": 4,
    "STUDENT_ID": 3,
    "COURT_CASE_NUMBER": 5,
    "INVESTMENT_ACCOUNT_NUMBER": 4,
    "API_KEY": 16,
    "SSH_KEY": 20,
    "SHIPPING_TRACKING_NUMBER": 10,
}

def should_keep_detection(label: str, value: str, neighborhood: str = "", source: str = "") -> bool:
    v = normalize_candidate(value)
    n = neighborhood.lower()

    # For field_label/context sources, skip neighborhood-based keyword rejection.
    # The explicit field label (e.g., "DOB:", "SSN:") already provides strong context.
    # Only apply format validation for these sources.
    _skip_neg_keywords = source in ("field_label", "context")

    # Minimum length check (applies to all sources)
    min_len = MIN_LENGTH_RULES.get(label)
    if min_len and len(v) < min_len:
        return False

    if label == "SSN" and not _skip_neg_keywords:
        if "audit" in n or "tracking code" in n or "case id" in n:
            return False
        if any(kw in n for kw in ["order number", "confirmation", "reference number",
                                   "invoice", "serial number", "batch", "model number"]):
            return False

    # Apply Luhn to Credit Cards
    if label == "CREDIT_CARD_NUMBER" or label == "ACCOUNT_NUMBER_AMEX":
        return luhn_checksum(v)

    if label in {"INCOME", "PERFORMANCE_RATING"}:
        if v.lower() in {"salary", "wages", "performance rating", "rating"}:
            return False

    # TAX_ID format validation: must be EIN (XX-XXXXXXX), ITIN (9XX-XX-XXXX), or 9 bare digits
    if label == "TAX_ID":
        if re.fullmatch(r"\d{2}-\d{7}", v):           # EIN
            return True
        if re.fullmatch(r"9\d{2}-\d{2}-\d{4}", v):    # ITIN
            return True
        if re.fullmatch(r"\d{9}", v):                  # Bare 9-digit EIN/ITIN
            return True
        return False

    if label == "CARD_EXPIRATION_DATE" and not _skip_neg_keywords:
        if "(r-" in n or "rev" in n or "form" in n:
            return False

    if label in {"STREET_ADDRESS", "EMPLOYER_NAME"}:
        return True

    if label == "EMAIL_ADDRESS":
        return is_valid_email(v)
    if label == "PHONE_NUMBER":
        return is_valid_phone(v)
    if label == "SSN":
        return is_valid_ssn(v)
    if label == "IP_ADDRESS":
        return is_valid_ip(v)
    if label == "PASSPORT_NUMBER":
        return is_valid_passport(v)
    if label == "DRIVERS_LICENSE_NUMBER":
        return is_valid_drivers_license(v)
    if label == "MR_NUMBER":
        return is_valid_mr_number(v)
    if label == "PAR_ID":
        return is_valid_par(v)
    if label == "AMEX_ACCOUNT_ID":
        return is_valid_amex_account_id(v)
    if label == "CUSTOMER_ID":
        return is_valid_customer_id(v)
    if label == "EMPLOYEE_ID":
        return is_valid_employee_id(v)
    if label == "TRACK_DATA":
        return is_valid_track_data(v)
    if label == "ROUTING_NUMBER":
        return is_valid_routing_number(v)
    if label == "BANK_ACCOUNT_NUMBER":
        return is_valid_bank_account_number(v)
    if label == "DATE_OF_BIRTH":
        if not _skip_neg_keywords:
            if any(kw in n for kw in ["transaction", "spent at", "purchase", "posted",
                                       "created", "modified", "updated", "logged",
                                       "effective date", "statement date", "billing date",
                                       "closing date", "due date", "paid on", "renewal",
                                       "start date", "end date", "expir"]):
                return False
        return is_valid_date_of_birth(v)
    if label == "CARD_EXPIRATION_DATE":
        return is_valid_card_expiration(v)
    if label == "CARD_SECURITY_CODE":
        return is_valid_card_security_code(v)
    if label == "PRECISE_GEOLOCATION":
        return is_valid_precise_geo(v)
    if label == "DEVICE_APP_DATA":
        return is_valid_device_app_data(v)
    if label == "USERNAME":
        return is_valid_username(v)
    if label == "PASSWORD":
        return is_valid_password(v)
    if label in {"PERSON_FULL_NAME", "PERSON_FIRST_NAME", "PERSON_LAST_NAME"}:
        return is_valid_person_name(v)
    if label == "MOTHERS_MAIDEN_NAME":
        return is_valid_person_name(v)
    if label == "PIN":
        return bool(re.fullmatch(r"\d{3,6}", v))
    if label == "ONE_TIME_CODE":
        return bool(re.fullmatch(r"\d{6,8}|[A-Z0-9\-]{6,30}", v, re.I))
    if label == "ACCOUNT_LAST4":
        return bool(re.fullmatch(r"\d{4}", v))
    if label == "SSN_LAST4":
        return bool(re.fullmatch(r"\d{4}", v))
    if label == "BANK_ACCOUNT_LAST4":
        return bool(re.fullmatch(r"\d{4}", v))

    # ── Contextual / Sensitive validators ──
    if label == "AGE":
        try:
            age_val = int(re.sub(r"\D", "", v))
            return 0 <= age_val <= 120
        except ValueError:
            return False

    if label == "GENDER":
        valid_genders = {"male", "female", "non-binary", "nonbinary", "transgender",
                         "trans", "other", "prefer not to say"}
        return v.lower() in valid_genders

    if label == "FICO_SCORE":
        try:
            score_val = int(re.sub(r"\D", "", v))
            return 300 <= score_val <= 850
        except ValueError:
            return False

    if label == "MARITAL_OR_FAMILIAL_STATUS":
        valid = {"single", "married", "divorced", "widowed", "separated",
                 "domestic partner", "civil union"}
        return v.lower() in valid

    if label == "CUSTOMER_STATUS":
        valid = {"active", "inactive", "suspended", "delinquent", "deceased",
                 "closed", "frozen", "collections", "charged off", "bankrupt"}
        return v.lower() in valid

    if label == "INSURANCE_POLICY":
        return bool(re.fullmatch(r"[A-Z]{0,3}\d{5,15}[A-Z]?", v, re.I)) or \
               bool(re.fullmatch(r"[A-Z0-9\-]{6,20}", v, re.I))

    if label == "MORTGAGE_LOAN_INFO":
        return bool(re.fullmatch(r"[A-Z0-9\-]{6,20}", v, re.I))

    if label == "GREEN_CARD_NUMBER":
        return bool(re.fullmatch(r"A?\d{7,9}", v, re.I)) or \
               bool(re.fullmatch(r"[A-Z0-9]{7,13}", v, re.I))

    if label == "NATIONAL_ID":
        return bool(re.fullmatch(r"[A-Z0-9\-]{5,20}", v, re.I))

    if label == "NON_DRIVERS_ID":
        return bool(re.fullmatch(r"[A-Z0-9\-]{5,15}", v, re.I))

    if label in {"SOCIAL_MEDIA_PROFILE"}:
        social_domains = ["facebook.com", "fb.com", "twitter.com", "x.com",
                          "instagram.com", "linkedin.com", "tiktok.com",
                          "snapchat.com", "reddit.com", "youtube.com"]
        return any(d in v.lower() for d in social_domains) or v.startswith("@")

    if label == "ONLINE_UNIQUE_IDENTIFIER":
        # UUID or hex/base64 token >= 16 chars
        return bool(re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", v, re.I)) or \
               bool(re.fullmatch(r"[A-Za-z0-9+/=_\-]{16,128}", v))

    if label == "OCCUPATION":
        if len(v) < 3 or len(v) > 50:
            return False
        # Reject if it looks like a sentence (too many words)
        if len(v.split()) > 6:
            return False
        return True

    if label == "EDUCATIONAL_AFFILIATIONS":
        if len(v) < 5:
            return False
        return True

    if label == "EMPLOYER_NAME":
        if len(v) < 2 or v.lower() in BAD_ORG_VALUES:
            return False
        return True

    # ── New PII type validators ──

    if label == "VIN":
        return bool(re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", v))

    if label == "LICENSE_PLATE_NUMBER":
        return bool(re.fullmatch(r"[A-Z0-9]{1,4}[\s\-]?[A-Z0-9]{2,5}", v, re.I)) and len(v) >= 4

    if label == "SWIFT_BIC_CODE":
        return bool(re.fullmatch(r"[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?", v))

    if label == "CUSIP_NUMBER":
        return bool(re.fullmatch(r"[A-Z0-9]{6}[A-Z0-9]{2}[0-9]", v))

    if label == "MEDICAL_RECORD_NUMBER":
        return bool(re.fullmatch(r"[A-Z0-9\-]{5,20}", v, re.I))

    if label == "MEDICARE_MEDICAID_ID":
        # MBI without dashes: 1EK4TA2HY74
        if re.fullmatch(r"[0-9][A-Z][A-Z0-9]{8,9}", v):
            return True
        # MBI with dashes: 1EK4-TA2-HY74
        if re.fullmatch(r"[1-9][A-Z][A-Z0-9]\d-[A-Z][A-Z0-9]\d-[A-Z][A-Z0-9]\d{2}", v):
            return True
        # Generic alphanumeric (Medicaid IDs vary by state)
        return bool(re.fullmatch(r"[A-Z0-9]{8,15}", v, re.I))

    if label == "PRESCRIPTION_NUMBER":
        return bool(re.fullmatch(r"[A-Z0-9\-]{6,15}", v, re.I))

    if label == "STUDENT_ID":
        return bool(re.fullmatch(r"[A-Z0-9\-]{4,15}", v, re.I))

    if label == "PAYROLL_NUMBER":
        return bool(re.fullmatch(r"[A-Z0-9\-]{4,15}", v, re.I))

    if label == "INVESTMENT_ACCOUNT_NUMBER":
        return bool(re.fullmatch(r"[A-Z0-9\-]{5,20}", v, re.I))

    if label == "CREDIT_REPORT_ID":
        return bool(re.fullmatch(r"[A-Z0-9\-]{6,20}", v, re.I))

    if label == "API_KEY":
        return bool(re.fullmatch(r"[A-Za-z0-9+/=_\-]{16,128}", v))

    if label == "SSH_KEY":
        return v.startswith("ssh-") or len(v) >= 20

    if label == "DIGITAL_PAYMENT_ACCOUNT":
        return "@" in v or v.startswith("$") or len(v) >= 5

    if label == "WIRE_TRANSFER_REFERENCE":
        return bool(re.fullmatch(r"[A-Z0-9\-]{8,25}", v, re.I))

    if label == "SHIPPING_TRACKING_NUMBER":
        return bool(re.fullmatch(r"[A-Z0-9]{10,30}", v, re.I))

    if label == "COURT_CASE_NUMBER":
        return bool(re.fullmatch(r"[A-Z0-9\-:]{5,25}", v, re.I))

    if label == "UTILITY_ACCOUNT_NUMBER":
        return bool(re.fullmatch(r"[A-Z0-9\-]{6,20}", v, re.I))

    if label in {"BANKRUPTCY_CASE_NUMBER", "ESCROW_ACCOUNT_NUMBER", "FINRA_CRD_NUMBER",
                 "PROPERTY_TAX_PARCEL", "CHILD_SUPPORT_CASE_NUMBER",
                 "HEALTH_INSURANCE_ID", "WORKERS_COMP_CLAIM_NUMBER",
                 "E_VERIFY_CASE_NUMBER", "VISA_NUMBER", "KNOWN_TRAVELER_NUMBER",
                 "PROFESSIONAL_LICENSE_NUMBER", "SELECTIVE_SERVICE_NUMBER",
                 "VOTER_REGISTRATION_NUMBER", "EBT_CARD_NUMBER", "GIFT_CARD_NUMBER",
                 "CABLE_INTERNET_ACCOUNT"}:
        return bool(re.fullmatch(r"[A-Z0-9\-]{4,25}", v, re.I))

    # Labels that are keyword-based and always valid if matched by regex
    if label in {"PHI", "BIOMETRIC_INFORMATION", "CRIMINAL_RECORD", "DISABILITY_STATUS",
                 "GENETIC_INFORMATION", "SEXUAL_ORIENTATION", "VETERAN_STATUS",
                 "LEAVE_OF_ABSENCE_TYPE", "CHILDBEARING_STATUS", "LEGAL_JUDGEMENT",
                 "PUBLIC_ASSISTANCE_RECEIPT", "PAYMENT_HISTORY", "MARKETING_PREFERENCES",
                 "MEMBERSHIP_OR_TRADE_UNIONS", "POLITICAL_OPINIONS", "RELIGION",
                 "ETHNICITY_OR_RACE", "NATIONALITY",
                 "SEX_OFFENDER_REGISTRY", "PROTECTIVE_ORDER_NUMBER",
                 "CDR_CALL_DETAIL_RECORD", "BROWSER_FINGERPRINT",
                 "SECURITY_CLEARANCE", "I9_EMPLOYMENT_DATA",
                 "OCCUPATIONAL_CERTIFICATION", "BACKGROUND_CHECK_REFERENCE",
                 "FAFSA_DATA", "TRANSCRIPT", "PLACE_OF_BIRTH", "ALIAS_NICKNAME",
                 "SIGNATURE", "PHOTOGRAPH", "TAX_RETURN_DATA",
                 "ARREST_BOOKING_NUMBER", "INMATE_OFFENDER_ID", "PROBATION_PAROLE_ID",
                 "FBI_NUMBER", "STATE_CRIMINAL_ID", "BOAT_HULL_ID", "AIRCRAFT_REGISTRATION",
                 "VEHICLE_REGISTRATION_NUMBER",
                 "CLINICAL_TRIAL_ID", "LAB_ACCESSION_NUMBER", "MEDICAL_DEVICE_SERIAL",
                 "HEALTH_PLAN_BENEFICIARY_NUMBER",
                 "CREDIT_FREEZE_PIN", "DIGITAL_CERTIFICATE", "PGP_KEY"}:
        return len(v) >= 3

    # ── Identifier labels that require alphanumeric format ──
    if label in {"REFERENCE_IDENTIFIER", "TRACK_LOG_ID", "NETWORK_TOKEN_ID",
                 "CARDMEMBER_ID", "PARTNER_ACCOUNT_NUMBER", "TICKET_NUMBER"}:
        return bool(re.fullmatch(r"[A-Z0-9\-]{4,30}", v, re.I))

    # ── Text-value labels that need minimum length ──
    if label in {"SECURITY_QUESTION", "SECURITY_ANSWER", "EMERGENCY_CONTACT_INFORMATION",
                 "CHILDREN_INFORMATION", "EMPLOYMENT_HISTORY", "AUTHORIZED_AGENT_INFO",
                 "CUSTOMER_PURCHASE_DATA", "CHARITABLE_CONTRIBUTION",
                 "HIGH_VALUE_INDICATOR", "ROC_DATA", "SOLE_TRADER_DATA"}:
        return len(v) >= 3

    # ── Reject UNKNOWN_IDENTIFIER that survived (shouldn't exist after regex removal) ──
    if label == "UNKNOWN_IDENTIFIER":
        return False

    return True