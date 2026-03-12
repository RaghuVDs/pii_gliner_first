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
EMPLOYEE_ID_RE = re.compile(r"^(AXP-EMP-[A-Z]{2}-[A-Z]{3}-\d{3,}|[0-9]{6,10})$", re.I)
TRACK_DATA_RE = re.compile(r"^[%;]?[A-Z0-9\^\?=/\-\.]{20,}$", re.I)
ROUTING_RE = re.compile(r"^\d{9}$")
BANK_ACCOUNT_RE = re.compile(r"^\d{8,17}$")
DATE_MMDDYYYY_RE = re.compile(r"^(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](19|20)\d{2}$")
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
    "optima", "representative", "supervisor", "manager", "associate"
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
    return bool(DATE_MMDDYYYY_RE.fullmatch(normalize_candidate(value)))

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
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z.\-']*(?:\s+[A-Za-z][A-Za-z.\-']*){0,4}", v))

def should_keep_detection(label: str, value: str, neighborhood: str = "") -> bool:
    v = normalize_candidate(value)
    n = neighborhood.lower()

    if label == "SSN":
        if "audit" in n or "tracking code" in n or "case id" in n:
            return False

    # Apply Luhn to Credit Cards
    if label == "CREDIT_CARD_NUMBER" or label == "ACCOUNT_NUMBER_AMEX":
        return luhn_checksum(v)
    
        # Reroute ITINs dynamically
    if "itin" in n or "tax id" in n:
        pass

    if label in {"INCOME", "PERFORMANCE_RATING"}:
        if v.lower() in {"salary", "wages", "performance rating", "rating"}:
            return False

    if label == "CARD_EXPIRATION_DATE":
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
        if "transaction" in n or "spent at" in n or "purchase" in n or "posted" in n:
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
    return True