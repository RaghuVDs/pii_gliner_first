from __future__ import annotations

import re
from typing import Dict, List, Optional
from rapidfuzz import fuzz
from app.models import Detection


class ContextDetector:
    def __init__(self, context_rules: Dict | None = None):
        self.context_rules = context_rules or {}
        
        # Compile dynamic rules from YAML if provided, allowing rulebook expansions
        self.dynamic_rules = {}
        for label, patterns in self.context_rules.items():
            compiled_patterns = []
            for p in patterns:
                try:
                    compiled_patterns.append(re.compile(p, re.IGNORECASE))
                except re.error:
                    pass
            self.dynamic_rules[label] = compiled_patterns

        # Core robust fallbacks - Expanded with massive synonym lists
        self.security_question_re = re.compile(
            r"(mother[’']?s maiden name|security question|challenge question|stored answer|primary question|secondary question|tertiary question)",
            re.IGNORECASE,
        )
        self.last4_re = re.compile(
            r"\b(last\s+(?:four|4)|ending\s+in|ends?\s+with)\b",
            re.IGNORECASE,
        )
        self.routing_re = re.compile(r"\b(routing|aba|transit)\b", re.IGNORECASE)
        
        self.bank_account_re = re.compile(
            r"\b(bank\s+account|checking\s+account|savings\s+account|account number|acct\s*#|account)\b",
            re.IGNORECASE,
        )

        self.card_re = re.compile(
            r"\b(card|amex|account|pan|card number|visa|mastercard|discover|credit\s*card|debit\s*card)\b",
            re.IGNORECASE,
        )
        self.dob_re = re.compile(r"\b(date of birth|dob|birthdate|born on)\b", re.IGNORECASE)
        self.transaction_re = re.compile(
            r"\b(transaction|spent at|purchase|posted|charge|payment|auth|authorization)\b",
            re.IGNORECASE,
        )
        self.track_re = re.compile(r"\b(track data|track 2|magstripe|magnetic stripe)\b", re.IGNORECASE)
        self.mr_re = re.compile(
            r"\b(membership rewards|mr account|mr number|master mr id|loyalty points)\b",
            re.IGNORECASE,
        )
        self.otp_re = re.compile(
            r"\b(one.?time|otp|verification code|backup code|2fa|mfa|security code sent to)\b",
            re.IGNORECASE,
        )
        self.pin_re = re.compile(r"\b(pin|passcode|ivr|portal access pin|personal identification number)\b", re.IGNORECASE)
        self.csc_re = re.compile(r"\b(cid|cvv|cvc|security code|card verification)\b", re.IGNORECASE)
        self.exp_re = re.compile(r"\b(exp|expires|expiration|valid thru)\b", re.IGNORECASE)
        self.ssn_re = re.compile(r"\b(ssn|social security|social|tin)\b", re.IGNORECASE)
        self.passport_re = re.compile(r"\b(passport|travel document)\b", re.IGNORECASE)
        self.dl_re = re.compile(
            r"\b(driver'?s license|drivers license|driver’s license|dl|state id)\b",
            re.IGNORECASE,
        )
        self.geo_re = re.compile(
            r"\b(gps|logged location|precise geo|precise geolocation|coordinates|lat/long|latitude)\b",
            re.IGNORECASE,
        )
        self.device_re = re.compile(
            r"\b(device|app|ios|android|mac address|advertising id|device token|device id|imei|uuid)\b",
            re.IGNORECASE,
        )
        self.itin_re = re.compile(r"\b(itin|tax id|tax identification|employer id|ein)\b", re.IGNORECASE)


    def _get_fuzzy_neighborhood_match(self, neighborhood: str, target_label: str, threshold: int = 85) -> bool:
        """Checks if any keyword for a label exists in the neighborhood with a fuzzy match."""
        keywords = self.context_rules.get(target_label, [])
        for word in neighborhood.lower().split():
            for kw in keywords:
                if fuzz.ratio(word, kw.lower()) >= threshold:
                    return True
        return False    

    def detect(self, text: str, detections: List[Detection]) -> List[Detection]:
        updated_detections: List[Detection] = []

        for d in detections:
            # Expand neighborhood for "super strong" context detection (150 chars each way)
            neighborhood = self._get_neighborhood(text, d.start, d.end, window=150)
            
            promoted = self.classify_by_context(d.label, d.text, neighborhood)

            if promoted and promoted != d.label:
                updated_detections.append(
                    Detection(
                        label=promoted,
                        text=d.text,
                        start=d.start,
                        end=d.end,
                        score=max(d.score, 0.95),  # Boost score to win overlap resolutions
                        source="context",
                        meta={"promoted_from": d.label, "neighborhood": neighborhood[:250]},
                    )
                )
            else:
                updated_detections.append(d)

        return updated_detections

    def _get_neighborhood(self, text: str, start: int, end: int, window: int = 150) -> str:
        left = max(0, start - window)
        right = min(len(text), end + window)
        
        # SMART BOUNDARIES: Snap to nearest newline if within the window
        # This prevents contextual bleed across distinct log entries or paragraphs
        nearest_newline_left = text.rfind('\n', left, start)
        if nearest_newline_left != -1:
            left = nearest_newline_left + 1
            
        nearest_newline_right = text.find('\n', end, right)
        if nearest_newline_right != -1:
            right = nearest_newline_right
            
        return text[left:right]

    def _matches_dynamic_rule(self, label: str, neighborhood: str) -> bool:
        if label not in self.dynamic_rules:
            return False
        for pattern in self.dynamic_rules[label]:
            if pattern.search(neighborhood):
                return True
        return False

    def classify_by_context(self, label: str, value: str, neighborhood: str) -> Optional[str]:
        v = value.strip()
        n = neighborhood.lower()

        if not v:
            return None

        # 1. Direct Re-routes based on strong context keywords
        if label == "SSN" and self.itin_re.search(n):
            return "TAX_ID"

        # 2. Check pure numeric promotions
        numeric_label = self.classify_numeric_by_context(v, n)
        if numeric_label:
            return numeric_label

        # 3. Dynamic overrides from YAML context_rules (if provided)
        for dyn_label in self.dynamic_rules:
            if self._matches_dynamic_rule(dyn_label, n):
                return dyn_label

        # 4. Fallback Name/Security Checks
        if label in {"PERSON_FULL_NAME", "PERSON_FIRST_NAME", "PERSON_LAST_NAME"}:
            if self.security_question_re.search(n) and "mother" in n and "maiden" in n:
                return "MOTHERS_MAIDEN_NAME"

        # 5. Identifier Upgrades
        if label in {"UNKNOWN_IDENTIFIER", "REFERENCE_IDENTIFIER", "ID"}:
            if self.routing_re.search(n) and re.fullmatch(r"\d{9}", v):
                return "ROUTING_NUMBER"
            if self.bank_account_re.search(n) and re.fullmatch(r"\d{8,17}", v):
                return "BANK_ACCOUNT_NUMBER"
            if self.mr_re.search(n):
                return "MR_NUMBER"
            if self.track_re.search(n):
                return "TRACK_LOG_ID"
            if self.passport_re.search(n):
                return "PASSPORT_NUMBER"
            if self.dl_re.search(n):
                return "DRIVERS_LICENSE_NUMBER"
            if self.geo_re.search(n):
                return "PRECISE_GEOLOCATION"
            if self.device_re.search(n):
                return "DEVICE_APP_DATA"

        # 6. Secret Upgrades
        if label in {"UNKNOWN_SECRET", "UNKNOWN_SENSITIVE", "CODE"}:
            if self.otp_re.search(n):
                return "ONE_TIME_CODE"
            if self.pin_re.search(n) and re.fullmatch(r"\d{3,6}", v):
                return "PIN"
            if self.device_re.search(n):
                return "DEVICE_APP_DATA"
            if self.track_re.search(n):
                return "TRACK_DATA"

        return None

    def classify_numeric_by_context(self, value: str, neighborhood: str) -> Optional[str]:
        v = value.strip()
        n = neighborhood.lower()

        if not v:
            return None

        # Extremely robust last 4 detection
        if re.fullmatch(r"\d{4}", v) and self.last4_re.search(n):
            if self.routing_re.search(n):
                return None
            if self.bank_account_re.search(n):
                return "BANK_ACCOUNT_LAST4"
            if self.card_re.search(n):
                return "ACCOUNT_LAST4"
            if self.ssn_re.search(n):
                return "SSN_LAST4"
            return "GENERIC_LAST4"

        if re.fullmatch(r"\d{9}", v) and self.routing_re.search(n):
            return "ROUTING_NUMBER"

        if re.fullmatch(r"\d{6,8}", v) and self.otp_re.search(n):
            return "ONE_TIME_CODE"

        if re.fullmatch(r"\d{3,4}", v):
            if self.csc_re.search(n):
                return "CARD_SECURITY_CODE"
            if self.pin_re.search(n):
                return "PIN"

        if re.fullmatch(r"\d{8,17}", v) and self.bank_account_re.search(n):
            return "BANK_ACCOUNT_NUMBER"

        if re.fullmatch(r"\d{1,2}/\d{2}", v) and self.exp_re.search(n):
            return "CARD_EXPIRATION_DATE"

        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", v):
            if self.transaction_re.search(n):
                return None
            if self.dob_re.search(n):
                return "DATE_OF_BIRTH"
            
        if re.fullmatch(r"\d{7,10}", v):
            if "employee" in n or "badge" in n:
                return "EMPLOYEE_ID"
            if "tax" in n or "itin" in n or "ein" in n:
                return "TAX_ID"

        return None