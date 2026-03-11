from __future__ import annotations

import re
from typing import Dict, List, Optional

from app.models import Detection


class ContextDetector:
    def __init__(self, context_rules: Dict | None = None):
        self.context_rules = context_rules or {}

        self.security_question_re = re.compile(
            r"(mother[’']?s maiden name|security question|challenge question|stored answer|primary question|secondary question|tertiary question)",
            re.IGNORECASE,
        )
        self.last4_re = re.compile(
            r"\b(last\s+(?:four|4)|ending\s+in|ends?\s+with)\b",
            re.IGNORECASE,
        )
        self.routing_re = re.compile(r"\brouting\b", re.IGNORECASE)
        self.bank_account_re = re.compile(
            r"\b(bank\s+account|checking\s+account|savings\s+account|account number)\b",
            re.IGNORECASE,
        )
        self.card_re = re.compile(
            r"\b(card|amex|account|pan|card number)\b",
            re.IGNORECASE,
        )
        self.dob_re = re.compile(r"\b(date of birth|dob)\b", re.IGNORECASE)
        self.transaction_re = re.compile(
            r"\b(transaction|spent at|purchase|posted|charge|payment)\b",
            re.IGNORECASE,
        )
        self.track_re = re.compile(r"\b(track data|track 2|magstripe)\b", re.IGNORECASE)
        self.mr_re = re.compile(
            r"\b(membership rewards|mr account|mr number|master mr id)\b",
            re.IGNORECASE,
        )
        self.otp_re = re.compile(
            r"\b(one.?time|otp|verification code|backup code)\b",
            re.IGNORECASE,
        )
        self.pin_re = re.compile(r"\b(pin|passcode|ivr|portal access pin)\b", re.IGNORECASE)
        self.csc_re = re.compile(r"\b(cid|cvv|cvc|security code)\b", re.IGNORECASE)
        self.exp_re = re.compile(r"\b(exp|expires|expiration)\b", re.IGNORECASE)
        self.ssn_re = re.compile(r"\b(ssn|social security)\b", re.IGNORECASE)
        self.passport_re = re.compile(r"\bpassport\b", re.IGNORECASE)
        self.dl_re = re.compile(
            r"\b(driver'?s license|drivers license|driver’s license|dl)\b",
            re.IGNORECASE,
        )
        self.geo_re = re.compile(
            r"\b(gps|logged location|precise geo|precise geolocation|coordinates)\b",
            re.IGNORECASE,
        )
        self.device_re = re.compile(
            r"\b(device|app|ios|android|mac address|advertising id|device token|device id)\b",
            re.IGNORECASE,
        )

    def detect(self, text: str, detections: List[Detection]) -> List[Detection]:
        new_detections: List[Detection] = []

        for d in detections:
            neighborhood = self._get_neighborhood(text, d.start, d.end)
            promoted = self.classify_by_context(d.label, d.text, neighborhood)

            if promoted and promoted != d.label:
                new_detections.append(
                    Detection(
                        label=promoted,
                        text=d.text,
                        start=d.start,
                        end=d.end,
                        score=max(d.score, 0.85),
                        source="context",
                        meta={
                            "promoted_from": d.label,
                            "neighborhood": neighborhood[:200],
                        },
                    )
                )

        return new_detections

    def _get_neighborhood(self, text: str, start: int, end: int, window: int = 100) -> str:
        left = max(0, start - window)
        right = min(len(text), end + window)
        return text[left:right]

    def classify_by_context(self, label: str, value: str, neighborhood: str) -> Optional[str]:
        v = value.strip()
        n = neighborhood.lower()

        if not v:
            return None

        numeric_label = self.classify_numeric_by_context(v, neighborhood)
        if numeric_label:
            return numeric_label

        if label in {"PERSON_FULL_NAME", "PERSON_FIRST_NAME", "PERSON_LAST_NAME"}:
            if self.security_question_re.search(n) and "mother" in n and "maiden" in n:
                return "MOTHERS_MAIDEN_NAME"

        if label in {"UNKNOWN_IDENTIFIER", "REFERENCE_IDENTIFIER"}:
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

        if label in {"UNKNOWN_SECRET", "UNKNOWN_SENSITIVE"}:
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

        if re.fullmatch(r"\d{4}", v) and self.last4_re.search(n):
            if self.routing_re.search(n):
                return None
            if self.bank_account_re.search(n):
                return "BANK_ACCOUNT_LAST4"
            if self.card_re.search(n):
                return "ACCOUNT_LAST4"
            if self.ssn_re.search(n):
                return "SSN_LAST4"

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

        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", v):
            if self.transaction_re.search(n):
                return None
            if self.dob_re.search(n):
                return "DATE_OF_BIRTH"
            
        # In classify_numeric_by_context
        if re.fullmatch(r"\d{7,9}", v):
            if "employee" in n or "badge" in n:
                return "EMPLOYEE_ID"
            if "tax" in n or "itin" in n:
                return "TAX_ID"

        return None