from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from app.models import Detection


class ContextDetector:
    def __init__(self, context_rules: Dict):
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
        self.bank_account_re = re.compile(r"\b(bank\s+account|checking\s+account|account number)\b", re.IGNORECASE)
        self.card_re = re.compile(r"\b(card|amex|account)\b", re.IGNORECASE)
        self.dob_re = re.compile(r"\b(date of birth|dob)\b", re.IGNORECASE)
        self.transaction_re = re.compile(r"\b(transaction|spent at|purchase|posted|charge|payment)\b", re.IGNORECASE)
        self.track_re = re.compile(r"\b(track data|track 2|magstripe)\b", re.IGNORECASE)
        self.mr_re = re.compile(r"\b(membership rewards|mr account|mr number|master mr id)\b", re.IGNORECASE)

    def detect(self, text: str) -> List[Detection]:
        detections: List[Detection] = []
        lines = text.splitlines()

        offset = 0
        for i, line in enumerate(lines):
            clean = line.strip()
            prev_line = lines[i - 1] if i > 0 else ""
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            neighborhood = f"{prev_line}\n{line}\n{next_line}"

            for label, match in self._detect_contexts(clean, neighborhood):
                start = offset + match.start()
                end = offset + match.end()
                detections.append(
                    Detection(
                        label=label,
                        text=match.group(0),
                        start=start,
                        end=end,
                        score=0.85,
                        source="context",
                        meta={"line": clean},
                    )
                )
            offset += len(line) + 1

        return detections

    def _detect_contexts(self, line: str, neighborhood: str) -> List[Tuple[str, re.Match]]:
        hits: List[Tuple[str, re.Match]] = []

        m = self.security_question_re.search(neighborhood)
        if m:
            hits.append(("SECURITY_CONTEXT", m))

        m = self.track_re.search(neighborhood)
        if m:
            hits.append(("TRACK_CONTEXT", m))

        m = self.mr_re.search(neighborhood)
        if m:
            hits.append(("MR_CONTEXT", m))

        m = self.dob_re.search(neighborhood)
        if m:
            hits.append(("DOB_CONTEXT", m))

        m = self.transaction_re.search(neighborhood)
        if m:
            hits.append(("TRANSACTION_CONTEXT", m))

        m = self.last4_re.search(neighborhood)
        if m:
            hits.append(("LAST4_CONTEXT", m))

        m = self.routing_re.search(neighborhood)
        if m:
            hits.append(("ROUTING_CONTEXT", m))

        m = self.bank_account_re.search(neighborhood)
        if m:
            hits.append(("BANK_ACCOUNT_CONTEXT", m))

        m = self.card_re.search(neighborhood)
        if m:
            hits.append(("CARD_CONTEXT", m))

        return hits

    def classify_numeric_by_context(self, value: str, neighborhood: str) -> Optional[str]:
        v = value.strip()

        if not v:
            return None

        if self.last4_re.search(neighborhood):
            if self.routing_re.search(neighborhood):
                return None
            if self.bank_account_re.search(neighborhood):
                return "BANK_ACCOUNT_LAST4"
            if self.card_re.search(neighborhood):
                return "ACCOUNT_LAST4"
            if re.search(r"\bssn|social security\b", neighborhood, re.IGNORECASE):
                return "SSN_LAST4"

        if re.fullmatch(r"\d{9}", v) and self.routing_re.search(neighborhood):
            return "ROUTING_NUMBER"

        if re.fullmatch(r"\d{6}", v) and re.search(r"\b(one.?time|otp|verification code|backup code)\b", neighborhood, re.IGNORECASE):
            return "ONE_TIME_CODE"

        if re.fullmatch(r"\d{3,4}", v):
            if re.search(r"\b(cid|cvv|cvc|security code)\b", neighborhood, re.IGNORECASE):
                return "CARD_SECURITY_CODE"
            if re.search(r"\bpin|passcode|ivr\b", neighborhood, re.IGNORECASE):
                return "PIN"

        if re.fullmatch(r"\d{8,17}", v):
            if self.bank_account_re.search(neighborhood):
                return "BANK_ACCOUNT_NUMBER"

        if re.fullmatch(r"\d{1,2}/\d{2}", v) and re.search(r"\b(exp|expires|expiration)\b", neighborhood, re.IGNORECASE):
            return "CARD_EXPIRATION_DATE"

        return None