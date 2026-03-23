from __future__ import annotations

import re
from typing import Dict, List, Optional
from rapidfuzz import fuzz
from app.models import Detection


class ContextDetector:
    # Labels where negative context should also apply to regex/field sources (tight window)
    # Labels where dynamic context rules may promote to a more specific label
    _PROMOTABLE_LABELS = {
        "UNKNOWN_IDENTIFIER", "UNKNOWN_PII", "UNKNOWN_SECRET", "UNKNOWN_SENSITIVE",
        "REFERENCE_IDENTIFIER", "ID", "CODE",
    }

    _STRICT_NEG_LABELS = {
        "SSN", "TAX_ID", "DATE_OF_BIRTH", "PHONE_NUMBER", "INCOME",
        "STREET_ADDRESS", "CREDIT_CARD_NUMBER", "BANK_ACCOUNT_NUMBER",
    }

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
            r"\b(bank\s+account|checking\s+account|savings\s+account|account\s*number|acct\s*#)\b",
            re.IGNORECASE,
        )

        self.card_re = re.compile(
            r"\b(card|amex|pan|card number|visa|mastercard|discover|credit\s*card|debit\s*card)\b",
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

        # ── NEGATIVE CONTEXT RULES ──
        # Patterns that indicate a detection is NOT PII (false positive indicators)
        self.negative_rules = {
            "SSN": re.compile(
                r"\b(order\s*(?:number|#|no)|confirmation\s*(?:number|#|no|code)|"
                r"reference\s*(?:number|#|no|code|id)|tracking\s*(?:number|#|no|code)|"
                r"case\s*(?:number|#|no|id)|invoice\s*(?:number|#|no)|"
                r"ticket\s*(?:number|#|no)|claim\s*(?:number|#|no)|"
                r"serial\s*(?:number|#|no)|model\s*(?:number|#|no)|"
                r"audit|batch\s*(?:number|#|no|id))\b",
                re.IGNORECASE,
            ),
            "TAX_ID": re.compile(
                r"\b(order\s*(?:number|#|no)|confirmation\s*(?:number|#|no|code)|"
                r"reference\s*(?:number|#|no|code|id)|tracking\s*(?:number|#|no|code)|"
                r"case\s*(?:number|#|no|id)|invoice\s*(?:number|#|no)|"
                r"ticket\s*(?:number|#|no)|claim\s*(?:number|#|no)|"
                r"serial\s*(?:number|#|no)|model\s*(?:number|#|no)|"
                r"audit|batch\s*(?:number|#|no|id))\b",
                re.IGNORECASE,
            ),
            "DATE_OF_BIRTH": re.compile(
                r"\b(created|modified|updated|logged|processed|"
                r"transaction\s*date|effective\s*date|statement\s*date|"
                r"billing\s*date|closing\s*date|opening\s*date|"
                r"posted\s*(?:on|date)|paid\s*(?:on|date)|due\s*date|"
                r"expir|renewal|start\s*date|end\s*date)\b",
                re.IGNORECASE,
            ),
            "PERSON_FULL_NAME": re.compile(
                r"\b(product\s*name|company\s*name|brand\s*name|"
                r"service\s*name|feature\s*name|plan\s*name|"
                r"card\s*(?:name|type)|program\s*name|project\s*name|"
                r"merchant\s*name|vendor\s*name|provider\s*name)\b",
                re.IGNORECASE,
            ),
            "PERSON_FIRST_NAME": re.compile(
                r"\b(product\s*name|company\s*name|brand\s*name|"
                r"service\s*name|feature\s*name|plan\s*name|"
                r"card\s*(?:name|type)|program\s*name|project\s*name)\b",
                re.IGNORECASE,
            ),
            "PERSON_LAST_NAME": re.compile(
                r"\b(product\s*name|company\s*name|brand\s*name|"
                r"service\s*name|feature\s*name|plan\s*name|"
                r"card\s*(?:name|type)|program\s*name|project\s*name)\b",
                re.IGNORECASE,
            ),
            "PHONE_NUMBER": re.compile(
                r"\b(version|build\s*#|"
                r"zip\s*code|postal\s*code|part\s*(?:number|#|no)|"
                r"npi|national\s*provider|"
                r"policy\s*(?:number|#|no)?|claim\s*(?:number|#|no))\b",
                re.IGNORECASE,
            ),
            "STREET_ADDRESS": re.compile(
                r"\b(email\s*address|web\s*address|ip\s*address|"
                r"mac\s*address|bitcoin\s*address|wallet\s*address|"
                r"url|endpoint)\b",
                re.IGNORECASE,
            ),
            "INCOME": re.compile(
                r"\b(transaction\s*amount|payment\s*amount|balance|"
                r"minimum\s*(?:due|payment)|credit\s*limit|"
                r"available\s*credit|annual\s*fee|interest\s*rate|apr)\b",
                re.IGNORECASE,
            ),
            "CREDIT_CARD_NUMBER": re.compile(
                r"\b(test\s*card|example|sample|dummy|fake|mock|sandbox)\b",
                re.IGNORECASE,
            ),
            "ACCOUNT_NUMBER_AMEX": re.compile(
                r"\b(reference\s*(?:number|#|no|code)|confirmation\s*(?:number|#|no|code)|"
                r"case\s*(?:number|#|no)|tracking\s*(?:number|#|no))\b",
                re.IGNORECASE,
            ),
            "BANK_ACCOUNT_NUMBER": re.compile(
                r"\b(balance|amount|total|payment\s*amount|"
                r"credit\s*limit|available\s*credit|minimum\s*due|"
                r"transaction\s*amount|annual\s*fee)\b",
                re.IGNORECASE,
            ),
            "ROUTING_NUMBER": re.compile(
                r"\b(phone|serial\s*(?:number|#|no)|model\s*(?:number|#|no)|"
                r"zip\s*code|postal\s*code|version)\b",
                re.IGNORECASE,
            ),
            "EMAIL_ADDRESS": re.compile(
                r"\b(format|template|placeholder|example\.com|test@|noreply@|no-reply@)\b",
                re.IGNORECASE,
            ),
        }

    def _get_fuzzy_neighborhood_match(self, neighborhood: str, target_label: str, threshold: int = 85) -> bool:
        """Checks if any keyword for a label exists in the neighborhood with a fuzzy match.

        Uses length-aware matching:
        - Keywords <= 4 chars: exact substring match only (fuzzy is unreliable on short strings)
        - Keywords 5-6 chars: stricter threshold of 90%
        - Keywords 7+ chars: standard threshold (default 85%)
        """
        keywords = self.context_rules.get(target_label, [])
        neighborhood_lower = neighborhood.lower()
        words = neighborhood_lower.split()

        for kw in keywords:
            kw_lower = kw.lower()
            kw_len = len(kw_lower)

            # Very short keywords: exact substring match only
            if kw_len <= 4:
                if kw_lower in neighborhood_lower:
                    return True
                continue

            # Medium keywords: stricter fuzzy threshold
            effective_threshold = 90 if kw_len <= 6 else threshold

            for word in words:
                if fuzz.ratio(word, kw_lower) >= effective_threshold:
                    return True

        return False

    def _is_negative_context(self, label: str, neighborhood: str) -> bool:
        """Check if neighborhood contains negative context indicators for this label."""
        pattern = self.negative_rules.get(label)
        if pattern and pattern.search(neighborhood):
            return True
        return False

    def detect(self, text: str, detections: List[Detection]) -> List[Detection]:
        updated_detections: List[Detection] = []
        # Track detections that remain as unknown/generic after all context passes
        self._last_unknown_candidates: List[Dict] = []

        for d in detections:
            # Expand neighborhood for "super strong" context detection (150 chars each way)
            neighborhood = self._get_neighborhood(text, d.start, d.end, window=150)

            # Reject detections with strong negative context
            # For regex/field_label: use a tighter window (50 chars) and only for FP-prone labels
            # For gliner/derived: use the full 150-char neighborhood
            if d.source in ("gliner", "derived") and self._is_negative_context(d.label, neighborhood):
                continue
            if d.source in ("regex", "field_label") and d.label in self._STRICT_NEG_LABELS:
                tight_neighborhood = self._get_neighborhood(text, d.start, d.end, window=50)
                if self._is_negative_context(d.label, tight_neighborhood):
                    continue

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
                # Collect unknown/generic detections that couldn't be promoted
                if d.label in self._PROMOTABLE_LABELS:
                    self._last_unknown_candidates.append({
                        "value": d.text,
                        "original_label": d.label,
                        "neighborhood": neighborhood[:300],
                        "start": d.start,
                        "end": d.end,
                        "score": d.score,
                        "source": d.source,
                    })

        return updated_detections

    def get_unknown_candidates(self) -> List[Dict]:
        """Return detections that remained unknown after the last detect() call."""
        return getattr(self, "_last_unknown_candidates", [])

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

    def classify_unknown_by_similarity(self, value: str, neighborhood: str, min_hits: int = 2) -> Optional[str]:
        """Strategy 1: Semantic similarity classification for unknown/novel PII.

        When no hardcoded rule matches an UNKNOWN_PII detection, score every
        label in context_rules by counting how many of its keywords appear
        (exact or fuzzy) in the neighborhood. Require `min_hits` keyword
        matches for confidence.

        Returns the best-matching label, or None if no label meets the threshold.
        """
        if not neighborhood:
            return None

        n_lower = neighborhood.lower()
        n_words = n_lower.split()

        best_label: Optional[str] = None
        best_score: int = 0

        for label, compiled_patterns in self.dynamic_rules.items():
            hits = 0
            for pattern in compiled_patterns:
                if pattern.search(n_lower):
                    hits += 1
            if hits > best_score:
                best_score = hits
                best_label = label

        # Also check raw keyword strings with fuzzy matching for typos/OCR errors
        for label, keywords in self.context_rules.items():
            fuzzy_hits = 0
            for kw in keywords:
                kw_lower = kw.lower()
                # Skip regex-style keywords for fuzzy (they have \b, .?, etc.)
                if any(ch in kw_lower for ch in r"\b.?+*[](){}|^$"):
                    continue
                kw_len = len(kw_lower)
                if kw_len <= 3:
                    if kw_lower in n_lower:
                        fuzzy_hits += 1
                    continue
                # Multi-word keywords: substring match (avoids splitting compound phrases)
                if " " in kw_lower:
                    if kw_lower in n_lower:
                        fuzzy_hits += 1
                    continue
                # Single-word keywords: fuzzy word match (handles typos/OCR)
                for word in n_words:
                    if fuzz.ratio(word, kw_lower) >= 85:
                        fuzzy_hits += 1
                        break
            # Take the max of compiled-pattern hits and fuzzy hits
            total = max(
                sum(1 for p in self.dynamic_rules.get(label, []) if p.search(n_lower)),
                fuzzy_hits,
            )
            if total > best_score:
                best_score = total
                best_label = label

        if best_score >= min_hits:
            return best_label
        return None

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
        # ONLY promote detections with ambiguous/generic labels — never override
        # specific labels like PERSON_FULL_NAME, DATE_OF_BIRTH, INCOME, etc.
        if label in self._PROMOTABLE_LABELS:
            for dyn_label in self.dynamic_rules:
                if self._matches_dynamic_rule(dyn_label, n):
                    return dyn_label

            # 3b. Fuzzy similarity fallback for truly unknown PII
            # When no exact dynamic rule matched, try semantic similarity
            # across ALL context_rules keywords (requires 2+ keyword hits)
            similarity_label = self.classify_unknown_by_similarity(v, neighborhood)
            if similarity_label:
                return similarity_label

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

        # CSC/PIN check MUST come before generic card-number check
        if re.fullmatch(r"\d{3,4}", v):
            if self.csc_re.search(n):
                return "CARD_SECURITY_CODE"
            if self.pin_re.search(n):
                return "PIN"

        # 4-digit number near "card" context but NOT a CSC/PIN → ACCOUNT_LAST4
        if re.fullmatch(r"\d{4}", v) and self.card_re.search(n):
            return "ACCOUNT_LAST4"

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
            # Use word boundaries to prevent "ein" matching inside "Spine", etc.
            if re.search(r"\btax\b", n) or re.search(r"\bitin\b", n) or re.search(r"\bein\b", n):
                return "TAX_ID"

        return None