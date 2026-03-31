"""Universal identifier spotter for free-text PII detection.

Catches identifiers being exchanged in text that no other detector recognizes.
Works by recognizing the STRUCTURAL signature of identifiers vs normal words:
  - Mixed alphanumeric with separators (MWM-0092847-JC, SF-HO-4829174-01)
  - Digit sequences in non-numeric contexts (account 7291048356)
  - Alphanumeric codes with prefixes (VG-82947103-IRA, 401K-F-2847391)

This is the "catch-all safety net" — anything that LOOKS like an identifier
but wasn't caught by GLiNER, regex, field, context, entropy, or intent
detectors gets flagged as UNKNOWN_PII for the adaptive learning loop.
"""
from __future__ import annotations

import re
import logging
from typing import List, Set

from app.models import Detection

logger = logging.getLogger("pii_engine.identifier")

# Structural patterns for identifiers (order matters — most specific first)
IDENTIFIER_PATTERNS = [
    # Alphanumeric with 2+ separators: MWM-0092847-JC, SF-HO-4829174-01
    re.compile(r'\b([A-Z]{1,6}[\-][A-Z0-9]{1,12}[\-][A-Z0-9]{1,12}(?:[\-][A-Z0-9]{1,6})?)\b'),

    # Prefix + digits: AFS-22918743, HLC-9284-7103
    re.compile(r'\b([A-Z]{2,6}[\-][0-9]{4,12}(?:[\-][0-9]{2,6})?)\b'),

    # Digits + letter suffix: 82947103-IRA, 2847391-JC
    re.compile(r'\b([0-9]{4,12}[\-][A-Z]{1,5})\b'),

    # Long digit strings (8+ digits) near identifier context words
    re.compile(r'(?i)(?:account|acct|number|no\.|id|policy|member|reference|ref)\s*(?:#|:|\s)\s*(\d{8,17})\b'),

    # Alphanumeric codes: 401K-F-2847391, GE-8847291
    re.compile(r'\b(\d{2,4}[A-Z][\-][A-Z][\-][0-9]{5,10})\b'),
]

# Words/patterns to NEVER flag as identifiers
BLOCKLIST_PATTERNS = frozenset({
    # Dates, times, durations that look like identifiers
    "AM", "PM", "EST", "PST", "UTC", "GMT",
})

# Context words that indicate the nearby value is an identifier
IDENTIFIER_CONTEXT = re.compile(
    r'(?i)\b(?:account|acct|number|no\.?|id|policy|member|reference|ref|'
    r'license|routing|serial|code|badge|certificate|registration|'
    r'claim|case|ticket|order|invoice|parcel|ein|ssn|vin|'
    r'iban|swift|cusip|npi|mrn|rx|heloc|wallet)\b'
)


class UniversalIdentifierSpotter:
    """Catches identifier-shaped tokens that other detectors missed."""

    def __init__(self, context_window: int = 120):
        """
        Args:
            context_window: Characters to search around each candidate for context words.
        """
        self.context_window = context_window

    def detect(
        self, text: str, existing_detections: List[Detection] = None
    ) -> List[Detection]:
        """Scan for identifier-shaped tokens not covered by existing detections.

        Args:
            text: Full input text.
            existing_detections: Detections from all prior pipeline stages.

        Returns:
            List of Detection objects with label UNKNOWN_PII.
        """
        covered = self._build_covered_set(existing_detections or [])
        detections: List[Detection] = []
        seen_spans: Set[tuple] = set()

        for pattern in IDENTIFIER_PATTERNS:
            for m in pattern.finditer(text):
                value = m.group(1) if pattern.groups else m.group(0)
                start = m.start(1) if pattern.groups else m.start(0)
                end = start + len(value)

                # Skip if already covered by existing detection
                if self._is_covered(start, end, covered):
                    continue

                # Skip duplicate spans
                span_key = (start, end)
                if span_key in seen_spans:
                    continue
                seen_spans.add(span_key)

                # Skip blocklisted patterns
                if value.upper() in BLOCKLIST_PATTERNS:
                    continue

                # Skip very short values
                if len(value) < 5:
                    continue

                # Skip if it's just digits and looks like a year or small number
                stripped = value.replace("-", "").replace(" ", "")
                if stripped.isdigit() and int(stripped) < 100000:
                    continue

                # Check for identifier context within window
                ctx_start = max(0, start - self.context_window)
                ctx_end = min(len(text), end + self.context_window)
                neighborhood = text[ctx_start:ctx_end]

                has_context = bool(IDENTIFIER_CONTEXT.search(neighborhood))

                # Must have identifier context OR strong structural signal
                # (2+ separators with mixed alpha+digit = strong structural signal)
                has_separators = value.count("-") >= 2
                has_mixed = (any(c.isalpha() for c in value) and any(c.isdigit() for c in value))

                if not has_context and not (has_separators and has_mixed):
                    continue

                # Score based on confidence signals
                score = 0.40
                if has_context:
                    score += 0.10
                if has_separators:
                    score += 0.05
                if has_mixed:
                    score += 0.05
                score = min(0.60, score)

                detections.append(Detection(
                    label="UNKNOWN_PII",
                    text=value,
                    start=start,
                    end=end,
                    score=round(score, 4),
                    source="identifier",
                    meta={"has_context": has_context, "structural": has_separators and has_mixed},
                ))

        return self._deduplicate(detections)

    @staticmethod
    def _build_covered_set(detections: List[Detection]) -> Set[int]:
        covered: Set[int] = set()
        for d in detections:
            covered.update(range(d.start, d.end))
        return covered

    @staticmethod
    def _is_covered(start: int, end: int, covered: Set[int]) -> bool:
        span_len = end - start
        if span_len <= 0:
            return True
        overlap = len(set(range(start, end)) & covered)
        return overlap > span_len * 0.5

    @staticmethod
    def _deduplicate(detections: List[Detection]) -> List[Detection]:
        if not detections:
            return detections
        detections.sort(key=lambda d: (d.start, -d.score))
        kept: List[Detection] = []
        for d in detections:
            overlaps = False
            for k in kept:
                if d.start < k.end and d.end > k.start:
                    if d.score > k.score:
                        kept.remove(k)
                        kept.append(d)
                    overlaps = True
                    break
            if not overlaps:
                kept.append(d)
        return kept
