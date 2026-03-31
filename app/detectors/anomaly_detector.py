"""Contextual anomaly detector for unknown PII in field-labeled positions.

Catches values that appear in PII-like syntactic contexts but were not
detected by any other detector:
  1. Unknown field labels: "SomeNewLabel: value" where SomeNewLabel is NOT
     in field_patterns.yaml and looks like a data field (not conversation)
  2. Uncaught context values: compact alphanumeric tokens (8+ chars) near
     known context keywords but not detected by any prior stage
"""
from __future__ import annotations

import re
import logging
from typing import Dict, List, Set

from app.models import Detection

logger = logging.getLogger("pii_engine.anomaly")

# Common non-PII labels found in transcripts, forms, and documents.
# These look like "Label: value" but are structural, not PII fields.
NON_PII_FIELD_LABELS = frozenset({
    "agent", "customer", "caller", "representative", "rep", "advisor",
    "note", "notes", "comment", "comments", "description", "summary",
    "date", "time", "duration", "channel", "call date", "call time",
    "status", "type", "category", "priority", "subject", "topic",
    "action", "result", "outcome", "resolution", "response",
    "question", "answer", "issue", "reason", "details", "info",
    "section", "step", "item", "page", "line", "row", "column",
    "from", "to", "cc", "bcc", "re", "fwd", "sent", "received",
    "total", "amount", "balance", "count", "number", "quantity",
    "yes", "no", "na", "n/a", "none", "other", "all", "any",
    "example", "e.g", "i.e", "note that", "please", "thank",
    "hi", "hello", "dear", "regards", "sincerely",
    "mr", "mrs", "ms", "dr", "prof",
    "icd", "icd-10", "cpt", "code",
})


class ContextualAnomalyDetector:
    """Detects PII-like values in structural positions missed by other detectors."""

    def __init__(
        self,
        known_field_labels: Set[str],
        context_keywords: Set[str],
        min_value_len: int = 6,
        max_value_len: int = 60,
    ):
        self.known_field_labels = known_field_labels
        self.context_keywords = context_keywords
        self.min_value_len = min_value_len
        self.max_value_len = max_value_len

        # Pattern: "FieldLabel" followed by separator and a COMPACT value
        # The value must look like an identifier/code, not a sentence.
        # Requires at least one digit or hyphen in the value to avoid matching prose.
        self._field_re = re.compile(
            r'\b([A-Za-z][A-Za-z0-9_ ]{2,25}?)\s*[:=]\s*'
            r'([A-Za-z0-9][\w\-\.#/]{' + str(min_value_len - 1) + r',' + str(max_value_len) + r'})',
        )

        # Pattern: standalone compact token (8+ chars, must contain digit + letter)
        self._value_re = re.compile(
            r'\b([A-Za-z0-9][\w\-\.]{7,' + str(max_value_len) + r'})\b'
        )

    def detect(
        self, text: str, existing_detections: List[Detection] = None
    ) -> List[Detection]:
        """Scan for PII-like values in structural positions missed by other detectors."""
        covered = self._build_covered_ranges(existing_detections or [])
        detections: List[Detection] = []

        # Strategy 1: Unknown field labels with compact values
        for m in self._field_re.finditer(text):
            label_text = m.group(1).strip().lower()
            value_text = m.group(2).strip()
            value_start = m.start(2)
            value_end = m.start(2) + len(value_text)

            # Skip known field labels
            if label_text in self.known_field_labels:
                continue

            # Skip common non-PII labels (conversation, structural)
            if label_text in NON_PII_FIELD_LABELS:
                continue

            # Skip single-word labels that are common English words
            if len(label_text.split()) == 1 and len(label_text) <= 4:
                continue

            # Skip if value is already covered by existing detection
            if self._is_covered(value_start, value_end, covered):
                continue

            # Value must look like an identifier: contain at least one digit
            if not any(c.isdigit() for c in value_text):
                continue

            # Value must be compact (not a sentence) — max 4 words
            if len(value_text.split()) > 4:
                continue

            detections.append(Detection(
                label="UNKNOWN_PII",
                text=value_text,
                start=value_start,
                end=value_end,
                score=0.55,
                source="contextual_anomaly",
                meta={"discovered_field": label_text},
            ))

        # Strategy 2: Uncaught compact tokens near context keywords
        for m in self._value_re.finditer(text):
            value = m.group(1)
            start = m.start(1)
            end = m.end(1)

            if self._is_covered(start, end, covered):
                continue

            # Must contain both letters and digits (identifiers, not words or numbers)
            has_letter = any(c.isalpha() for c in value)
            has_digit = any(c.isdigit() for c in value)
            if not (has_letter and has_digit):
                continue

            # Must be compact — single token, no spaces
            if " " in value:
                continue

            if len(value) < 8:
                continue

            # Check if near a context keyword (within 100 chars)
            neighborhood = text[max(0, start - 100):min(len(text), end + 100)].lower()
            matching_keywords = [kw for kw in self.context_keywords if kw in neighborhood]

            # Require at least 2 keyword matches for higher confidence
            if len(matching_keywords) < 2:
                continue

            detections.append(Detection(
                label="UNKNOWN_PII",
                text=value,
                start=start,
                end=end,
                score=0.45,
                source="contextual_anomaly",
                meta={"nearby_keywords": matching_keywords[:5]},
            ))

        return self._deduplicate(detections)

    @staticmethod
    def _build_covered_ranges(detections: List[Detection]) -> List[tuple]:
        """Build sorted list of (start, end) ranges already covered."""
        return sorted((d.start, d.end) for d in detections)

    @staticmethod
    def _is_covered(start: int, end: int, ranges: List[tuple]) -> bool:
        """Check if a span overlaps >50% with any existing detection."""
        span_len = end - start
        if span_len <= 0:
            return True
        for rs, re_ in ranges:
            if rs >= end:
                break
            if re_ <= start:
                continue
            overlap = min(end, re_) - max(start, rs)
            if overlap > span_len * 0.5:
                return True
        return False

    @staticmethod
    def _deduplicate(detections: List[Detection]) -> List[Detection]:
        """Remove overlapping detections, keeping highest score."""
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
