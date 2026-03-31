"""Conversational intent detector for unknown PII in free text.

Detects PII-like values being exchanged in conversation by recognizing
the INTENT patterns that surround PII disclosure:
  - "my X is Y" / "X is 12345"
  - "can you verify your X?" → next value is likely PII
  - "here's my X: Y"
  - "it's 12345" (in a verification context)

This catches PII that no regex, GLiNER label, or keyword rule covers —
truly novel PII types being exchanged in conversation for the first time.
"""
from __future__ import annotations

import re
import logging
from typing import List, Set, Tuple

from app.models import Detection

logger = logging.getLogger("pii_engine.intent")

# Intent patterns that indicate a PII value follows.
# Each pattern captures (optional_label, value).
INTENT_PATTERNS = [
    # "my/the/your X is Y" — possessive disclosure
    re.compile(
        r'(?:my|the|your|his|her|our|their)\s+'
        r'([a-zA-Z][a-zA-Z\s]{1,30}?)\s+'
        r'(?:is|was|are|reads?|shows?)\s+'
        r'([A-Za-z0-9][\w\-\.#/]{3,50})',
        re.IGNORECASE,
    ),
    # "X number/code/id is Y" — labeled disclosure
    re.compile(
        r'([a-zA-Z][a-zA-Z\s]{1,25}?'
        r'(?:number|code|id|ID|ref|reference|no\.?))\s+'
        r'(?:is|was|reads?|shows?)\s+'
        r'([A-Za-z0-9][\w\-\.#/]{3,50})',
        re.IGNORECASE,
    ),
    # "it's Y" / "that's Y" — pronoun reference (value after possessive)
    re.compile(
        r"(?:it'?s|that'?s|here'?s)\s+"
        r'([A-Za-z0-9][\w\-\.#/]{5,50})',
        re.IGNORECASE,
    ),
    # "verify/confirm your X" ... then a value on same or next line
    re.compile(
        r'(?:verify|confirm|provide|give)\s+'
        r'(?:your|the|my)\s+'
        r'([a-zA-Z][a-zA-Z\s]{1,30}?)\s*'
        r'(?:\?|—|--|-)\s*'
        r'([A-Za-z0-9][\w\-\.#/]{3,50})',
        re.IGNORECASE,
    ),
]

# Values that are clearly NOT PII (common words that follow "my X is")
VALUE_BLOCKLIST = frozenset({
    "fine", "good", "great", "okay", "ok", "correct", "right", "yes", "no",
    "true", "false", "done", "ready", "available", "not", "still", "also",
    "that", "this", "here", "there", "what", "which", "where", "when",
    "very", "really", "just", "about", "already", "currently", "actually",
})

# Label hints that are clearly NOT PII fields
LABEL_BLOCKLIST = frozenset({
    "name is", "question is", "answer is", "issue is", "problem is",
    "concern is", "reason is", "purpose is", "goal is", "plan is",
    "status is", "situation is", "schedule is", "preference is",
    "appointment is", "favorite is", "opinion is",
})


class ConversationalIntentDetector:
    """Detects PII values disclosed via conversational intent patterns."""

    def __init__(self):
        self.patterns = INTENT_PATTERNS

    def detect(
        self, text: str, existing_detections: List[Detection] = None
    ) -> List[Detection]:
        """Scan for PII values in conversational disclosure patterns.

        Args:
            text: Full input text.
            existing_detections: Detections from prior pipeline stages.

        Returns:
            List of Detection objects with label UNKNOWN_PII.
        """
        covered = self._build_covered_set(existing_detections or [])
        detections: List[Detection] = []

        for pattern in self.patterns:
            for m in pattern.finditer(text):
                groups = m.groups()

                if len(groups) == 1:
                    # Pattern with only value (e.g., "it's Y")
                    label_hint = ""
                    value = groups[0].strip()
                    value_start = m.start(1)
                elif len(groups) == 2:
                    label_hint = groups[0].strip().lower()
                    value = groups[1].strip()
                    value_start = m.start(2)
                else:
                    continue

                # Strip trailing punctuation from value
                while value and value[-1] in ".,:;!?)]}":
                    value = value[:-1]
                value_end = value_start + len(value)

                if not value or len(value) < 3:
                    continue

                # Skip if value is already covered by existing detection
                if self._is_covered(value_start, value_end, covered):
                    continue

                # Skip blocklisted values (common words)
                if value.lower() in VALUE_BLOCKLIST:
                    continue

                # Skip blocklisted label hints
                if any(bl in label_hint for bl in LABEL_BLOCKLIST):
                    continue

                # Value must contain at least one digit (identifiers, not words)
                if not any(c.isdigit() for c in value):
                    continue

                # Value must be compact — single token or short code, not a sentence
                if len(value.split()) > 3:
                    continue

                # Skip very short values (likely noise)
                if len(value) < 4:
                    continue

                meta = {}
                if label_hint:
                    meta["intent_label"] = label_hint

                detections.append(Detection(
                    label="UNKNOWN_PII",
                    text=value,
                    start=value_start,
                    end=value_end,
                    score=0.50,
                    source="intent",
                    meta=meta,
                ))

        return self._deduplicate(detections)

    @staticmethod
    def _build_covered_set(detections: List[Detection]) -> Set[int]:
        """Build set of character positions covered by existing detections."""
        covered: Set[int] = set()
        for d in detections:
            covered.update(range(d.start, d.end))
        return covered

    @staticmethod
    def _is_covered(start: int, end: int, covered: Set[int]) -> bool:
        """Check if >50% of span is already covered."""
        span_len = end - start
        if span_len <= 0:
            return True
        overlap = len(set(range(start, end)) & covered)
        return overlap > span_len * 0.5

    @staticmethod
    def _deduplicate(detections: List[Detection]) -> List[Detection]:
        """Remove overlapping detections, keep highest score."""
        if not detections:
            return detections
        detections.sort(key=lambda d: (d.start, -d.score))
        kept: List[Detection] = []
        for d in detections:
            overlaps = False
            for k in kept:
                if d.start < k.end and d.end > k.start:
                    overlaps = True
                    break
            if not overlaps:
                kept.append(d)
        return kept
