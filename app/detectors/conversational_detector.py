"""Conversational PII Detector — discovers PII from dialogue context.

In call-center transcripts, PII is often revealed through Q&A patterns:

    Agent: What is your mother's maiden name?
    Customer: Fitzpatrick.

    Agent: Can you verify the PIN?
    Customer: 7392.

    Agent: The third-party vendor, Robert Williams, was contacted.

None of these trigger regex (no format pattern), GLiNER (too subtle), or
field detection (no `Label: Value` separator). This detector finds them by
scanning for conversational trigger phrases and extracting the likely PII
value from the surrounding text.

Each trigger maps a question/phrase regex to:
  - An expected PII label
  - A value extraction pattern (what to capture after the trigger)
  - A search window (how far ahead to look for the answer)

The detector is config-driven: triggers are loaded from
`app/config/conversational_triggers.yaml` so users can add domain-specific
patterns without touching code.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import yaml

from app.models import Detection

logger = logging.getLogger("pii_engine.conversational")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config", "conversational_triggers.yaml")


class ConversationalPIIDetector:
    """Discovers PII from conversational Q&A patterns in transcripts."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self._triggers: List[_CompiledTrigger] = []
        self._load_config(config_path)

    def _load_config(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning(f"[conversational] config not found at {path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            triggers = data.get("triggers", [])
            for entry in triggers:
                if not isinstance(entry, dict):
                    continue
                phrase = entry.get("phrase")
                label = entry.get("label")
                extract = entry.get("extract", "next_word")
                window = int(entry.get("window", 100))
                score = float(entry.get("score", 0.75))
                if not phrase or not label:
                    continue
                try:
                    compiled = re.compile(phrase, re.IGNORECASE)
                    self._triggers.append(_CompiledTrigger(
                        pattern=compiled,
                        label=label,
                        extract=extract,
                        window=window,
                        score=score,
                    ))
                except re.error as e:
                    logger.debug(f"[conversational] bad regex '{phrase}': {e}")

            logger.info(f"[conversational] loaded {len(self._triggers)} trigger patterns")
        except Exception as e:
            logger.warning(f"[conversational] failed to load config: {e}")

    def detect(
        self,
        text: str,
        existing_detections: Optional[List[Detection]] = None,
    ) -> List[Detection]:
        """Scan text for conversational PII patterns.

        Args:
            text: Full input text.
            existing_detections: Detections from prior stages, used to skip
                spans that are already detected (avoid duplicates).

        Returns:
            New Detection objects discovered from conversational context.
        """
        if not self._triggers:
            return []

        covered = _build_covered_set(existing_detections or [])
        detections: List[Detection] = []

        for trigger in self._triggers:
            for match in trigger.pattern.finditer(text):
                trigger_end = match.end()
                # Look ahead in the window after the trigger phrase
                window_end = min(len(text), trigger_end + trigger.window)
                after_text = text[trigger_end:window_end]

                extracted = _extract_value(after_text, trigger.extract)
                if not extracted:
                    continue

                value, local_start, local_end = extracted
                abs_start = trigger_end + local_start
                abs_end = trigger_end + local_end

                # Skip if this span is already covered by another detection
                if _is_covered(abs_start, abs_end, covered):
                    continue

                # Skip very short values (likely noise)
                if len(value.strip()) < 2:
                    continue

                detections.append(Detection(
                    label=trigger.label,
                    text=value,
                    start=abs_start,
                    end=abs_end,
                    score=trigger.score,
                    source="conversational_context",
                    meta={
                        "trigger_phrase": match.group(0)[:60],
                        "extract_mode": trigger.extract,
                    },
                ))
                # Add to covered set to avoid duplicate extractions
                for i in range(abs_start, abs_end):
                    covered.add(i)

        return detections


class _CompiledTrigger:
    """A compiled trigger pattern with its extraction config."""
    __slots__ = ("pattern", "label", "extract", "window", "score")

    def __init__(
        self,
        pattern: re.Pattern,
        label: str,
        extract: str,
        window: int,
        score: float,
    ):
        self.pattern = pattern
        self.label = label
        self.extract = extract
        self.window = window
        self.score = score


# ── Extraction helpers ──────────────────────────────────────────────────

# Common dialogue markers that should be stripped when extracting values
_DIALOGUE_MARKER_RE = re.compile(
    r'^\s*(?:\[[\d:]+\]\s*)?(?:(?:Customer|Agent|Caller|Rep|Representative)\s*:\s*)?',
    re.IGNORECASE,
)

# Extraction patterns for different value types
_EXTRACTORS = {
    # Capitalized word(s) — for names, places
    "name": re.compile(
        r'(?:^|[?\s.!,;:]*\n)' + _DIALOGUE_MARKER_RE.pattern +
        r'(?:(?:it\'?s?|that\'?s?|is)\s+)?'
        r'([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,3})'
        r'(?=[,.\s\n]|$)',
        re.MULTILINE | re.IGNORECASE,
    ),
    # Next word (any non-whitespace token after optional dialogue markers)
    "next_word": re.compile(
        r'(?:^|[?\s.!,;:]*\n)' + _DIALOGUE_MARKER_RE.pattern +
        r'(?:(?:it\'?s?|that\'?s?|is)\s+)?'
        r'(\S+)',
        re.MULTILINE | re.IGNORECASE,
    ),
    # Number (4-10 digits, possibly with separators)
    "number": re.compile(
        r'(?:^|[?\s.!,;:]*\n)' + _DIALOGUE_MARKER_RE.pattern +
        r'(?:(?:it\'?s?|that\'?s?|is)\s+)?'
        r'(\d[\d\-\s]{2,15}\d)',
        re.MULTILINE | re.IGNORECASE,
    ),
    # Short number (4-6 digits — for PINs, short codes)
    # Allow punctuation/whitespace before the newline (trigger line may end with "?")
    "short_number": re.compile(
        r'(?:^|[?\s.!,;]\s*\n)' + _DIALOGUE_MARKER_RE.pattern +
        r'(?:(?:it\'?s?|that\'?s?|is)\s+)?'
        r'(\d{4,6})\b',
        re.MULTILINE | re.IGNORECASE,
    ),
    # Alphanumeric identifier (mixed letters+digits, possibly with dashes)
    "identifier": re.compile(
        r'(?:^|[?\s.!,;:]*\n)' + _DIALOGUE_MARKER_RE.pattern +
        r'(?:(?:it\'?s?|that\'?s?|is)\s+)?'
        r'([A-Za-z0-9][\w\-\.]{3,30})',
        re.MULTILINE | re.IGNORECASE,
    ),
    # Full line value — captures everything until end of line
    "line": re.compile(
        r'(?:^|[?\s.!,;:]*\n)' + _DIALOGUE_MARKER_RE.pattern +
        r'(?:(?:it\'?s?|that\'?s?|is)\s+)?'
        r'(.+?)(?:\n|$)',
        re.MULTILINE | re.IGNORECASE,
    ),
    # Inline value — captures the next non-trivial token on the SAME line
    "inline_name": re.compile(
        r',?\s+([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,3})'
        r'(?=[,.\s]|$)',
    ),
}


def _extract_value(
    after_text: str,
    mode: str,
) -> Optional[Tuple[str, int, int]]:
    """Extract a PII value from text following a trigger phrase.

    Args:
        after_text: The text window after the trigger phrase.
        mode: Extraction mode (name, number, identifier, etc.)

    Returns:
        (value, local_start, local_end) or None if nothing found.
        Positions are relative to after_text (the caller adds chunk_start).
    """
    extractor = _EXTRACTORS.get(mode)
    if extractor is None:
        return None

    # The after_text often starts with punctuation and a line break before
    # the answer (e.g., "?\nCustomer: Fitzpatrick."). We try the extractor
    # directly first, and if it fails we strip leading punctuation/whitespace
    # to give it a second chance on the "clean" content, tracking the offset
    # so positions remain valid in the original after_text.
    m = extractor.search(after_text)
    offset = 0
    if not m:
        # Strip leading punctuation, whitespace, and dialogue markers
        stripped = _LEADING_JUNK_RE.sub("", after_text)
        offset = len(after_text) - len(stripped)
        if stripped:
            m = extractor.search(stripped)

    if not m:
        return None

    value = m.group(1).strip().rstrip(".,;:!?")
    if not value:
        return None

    start = offset + m.start(1)
    end = start + len(value)
    return value, start, end


# Strip leading punctuation, newlines, and dialogue markers from after_text
_LEADING_JUNK_RE = re.compile(
    r'^[?\s.!,;:\-]*'
    r'(?:\[[\d:]+\]\s*)?'
    r'(?:(?:Customer|Agent|Caller|Rep|Representative)\s*:\s*)?',
    re.IGNORECASE | re.MULTILINE,
)


def _build_covered_set(detections: List[Detection]) -> set:
    """Build a set of covered character positions from existing detections."""
    covered = set()
    for d in detections:
        for i in range(d.start, d.end):
            covered.add(i)
    return covered


def _is_covered(start: int, end: int, covered: set) -> bool:
    """Check if >50% of a span is already covered."""
    span_len = end - start
    if span_len <= 0:
        return True
    overlap = sum(1 for i in range(start, end) if i in covered)
    return overlap / span_len > 0.5
