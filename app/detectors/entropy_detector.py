"""Entropy-based anomaly detector for unknown secrets, tokens, and hashes.

Scans text for high-entropy spans that look like sensitive data (API keys,
tokens, cryptographic hashes, passwords) that no other detector would catch.

Uses Shannon entropy over character distribution in a sliding window.
High entropy (>= threshold) + high alphanumeric ratio = likely secret.
"""
from __future__ import annotations

import math
import re
import logging
from collections import Counter
from typing import List, Optional, Set, Tuple, TYPE_CHECKING

from app.models import Detection

if TYPE_CHECKING:
    from app.perf.pii_regions import PIIRegionMask

logger = logging.getLogger("pii_engine.entropy")

# Defaults — can be overridden via anomaly_config.yaml
DEFAULT_ENTROPY_THRESHOLD = 4.2   # bits per character (natural language ≈ 3.5-4.0, pure random ≈ 5.0)
DEFAULT_MIN_SPAN_LEN = 16
DEFAULT_MAX_SPAN_LEN = 128
DEFAULT_ALNUM_RATIO = 0.75        # at least 75% alphanumeric
DEFAULT_STEP = 4


class EntropyAnomalyDetector:
    """Detects high-entropy spans that are likely secrets or tokens."""

    def __init__(
        self,
        entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
        min_span_len: int = DEFAULT_MIN_SPAN_LEN,
        max_span_len: int = DEFAULT_MAX_SPAN_LEN,
        alnum_ratio: float = DEFAULT_ALNUM_RATIO,
        step: int = DEFAULT_STEP,
    ):
        self.entropy_threshold = entropy_threshold
        self.min_span_len = min_span_len
        self.max_span_len = max_span_len
        self.alnum_ratio = alnum_ratio
        self.step = step

        # Token-like pattern: continuous non-whitespace runs of min length
        self._token_re = re.compile(r'\S{' + str(min_span_len) + r',}')

    def detect(
        self,
        text: str,
        existing_detections: List[Detection] = None,
        pii_regions: Optional["PIIRegionMask"] = None,
    ) -> List[Detection]:
        """Scan text for high-entropy spans not already covered by other detectors.

        Args:
            text: Full input text.
            existing_detections: Detections from prior pipeline stages (used to
                skip spans that are already detected).
            pii_regions: When provided (token-dropping mode), scan only within
                these regions instead of the full text.

        Returns:
            List of Detection objects with label UNKNOWN_SECRET.
        """
        covered = self._build_covered_set(existing_detections or [])
        candidates: List[Detection] = []

        # When PII regions are provided, scan only within those regions.
        if pii_regions is not None and not pii_regions.is_empty():
            text_slices = pii_regions.extract_regions(text)
        else:
            text_slices = [(text, 0, len(text))]

        for sub_text, region_start, _region_end in text_slices:
            for match in self._token_re.finditer(sub_text):
                token = match.group()
                token_start = region_start + match.start()

                if len(token) > self.max_span_len:
                    for offset in range(0, len(token) - self.min_span_len + 1, self.step):
                        for win_len in (64, 32, self.min_span_len):
                            if offset + win_len > len(token):
                                continue
                            span = token[offset:offset + win_len]
                            abs_start = token_start + offset
                            abs_end = abs_start + win_len
                            det = self._evaluate_span(span, abs_start, abs_end, covered)
                            if det is not None:
                                candidates.append(det)
                else:
                    det = self._evaluate_span(token, token_start, token_start + len(token), covered)
                    if det is not None:
                        candidates.append(det)

        # Deduplicate overlapping entropy detections — keep highest score
        return self._deduplicate(candidates)

    def _evaluate_span(
        self, span: str, start: int, end: int, covered: Set[int]
    ) -> Detection | None:
        """Check if a span is high-entropy and not already covered."""
        # Skip if mostly covered by existing detections
        span_positions = set(range(start, end))
        if len(span_positions & covered) > len(span_positions) * 0.5:
            return None

        # Check alphanumeric ratio
        alnum_count = sum(1 for c in span if c.isalnum())
        if alnum_count / len(span) < self.alnum_ratio:
            return None

        # Reject if it looks like a natural language word (all alpha, no mixed case/digits)
        if span.isalpha():
            return None

        # Reject if it looks like a field label (ends with colon, CamelCase word)
        if span.endswith(":") or span.endswith("="):
            return None

        entropy = self._shannon_entropy(span)
        if entropy < self.entropy_threshold:
            return None

        # Confidence: normalize entropy to [0.45, 0.85]
        max_entropy = math.log2(len(set(span))) if len(set(span)) > 1 else 5.0
        raw_conf = (entropy - self.entropy_threshold) / max(max_entropy - self.entropy_threshold, 0.5)
        confidence = min(0.85, max(0.45, 0.45 + raw_conf * 0.40))

        return Detection(
            label="UNKNOWN_SECRET",
            text=span,
            start=start,
            end=end,
            score=round(confidence, 4),
            source="entropy_anomaly",
            meta={"entropy": round(entropy, 3)},
        )

    @staticmethod
    def _shannon_entropy(text: str) -> float:
        """Compute Shannon entropy (bits per character) of a string."""
        if not text:
            return 0.0
        freq = Counter(text)
        length = len(text)
        return -sum(
            (count / length) * math.log2(count / length)
            for count in freq.values()
        )

    @staticmethod
    def _build_covered_set(detections: List[Detection]) -> Set[int]:
        """Build a set of character positions already covered by detections."""
        covered: Set[int] = set()
        for d in detections:
            covered.update(range(d.start, d.end))
        return covered

    @staticmethod
    def _deduplicate(candidates: List[Detection]) -> List[Detection]:
        """Remove overlapping entropy detections, keeping highest score."""
        if not candidates:
            return candidates
        # Sort by start, then by score descending
        candidates.sort(key=lambda d: (d.start, -d.score))
        kept: List[Detection] = []
        for c in candidates:
            # Check if overlapping with any kept detection
            overlaps = False
            for k in kept:
                if c.start < k.end and c.end > k.start:
                    # Overlap — keep the one with higher score
                    if c.score > k.score:
                        kept.remove(k)
                        kept.append(c)
                    overlaps = True
                    break
            if not overlaps:
                kept.append(c)
        return kept
