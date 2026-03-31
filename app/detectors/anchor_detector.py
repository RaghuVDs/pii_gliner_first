"""Anchor-and-Expand PII detector.

Detects reliable anchor signals (zip codes, @ signs, country codes, state
abbreviations) and expands outward to capture the full PII entity.

This is fundamentally different from regex — we don't hardcode address formats.
Instead we detect ANCHORS and let the surrounding context define the boundary.

Example:
    "Rd. #29 Milwaukee Oregon 97267"
    Anchor: "97267" (5-digit zip code)
    Expand left: grab text until sentence boundary → full address
"""
from __future__ import annotations

import re
import logging
from typing import List, Set, Tuple, Optional

from app.models import Detection

logger = logging.getLogger("pii_engine.anchor")


class AnchorExpandDetector:
    """Detects PII by finding anchor signals and expanding to capture full entities."""

    # US state names and abbreviations for address anchoring
    _US_STATES = {
        "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
        "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
        "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
        "maine", "maryland", "massachusetts", "michigan", "minnesota",
        "mississippi", "missouri", "montana", "nebraska", "nevada",
        "new hampshire", "new jersey", "new mexico", "new york",
        "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
        "pennsylvania", "rhode island", "south carolina", "south dakota",
        "tennessee", "texas", "utah", "vermont", "virginia", "washington",
        "west virginia", "wisconsin", "wyoming",
        # Abbreviations
        "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
        "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
        "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
        "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
        "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
        "dc",
    }

    # Zip code pattern (5 digits, optionally +4)
    _ZIP_RE = re.compile(r'(?<!\d)(\d{5})(?:-\d{4})?(?!\d)')

    # Sentence/clause boundaries for expansion limits
    _BOUNDARY_RE = re.compile(r'[;.!?\n]|(?:Customer|Agent|Rep)\s*[-:]')

    def detect(
        self, text: str, existing_detections: List[Detection] = None
    ) -> List[Detection]:
        """Detect PII by finding anchors and expanding outward."""
        covered = self._build_covered_set(existing_detections or [])
        detections: List[Detection] = []

        # Strategy 1: Zip code → expand to full address
        for m in self._ZIP_RE.finditer(text):
            zip_code = m.group(1)
            zip_start = m.start(1)
            zip_end = m.end(0)

            # Skip if already covered
            if self._is_covered(zip_start, zip_end, covered):
                continue

            # Check if a US state name/abbreviation is nearby (within 30 chars before)
            pre_text = text[max(0, zip_start - 30):zip_start].lower()
            has_state = any(state in pre_text for state in self._US_STATES)

            if not has_state:
                continue  # Zip without state context — might be a generic number

            # Expand LEFT to find the start of the address
            # Look for the nearest clause boundary or start of line
            expand_start = max(0, zip_start - 80)
            pre_region = text[expand_start:zip_start]

            # Find the last boundary in the pre-region
            boundary_match = None
            for bm in self._BOUNDARY_RE.finditer(pre_region):
                boundary_match = bm

            if boundary_match:
                addr_start = expand_start + boundary_match.end()
            else:
                addr_start = expand_start

            # Trim leading whitespace and punctuation
            while addr_start < zip_start and text[addr_start] in " \t\n-;,":
                addr_start += 1

            address_text = text[addr_start:zip_end].strip()

            if len(address_text) < 8:
                continue  # Too short to be a meaningful address

            # Skip if mostly covered by existing detections
            if self._is_covered(addr_start, zip_end, covered):
                continue

            detections.append(Detection(
                label="STREET_ADDRESS",
                text=address_text,
                start=addr_start,
                end=zip_end,
                score=0.75,
                source="anchor",
                meta={"anchor": "zip_code", "zip": zip_code},
            ))

        return detections

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
