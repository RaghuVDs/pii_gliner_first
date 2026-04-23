"""PII Region Mask — token dropping for downstream detectors.

After GLiNER + regex produce detections, this module builds a set of
non-overlapping, padded character-offset regions that contain PII.
Downstream detectors (field, entropy, anomaly) scan only these regions
instead of the full 80k-char document, which can cut their cost by
10-20x on sparse/medium-density rows.

The offset translation pattern used here is identical to
GLiNERDetector._chunk_within_windows: each detector receives
(substring, region_start) and adds region_start to all match positions
to get original-text coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models import Detection


@dataclass
class PIIRegionMask:
    """Non-overlapping, sorted (start, end) char-offset regions containing PII.

    Built once per row after GLiNER + regex produce detections. Passed to
    downstream detectors so they scan only within these regions.
    """
    regions: List[Tuple[int, int]] = field(default_factory=list)
    text_length: int = 0

    @staticmethod
    def from_detections(
        detections: List["Detection"],
        text_length: int,
        pad_chars: int = 150,
    ) -> "PIIRegionMask":
        """Build mask from detections.

        Each detection span is expanded by ±pad_chars, then overlapping
        intervals are merged. The pad provides surrounding context for
        downstream pattern matching (field labels precede values by
        ~20-50 chars; context detector extracts 150-char neighborhoods).

        Args:
            detections: Combined output from GLiNER + regex stages.
            text_length: Length of the original text (for clamping).
            pad_chars: How many characters to expand each detection span.

        Returns:
            A PIIRegionMask with sorted, non-overlapping regions.
        """
        if not detections or text_length <= 0:
            return PIIRegionMask(regions=[], text_length=text_length)

        # Build padded intervals
        intervals = sorted(
            (max(0, d.start - pad_chars), min(text_length, d.end + pad_chars))
            for d in detections
        )

        # Merge overlapping intervals (same algorithm as
        # CandidateWindowScanner._merge_windows)
        merged: List[Tuple[int, int]] = []
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))

        return PIIRegionMask(regions=merged, text_length=text_length)

    def coverage(self) -> float:
        """Fraction of text covered by PII regions."""
        if self.text_length <= 0:
            return 0.0
        return sum(e - s for s, e in self.regions) / self.text_length

    def extract_regions(self, text: str) -> List[Tuple[str, int, int]]:
        """Return (substring, original_start, original_end) for each region.

        Downstream detectors iterate this list, run finditer on the
        substring, and translate offsets back to original-text coordinates
        by adding original_start to match positions.
        """
        return [(text[s:e], s, e) for s, e in self.regions]

    def is_empty(self) -> bool:
        """True if no PII regions were found (nothing to scan)."""
        return len(self.regions) == 0
