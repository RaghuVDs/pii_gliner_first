from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from app.models import Detection

if TYPE_CHECKING:
    from app.perf.pii_regions import PIIRegionMask


class PatternFieldDetector:
    def __init__(self, field_patterns: Dict):
        self.field_patterns = field_patterns or {}
        self.compiled = self._compile_patterns(self.field_patterns)

    # Max characters for the captured value per label category
    MAX_VALUE_LENGTH = {
        "PERSON_FULL_NAME": 50,
        "PERSON_FIRST_NAME": 25,
        "PERSON_LAST_NAME": 25,
        "EMAIL_ADDRESS": 80,
        "PHONE_NUMBER": 25,
        "DATE_OF_BIRTH": 20,
        "STREET_ADDRESS": 120,
        "SSN": 15,
        "USERNAME": 40,
    }
    DEFAULT_MAX_VALUE_LENGTH = 100

    def _compile_patterns(self, field_patterns: Dict) -> Dict[str, List[re.Pattern]]:
        compiled: Dict[str, List[re.Pattern]] = {}
        for label, aliases in field_patterns.items():
            compiled[label] = []
            for alias in aliases:
                escaped = re.escape(alias).replace(r"\ ", r"\s+")

                # ALL field patterns require an explicit separator (: = - ; —)
                # to avoid matching conversational text like "my name is John"
                pattern = re.compile(
                    rf"(?P<full>(?P<field>{escaped})\s*[:=\-;\u2014]\s*(?P<value>[^\n;]+))",
                    re.IGNORECASE,
                )
                compiled[label].append(pattern)
        return compiled

    def detect(
        self,
        text: str,
        pii_regions: Optional["PIIRegionMask"] = None,
    ) -> List[Detection]:
        detections: List[Detection] = []

        # When PII regions are provided (token-dropping mode), scan only
        # within those regions instead of the full text. Each region's
        # match offsets are translated back to original-text coordinates.
        if pii_regions is not None and not pii_regions.is_empty():
            text_slices = pii_regions.extract_regions(text)
        else:
            text_slices = [(text, 0, len(text))]

        for label, patterns in self.compiled.items():
            max_len = self.MAX_VALUE_LENGTH.get(label, self.DEFAULT_MAX_VALUE_LENGTH)
            for pattern in patterns:
                for sub_text, region_start, _region_end in text_slices:
                    for m in pattern.finditer(sub_text):
                        value = m.group("value").strip().strip("\"'")
                        if not value:
                            continue

                        # Skip overly long values for the given label type
                        if len(value) > max_len:
                            continue

                        start = region_start + m.start("value")
                        end = region_start + m.end("value")

                        detections.append(
                            Detection(
                                label=label,
                                text=value,
                                start=start,
                                end=end,
                                score=0.99,
                                source="field_label",
                                meta={
                                    "field": m.group("field"),
                                    "full_match": m.group("full"),
                                },
                            )
                        )

        return detections
