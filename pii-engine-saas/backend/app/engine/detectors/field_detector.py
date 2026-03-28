"""Field label detector - identical to original with updated imports."""

from __future__ import annotations

import re
from typing import Dict, List

from app.engine.models import Detection


class PatternFieldDetector:
    def __init__(self, field_patterns: Dict):
        self.field_patterns = field_patterns or {}
        self.compiled = self._compile_patterns(self.field_patterns)

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
                pattern = re.compile(
                    rf"(?P<full>(?P<field>{escaped})\s*[:=\-;\u2014]\s*(?P<value>[^\n;]+))",
                    re.IGNORECASE,
                )
                compiled[label].append(pattern)
        return compiled

    def detect(self, text: str) -> List[Detection]:
        detections: List[Detection] = []

        for label, patterns in self.compiled.items():
            max_len = self.MAX_VALUE_LENGTH.get(label, self.DEFAULT_MAX_VALUE_LENGTH)
            for pattern in patterns:
                for m in pattern.finditer(text):
                    value = m.group("value").strip().strip("\"'")
                    if not value:
                        continue
                    if len(value) > max_len:
                        continue

                    start = m.start("value")
                    end = m.end("value")

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
