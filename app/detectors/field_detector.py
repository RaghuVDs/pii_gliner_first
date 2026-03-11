from __future__ import annotations

import re
from typing import Dict, List, Tuple

from app.models import Detection


class PatternFieldDetector:
    def __init__(self, field_patterns: Dict):
        self.field_patterns = field_patterns or {}
        self.compiled = self._compile_patterns(self.field_patterns)

    def _compile_patterns(self, field_patterns: Dict) -> Dict[str, List[re.Pattern]]:
        compiled: Dict[str, List[re.Pattern]] = {}
        for label, aliases in field_patterns.items():
            compiled[label] = []
            for alias in aliases:
                escaped = re.escape(alias).replace(r"\ ", r"\s+")
                # Supports:
                # Label: value
                # Label - value
                # Label = value
                # Label value
                pattern = re.compile(
                    rf"(?P<full>(?P<field>{escaped})\s*(?:[:=\-]\s*|\s+)(?P<value>[^\n;]+))",
                    re.IGNORECASE,
                )
                compiled[label].append(pattern)
        return compiled

    def detect(self, text: str) -> List[Detection]:
        detections: List[Detection] = []

        for label, patterns in self.compiled.items():
            for pattern in patterns:
                for m in pattern.finditer(text):
                    value = m.group("value").strip().strip("\"'“”‘’")
                    if not value:
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