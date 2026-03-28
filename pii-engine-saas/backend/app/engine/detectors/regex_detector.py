"""Regex-based PII detector - identical to original with updated imports."""

import re
import logging
from typing import Dict, List
from app.engine.models import Detection

logger = logging.getLogger(__name__)


class RegexDetector:
    _CASE_SENSITIVE_LABELS = {"PERSON_FULL_NAME"}

    def __init__(self, regex_rules: Dict):
        self.regex_rules = regex_rules or {}
        self.compiled = self._compile_rules(self.regex_rules)

    def _compile_rules(self, regex_rules: Dict) -> Dict[str, List[re.Pattern]]:
        compiled: Dict[str, List[re.Pattern]] = {}
        for label, patterns in regex_rules.items():
            compiled[label] = []
            for pattern in patterns:
                try:
                    if label in self._CASE_SENSITIVE_LABELS and "(?i)" not in pattern:
                        compiled[label].append(re.compile(pattern))
                    else:
                        compiled[label].append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning("Invalid regex pattern for %s: %s — %s", label, pattern, e)
        return compiled

    def detect(self, text: str) -> List[Detection]:
        detections: List[Detection] = []
        for label, patterns in self.compiled.items():
            for rx in patterns:
                for m in rx.finditer(text):
                    text_val = m.group(1) if m.lastindex else m.group(0)
                    start_idx = m.start(1) if m.lastindex else m.start()
                    end_idx = m.end(1) if m.lastindex else m.end()

                    detections.append(
                        Detection(
                            label=label,
                            start=start_idx,
                            end=end_idx,
                            text=text_val,
                            score=0.98,
                            source="regex",
                        )
                    )
        return detections
