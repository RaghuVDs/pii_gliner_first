import re
from typing import Dict, List
from app.models import Detection

class RegexDetector:
    def __init__(self, regex_rules: Dict):
        self.regex_rules = regex_rules or {}
        self.compiled = self._compile_rules(self.regex_rules)

    def _compile_rules(self, regex_rules: Dict) -> Dict[str, List[re.Pattern]]:
        compiled: Dict[str, List[re.Pattern]] = {}
        for label, patterns in regex_rules.items():
            compiled[label] = []
            for pattern in patterns:
                try:
                    compiled[label].append(re.compile(pattern, re.IGNORECASE))
                except re.error:
                    pass
        return compiled

    def detect(self, text: str) -> List[Detection]:
        detections: List[Detection] = []
        for label, patterns in self.compiled.items():
            for rx in patterns:
                for m in rx.finditer(text):
                    # Extract Capture Group 1 if it exists, otherwise the full match
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