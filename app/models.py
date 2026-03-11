from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class Detection:
    label: str
    text: str
    start: int
    end: int
    score: float
    source: str
    meta: Dict[str, Any] = field(default_factory=dict)
    replacement_tag: str = "" # Holds the dynamic <LABEL_1> tag

@dataclass
class RedactionResult:
    original_text: str
    detections: List[Detection]
    redacted_text: str
    unknown_candidates: List[Any] = field(default_factory=list)