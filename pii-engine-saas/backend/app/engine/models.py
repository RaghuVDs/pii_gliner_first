"""Detection and RedactionResult dataclasses - identical to the original engine models."""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class Detection:
    """A single PII detection result."""
    label: str
    text: str
    start: int
    end: int
    score: float
    source: str  # "gliner", "regex", "field_label", "context", "pattern_lstm", "derived", "propagated"
    meta: Dict[str, Any] = field(default_factory=dict)
    replacement_tag: str = ""  # Holds the dynamic <LABEL_1> tag

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "label": self.label,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "source": self.source,
            "meta": self.meta,
            "replacement_tag": self.replacement_tag,
        }


@dataclass
class RedactionResult:
    """Result of a full detect + redact pipeline run."""
    original_text: str
    detections: List[Detection]
    redacted_text: str
    unknown_candidates: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "original_text": self.original_text,
            "detections": [d.to_dict() for d in self.detections],
            "redacted_text": self.redacted_text,
            "unknown_candidates": self.unknown_candidates,
        }
