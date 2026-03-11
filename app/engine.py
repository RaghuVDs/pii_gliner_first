from __future__ import annotations
import os
from typing import List
import yaml

from app.models import RedactionResult, Detection
from app.detectors.regex_detector import RegexDetector
from app.detectors.gliner_detector import GLiNERDetector
from app.resolver import resolve_detections
from app.postprocessing import add_instance_numbers, remove_false_positives, split_person_names
from app.policy_engine import MaskingPolicyEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_yaml(filename: str):
    path = os.path.join(BASE_DIR, "config", filename)
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class HybridPIIEngine:
    def __init__(
        self,
        use_gliner: bool = True,
        gliner_model_name: str = "knowledgator/gliner-pii-large-v1.0",
        gliner_threshold: float = 0.35,
    ):
        self.regex_rules = load_yaml("regex_rules.yaml")
        self.masking_rules = load_yaml("masking_rules.yaml")
        taxonomy_path = os.path.join(BASE_DIR, "config", "pii_taxonomy.yaml")

        self.gliner_detector = GLiNERDetector(
            model_name=gliner_model_name,
            threshold=gliner_threshold,
            enabled=use_gliner,
            taxonomy_path=taxonomy_path
        )
        self.regex_detector = RegexDetector(self.regex_rules)
        self.masker = MaskingPolicyEngine(self.masking_rules)

    def detect(self, text: str) -> List[Detection]:
        detections: List[Detection] = []

        if self.gliner_detector is not None:
            detections.extend(self.gliner_detector.detect(text))

        detections.extend(self.regex_detector.detect(text))

        # 1. Drop bad GLiNERguesses  using your preprocessing.py logic
        detections = remove_false_positives(text, detections)
        
        # 2. Execute Granular First/Last name splitting
        detections = split_person_names(text, detections)
        
        # 3. Resolve Overlaps
        detections = resolve_detections(detections)

        # 4. Add strict <LABEL_N> numbering
        detections = add_instance_numbers(detections)

        return detections

    def redact(self, text: str) -> RedactionResult:
        detections = self.detect(text)
        redacted_text = self.masker.redact(text, detections)

        return RedactionResult(
            original_text=text,
            detections=detections,
            redacted_text=redacted_text,
            unknown_candidates=[],
        )