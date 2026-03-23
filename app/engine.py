from __future__ import annotations
import os
from typing import List, Dict, Any
import yaml

from app.models import RedactionResult, Detection
from app.detectors.regex_detector import RegexDetector
from app.detectors.gliner_detector import GLiNERDetector
from app.detectors.context_detector import ContextDetector
from app.detectors.field_detector import PatternFieldDetector
from app.resolver import resolve_detections, apply_cross_validation_bonus
from app.postprocessing import add_instance_numbers, remove_false_positives, split_person_names, propagate_person_names
from app.policy_engine import MaskingPolicyEngine
from app.adaptive_learning import (
    UnknownPIIAccumulator,
    promote_pending_rules,
    manual_promote,
    review_pending_rules,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_yaml(filename: str):
    path = os.path.join(BASE_DIR, "config", filename)
    with open(path, 'r', encoding='utf-8') as f:
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
        self.field_patterns = load_yaml("field_patterns.yaml") # Load field patterns
        taxonomy_path = os.path.join(BASE_DIR, "config", "pii_taxonomy.yaml")

        # Initialize Detectors
        self.field_detector = PatternFieldDetector(self.field_patterns) # Wired up
        self.gliner_detector = GLiNERDetector(
            model_name=gliner_model_name,
            threshold=gliner_threshold,
            enabled=use_gliner,
            taxonomy_path=taxonomy_path
        )
        self.regex_detector = RegexDetector(self.regex_rules)
        self.context_rules = load_yaml("context_rules.yaml")
        self.context_detector = ContextDetector(context_rules=self.context_rules)
        self.masker = MaskingPolicyEngine(self.masking_rules)

        # Strategy 3: Adaptive Learning — accumulates unknown PII for self-learning
        self.accumulator = UnknownPIIAccumulator()

    def detect(self, text: str) -> List[Detection]:
        detections: List[Detection] = []

        # ── GLiNER-FIRST ARCHITECTURE ──
        # GLiNER is the PRIMARY detector (semantic understanding for diverse transcripts).
        # Field labels and regex are FALLBACK (structured patterns, format validation).

        # 1. AI Scan (PRIMARY — handles diverse/unique transcripts semantically)
        if self.gliner_detector is not None:
            detections.extend(self.gliner_detector.detect(text))

        # 2. Field Label Scan (SUPPLEMENT — catches explicit label:value patterns)
        detections.extend(self.field_detector.detect(text))

        # 3. Regex Scan (FALLBACK — format-validated patterns: SSN, email, Luhn, etc.)
        detections.extend(self.regex_detector.detect(text))

        # 4. Context Promotion (Refines labels: SSN→TAX_ID, unknown→specific)
        #    Also uses Strategy 1 (similarity classifier) for unknown PII
        detections = self.context_detector.detect(text, detections)

        # 4b. Adaptive Learning: record unclassified detections for self-learning
        unknown_candidates = self.context_detector.get_unknown_candidates()
        if unknown_candidates:
            self.accumulator.record(unknown_candidates)
            self.accumulator.flush()

        # 5. Drop bad guesses (source-aware filtering)
        detections = remove_false_positives(text, detections)

        # 5b. Cross-validation bonus: boost when GLiNER + regex agree on same span
        detections = apply_cross_validation_bonus(detections)

        # 6. Execute Granular First/Last name splitting
        detections = split_person_names(text, detections)

        # 7. Resolve Overlaps (GLiNER wins by default; format-validated regex overrides)
        detections = resolve_detections(detections)

        # 8. Propagate known names to undetected re-mentions
        #    (catches "Dr. Patel", "Mr. Doe", "Hi Emily" etc.)
        detections = propagate_person_names(text, detections)

        # 9. Add strict <LABEL_N> numbering
        detections = add_instance_numbers(detections)

        return detections

    def redact(self, text: str) -> RedactionResult:
        detections = self.detect(text)
        redacted_text = self.masker.redact(text, detections)

        # Include unknown candidates so callers can see what wasn't classified
        unknown_candidates = self.context_detector.get_unknown_candidates()

        return RedactionResult(
            original_text=text,
            detections=detections,
            redacted_text=redacted_text,
            unknown_candidates=unknown_candidates,
        )

    # ── Adaptive Learning API ──────────────────────────────────────────

    def review_pending(self) -> List[Dict[str, Any]]:
        """Review all pending PII rules accumulated by the self-learning system.

        Returns a list sorted by seen_count (highest first), showing:
        - group_key, suggested_label, seen_count, keywords, example_contexts
        """
        return review_pending_rules()

    def promote_rules(self, threshold: int = 3, auto: bool = False) -> List[Dict[str, Any]]:
        """Promote pending rules that have been seen >= threshold times.

        If auto=False (default): returns candidates for review without promoting.
        If auto=True: promotes to context_rules.yaml and reloads the context detector.

        Returns list of promoted/promotable entries.
        """
        result = promote_pending_rules(threshold=threshold, auto_confirm=auto)
        if auto and result:
            # Hot-reload: re-read context_rules.yaml and rebuild the context detector
            self.context_rules = load_yaml("context_rules.yaml")
            self.context_detector = ContextDetector(context_rules=self.context_rules)
        return result

    def manual_promote_rule(self, group_key: str, label: str, keywords: List[str] = None) -> bool:
        """Manually promote a specific pending rule with the correct PII label.

        Use this when auto-suggested label is wrong — provide the right label.
        Hot-reloads the context detector after promotion.

        Args:
            group_key: The group key from review_pending() output.
            label: The correct PII taxonomy label (e.g., "BIOMETRIC_TEMPLATE_ID").
            keywords: Optional keyword overrides.

        Returns:
            True if promoted successfully.
        """
        ok = manual_promote(group_key=group_key, label=label, keywords=keywords)
        if ok:
            self.context_rules = load_yaml("context_rules.yaml")
            self.context_detector = ContextDetector(context_rules=self.context_rules)
        return ok

    def learning_stats(self) -> Dict[str, Any]:
        """Get stats about the adaptive learning system.

        Returns:
            Dict with total_groups, ready_to_promote, total_sightings.
        """
        return self.accumulator.get_stats()