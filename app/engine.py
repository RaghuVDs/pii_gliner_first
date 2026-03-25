from __future__ import annotations
import os
import logging
from typing import List, Dict, Any
import yaml

logger = logging.getLogger("pii_engine")

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
    DetectionStatsTracker,
    promote_pending_rules,
    manual_promote,
    review_pending_rules,
)
from app.ml.trainer import SelfTrainer

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

        # Strategy 3: Adaptive Learning — accumulates unknown PII for self-learning (PII-safe)
        self.accumulator = UnknownPIIAccumulator()
        self.stats_tracker = DetectionStatsTracker()

        # Pattern LSTM classifier — learns from GLiNER detections, upgrades unknowns
        self.self_trainer = SelfTrainer()

        # Auto-promotion counter — promotes pending rules every N runs without human review
        self._run_count = 0
        self._auto_promote_interval = 50  # Check for auto-promotion every 50 detect() calls

        if self.self_trainer.is_ready():
            labels = list(self.self_trainer.label_to_idx.keys())
            logger.info(f"[LSTM] Model loaded on startup — {len(labels)} labels known: {labels}")

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

        # 4b. Adaptive Learning: record unclassified detections (PII-safe)
        unknown_candidates = self.context_detector.get_unknown_candidates()
        if unknown_candidates:
            self.accumulator.record(unknown_candidates, detections, text)
            self.accumulator.flush()

        # 4c. Pattern LSTM: upgrade UNKNOWN_PII / low-confidence using learned patterns
        if self.self_trainer.is_ready():
            before_count = sum(1 for d in detections if d.source == "pattern_lstm")
            detections = self.self_trainer.pseudo_label(detections, text)
            after_count = sum(1 for d in detections if d.source == "pattern_lstm")
            upgrades = after_count - before_count
            if upgrades > 0:
                upgraded = [d for d in detections if d.source == "pattern_lstm"]
                labels = [f"{d.label}({d.score:.2f})" for d in upgraded[-upgrades:]]
                logger.info(f"[LSTM] Upgraded {upgrades} detections: {labels}")

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

        # 10. Record aggregate stats (PII-safe: labels, scores, sources only)
        self.stats_tracker.record_run(detections)

        # 11. Collect training data from high-confidence detections (PII-safe)
        n_collected = self.self_trainer.collect_from_detections(detections, text)
        self.self_trainer.flush_to_disk()

        # 12. Auto-retrain if enough new data accumulated
        if self.self_trainer.should_retrain():
            logger.info("[LSTM] Auto-retrain triggered — training on accumulated data...")
            result = self.self_trainer.retrain()
            if result.get("status") == "trained":
                labels = list(self.self_trainer.label_to_idx.keys())
                logger.info(
                    f"[LSTM] Training complete — {result['num_examples']} examples, "
                    f"{result['num_labels']} labels, val_f1={result.get('val_weighted_f1', 0):.3f}, "
                    f"best_epoch={result.get('best_epoch')}/{result.get('stopped_epoch')}"
                )
                logger.info(f"[LSTM] Known labels: {labels}")

        # 13. Auto-promote pending rules (no human reviewer needed)
        self._run_count += 1
        if self._run_count % self._auto_promote_interval == 0:
            promoted = self.promote_rules(threshold=3, auto=True)
            if promoted:
                promoted_labels = [r.get("label", "?") for r in promoted]
                logger.info(f"[AUTO-PROMOTE] Promoted {len(promoted)} rules: {promoted_labels}")

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
        If auto=True: promotes to context_rules.yaml, reloads the context detector,
                       feeds promoted rules into the LSTM trainer as human-verified
                       training data, and triggers a retrain.

        Returns list of promoted/promotable entries.
        """
        result = promote_pending_rules(threshold=threshold, auto_confirm=auto)
        if auto and result:
            # Hot-reload: re-read context_rules.yaml and rebuild the context detector
            self.context_rules = load_yaml("context_rules.yaml")
            self.context_detector = ContextDetector(context_rules=self.context_rules)

            # Feed promoted rules into the LSTM trainer as high-confidence training data
            n_added = self.self_trainer.collect_from_promoted(result)
            if n_added > 0:
                self.self_trainer.flush_to_disk()
                # Retrain with the new high-quality data
                self.self_trainer.retrain()
        return result

    def manual_promote_rule(self, group_key: str, label: str, keywords: List[str] = None) -> bool:
        """Manually promote a specific pending rule with the correct PII label.

        Use this when auto-suggested label is wrong — provide the right label.
        Hot-reloads the context detector after promotion.
        Feeds the promoted rule into the LSTM trainer and triggers retrain.

        Args:
            group_key: The group key from review_pending() output.
            label: The correct PII taxonomy label (e.g., "BIOMETRIC_TEMPLATE_ID").
            keywords: Optional keyword overrides.

        Returns:
            True if promoted successfully.
        """
        # Get the full entry before promoting (need structure_patterns, co_labels)
        pending = self.accumulator.get_pending()
        entry = pending.get(group_key, {})

        ok = manual_promote(group_key=group_key, label=label, keywords=keywords)
        if ok:
            self.context_rules = load_yaml("context_rules.yaml")
            self.context_detector = ContextDetector(context_rules=self.context_rules)

            # Feed into LSTM trainer as human-verified
            promoted_data = [{
                "label": label,
                "keywords": keywords or entry.get("suggested_keywords", []),
                "structure_patterns": entry.get("structure_patterns", []),
                "co_occurring_labels": entry.get("co_occurring_labels", []),
                "seen_count": entry.get("seen_count", 1),
            }]
            n_added = self.self_trainer.collect_from_promoted(promoted_data)
            if n_added > 0:
                self.self_trainer.flush_to_disk()
                self.self_trainer.retrain()
        return ok

    def learning_stats(self) -> Dict[str, Any]:
        """Get stats about the adaptive learning system.

        Returns:
            Dict with total_groups, ready_to_promote, total_sightings.
        """
        return self.accumulator.get_stats()

    def detection_stats(self) -> Dict[str, Any]:
        """Get aggregate detection statistics across all runs.

        Returns:
            Dict with label_counts, source_counts, label_source_counts, total_runs.
        """
        return self.stats_tracker.get_stats()

    def lstm_known_labels(self) -> List[str]:
        """Return the list of PII labels the LSTM can currently predict.

        Empty list means the model hasn't been trained yet — it's still collecting data.
        Once trained, any label in this list will be auto-detected on future transcripts.
        """
        return self.self_trainer.known_labels()

    def pattern_lstm_stats(self) -> Dict[str, Any]:
        """Get stats about the pattern LSTM classifier.

        Returns training data counts, model readiness, label distribution.
        """
        return self.self_trainer.get_stats()

    def retrain_pattern_model(self, epochs: int = 50) -> Dict[str, Any]:
        """Manually trigger retraining of the pattern LSTM classifier.

        Use this after processing many transcripts to force a model update.
        Returns full metrics: per-label P/R/F1, confusion matrix, training curves.
        """
        return self.self_trainer.retrain(epochs=epochs)

    def metrics_history(self) -> Dict[str, Any]:
        """Return the full metrics log across all training runs.

        Each run contains: per-label precision/recall/F1, confusion matrix,
        macro/weighted averages, hyperparameters, timestamps.
        """
        return self.self_trainer.get_metrics_history()