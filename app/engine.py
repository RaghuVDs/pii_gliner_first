from __future__ import annotations
import os
import logging
from typing import List, Dict, Any, Set
import yaml

logger = logging.getLogger("pii_engine")

from app.models import RedactionResult, Detection
from app.detectors.regex_detector import RegexDetector
from app.detectors.gliner_detector import GLiNERDetector
from app.detectors.context_detector import ContextDetector
from app.detectors.field_detector import PatternFieldDetector
from app.detectors.entropy_detector import EntropyAnomalyDetector
from app.detectors.anomaly_detector import ContextualAnomalyDetector
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
from app.ml.few_shot import PrototypicalFewShotClassifier
from app.ml.calibration import CalibrationManager
from app.active_learning import ActiveLearningManager

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
        self.field_detector = PatternFieldDetector(self.field_patterns)
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

        # ── Anomaly Detectors (unknown PII discovery) ────────────────
        self.entropy_detector = EntropyAnomalyDetector()
        # Build known field label set for contextual anomaly detector
        known_fields: Set[str] = set()
        for label_data in self.field_patterns.values():
            if isinstance(label_data, list):
                known_fields.update(f.lower().strip() for f in label_data)
            elif isinstance(label_data, str):
                known_fields.add(label_data.lower().strip())
        context_keywords: Set[str] = set()
        for kw_list in self.context_rules.values():
            if isinstance(kw_list, list):
                context_keywords.update(kw.lower() for kw in kw_list)
        self.contextual_anomaly_detector = ContextualAnomalyDetector(
            known_field_labels=known_fields,
            context_keywords=context_keywords,
        )

        # ── Adaptive Learning ────────────────────────────────────────
        self.accumulator = UnknownPIIAccumulator()
        self.stats_tracker = DetectionStatsTracker()

        # Pattern LSTM classifier — learns from GLiNER detections, upgrades unknowns
        self.self_trainer = SelfTrainer()

        # ── Few-Shot Classifier ──────────────────────────────────────
        self.few_shot = PrototypicalFewShotClassifier()

        # ── Confidence Calibration ───────────────────────────────────
        self.calibration = CalibrationManager()

        # ── Active Learning ──────────────────────────────────────────
        self.active_learner = ActiveLearningManager()

        # Auto-promotion counter — promotes pending rules every N runs without human review
        self._run_count = 0
        self._auto_promote_interval = 50

        if self.self_trainer.is_ready():
            labels = list(self.self_trainer.label_to_idx.keys())
            logger.info(f"[LSTM] Model loaded on startup — {len(labels)} labels known: {labels}")

    def detect(self, text: str) -> List[Detection]:
        detections: List[Detection] = []

        # ══════════════════════════════════════════════════════════════
        # HYBRID PII DETECTION PIPELINE
        #
        # 6 core detectors + self-healing learning loop.
        # GLiNER handles semantics, regex validates formats,
        # anomaly detectors discover unknowns, LSTM learns over time.
        # ══════════════════════════════════════════════════════════════

        # ── 1. GLiNER AI (primary semantic detection, zero-shot NER) ──
        if self.gliner_detector is not None:
            detections.extend(self.gliner_detector.detect(text))

        # ── 2. Field Label Scan (explicit label:value patterns) ──
        detections.extend(self.field_detector.detect(text))

        # ── 3. Regex Scan (format-validated: SSN, email, Luhn, etc.) ──
        detections.extend(self.regex_detector.detect(text))

        # ── 4. Context Promotion (reclassify + filter by keywords) ──
        detections = self.context_detector.detect(text, detections)

        # Record unclassified detections for adaptive learning
        unknown_candidates = self.context_detector.get_unknown_candidates()
        if unknown_candidates:
            self.accumulator.record(unknown_candidates, detections, text)
            self.accumulator.flush()

        # ── 5. Entropy Anomaly (secrets, tokens, API keys, hashes) ──
        entropy_hits = self.entropy_detector.detect(text, existing_detections=detections)
        if entropy_hits:
            detections.extend(entropy_hits)
            logger.info(f"[ENTROPY] Detected {len(entropy_hits)} high-entropy spans")

        # ── 6. Contextual Anomaly (unknown field patterns, novel PII) ──
        ctx_anomalies = self.contextual_anomaly_detector.detect(text, existing_detections=detections)
        if ctx_anomalies:
            detections.extend(ctx_anomalies)
            logger.info(f"[ANOMALY] Detected {len(ctx_anomalies)} contextual anomalies")

        # ── 7. LSTM Pseudo-labeling (upgrades unknowns with learned patterns) ──
        if self.self_trainer.is_ready():
            before_count = sum(1 for d in detections if d.source == "pattern_lstm")
            detections = self.self_trainer.pseudo_label(detections, text)
            after_count = sum(1 for d in detections if d.source == "pattern_lstm")
            upgrades = after_count - before_count
            if upgrades > 0:
                upgraded = [d for d in detections if d.source == "pattern_lstm"]
                labels = [f"{d.label}({d.score:.2f})" for d in upgraded[-upgrades:]]
                logger.info(f"[LSTM] Upgraded {upgrades} detections: {labels}")

        # ── 8. Few-Shot Classification (instant detection from prototypes) ──
        if self.few_shot.is_ready():
            detections = self._apply_few_shot(detections, text)

        # ── 9. False Positive Filtering (grammar gates) ──
        detections = remove_false_positives(text, detections)

        # ── 10. Cross-Validation Bonus (GLiNER + regex/field agreement) ──
        detections = apply_cross_validation_bonus(detections)

        # ── 11. Name Splitting + Overlap Resolution + Propagation ──
        detections = split_person_names(text, detections)
        detections = resolve_detections(detections)
        detections = propagate_person_names(text, detections)
        detections = add_instance_numbers(detections)

        # ── 12. Self-Healing Loop (collect → retrain → promote) ──
        self.stats_tracker.record_run(detections)
        self.self_trainer.collect_from_detections(detections, text)
        self.self_trainer.flush_to_disk()
        self.active_learner.score_detections(detections)
        self.active_learner.flush()

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

        self._run_count += 1
        if self._run_count % self._auto_promote_interval == 0:
            promoted = self.promote_rules(threshold=3, auto=True)
            if promoted:
                promoted_labels = [r.get("label", "?") for r in promoted]
                logger.info(f"[AUTO-PROMOTE] Promoted {len(promoted)} rules: {promoted_labels}")

        return detections

    def _apply_few_shot(self, detections: List[Detection], text: str) -> List[Detection]:
        """Apply few-shot classification to unknown/low-confidence detections."""
        from app.adaptive_learning import _value_to_structure, _extract_safe_neighborhood
        from app.ml.model import encode_pattern
        import torch

        if not self.self_trainer.is_ready():
            return detections

        upgraded = 0
        for d in detections:
            # Only try few-shot on unknowns or low-confidence detections
            if d.label not in {"UNKNOWN_PII", "UNKNOWN_IDENTIFIER", "UNKNOWN_SECRET", "CODE"}:
                if d.score > 0.55:
                    continue

            try:
                # Build feature vector using LSTM encoder
                structure = _value_to_structure(d.text)
                safe_neighborhood = _extract_safe_neighborhood(text, d.start, d.end, detections, pad=150)
                keywords = self.self_trainer._extract_keywords(safe_neighborhood)
                co_labels = sorted(set(
                    o.label for o in detections
                    if o is not d and abs(o.start - d.start) < 500 and o.label != d.label
                ))

                # Encode through LSTM to get 288-dim features
                char_ids = torch.tensor([encode_pattern(structure)], dtype=torch.long,
                                        device=self.self_trainer.device)
                ctx = self.self_trainer._build_context_features({
                    "structure": structure,
                    "keywords": keywords[:12],
                    "co_labels": co_labels[:10],
                    "source": d.source,
                    "score": d.score,
                    "length": len(d.text),
                })
                ctx_tensor = ctx.unsqueeze(0).to(self.self_trainer.device)

                with torch.no_grad():
                    features = self.self_trainer.model.get_features(char_ids, ctx_tensor)
                feature_list = features.squeeze(0).cpu().tolist()

                # Classify via few-shot
                result = self.few_shot.classify(feature_list)
                if result is not None:
                    new_label, similarity = result
                    d.label = new_label
                    d.score = min(0.85, similarity)
                    d.source = "few_shot"
                    d.meta["few_shot_similarity"] = similarity
                    upgraded += 1

            except Exception:
                continue  # Silently skip on any encoding error

        if upgraded > 0:
            logger.info(f"[FEW-SHOT] Classified {upgraded} detections via prototypes")
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
        Also creates a few-shot prototype for instant future detection.

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

            # Create few-shot prototype for instant future detection
            self._create_few_shot_prototype(label, entry)
        return ok

    def _create_few_shot_prototype(self, label: str, entry: Dict) -> None:
        """Extract a prototype feature vector from LSTM and store for few-shot."""
        if not self.self_trainer.is_ready():
            return
        try:
            import torch
            from app.ml.model import encode_pattern

            structures = entry.get("structure_patterns", [])
            kw = entry.get("suggested_keywords", [])
            co = entry.get("co_occurring_labels", [])

            for struct in structures[:3]:  # At most 3 prototypes per promotion
                if not struct:
                    continue
                char_ids = torch.tensor([encode_pattern(struct)], dtype=torch.long,
                                        device=self.self_trainer.device)
                ctx = self.self_trainer._build_context_features({
                    "structure": struct, "keywords": kw[:12], "co_labels": co[:10],
                    "source": "context", "score": 0.95, "length": len(struct),
                })
                ctx_tensor = ctx.unsqueeze(0).to(self.self_trainer.device)

                with torch.no_grad():
                    features = self.self_trainer.model.get_features(char_ids, ctx_tensor)
                self.few_shot.add_prototype(label, features.squeeze(0).cpu().tolist())
        except Exception as e:
            logger.warning(f"[FEW-SHOT] Failed to create prototype: {e}")

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

    # ── Active Learning API ───────────────────────────────────────────

    def get_review_queue(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get the top-K most informative candidates for human review.

        Returns candidates ranked by informativeness (uncertainty + disagreement + novelty).
        More efficient than passive 3-sighting promotion — focuses human effort
        where it matters most.
        """
        return self.active_learner.get_review_queue(top_k=top_k)

    def submit_review(self, key: str, correct_label: str) -> bool:
        """Submit a human review for an active learning candidate.

        The reviewed example feeds into LSTM training with 2-3x weight,
        and creates a few-shot prototype for instant future detection.
        """
        ok = self.active_learner.submit_review(key, correct_label)
        if ok:
            # Create few-shot prototype from the reviewed entry
            entry = self.active_learner._queue.get(key, {})
            self._create_few_shot_prototype(correct_label, {
                "structure_patterns": [entry.get("structure", "")],
                "suggested_keywords": [],
                "co_occurring_labels": [],
            })
        return ok

    def active_learning_stats(self) -> Dict[str, Any]:
        """Get active learning queue statistics."""
        return self.active_learner.stats()

    # ── Few-Shot API ─────────────────────────────────────────────────

    def few_shot_labels(self) -> List[str]:
        """Labels with few-shot prototypes for instant detection."""
        return self.few_shot.known_labels()

    def few_shot_stats(self) -> Dict[str, Any]:
        """Get few-shot classifier statistics."""
        return self.few_shot.stats()

    # ── Synthetic Data API ───────────────────────────────────────────

    def generate_synthetic_data(self, n_per_label: int = 20, n_documents: int = 50) -> Dict[str, Any]:
        """Generate synthetic training data to augment LSTM training.

        Creates fake PII values in realistic document contexts using the faker library.
        Excludes person name labels (too generic for pattern learning).

        Args:
            n_per_label: Number of isolated examples per PII label.
            n_documents: Number of full synthetic documents to generate.

        Returns:
            Dict with counts of generated examples and labels covered.
        """
        from app.ml.synthetic_data import SyntheticPIIGenerator

        gen = SyntheticPIIGenerator()
        isolated = gen.generate(n_per_label=n_per_label)
        doc_based = gen.generate_documents(n_documents=n_documents)
        all_examples = isolated + doc_based

        # Feed into LSTM trainer with synthetic confidence type
        if all_examples:
            with self.self_trainer._lock:
                self.self_trainer._buffer.extend(all_examples)
            self.self_trainer.flush_to_disk()

        labels_covered = set(ex["label"] for ex in all_examples)
        return {
            "total_examples": len(all_examples),
            "isolated_examples": len(isolated),
            "document_examples": len(doc_based),
            "labels_covered": sorted(labels_covered),
        }