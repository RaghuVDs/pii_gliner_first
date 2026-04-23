from __future__ import annotations
import os
import logging
from typing import List, Dict, Any, Optional, Set
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
from app.perf.candidate_windows import CandidateWindowScanner
# ConversationalPIIDetector removed — replaced by conversation-aware
# chunking in GLiNERDetector which keeps Q&A pairs together so GLiNER
# sees the full context. No hardcoded trigger phrases needed.
from app.ml.trainer import SelfTrainer
from app.ml.few_shot import PrototypicalFewShotClassifier
from app.ml.calibration import CalibrationManager
from app.active_learning import ActiveLearningManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_yaml(filename: str):
    path = os.path.join(BASE_DIR, "config", filename)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ── Per-stage profiler helpers ──────────────────────────────────────────
import time as _time_mod


class _NullCtx:
    """No-op context manager for the disabled-profiler fast path."""
    def __enter__(self): return self
    def __exit__(self, *a): return False


_NULL_CTX = _NullCtx()


class _StageTimer:
    """Times one stage of detect() and accumulates into the engine totals."""
    __slots__ = ("engine", "name", "_t0")

    def __init__(self, engine, name: str):
        self.engine = engine
        self.name = name
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = _time_mod.perf_counter()
        return self

    def __exit__(self, *a):
        ms = (_time_mod.perf_counter() - self._t0) * 1000.0
        self.engine._stage_totals_ms[self.name] = (
            self.engine._stage_totals_ms.get(self.name, 0.0) + ms
        )
        self.engine._stage_calls[self.name] = (
            self.engine._stage_calls.get(self.name, 0) + 1
        )
        return False


class HybridPIIEngine:
    def __init__(
        self,
        use_gliner: bool = True,
        gliner_model_name: str = "knowledgator/gliner-pii-large-v1.0",
        gliner_threshold: float = 0.35,
        inference_only: bool = False,
        use_ultra_canonical: bool = False,
        use_onnx: bool = False,
        onnx_model_file: str = "onnx/model_quint8.onnx",
    ):
        # ── Bulk mode state ──────────────────────────────────────────────
        # When True, per-row I/O and the LSTM auto-retrain check are skipped.
        # All buffered learning state is flushed once via finalize_bulk() at
        # the end of a batch. End-of-bulk state on disk is equivalent to
        # online mode; the only semantic difference is that the LSTM is
        # retrained at most once per batch instead of every ~10 detections.
        self._bulk_mode: bool = False
        self._bulk_rows_seen: int = 0

        # ── Inference-only mode (for distributed Spark workers) ─────────
        # When True, ALL learning side effects are disabled — no accumulator
        # writes, no trainer buffer, no active learning queue, no LSTM
        # retraining or auto-promote. The detect() pipeline produces detections
        # but never mutates on-disk learning state. This is the right mode for
        # Spark executors so multiple workers can run in parallel without
        # racing on the same YAML files.
        self._inference_only: bool = inference_only
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
            taxonomy_path=taxonomy_path,
            use_ultra_canonical=use_ultra_canonical,
            use_onnx=use_onnx,
            onnx_model_file=onnx_model_file,
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

        # ── Candidate-window pre-scanner (Phase 4) ───────────────────────
        # Cheap pre-scan that finds regions of interest so GLiNER doesn't
        # have to scan filler dialogue. Built dynamically from the same
        # config dicts the detectors use — no separate label list maintained.
        self.window_scanner = CandidateWindowScanner(
            regex_rules=self.regex_rules,
            field_patterns=self.field_patterns,
            context_rules=self.context_rules,
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

        # ── Per-stage profiler (zero overhead unless enabled) ────────────
        # Set engine.profile_stages = True to accumulate per-stage wall time.
        # The benchmark prints get_stage_profile() at the end.
        self.profile_stages: bool = False
        self._stage_totals_ms: Dict[str, float] = {}
        self._stage_calls: Dict[str, int] = {}

        if self.self_trainer.is_ready():
            labels = list(self.self_trainer.label_to_idx.keys())
            logger.info(f"[LSTM] Model loaded on startup — {len(labels)} labels known: {labels}")

    def _stage(self, name: str):
        """Context manager that times a stage if profiling is enabled.

        Returns a no-op context manager when profile_stages is False so the
        per-row hot path stays free of timing overhead in normal operation.
        """
        if not self.profile_stages:
            return _NULL_CTX
        return _StageTimer(self, name)

    def get_stage_profile(self) -> Dict[str, Any]:
        """Return accumulated per-stage timings as a sorted summary."""
        if not self._stage_totals_ms:
            return {"stages": [], "total_ms": 0.0}
        items = sorted(
            self._stage_totals_ms.items(), key=lambda kv: kv[1], reverse=True
        )
        total = sum(self._stage_totals_ms.values())
        out = []
        for name, ms in items:
            calls = self._stage_calls.get(name, 0)
            out.append({
                "stage": name,
                "total_ms": round(ms, 1),
                "calls": calls,
                "avg_ms": round(ms / max(calls, 1), 2),
                "pct": round(ms / total * 100, 1) if total else 0.0,
            })
        return {"stages": out, "total_ms": round(total, 1)}

    def reset_stage_profile(self) -> None:
        self._stage_totals_ms.clear()
        self._stage_calls.clear()

    def detect(self, text: str) -> List[Detection]:
        detections: List[Detection] = []

        # ══════════════════════════════════════════════════════════════
        # MODEL-FIRST HYBRID PII DETECTION PIPELINE
        #
        # GLiNER runs first on the full text (no regex pre-scan), then
        # regex validates formats. The combined output defines PII
        # regions — downstream detectors scan only those regions (token
        # dropping). Anomaly detectors discover unknowns, LSTM learns.
        # ══════════════════════════════════════════════════════════════

        # ── 1. GLiNER AI (primary semantic detection, zero-shot NER) ──
        # Runs on the full text — the model is the PII region finder, not
        # regex. The broad "any sensitive personal information" ultra-
        # canonical concept catches novel PII types the specific labels miss.
        if self.gliner_detector is not None:
            with self._stage("01_gliner"):
                detections.extend(self.gliner_detector.detect(text, windows=None))

        # ── 2. Regex Scan (format validation: SSN, email, Luhn, etc.) ──
        # High-precision format validators that run on full text. These
        # catch deterministic patterns (SSN checksums, CC Luhn checks,
        # IBAN mod-97) more reliably than any model. The cost is modest
        # (~50-80 ms for 80k chars) compared to the old window_scan+regex
        # combined pass (~183 ms).
        with self._stage("02_regex"):
            detections.extend(self.window_scanner.regex_sweep_only(text))

        # ── BUILD PII REGION MASK (token dropping) ──────────────────
        # The union of GLiNER + regex detections defines where PII lives.
        # Expand each detection by ±150 chars and merge overlaps. Down-
        # stream stages scan only within these regions, skipping ~80-95%
        # of the text on sparse/medium rows.
        from app.perf.pii_regions import PIIRegionMask
        pii_mask: Optional[PIIRegionMask] = PIIRegionMask.from_detections(
            detections, len(text), pad_chars=150
        )
        mask_coverage = pii_mask.coverage() if pii_mask else 0.0
        if pii_mask and mask_coverage > 0.80:
            # PII-dense row: masking wouldn't save much. Fall back to full scan.
            pii_mask = None
            logger.info(f"[regions] coverage {mask_coverage:.0%} > 80% → full scan for downstream")
        elif pii_mask and not pii_mask.is_empty():
            logger.info(
                f"[regions] {len(pii_mask.regions)} regions, "
                f"covering {mask_coverage:.1%} of {len(text)} chars → token dropping active"
            )

        # ── 4. Field Label Scan (explicit label:value patterns) ──
        # Scans only within PII regions when token dropping is active.
        with self._stage("03_field"):
            detections.extend(self.field_detector.detect(text, pii_regions=pii_mask))

        # ── 5. Context Promotion (reclassify + filter by keywords) ──
        with self._stage("04_context"):
            detections = self.context_detector.detect(text, detections)

        # Record unclassified detections for adaptive learning.
        if not self._inference_only:
            unknown_candidates = self.context_detector.get_unknown_candidates()
            if unknown_candidates:
                self.accumulator.record(unknown_candidates, detections, text)
                if not self._bulk_mode:
                    self.accumulator.flush()

        # ── 6. Entropy Anomaly (secrets, tokens, API keys, hashes) ──
        # Scans only within PII regions when token dropping is active.
        with self._stage("05_entropy_anomaly"):
            entropy_hits = self.entropy_detector.detect(
                text, existing_detections=detections, pii_regions=pii_mask,
            )
            if entropy_hits:
                detections.extend(entropy_hits)
                logger.info(f"[ENTROPY] Detected {len(entropy_hits)} high-entropy spans")

        # ── 7. Contextual Anomaly (unknown field patterns, novel PII) ──
        # Scans only within PII regions when token dropping is active.
        with self._stage("06_contextual_anomaly"):
            ctx_anomalies = self.contextual_anomaly_detector.detect(
                text, existing_detections=detections, pii_regions=pii_mask,
            )
            if ctx_anomalies:
                detections.extend(ctx_anomalies)
                logger.info(f"[ANOMALY] Detected {len(ctx_anomalies)} contextual anomalies")

        # ── 8. LSTM Pseudo-labeling (upgrades unknowns with learned patterns) ──
        with self._stage("07_lstm_pseudo"):
            if self.self_trainer.is_ready():
                before_count = sum(1 for d in detections if d.source == "pattern_lstm")
                detections = self.self_trainer.pseudo_label(detections, text)
                after_count = sum(1 for d in detections if d.source == "pattern_lstm")
                upgrades = after_count - before_count
                if upgrades > 0:
                    upgraded = [d for d in detections if d.source == "pattern_lstm"]
                    labels = [f"{d.label}({d.score:.2f})" for d in upgraded[-upgrades:]]
                    logger.info(f"[LSTM] Upgraded {upgrades} detections: {labels}")

        # ── 9. Few-Shot Classification (instant detection from prototypes) ──
        with self._stage("08_few_shot"):
            if self.few_shot.is_ready():
                detections = self._apply_few_shot(detections, text)

        # ── 10. False Positive Filtering (grammar gates) ──
        with self._stage("09_false_positive"):
            detections = remove_false_positives(text, detections)

        # ── 11. Cross-Validation Bonus (GLiNER + regex/field agreement) ──
        with self._stage("10_xval_bonus"):
            detections = apply_cross_validation_bonus(detections)

        # ── 12. Name Splitting + Overlap Resolution + Propagation ──
        with self._stage("11_postprocess"):
            detections = split_person_names(text, detections)
            detections = resolve_detections(detections)
            detections = propagate_person_names(text, detections)
            detections = add_instance_numbers(detections)

        # ── 12. Self-Healing Loop (collect → retrain → promote) ──
        # Stats and in-memory collection are always cheap; we keep them.
        # In inference_only mode we skip the entire learning loop — Spark
        # workers must not mutate any on-disk training state.
        if self._inference_only:
            return detections

        self.stats_tracker.record_run(detections)
        self.self_trainer.collect_from_detections(detections, text)
        self.active_learner.score_detections(detections)

        if not self._bulk_mode:
            # Online mode: flush per-row, retrain when threshold reached, periodic auto-promote.
            self.self_trainer.flush_to_disk()
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
        else:
            # Bulk mode: defer all I/O and retraining to finalize_bulk().
            self._bulk_rows_seen += 1

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

    # ── Bulk processing API ──────────────────────────────────────────────

    def detect_batch(self, texts: List[str]) -> List[List[Detection]]:
        """Process many rows with per-row I/O and LSTM retraining suppressed.

        Behaviour vs detect(): identical detection logic per row, but the
        per-row YAML flushes (accumulator, trainer buffer, active learner)
        and the LSTM auto-retrain are deferred to finalize_bulk(), which
        runs once at the end. End-of-batch on-disk state is equivalent to
        processing the rows individually with detect().

        For 200k rows this collapses 200k YAML writes and ~20k LSTM
        retrains (each ~30s) down to 1 of each. The LSTM model used for
        pseudo-labeling during the batch is the one loaded at engine
        startup; it does not evolve mid-batch.

        Args:
            texts: List of input strings.

        Returns:
            List of detection lists, one per input in the same order.
        """
        results: List[List[Detection]] = []
        prev_bulk = self._bulk_mode
        self._bulk_mode = True
        self._bulk_rows_seen = 0
        try:
            for text in texts:
                results.append(self.detect(text))
        finally:
            try:
                self.finalize_bulk()
            finally:
                self._bulk_mode = prev_bulk
        return results

    def finalize_bulk(self) -> Dict[str, Any]:
        """Drain all deferred bulk-mode state to disk with exactly ONE retrain.

        Order matters here. Auto-promote feeds new "human-verified" training
        examples into the trainer buffer. If we retrain *before* auto-promote,
        we'd retrain twice (once now, once after promote). Order:

          1. Flush accumulator + active learner (no training side-effects)
          2. Auto-promote pending rules → adds examples to trainer buffer + reloads
             context detector. Bypasses self.promote_rules() to skip its inner retrain.
          3. Flush trainer buffer (now contains both detection-derived and promoted examples)
          4. ONE retrain at the end if the threshold is met

        Idempotent. Safe to call after a sequence of detect() calls or as a
        shutdown hook regardless of whether _bulk_mode is currently set.
        """
        summary: Dict[str, Any] = {
            "rows_finalized": self._bulk_rows_seen,
            "accumulator_flushed": False,
            "trainer_flushed": 0,
            "active_learner_flushed": False,
            "retrained": False,
            "promoted": 0,
            "inference_only": self._inference_only,
        }

        # Inference-only mode: nothing to flush, no learning state to update.
        if self._inference_only:
            self._bulk_rows_seen = 0
            return summary

        # 1. Flush accumulator (unknown PII candidates)
        try:
            self.accumulator.flush()
            summary["accumulator_flushed"] = True
        except Exception as e:
            logger.warning(f"[finalize_bulk] accumulator flush failed: {e}")

        # 2. Flush active learner queue
        try:
            self.active_learner.flush()
            summary["active_learner_flushed"] = True
        except Exception as e:
            logger.warning(f"[finalize_bulk] active learner flush failed: {e}")

        # 3. Auto-promote — bypass self.promote_rules() because that triggers
        #    an inner retrain. We do the same work but defer retraining to step 5.
        try:
            self._run_count += self._bulk_rows_seen
            promoted = promote_pending_rules(threshold=3, auto_confirm=True)
            summary["promoted"] = len(promoted) if promoted else 0
            if promoted:
                # Hot-reload the context detector with the new rules
                self.context_rules = load_yaml("context_rules.yaml")
                self.context_detector = ContextDetector(context_rules=self.context_rules)
                # Feed promoted rules into the trainer buffer (in-memory only here)
                n_added = self.self_trainer.collect_from_promoted(promoted)
                if n_added:
                    logger.info(f"[finalize_bulk] auto-promoted {len(promoted)} rules → {n_added} training examples")
        except Exception as e:
            logger.warning(f"[finalize_bulk] auto-promote failed: {e}")

        # 4. Flush trainer buffer to disk (now contains detection-derived + promoted examples)
        try:
            n = self.self_trainer.flush_to_disk()
            summary["trainer_flushed"] = n
        except Exception as e:
            logger.warning(f"[finalize_bulk] trainer flush failed: {e}")

        # 5. ONE LSTM retrain at the very end — only if threshold reached
        try:
            if self.self_trainer.should_retrain():
                logger.info(f"[finalize_bulk] retraining LSTM after {self._bulk_rows_seen} bulk rows (one pass only)...")
                result = self.self_trainer.retrain()
                if result.get("status") == "trained":
                    summary["retrained"] = True
                    logger.info(
                        f"[finalize_bulk] retrain done — {result['num_examples']} examples, "
                        f"{result['num_labels']} labels, val_f1={result.get('val_weighted_f1', 0):.3f}"
                    )
        except Exception as e:
            logger.warning(f"[finalize_bulk] retrain failed: {e}")

        self._bulk_rows_seen = 0
        return summary

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