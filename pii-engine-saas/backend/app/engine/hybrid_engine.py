"""
HybridPIIEngine - Multi-tenant, config-injected version.

This is the refactored engine that accepts pre-built configuration dicts
instead of loading from YAML files. The detection pipeline (13 stages)
remains identical to the original engine.

Usage:
    # Configs are assembled from the database by ConfigAssembler
    engine = HybridPIIEngine(
        taxonomy_config=assembled_taxonomy,
        regex_rules=assembled_regex_rules,
        masking_rules=assembled_masking_rules,
        field_patterns=assembled_field_patterns,
        context_rules=assembled_context_rules,
    )
    detections = engine.detect(text)
    result = engine.redact(text)
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from app.engine.models import Detection, RedactionResult

# Import detectors from the original engine package
# These are copied into backend/app/engine/detectors/ from the original project
from app.engine.detectors.gliner_detector import GLiNERDetector
from app.engine.detectors.regex_detector import RegexDetector
from app.engine.detectors.field_detector import PatternFieldDetector
from app.engine.detectors.context_detector import ContextDetector
from app.engine.resolver import resolve_detections, apply_cross_validation_bonus
from app.engine.postprocessing import (
    add_instance_numbers,
    remove_false_positives,
    split_person_names,
    propagate_person_names,
)
from app.engine.policy_engine import MaskingPolicyEngine

logger = logging.getLogger("pii_engine")


class HybridPIIEngine:
    """Multi-tenant PII detection engine with config injection.

    Instead of loading YAML files directly, this engine receives
    pre-assembled configuration dicts from the database via ConfigAssembler.
    This enables per-tenant configuration without touching the detection pipeline.
    """

    def __init__(
        self,
        taxonomy_config: Dict[str, Any],
        regex_rules: Dict[str, Any],
        masking_rules: Dict[str, Any],
        field_patterns: Dict[str, Any],
        context_rules: Dict[str, Any],
        use_gliner: bool = True,
        gliner_model_name: str = "knowledgator/gliner-pii-large-v1.0",
        gliner_threshold: float = 0.35,
        tenant_id: Optional[str] = None,
        model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        gliner_model: Optional[Any] = None,
    ):
        """Initialize the PII engine with injected configurations.

        Args:
            taxonomy_config: PII taxonomy dict (same structure as pii_taxonomy.yaml).
                             Categories -> PII types with gliner_aliases and thresholds.
            regex_rules: Regex patterns dict (same structure as regex_rules.yaml).
                         PII type name -> list of regex pattern strings.
            masking_rules: Masking strategy dict (same structure as masking_rules.yaml).
            field_patterns: Field label patterns dict (same structure as field_patterns.yaml).
                           PII type name -> list of label strings.
            context_rules: Context keyword rules dict (same structure as context_rules.yaml).
                          PII type name -> list of keyword/regex patterns.
            use_gliner: Whether to enable the GLiNER AI detector.
            gliner_model_name: HuggingFace model ID for GLiNER.
            gliner_threshold: Default confidence threshold for GLiNER.
            tenant_id: The tenant this engine instance belongs to (for logging).
            model_path: Path to tenant-specific LSTM model file (.pt).
            vocab_path: Path to tenant-specific LSTM vocabulary file.
        """
        self.tenant_id = tenant_id
        self.regex_rules = regex_rules
        self.masking_rules = masking_rules
        self.field_patterns = field_patterns
        self.context_rules = context_rules
        self.taxonomy_config = taxonomy_config

        # Initialize detectors with injected configs
        self.field_detector = PatternFieldDetector(self.field_patterns)

        self.gliner_detector = GLiNERDetector(
            model_name=gliner_model_name,
            threshold=gliner_threshold,
            enabled=use_gliner,
            taxonomy_config=taxonomy_config,
            preloaded_model=gliner_model,
        )

        self.regex_detector = RegexDetector(self.regex_rules)
        self.context_detector = ContextDetector(context_rules=self.context_rules)
        self.masker = MaskingPolicyEngine(self.masking_rules)

        # Pattern LSTM classifier - optional, loaded from tenant-specific paths
        self.self_trainer = None
        self._lstm_ready = False

        if model_path and vocab_path:
            try:
                from app.engine.ml.trainer import SelfTrainer
                self.self_trainer = SelfTrainer(
                    model_path=model_path,
                    vocab_path=vocab_path,
                )
                self._lstm_ready = self.self_trainer.is_ready()
                if self._lstm_ready:
                    labels = self.self_trainer.known_labels()
                    logger.info(
                        f"[LSTM] tenant={tenant_id} Model loaded — "
                        f"{len(labels)} labels known"
                    )
            except Exception as e:
                logger.warning(f"[LSTM] tenant={tenant_id} Failed to load: {e}")
                self.self_trainer = None
                self._lstm_ready = False

        logger.info(
            f"[Engine] Initialized for tenant={tenant_id} | "
            f"GLiNER={'ON' if use_gliner else 'OFF'} | "
            f"LSTM={'ON' if self._lstm_ready else 'OFF'} | "
            f"Regex rules={len(self.regex_rules)} types | "
            f"Field patterns={len(self.field_patterns)} types | "
            f"Context rules={len(self.context_rules)} types"
        )

    def detect(self, text: str) -> List[Detection]:
        """Run the full 13-stage PII detection pipeline.

        The pipeline is identical to the original engine:
        1. GLiNER detection (PRIMARY - semantic understanding)
        2. Field label detection (SUPPLEMENT - structured patterns)
        3. Regex detection (FALLBACK - format validation)
        4. Context promotion (refines labels using keywords)
        5. Pattern LSTM pseudo-labeling (if model available)
        6. False positive removal (universal dynamic filters)
        7. Cross-validation bonus (GLiNER + regex agreement)
        8. Person name splitting (FULL_NAME -> FIRST/LAST/MIDDLE)
        9. Overlap resolution (priority-based dedup)
        10. Person name propagation (catch re-mentions)
        11. Instance numbering (<LABEL_N> tags)

        Note: Steps 12-13 from the original (stats tracking, auto-retrain,
        auto-promote) are handled externally by the SaaS service layer,
        not inside the engine, to support async/DB-backed operations.

        Args:
            text: The input text to scan for PII.

        Returns:
            List of Detection objects with all PII found.
        """
        detections: List[Detection] = []

        # 1. AI Scan (PRIMARY)
        if self.gliner_detector is not None:
            detections.extend(self.gliner_detector.detect(text))

        # 2. Field Label Scan (SUPPLEMENT)
        detections.extend(self.field_detector.detect(text))

        # 3. Regex Scan (FALLBACK)
        detections.extend(self.regex_detector.detect(text))

        # 4. Context Promotion
        detections = self.context_detector.detect(text, detections)

        # 5. Pattern LSTM pseudo-labeling (if available)
        if self._lstm_ready and self.self_trainer is not None:
            try:
                before_count = sum(1 for d in detections if d.source == "pattern_lstm")
                detections = self.self_trainer.pseudo_label(detections, text)
                after_count = sum(1 for d in detections if d.source == "pattern_lstm")
                upgrades = after_count - before_count
                if upgrades > 0:
                    logger.info(
                        f"[LSTM] tenant={self.tenant_id} "
                        f"Upgraded {upgrades} detections"
                    )
            except Exception as e:
                logger.warning(f"[LSTM] tenant={self.tenant_id} Pseudo-label error: {e}")

        # 6. Drop false positives (source-aware filtering)
        detections = remove_false_positives(text, detections)

        # 7. Cross-validation bonus
        detections = apply_cross_validation_bonus(detections)

        # 8. Person name splitting
        detections = split_person_names(text, detections)

        # 9. Resolve overlaps
        detections = resolve_detections(detections)

        # 10. Propagate known names
        detections = propagate_person_names(text, detections)

        # 11. Instance numbering
        detections = add_instance_numbers(detections)

        return detections

    def redact(self, text: str) -> RedactionResult:
        """Detect PII and redact it from the text.

        Args:
            text: The input text to scan and redact.

        Returns:
            RedactionResult with original text, detections, and redacted text.
        """
        detections = self.detect(text)
        redacted_text = self.masker.redact(text, detections)

        unknown_candidates = self.context_detector.get_unknown_candidates()

        return RedactionResult(
            original_text=text,
            detections=detections,
            redacted_text=redacted_text,
            unknown_candidates=unknown_candidates,
        )

    def get_unknown_candidates(self) -> List[Dict[str, Any]]:
        """Get unclassified detection candidates from the last run.

        These are detections the context detector couldn't classify,
        useful for feeding into the adaptive learning pipeline.
        """
        return self.context_detector.get_unknown_candidates()

    def get_taxonomy_labels(self) -> List[str]:
        """Get all PII type labels from the loaded taxonomy."""
        labels = []
        for category_data in self.taxonomy_config.values():
            if isinstance(category_data, dict):
                for label in category_data.keys():
                    if isinstance(category_data[label], dict):
                        labels.append(label)
        return labels
