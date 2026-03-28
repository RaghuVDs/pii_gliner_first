"""GLiNER detector - refactored to accept taxonomy config dict instead of YAML file path.

Key change from original: _load_taxonomy() now accepts a dict instead of a file path.
All detection logic (tiered passes, sliding window, dedup) is unchanged.
"""

from __future__ import annotations
from typing import Any, List, Optional, Tuple, Dict
from app.engine.models import Detection
import os
import logging

logger = logging.getLogger("pii_engine.gliner")

try:
    import torch
    from gliner import GLiNER
except ImportError:
    GLiNER = None
    torch = None


class GLiNERDetector:
    """GLiNER-based PII detector with tiered detection and sliding window chunking.

    Refactored to accept taxonomy_config dict instead of taxonomy_path file.
    """

    # Maps taxonomy group names to tier indices for tiered detection
    GROUP_TO_TIER = {
        "general": 0,
        "country_specific": 0,
        "financial_general": 1,
        "amex_specific": 1,
        "medical": 2,
        "education": 2,
        "employment": 2,
        "digital_security": 2,
        "vehicle_property": 3,
        "legal_criminal": 3,
        "communication_utility": 3,
        "contextual": 3,
        "fallback": 3,
    }

    def __init__(
        self,
        model_name: str = "knowledgator/gliner-pii-large-v1.0",
        threshold: float = 0.20,
        enabled: bool = True,
        taxonomy_config: Optional[Dict[str, Any]] = None,
        taxonomy_path: Optional[str] = None,
        preloaded_model: Optional[Any] = None,
    ):
        """Initialize GLiNER detector.

        Args:
            model_name: HuggingFace model ID.
            threshold: Default confidence threshold.
            enabled: Whether to enable detection.
            taxonomy_config: Pre-loaded taxonomy dict (preferred for SaaS mode).
            taxonomy_path: Path to taxonomy YAML file (legacy mode, used if taxonomy_config is None).
        """
        self.enabled = enabled and (preloaded_model is not None or GLiNER is not None)
        self.model_name = model_name
        self.threshold = threshold
        self.model = None

        if preloaded_model is not None:
            # Use pre-loaded cached model (no re-download)
            self.model = preloaded_model
            self.enabled = True
        elif self.enabled:
            try:
                device = "cpu"
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"

                logger.info(f"Loading GLiNER on {device.upper()}...")

                if device == "cpu":
                    cpu_cores = os.cpu_count() or 4
                    torch.set_num_threads(cpu_cores)

                self.model = GLiNER.from_pretrained(model_name).to(device)

            except Exception as e:
                self.enabled = False
                logger.error(f"GLiNER Load Error: {e}")

        self.alias_to_amex_label: Dict[str, str] = {}
        self.gliner_prompt_labels: List[str] = []
        self.label_thresholds: Dict[str, float] = {}
        self.tiered_labels: Dict[int, List[str]] = {0: [], 1: [], 2: [], 3: []}

        # Load taxonomy from config dict or file path
        if taxonomy_config is not None:
            self._load_taxonomy_from_config(taxonomy_config)
        elif taxonomy_path is not None:
            self._load_taxonomy_from_file(taxonomy_path)

    def _load_taxonomy_from_config(self, taxonomy: Dict[str, Any]):
        """Load taxonomy from a pre-built config dict (SaaS mode).

        The dict has the same structure as pii_taxonomy.yaml:
        {
            "general": {
                "PERSON_FULL_NAME": {"gliner_aliases": [...], "threshold": 0.40},
                ...
            },
            ...
        }
        """
        for group_name, group in taxonomy.items():
            if not isinstance(group, dict):
                continue
            tier = self.GROUP_TO_TIER.get(group_name, 2)
            for amex_label, data in group.items():
                if not isinstance(data, dict):
                    continue
                custom_threshold = data.get("threshold", self.threshold)
                for alias in data.get("gliner_aliases", []):
                    self.gliner_prompt_labels.append(alias)
                    self.alias_to_amex_label[alias] = amex_label
                    self.label_thresholds[alias] = custom_threshold
                    if alias not in self.tiered_labels[tier]:
                        self.tiered_labels[tier].append(alias)

        self.gliner_prompt_labels = list(set(self.gliner_prompt_labels))
        for tier in self.tiered_labels:
            self.tiered_labels[tier] = list(set(self.tiered_labels[tier]))

        all_thresholds = list(self.label_thresholds.values()) + [self.threshold]
        self._model_threshold = min(all_thresholds)

    def _load_taxonomy_from_file(self, path: str):
        """Load taxonomy from YAML file (legacy mode)."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            taxonomy = yaml.safe_load(f)
        self._load_taxonomy_from_config(taxonomy)

    def detect(self, text: str) -> List[Detection]:
        if not self.enabled or not self.model or not text.strip():
            return []

        detections: List[Detection] = []
        chunks = self._sliding_window_chunker(text, window_size=1500, overlap=300)

        if not chunks:
            return []

        chunk_texts = [c[0] for c in chunks]
        chunk_starts = [c[1] for c in chunks]

        # TIERED DETECTION: Run GLiNER in focused passes
        for tier_idx in sorted(self.tiered_labels.keys()):
            tier_labels = self.tiered_labels[tier_idx]
            if not tier_labels:
                continue

            for text_chunk, chunk_start in zip(chunk_texts, chunk_starts):
                preds = self.model.predict_entities(
                    text_chunk,
                    tier_labels,
                    threshold=self._model_threshold,
                )

                for pred in preds:
                    found_alias = pred["label"]
                    score = float(pred.get("score", 0.0))

                    if score < self.label_thresholds.get(found_alias, self.threshold):
                        continue

                    strict_amex_label = self.alias_to_amex_label.get(
                        found_alias, "UNKNOWN_PII"
                    )
                    start = chunk_start + int(pred["start"])
                    end = chunk_start + int(pred["end"])
                    value = text[start:end]

                    if not value.strip():
                        continue

                    detections.append(
                        Detection(
                            label=strict_amex_label,
                            text=value,
                            start=start,
                            end=end,
                            score=score,
                            source="gliner",
                            meta={"gliner_alias": found_alias},
                        )
                    )

        return self._deduplicate_overlap_detections(detections)

    @staticmethod
    def _deduplicate_overlap_detections(
        detections: List[Detection],
    ) -> List[Detection]:
        """Remove duplicate detections from chunk overlap zones and cross-tier conflicts."""
        if not detections:
            return detections

        sorted_dets = sorted(detections, key=lambda d: (d.start, d.end))

        def _sweep_dedup(
            dets: List[Detection], threshold: float, same_label_only: bool
        ) -> List[Detection]:
            kept: List[Detection] = []
            for d in dets:
                merged = False
                for i in range(len(kept) - 1, -1, -1):
                    k = kept[i]
                    if k.end <= d.start:
                        break
                    if same_label_only and d.label != k.label:
                        continue
                    overlap_len = max(0, min(d.end, k.end) - max(d.start, k.start))
                    shorter_len = min(d.end - d.start, k.end - k.start)
                    if shorter_len > 0 and overlap_len / shorter_len > threshold:
                        if d.score > k.score:
                            kept[i] = d
                        merged = True
                        break
                if not merged:
                    kept.append(d)
            return kept

        kept = _sweep_dedup(sorted_dets, threshold=0.80, same_label_only=True)
        final = _sweep_dedup(kept, threshold=0.90, same_label_only=False)
        return final

    def _sliding_window_chunker(
        self, text: str, window_size: int, overlap: int
    ) -> List[Tuple[str, int]]:
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + window_size, text_length)
            if end < text_length:
                last_break = text.rfind("\n", start, end)
                if last_break == -1:
                    last_break = text.rfind(" ", start, end)
                if last_break != -1:
                    end = last_break
            chunks.append((text[start:end], start))
            start = end - overlap
            if start <= chunks[-1][1]:
                start = end
        return chunks
