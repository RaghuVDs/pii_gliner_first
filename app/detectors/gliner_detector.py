from __future__ import annotations
from typing import List, Optional, Tuple, Dict
from app.models import Detection
import yaml
import os

try:
    import torch
    from gliner import GLiNER
except ImportError:
    GLiNER = None

class GLiNERDetector:
    def __init__(
        self,
        model_name: str = "knowledgator/gliner-pii-large-v1.0",
        threshold: float = 0.20,
        enabled: bool = True,
        taxonomy_path: str = "app/config/pii_taxonomy.yaml"
    ):
        self.enabled = enabled and GLiNER is not None
        self.model_name = model_name
        self.threshold = threshold
        self.model = None

        if self.enabled:
            try:
                # DYNAMIC DEVICE SELECTION (Massive Performance Boost)
                device = "cpu"
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                
                print(f"Loading GLiNER on {device.upper()}...")
                
                if device == "cpu":
                    cpu_cores = os.cpu_count() or 4
                    torch.set_num_threads(cpu_cores)
                    
                self.model = GLiNER.from_pretrained(model_name).to(device)
                
            except Exception as e:
                self.enabled = False
                print(f"GLiNER Load Error: {e}")

        self.alias_to_amex_label: Dict[str, str] = {}
        self.gliner_prompt_labels: List[str] = []
        self.label_thresholds: Dict[str, float] = {}
        self._load_taxonomy(taxonomy_path)

    # Maps taxonomy YAML group names to tier indices for tiered detection
    GROUP_TO_TIER = {
        "general": 0,               # Tier 0: Names, addresses, contact info
        "country_specific": 0,      # Tier 0: SSN, passport, DL, government IDs
        "financial_general": 1,     # Tier 1: Financial identifiers
        "amex_specific": 1,         # Tier 1: AMEX-specific IDs
        "medical": 2,               # Tier 2: HIPAA PHI, medical records
        "education": 2,             # Tier 2: FERPA, student records
        "employment": 2,            # Tier 2: HR, payroll, background checks
        "digital_security": 2,      # Tier 2: API keys, SSH, certificates
        "vehicle_property": 3,      # Tier 3: VIN, license plates
        "legal_criminal": 3,        # Tier 3: Court cases, criminal IDs
        "communication_utility": 3, # Tier 3: Utility accounts, shipping
        "contextual": 3,            # Tier 3: Sensitive contextual data
        "fallback": 3,              # Tier 3: Unknown/fallback
    }

    def _load_taxonomy(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            taxonomy = yaml.safe_load(f)

        # Build tiered label lists for focused GLiNER passes
        self.tiered_labels: Dict[int, List[str]] = {0: [], 1: [], 2: [], 3: []}

        for group_name, group in taxonomy.items():
            tier = self.GROUP_TO_TIER.get(group_name, 2)
            for amex_label, data in group.items():
                custom_threshold = data.get("threshold", self.threshold)
                for alias in data.get("gliner_aliases", []):
                    self.gliner_prompt_labels.append(alias)
                    self.alias_to_amex_label[alias] = amex_label
                    self.label_thresholds[alias] = custom_threshold
                    if alias not in self.tiered_labels[tier]:
                        self.tiered_labels[tier].append(alias)

        self.gliner_prompt_labels = list(set(self.gliner_prompt_labels))

        # Deduplicate within each tier
        for tier in self.tiered_labels:
            self.tiered_labels[tier] = list(set(self.tiered_labels[tier]))

        # Use the minimum of ALL thresholds (per-label AND user-provided) as the
        # model-level threshold so no valid detections are pre-filtered by predict_entities()
        all_thresholds = list(self.label_thresholds.values()) + [self.threshold]
        self._model_threshold = min(all_thresholds)

    def detect(self, text: str) -> List[Detection]:
        if not self.enabled or not self.model or not text.strip():
            return []

        detections: List[Detection] = []
        # INCREASED OVERLAP from 150 to 300 to ensure contextual clues are captured
        chunks = self._sliding_window_chunker(text, window_size=1500, overlap=300)

        if not chunks:
            return []

        chunk_texts = [c[0] for c in chunks]
        chunk_starts = [c[1] for c in chunks]

        # TIERED DETECTION: Run GLiNER in focused passes to reduce label confusion
        # Each tier has a smaller, focused set of labels → less probability mass spread
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

                    strict_amex_label = self.alias_to_amex_label.get(found_alias, "UNKNOWN_PII")
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
                            meta={"gliner_alias": found_alias}
                        )
                    )

        return self._deduplicate_overlap_detections(detections)

    @staticmethod
    def _deduplicate_overlap_detections(detections: List[Detection]) -> List[Detection]:
        """Remove duplicate detections from chunk overlap zones and cross-tier conflicts.

        Phase 1 (same-label): When two chunks overlap, the same entity may be
        detected twice with slightly different spans/scores. Keep the higher-scoring
        detection when overlap >80% of the shorter span.

        Phase 2 (cross-label): Different tiers may detect the same span with
        different labels. Keep the higher-scoring detection when overlap >90%.
        """
        if not detections:
            return detections

        sorted_dets = sorted(detections, key=lambda d: (d.start, d.end))
        kept: List[Detection] = []

        # Phase 1: Same-label dedup (chunk overlap)
        for d in sorted_dets:
            merged = False
            for i, k in enumerate(kept):
                if d.label != k.label:
                    continue
                overlap_start = max(d.start, k.start)
                overlap_end = min(d.end, k.end)
                overlap_len = max(0, overlap_end - overlap_start)
                shorter_len = min(d.end - d.start, k.end - k.start)

                if shorter_len > 0 and overlap_len / shorter_len > 0.80:
                    if d.score > k.score:
                        kept[i] = d
                    merged = True
                    break
            if not merged:
                kept.append(d)

        # Phase 2: Cross-label dedup (cross-tier conflicts)
        final: List[Detection] = []
        for d in kept:
            replaced = False
            for i, f in enumerate(final):
                overlap_start = max(d.start, f.start)
                overlap_end = min(d.end, f.end)
                overlap_len = max(0, overlap_end - overlap_start)
                shorter_len = min(d.end - d.start, f.end - f.start)

                if shorter_len > 0 and overlap_len / shorter_len > 0.90:
                    if d.score > f.score:
                        final[i] = d
                    replaced = True
                    break
            if not replaced:
                final.append(d)

        return final

    def _sliding_window_chunker(self, text: str, window_size: int, overlap: int) -> List[Tuple[str, int]]:
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + window_size, text_length)
            if end < text_length:
                # Snap to a newline if possible instead of just a space for cleaner boundaries
                last_break = text.rfind('\n', start, end)
                if last_break == -1:
                    last_break = text.rfind(' ', start, end)
                if last_break != -1: 
                    end = last_break
            chunks.append((text[start:end], start))
            start = end - overlap
            if start <= chunks[-1][1]: start = end
        return chunks