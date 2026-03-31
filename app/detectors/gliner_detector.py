from __future__ import annotations
import re
import logging
from typing import List, Optional, Tuple, Dict
from app.models import Detection
import yaml
import os

logger = logging.getLogger("pii_engine.gliner")

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
        self._bi_encoder = False  # Whether model supports bi-encoder optimization

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

        # ── Bi-Encoder Optimization ──────────────────────────────────
        # Pre-compute label embeddings per tier (one-time cost at init).
        # At inference, only text needs encoding — labels are cached.
        self.tier_label_embeds: Dict[int, object] = {}
        if self.enabled and self.model is not None:
            self._init_bi_encoder_cache()

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
        "non_pii_suppression": 0,   # Tier 0: Runs alongside names to suppress org/product FPs
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

    def _init_bi_encoder_cache(self):
        """Pre-compute and cache label embeddings per tier for bi-encoder mode.

        If the model supports encode_labels() (bi-encoder architecture),
        label embeddings are computed once at init and reused for every
        detect() call — only text needs encoding at inference time.
        """
        try:
            # Test if model supports bi-encoder API
            test_labels = self.tiered_labels.get(0, [])[:2]
            if not test_labels:
                return
            _ = self.model.encode_labels(test_labels, batch_size=8)
            self._bi_encoder = True

            for tier_idx, tier_labels in self.tiered_labels.items():
                if tier_labels:
                    self.tier_label_embeds[tier_idx] = self.model.encode_labels(
                        tier_labels, batch_size=8
                    )
            logger.info(
                f"[GLiNER] Bi-encoder cache initialized — "
                f"{sum(len(v) for v in self.tiered_labels.values())} label embeddings cached across {len(self.tiered_labels)} tiers"
            )
        except (NotImplementedError, AttributeError, TypeError):
            self._bi_encoder = False
            logger.info("[GLiNER] Model does not support bi-encoder API — using standard predict_entities()")

    def detect(self, text: str) -> List[Detection]:
        if not self.enabled or not self.model or not text.strip():
            return []

        detections: List[Detection] = []
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

            # ── Bi-encoder path: batch all chunks with cached label embeddings ──
            if self._bi_encoder and tier_idx in self.tier_label_embeds:
                try:
                    batch_results = self.model.batch_predict_with_embeds(
                        texts=chunk_texts,
                        labels_embeddings=self.tier_label_embeds[tier_idx],
                        labels=tier_labels,
                        threshold=self._model_threshold,
                        batch_size=min(16, len(chunk_texts)),
                    )
                    for chunk_idx, preds in enumerate(batch_results):
                        chunk_start = chunk_starts[chunk_idx]
                        for pred in preds:
                            det = self._pred_to_detection(pred, chunk_start, text)
                            if det is not None:
                                detections.append(det)
                    continue  # Skip standard path for this tier
                except Exception as e:
                    logger.warning(f"[GLiNER] Bi-encoder batch failed for tier {tier_idx}, falling back: {e}")

            # ── Standard path: per-chunk predict_entities() ──
            for text_chunk, chunk_start in zip(chunk_texts, chunk_starts):
                preds = self.model.predict_entities(
                    text_chunk,
                    tier_labels,
                    threshold=self._model_threshold,
                )
                for pred in preds:
                    det = self._pred_to_detection(pred, chunk_start, text)
                    if det is not None:
                        detections.append(det)

        return self._deduplicate_overlap_detections(detections)

    def _pred_to_detection(self, pred: dict, chunk_start: int, full_text: str) -> Optional[Detection]:
        """Convert a GLiNER prediction dict to a Detection, or None if filtered."""
        found_alias = pred["label"]
        score = float(pred.get("score", 0.0))

        if score < self.label_thresholds.get(found_alias, self.threshold):
            return None

        strict_amex_label = self.alias_to_amex_label.get(found_alias, "UNKNOWN_PII")
        start = chunk_start + int(pred["start"])
        end = chunk_start + int(pred["end"])
        value = full_text[start:end]

        if not value.strip():
            return None

        return Detection(
            label=strict_amex_label,
            text=value,
            start=start,
            end=end,
            score=score,
            source="gliner",
            meta={"gliner_alias": found_alias},
        )

    @staticmethod
    def _deduplicate_overlap_detections(detections: List[Detection]) -> List[Detection]:
        """Remove duplicate detections from chunk overlap zones and cross-tier conflicts.

        Uses a sweep-line approach: sort by start, then only compare against
        recent detections whose spans could still overlap (end > current start).
        This avoids O(n²) full scans for well-distributed detections.

        Phase 1 (same-label): overlap >80% of shorter span → keep higher score.
        Phase 2 (cross-label): overlap >90% of shorter span → keep higher score.
        """
        if not detections:
            return detections

        sorted_dets = sorted(detections, key=lambda d: (d.start, d.end))

        def _sweep_dedup(dets: List[Detection], threshold: float, same_label_only: bool) -> List[Detection]:
            kept: List[Detection] = []
            for d in dets:
                merged = False
                # Walk backwards through kept — only check items whose end > d.start
                for i in range(len(kept) - 1, -1, -1):
                    k = kept[i]
                    if k.end <= d.start:
                        break  # No more possible overlaps (sorted by start)
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

        # Phase 1: Same-label dedup (chunk overlap)
        kept = _sweep_dedup(sorted_dets, threshold=0.80, same_label_only=True)
        # Phase 2: Cross-label dedup (cross-tier conflicts)
        final = _sweep_dedup(kept, threshold=0.90, same_label_only=False)
        return final

    # Sentence-ending punctuation followed by whitespace and uppercase letter
    _SENTENCE_BOUNDARY_RE = re.compile(r'[.!?]\s+(?=[A-Z])')

    def _sliding_window_chunker(self, text: str, window_size: int, overlap: int) -> List[Tuple[str, int]]:
        """Split text into overlapping chunks, snapping to sentence boundaries.

        Priority order for break points:
          1. Sentence boundary ([.!?] followed by space + uppercase)
          2. Newline
          3. Space
        This produces cleaner chunks that preserve semantic context.
        """
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + window_size, text_length)
            if end < text_length:
                # Try sentence boundary first (best semantic split)
                best_break = -1
                # Search in the last 30% of the window for a sentence boundary
                search_start = start + int(window_size * 0.7)
                for m in self._SENTENCE_BOUNDARY_RE.finditer(text, search_start, end):
                    best_break = m.end()  # Position right after the punctuation + space
                if best_break == -1:
                    # Fall back to newline
                    best_break = text.rfind('\n', start, end)
                if best_break == -1:
                    # Fall back to space
                    best_break = text.rfind(' ', start, end)
                if best_break > start:
                    end = best_break
            chunks.append((text[start:end], start))
            start = end - overlap
            if start <= chunks[-1][1]:
                start = end
        return chunks