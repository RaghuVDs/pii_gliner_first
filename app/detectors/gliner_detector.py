from __future__ import annotations
import re
import logging
import warnings
from typing import List, Optional, Tuple, Dict
from app.models import Detection
from app.perf.device_manager import get_device_manager
from app.perf.autotune import autotune
import yaml
import os

logger = logging.getLogger("pii_engine.gliner")

# ── Auto-setup CUDA lib paths ─────────────────────────────────────────
# pip-installed nvidia-cu{12,13} wheels drop .so files into site-packages
# but don't add them to LD_LIBRARY_PATH. Torch and ORT both need them at
# import time. We resolve the paths once here so every subprocess works
# without manual env-var exports.
def _ensure_nvidia_lib_paths() -> None:
    """Add pip-installed nvidia lib dirs to LD_LIBRARY_PATH if present."""
    nvidia_packages = [
        "nvidia.cu13.lib",       # torch's cu13 runtime (nvrtc, cublas, etc.)
        "nvidia.cublas.lib",     # ORT needs cu12 cublas
        "nvidia.cudnn.lib",      # ORT needs cu12 cudnn
        "nvidia.curand.lib",
        "nvidia.cufft.lib",
        "nvidia.cuda_runtime.lib",
        "nvidia.cuda_nvrtc.lib",
    ]
    new_paths = []
    for pkg in nvidia_packages:
        try:
            mod = __import__(pkg, fromlist=["__path__"])
            for p in mod.__path__:
                if p not in os.environ.get("LD_LIBRARY_PATH", ""):
                    new_paths.append(p)
        except (ImportError, AttributeError):
            pass
    if new_paths:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths) + (":" + existing if existing else "")

_ensure_nvidia_lib_paths()

# Silence the deprecated batch_predict_entities warning — we use it intentionally
# until inference() is the universally-supported API across GLiNER versions.
warnings.filterwarnings("ignore", message=".*batch_predict_entities is deprecated.*")

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
        taxonomy_path: str = "app/config/pii_taxonomy.yaml",
        use_canonical_labels: bool = True,
        use_ultra_canonical: bool = False,
        ultra_canonical_path: str = "app/config/ultra_canonical_labels.yaml",
        use_onnx: bool = False,
        onnx_model_file: str = "onnx/model_quint8.onnx",
    ):
        self.enabled = enabled and GLiNER is not None
        self.model_name = model_name
        self.threshold = threshold
        self.use_canonical_labels = use_canonical_labels
        # Ultra-canonical mode: ~30 broad concept labels in a single forward
        # pass per chunk. Requires use_canonical_labels=True implicitly. Off
        # by default — opt in to trade some label specificity for ~6x speedup.
        self.use_ultra_canonical = use_ultra_canonical
        self.ultra_canonical_path = ultra_canonical_path
        self.ultra_canonical_labels: List[str] = []
        self.ultra_canonical_to_strict: Dict[str, str] = {}
        # ONNX runtime path: load the pre-built quantized .onnx file from the
        # same HF repo as the torch weights, executed via onnxruntime's
        # CUDAExecutionProvider. The gliner package handles session creation;
        # we just pass `load_onnx_model=True` and `map_location='cuda'`.
        # Off by default to preserve the established torch baseline.
        self.use_onnx = use_onnx
        self.onnx_model_file = onnx_model_file
        self._is_onnx = False
        self.model = None

        self.device = "cpu"
        self._use_fp16 = False
        self._tuning = None  # populated after model load if enabled
        if self.enabled:
            try:
                # ── Device selection via DeviceManager (no hardcoded strings) ──
                dm = get_device_manager()
                device = dm.primary.torch_device  # "cuda:0" | "mps" | "cpu"
                kind = dm.primary.kind

                load_path = "ONNX" if self.use_onnx else "torch"
                logger.info(
                    f"Loading GLiNER ({load_path}) on {dm.primary.name} ({device})..."
                )

                if kind == "cpu":
                    cpu_cores = os.cpu_count() or 4
                    torch.set_num_threads(cpu_cores)

                if self.use_onnx:
                    # ONNX path: gliner picks CUDAExecutionProvider when
                    # map_location starts with "cuda". The wrapper exposes the
                    # same predict_entities / batch_predict_entities API so the
                    # rest of this detector is identical between torch and ONNX.
                    onnx_map_location = "cuda" if kind == "cuda" else "cpu"
                    self.model = GLiNER.from_pretrained(
                        model_name,
                        load_onnx_model=True,
                        load_tokenizer=True,
                        onnx_model_file=self.onnx_model_file,
                        map_location=onnx_map_location,
                    )
                    self._is_onnx = True
                    logger.info(
                        f"[GLiNER] ONNX session active "
                        f"(file={self.onnx_model_file}, provider={'CUDA' if kind == 'cuda' else 'CPU'})"
                    )
                    # Skip .to() / .eval() / .half() — those are torch-only ops
                    # that don't apply to the ONNX wrapper. Quantization is
                    # baked into the .onnx file (UINT8 for model_quint8.onnx).
                else:
                    self.model = GLiNER.from_pretrained(model_name).to(device)

                    # Eval mode: disables dropout, batchnorm updates, etc.
                    try:
                        self.model.eval()
                    except Exception:
                        pass

                    # FP16 on CUDA: ~1.5-2x speedup, negligible accuracy impact for NER
                    if kind == "cuda":
                        try:
                            self.model.half()
                            self._use_fp16 = True
                            logger.info("[GLiNER] FP16 enabled")
                        except Exception as e:
                            logger.warning(f"[GLiNER] FP16 not available, staying FP32: {e}")

                self.device = device

            except Exception as e:
                self.enabled = False
                logger.error(f"GLiNER Load Error: {e}")

        self.alias_to_amex_label: Dict[str, str] = {}
        self.gliner_prompt_labels: List[str] = []
        self.label_thresholds: Dict[str, float] = {}
        self._load_taxonomy(taxonomy_path)

        # Combined (single-pass) label set: union of all tiers.
        # Used by the fast single-pass path; falls back to tiered if it fails.
        self.combined_labels: List[str] = []
        if self.enabled and self.model is not None:
            seen = set()
            for tl in self.tiered_labels.values():
                for lbl in tl:
                    if lbl not in seen:
                        seen.add(lbl)
                        self.combined_labels.append(lbl)
            logger.info(
                f"[GLiNER] labels: {len(self.combined_labels)} all-aliases, "
                f"{len(self.canonical_labels)} canonical (1 per strict label), "
                f"{sum(len(v) for v in self.tiered_labels.values())} across {len(self.tiered_labels)} tiers"
            )

        # Pick the active label set in priority order: ultra-canonical > canonical > tiered.
        # All three live in the same Dict[int, List[str]] structure so the
        # existing detect loop iterates them identically.
        if self.use_ultra_canonical and self.ultra_canonical_labels:
            self._active_label_groups: Dict[int, List[str]] = {0: list(self.ultra_canonical_labels)}
            n = len(self.ultra_canonical_labels)
            logger.info(
                f"[GLiNER] ULTRA-CANONICAL mode ON: {n} labels "
                f"→ ~{(n + 29) // 30} forward pass(es) per chunk"
            )
        elif self.use_canonical_labels and self.canonical_labels:
            self._active_label_groups = {0: list(self.canonical_labels)}
            logger.info(
                f"[GLiNER] canonical mode ON: {len(self.canonical_labels)} labels "
                f"→ ~{(len(self.canonical_labels) + 29) // 30} forward passes per chunk"
            )
        else:
            self._active_label_groups = self.tiered_labels
            total = sum(len(v) for v in self.tiered_labels.values())
            logger.info(
                f"[GLiNER] tiered mode: {total} labels "
                f"→ ~{(total + 29) // 30} forward passes per chunk"
            )

        # ── Auto-tuned performance knobs ─────────────────────────────────
        # All values below are discovered at startup by app.perf.autotune,
        # cached per (gpu, model) in app/perf/tuning_cache.yaml, and re-probed
        # only when hardware or model changes. Nothing here is hardcoded to
        # a specific GPU. See app/perf/autotune.py for the probing strategy.
        self.single_pass = False  # 617 labels is too many for uni-encoder; keep tiered
        if self.enabled and self.model is not None:
            try:
                # Use a runtime-specific cache key suffix so torch and ONNX
                # builds get independent tuning entries — their optimal
                # batch size, labels-per-call, and chunk window can differ
                # significantly even on the same GPU.
                tune_key = f"{model_name}#onnx-int8" if self._is_onnx else model_name
                self._tuning = autotune(self.model, tune_key)
                self.chunk_window = self._tuning.chunk_window
                self.chunk_overlap = self._tuning.chunk_overlap
                self.batch_size = self._tuning.batch_size
                self.max_labels_per_call = self._tuning.max_labels_per_call
                logger.info(
                    f"[GLiNER] auto-tuned: window={self.chunk_window}, "
                    f"overlap={self.chunk_overlap}, batch={self.batch_size}, "
                    f"labels/call={self.max_labels_per_call}"
                )
            except Exception as e:
                logger.error(f"[GLiNER] autotune failed ({e}); using safe defaults")
                self.chunk_window = 3000
                self.chunk_overlap = 150
                self.batch_size = 16
                self.max_labels_per_call = 25
        else:
            # Detector disabled or model failed to load — values will not be used
            self.chunk_window = 3000
            self.chunk_overlap = 150
            self.batch_size = 16
            self.max_labels_per_call = 25

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

        # Canonical labels: ONE alias per strict label (the first one, which the
        # taxonomy author conventionally places as the most generic / highest-recall
        # phrasing). Used in canonical mode to drop the GLiNER label count from
        # ~620 → ~160, which roughly cuts forward passes per chunk by 4x at the
        # cost of slightly less synonym coverage. Detection mapping is unchanged
        # because alias_to_amex_label still resolves canonical aliases correctly.
        self.canonical_labels: List[str] = []
        canonical_seen: set = set()

        for group_name, group in taxonomy.items():
            tier = self.GROUP_TO_TIER.get(group_name, 2)
            for amex_label, data in group.items():
                custom_threshold = data.get("threshold", self.threshold)
                aliases = data.get("gliner_aliases", []) or []
                for alias in aliases:
                    self.gliner_prompt_labels.append(alias)
                    self.alias_to_amex_label[alias] = amex_label
                    self.label_thresholds[alias] = custom_threshold
                    if alias not in self.tiered_labels[tier]:
                        self.tiered_labels[tier].append(alias)
                # Promote the FIRST alias to the canonical set
                if aliases:
                    first_alias = aliases[0]
                    if first_alias not in canonical_seen:
                        canonical_seen.add(first_alias)
                        self.canonical_labels.append(first_alias)

        self.gliner_prompt_labels = list(set(self.gliner_prompt_labels))

        # Deduplicate within each tier
        for tier in self.tiered_labels:
            self.tiered_labels[tier] = list(set(self.tiered_labels[tier]))

        # Use the minimum of ALL thresholds (per-label AND user-provided) as the
        # model-level threshold so no valid detections are pre-filtered by predict_entities()
        all_thresholds = list(self.label_thresholds.values()) + [self.threshold]
        self._model_threshold = min(all_thresholds)

        # Ultra-canonical mode label set — loaded from a separate YAML so it
        # can be edited without touching the main taxonomy. Each entry maps
        # a broad GLiNER concept (e.g. "person name") to a default strict
        # AMEX label (PERSON_FULL_NAME). Downstream context_detector promotes
        # the default to subtypes when the surrounding text supports it.
        self._load_ultra_canonical(self.ultra_canonical_path)

    def _load_ultra_canonical(self, path: str) -> None:
        """Load and validate the ultra-canonical concept set.

        On any error (missing file, malformed YAML, missing fields) the
        ultra-canonical list is left empty and the detector silently falls
        back to canonical mode regardless of the flag. We never crash startup
        because of this optional config.
        """
        try:
            if not os.path.exists(path):
                logger.info(f"[GLiNER] ultra-canonical file not found at {path}; mode unavailable")
                return
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            concepts = data.get("concepts") or []
            if not isinstance(concepts, list):
                logger.warning("[GLiNER] ultra_canonical_labels.yaml: 'concepts' must be a list")
                return
            for entry in concepts:
                if not isinstance(entry, dict):
                    continue
                concept = entry.get("concept")
                default_label = entry.get("default_label")
                if not concept or not default_label:
                    continue
                concept = str(concept).strip().lower()
                if concept and concept not in self.ultra_canonical_to_strict:
                    self.ultra_canonical_labels.append(concept)
                    self.ultra_canonical_to_strict[concept] = str(default_label)
            logger.info(
                f"[GLiNER] loaded {len(self.ultra_canonical_labels)} ultra-canonical concepts from {path}"
            )
        except Exception as e:
            logger.warning(f"[GLiNER] failed to load ultra-canonical config: {e}")
            self.ultra_canonical_labels = []
            self.ultra_canonical_to_strict = {}

    def detect(
        self,
        text: str,
        windows: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Detection]:
        """Run GLiNER over text.

        Args:
            text: The full document.
            windows: Optional list of (start, end) char-offset tuples in the
                original text. When provided, GLiNER chunks and predicts only
                inside these windows; chunk offsets are translated back to the
                original text coordinates so downstream detection offsets are
                unchanged. When None or empty list, the full text is scanned.

                An *empty* list means "no candidate windows found, skip GLiNER
                entirely" — return [] without invoking the model. None means
                "fall through to a full-text scan".
        """
        if not self.enabled or not self.model or not text.strip():
            return []

        # Empty list = nothing to scan, save the GPU forward entirely
        if windows is not None and len(windows) == 0:
            return []

        detections: List[Detection] = []
        if windows:
            chunks = self._chunk_within_windows(text, windows)
        else:
            chunks = self._sliding_window_chunker(
                text, window_size=self.chunk_window, overlap=self.chunk_overlap
            )

        if not chunks:
            return []

        # Diagnostic counter for benchmarking — total chunks processed across the
        # life of this detector. Lets the benchmark report avg chunks/row to
        # quantify the candidate-window speedup without instrumenting every call.
        self._chunks_processed_total = getattr(self, "_chunks_processed_total", 0) + len(chunks)
        self._calls_total = getattr(self, "_calls_total", 0) + 1

        chunk_texts = [c[0] for c in chunks]
        chunk_starts = [c[1] for c in chunks]

        def _ingest(batch_results):
            for chunk_idx, preds in enumerate(batch_results):
                chunk_start = chunk_starts[chunk_idx]
                for pred in preds:
                    det = self._pred_to_detection(pred, chunk_start, text)
                    if det is not None:
                        detections.append(det)

        # ── FAST PATH: single combined pass over ALL labels ──
        # One batched forward pass over all chunks with the full label set.
        # If the uni-encoder can't fit all labels in context, falls back to tiered.
        if self.single_pass and self.combined_labels:
            try:
                with torch.inference_mode():
                    batch_results = self.model.batch_predict_entities(
                        texts=chunk_texts,
                        labels=self.combined_labels,
                        threshold=self._model_threshold,
                        batch_size=min(self.batch_size, len(chunk_texts)),
                    )
                _ingest(batch_results)
                return self._deduplicate_overlap_detections(detections)
            except Exception as e:
                logger.warning(
                    f"[GLiNER] Single-pass failed ({len(self.combined_labels)} labels), "
                    f"falling back to tiered batched: {e}"
                )
                detections.clear()

        # ── TIERED BATCHED + LABEL-CHUNKED ──
        # Each tier is split into groups of `max_labels_per_call` labels.
        # Each group runs as one batched forward pass over all chunks.
        # Total forwards/row ≈ ceil(total_labels / max_labels_per_call).
        # In canonical mode, _active_label_groups is a single "tier" containing
        # ~160 deduplicated canonical labels rather than the full ~620 alias set.
        cap = max(1, self.max_labels_per_call)
        for tier_idx in sorted(self._active_label_groups.keys()):
            tier_labels = self._active_label_groups[tier_idx]
            if not tier_labels:
                continue
            for i in range(0, len(tier_labels), cap):
                label_group = tier_labels[i : i + cap]
                try:
                    with torch.inference_mode():
                        batch_results = self.model.batch_predict_entities(
                            texts=chunk_texts,
                            labels=label_group,
                            threshold=self._model_threshold,
                            batch_size=min(self.batch_size, len(chunk_texts)),
                        )
                    _ingest(batch_results)
                except Exception as e:
                    logger.warning(
                        f"[GLiNER] Tier {tier_idx} group {i}-{i+len(label_group)} "
                        f"batched failed, falling back to per-chunk: {e}"
                    )
                    with torch.inference_mode():
                        for text_chunk, chunk_start in zip(chunk_texts, chunk_starts):
                            preds = self.model.predict_entities(
                                text_chunk,
                                label_group,
                                threshold=self._model_threshold,
                            )
                            for pred in preds:
                                det = self._pred_to_detection(pred, chunk_start, text)
                                if det is not None:
                                    detections.append(det)

        return self._deduplicate_overlap_detections(detections)

    def _pred_to_detection(self, pred: dict, chunk_start: int, full_text: str) -> Optional[Detection]:
        """Convert a GLiNER prediction dict to a Detection, or None if filtered.

        Resolves the predicted alias → strict AMEX label via two paths:
          1. Ultra-canonical mode: ultra_canonical_to_strict (concept → default label)
          2. All other modes: alias_to_amex_label (full taxonomy mapping)

        Falls back to UNKNOWN_PII if neither map contains the alias — this can
        happen if the user changes the active label set but the model still
        emits a stale label, which is rare but worth catching.
        """
        found_alias = pred["label"]
        score = float(pred.get("score", 0.0))

        # Threshold check — ultra-canonical concepts use the user-provided
        # default since they don't have per-alias thresholds in the taxonomy.
        threshold = self.label_thresholds.get(found_alias, self.threshold)
        if score < threshold:
            return None

        if self.use_ultra_canonical and self.ultra_canonical_to_strict:
            strict_amex_label = self.ultra_canonical_to_strict.get(
                found_alias.lower(),
                self.alias_to_amex_label.get(found_alias, "UNKNOWN_PII"),
            )
        else:
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

    # Speaker-turn pattern: matches common dialogue formats dynamically.
    # No hardcoded speaker names — detects the STRUCTURAL pattern of
    # "SomeName:" or "[timestamp] SomeName:" at line start.
    _TURN_BOUNDARY_RE = re.compile(
        r'^'                             # start of line
        r'(?:\[[\d:.\s]+\]\s*)?'         # optional timestamp [00:01:23]
        r'[A-Z][A-Za-z\s]{0,25}'         # speaker name (capitalized, ≤25 chars)
        r'\s*:\s',                        # colon + space (the key structural signal)
        re.MULTILINE,
    )

    def _chunk_within_windows(
        self,
        text: str,
        windows: List[Tuple[int, int]],
    ) -> List[Tuple[str, int]]:
        """Produce chunks restricted to the given windows of the original text.

        For each window, slice text[ws:we] and run the same sliding-window
        chunker. Translate each chunk's local start back to the original-text
        coordinate by adding the window's offset.

        This is the candidate-window fast path: we typically chunk only ~5–10%
        of the text per row instead of the full document.
        """
        out: List[Tuple[str, int]] = []
        for ws, we in windows:
            if ws >= we or ws < 0 or we > len(text):
                continue
            sub = text[ws:we]
            sub_chunks = self._sliding_window_chunker(
                sub, window_size=self.chunk_window, overlap=self.chunk_overlap
            )
            for chunk_text, local_start in sub_chunks:
                out.append((chunk_text, ws + local_start))
        return out

    def _sliding_window_chunker(self, text: str, window_size: int, overlap: int) -> List[Tuple[str, int]]:
        """Split text into overlapping chunks, preserving conversation context.

        Conversation-aware: detects speaker-turn boundaries dynamically
        (any "Name: text" pattern, with optional timestamps) and avoids
        splitting Q&A pairs across chunks. When a question turn asks about
        PII and the answer is on the next turn, both stay in the same chunk
        so GLiNER sees the full context.

        No hardcoded speaker names or trigger phrases. The detector
        recognizes the STRUCTURAL pattern of dialogue (capitalized name
        followed by colon) and uses it as the preferred break point.

        Break point priority:
          1. Speaker turn boundary (keeps Q&A pairs together)
          2. Sentence boundary ([.!?] + space + uppercase)
          3. Newline
          4. Space

        Falls back to regular sliding window for non-conversational text
        (text without speaker-turn patterns).
        """
        chunks = []
        start = 0
        text_length = len(text)

        # Pre-scan: find all speaker-turn boundaries once. O(n) regex over
        # the full text, cached for all chunks. These are the PREFERRED
        # break points because they keep Q&A pairs together.
        turn_starts = [m.start() for m in self._TURN_BOUNDARY_RE.finditer(text)]

        while start < text_length:
            end = min(start + window_size, text_length)
            if end < text_length:
                best_break = -1
                # Search zone: last 30% of the window
                search_start = start + int(window_size * 0.7)

                # Priority 1: find the LAST speaker turn boundary in the
                # search zone. This ensures the chunk ends at a turn
                # boundary, and the NEXT chunk starts with a fresh turn
                # (which includes the question context for its answer).
                for ts in reversed(turn_starts):
                    if search_start <= ts < end:
                        best_break = ts
                        break
                    if ts < search_start:
                        break  # turn_starts is sorted, no point looking further

                # Priority 2: sentence boundary
                if best_break == -1:
                    for m in self._SENTENCE_BOUNDARY_RE.finditer(text, search_start, end):
                        best_break = m.end()

                # Priority 3: newline
                if best_break == -1:
                    best_break = text.rfind('\n', start, end)

                # Priority 4: space
                if best_break == -1:
                    best_break = text.rfind(' ', start, end)

                if best_break > start:
                    end = best_break

            chunks.append((text[start:end], start))
            start = end - overlap
            if start <= chunks[-1][1]:
                start = end
        return chunks