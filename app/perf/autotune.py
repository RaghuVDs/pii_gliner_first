"""GLiNER auto-tuner — finds optimal (chunk_window, batch_size, max_labels_per_call)
for the live (GPU, model) combo, targeting ~90% VRAM utilization.

Why this exists:
  The pipeline used to hardcode chunk_window=3000, batch_size=32, max_labels_per_call=25.
  Those numbers were tuned for one machine and one model. They are wrong on every
  other GPU. This module discovers the right values at startup and caches them.

Probing strategy (cheap, runs once per (gpu, model) and is cached):
  1. chunk_window: try {1500, 2000, 3000, 4000, 6000} chars. Pick the largest that
     keeps per-token throughput within 5% of the best (larger window = fewer chunks
     but quadratic attention; there's a knee).
  2. batch_size: double 1 → 128. Stop when peak VRAM crosses VRAM_TARGET_FRACTION
     (default 0.90) OR throughput gains drop below 5% per doubling. Take the largest
     value that satisfies both.
  3. max_labels_per_call: probe {10, 15, 25, 40, 60}. Uni-encoder GLiNER has a knee
     in cost-per-(chunk × label); pick the value that minimises ms per (chunk × label).

Cache:
  Results are persisted to app/perf/tuning_cache.yaml keyed by
  (device_kind, gpu_name, model_name, torch_version). Re-probe is automatic when
  any key component changes.

Failure modes:
  - OOM during probe → back off batch_size by 2x and continue
  - Probe wholly fails → fall back to conservative defaults and log loudly
  - On CPU → skip probing entirely, use small conservative values
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from app.perf.device_manager import DeviceInfo, get_device_manager

logger = logging.getLogger("pii_engine.perf.autotune")

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning_cache.yaml")

# Bump this whenever the probe logic changes in a way that would yield
# different results — entries with an older version are ignored on load,
# forcing a re-probe. This avoids stale cached results haunting future runs.
TUNING_VERSION = 4

# Tunable targets — these are the only "constants" in this file, and they
# describe the *goal*, not a specific GPU. The actual numbers are derived.
VRAM_TARGET_FRACTION = 0.90  # use up to 90% of VRAM
VRAM_SAFETY_MARGIN = 0.05    # back off if we cross 95% during probe
THROUGHPUT_GAIN_THRESHOLD = 0.05  # stop scaling if next step gains <5%

# Token budget safety: leave headroom for special tokens, label tokens, padding.
# The chunker targets (model_max_len - TOKEN_HEADROOM) tokens worth of chars.
TOKEN_HEADROOM = 64
# Multiplicative safety on chars-per-token to handle worst-case dense content
TOKEN_BUDGET_SAFETY = 0.85

# Conservative fallbacks if probing fails entirely
CPU_FALLBACK = {"chunk_window": 1500, "chunk_overlap": 100, "batch_size": 4, "max_labels_per_call": 15}
GPU_FALLBACK = {"chunk_window": 3000, "chunk_overlap": 150, "batch_size": 16, "max_labels_per_call": 25}


@dataclass
class TuningResult:
    """Auto-tuned hyperparameters for one (device, model) combo."""
    chunk_window: int
    chunk_overlap: int
    batch_size: int
    max_labels_per_call: int
    # Provenance — useful for debugging and cache validation
    device_name: str = ""
    device_kind: str = ""
    model_name: str = ""
    torch_version: str = ""
    probed_at: float = 0.0
    probe_seconds: float = 0.0
    measured_vram_peak_gb: float = 0.0
    measured_ms_per_chunk: float = 0.0
    # Hard model limits (introspected, not tuned) — persisted so we can detect drift
    model_max_seq_tokens: int = 0
    model_max_labels: int = 0
    chars_per_token_p99: float = 0.0
    # Probe-logic version. Cache entries from older versions are ignored.
    tuning_version: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Cache I/O ───────────────────────────────────────────────────────────


def _load_cache() -> Dict[str, Any]:
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"[autotune] cache read failed ({e}); will re-probe")
        return {}


def _coerce_yaml_safe(obj: Any) -> Any:
    """Recursively coerce values to yaml-representable primitives.

    PyYAML's safe_dump rejects custom types like torch.version.TorchVersion.
    Coerce anything that isn't a primitive to its str() form, defensively.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _coerce_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_yaml_safe(v) for v in obj]
    return str(obj)


def _save_cache(cache: Dict[str, Any]) -> None:
    try:
        safe = _coerce_yaml_safe(cache)
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(safe, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp, CACHE_PATH)
    except Exception as e:
        logger.warning(f"[autotune] cache write failed ({e}); continuing without cache")


# ── Model introspection ─────────────────────────────────────────────────


@dataclass
class ModelLimits:
    """Hard limits exposed by the GLiNER model itself.

    These are not tunable. They constrain what the auto-tuner can probe —
    e.g., probing batch_size at a chunk window that overflows max_seq_tokens
    would silently truncate input and produce wrong detections.
    """
    max_seq_tokens: int            # transformer max sequence length (tokens)
    max_labels_per_call: int       # model's hard cap on labels per forward pass
    max_entity_width_tokens: int   # max span width in tokens
    chars_per_token_p99: float     # measured upper bound on chars/token

    @property
    def safe_chunk_chars(self) -> int:
        """Largest char-window guaranteed to fit under max_seq_tokens."""
        usable_tokens = max(64, self.max_seq_tokens - TOKEN_HEADROOM)
        return int(usable_tokens * self.chars_per_token_p99 * TOKEN_BUDGET_SAFETY)


def _measure_chars_per_token(model) -> float:
    """Measure a representative chars-per-token ratio for typical PII content.

    Strategy: tokenize several samples spanning realistic content styles,
    sort the ratios, and take the **median**. This is robust to one pathological
    outlier (e.g., a sample of pure digits) without being so conservative that
    we waste capacity on real text.

    The chunker hard-caps every chunk at `window_size` chars regardless, so
    even if a row contains a denser-than-median region the chunk count just
    grows; nothing overflows the model's max_seq_tokens because the safety
    factor (TOKEN_BUDGET_SAFETY) absorbs the variance.
    """
    samples = [
        # Natural prose — best case, ~4-5 chars/token on most BPE tokenizers
        "The customer called this morning about a problem with their recent statement. "
        "They said they were not happy with the resolution and wanted to escalate.",
        # Mixed prose with light PII — typical of conversation transcripts
        "Customer John Smith called from 555-123-4567 about account 1234-5678-9012. "
        "His email is john.smith@example.com and his date of birth is 1985-03-15.",
        # Mixed structured PII — typical of form/record data
        "Name: Jane Doe. Address: 100 Main Street, Springfield, IL 62701. "
        "Phone: 217-555-0199. SSN: 123-45-6789. Account: 4111111111111234.",
        # Token-dense alphanumeric IDs — represents the worst realistic case
        "Account A8F2K9L3, transaction TXN-20260101-998877, ref REF/Q1/00451/AX.",
    ]
    tokenizer = None
    for attr in ("tokenizer", "data_processor"):
        obj = getattr(model, attr, None)
        if obj is not None:
            t = getattr(obj, "tokenizer", None) or getattr(obj, "transformer_tokenizer", None)
            if t is not None:
                tokenizer = t
                break
    if tokenizer is None:
        logger.warning("[autotune] tokenizer not introspectable; assuming 3.0 chars/token")
        return 3.0

    ratios: List[float] = []
    for s in samples:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                ratios.append(len(s) / len(ids))
        except Exception:
            continue

    if not ratios:
        return 3.0
    ratios.sort()
    # Median — robust to a single adversarial outlier in either direction.
    # The TOKEN_BUDGET_SAFETY multiplier (in safe_chunk_chars) handles variance
    # within real rows.
    median = ratios[len(ratios) // 2]
    logger.debug(f"[autotune] chars/token samples sorted={[f'{r:.2f}' for r in ratios]} median={median:.2f}")
    return median


def _read_model_limits(model) -> ModelLimits:
    """Pull hard limits from model.config and the live tokenizer.

    Falls back to safe defaults if the attributes are missing (older GLiNER
    versions, custom checkpoints, etc.).
    """
    cfg = getattr(model, "config", None)
    max_len = getattr(cfg, "max_len", None) if cfg is not None else None
    max_types = getattr(cfg, "max_types", None) if cfg is not None else None
    max_width = getattr(cfg, "max_width", None) if cfg is not None else None

    # Defensive fallbacks — never crash startup over missing attributes
    if not isinstance(max_len, int) or max_len <= 0:
        max_len = 512
        logger.warning(f"[autotune] model.config.max_len missing; defaulting to {max_len}")
    if not isinstance(max_types, int) or max_types <= 0:
        max_types = 25
        logger.warning(f"[autotune] model.config.max_types missing; defaulting to {max_types}")
    if not isinstance(max_width, int) or max_width <= 0:
        max_width = 100

    cpt = _measure_chars_per_token(model)
    limits = ModelLimits(
        max_seq_tokens=max_len,
        max_labels_per_call=max_types,
        max_entity_width_tokens=max_width,
        chars_per_token_p99=cpt,
    )
    logger.info(
        f"[autotune] model limits: max_tokens={max_len}, max_labels={max_types}, "
        f"chars/token (worst-case)={cpt:.2f} → safe chunk chars={limits.safe_chunk_chars}"
    )
    return limits


# ── Probing primitives ──────────────────────────────────────────────────


def _make_probe_text(n_chars: int) -> str:
    """Build a synthetic probe document of approximately n_chars characters.

    Uses realistic-looking sentences with mixed structure so the tokenizer
    behaves like it would on real input. Deterministic — same n_chars → same text.
    """
    sentence = (
        "The customer John Smith called from 555-123-4567 about account 1234-5678-9012. "
        "His email is john.smith@example.com and his SSN is 123-45-6789. "
        "Address: 100 Main Street, Springfield, IL 62701. DOB 1985-03-15. "
    )
    out: List[str] = []
    total = 0
    while total < n_chars:
        out.append(sentence)
        total += len(sentence)
    return "".join(out)[:n_chars]


def _chunk_for_probe(text: str, window: int, overlap: int) -> List[str]:
    """Mirror gliner_detector chunking, simplified — used only by the probe."""
    if window <= overlap:
        overlap = max(0, window // 10)
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + window, n)
        if end < n:
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        chunks.append(text[start:end])
        nxt = end - overlap
        if nxt <= start:
            nxt = end
        start = nxt
    return chunks


def _vram_peak_gb() -> float:
    if torch is None or not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        return 0.0


def _vram_reset() -> None:
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        except Exception:
            pass


def _sync() -> None:
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _time_call(model, texts: List[str], labels: List[str], batch_size: int,
               threshold: float = 0.2, n_runs: int = 2) -> float:
    """Median wall-time for one batch_predict_entities call, in seconds."""
    assert torch is not None
    # Warmup
    with torch.inference_mode():
        _ = model.batch_predict_entities(texts, labels, threshold=threshold, batch_size=batch_size)
    _sync()
    times: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model.batch_predict_entities(texts, labels, threshold=threshold, batch_size=batch_size)
        _sync()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


# ── Probe stages ────────────────────────────────────────────────────────


def _probe_chunk_window(model, base_labels: List[str], device: DeviceInfo,
                         limits: "ModelLimits") -> int:
    """Find the chunk window with best per-token throughput.

    Hard upper bound = limits.safe_chunk_chars (the largest char-window that
    is guaranteed to fit under the model's max_seq_tokens). Probing larger
    values would silently truncate input — wrong, not just slow.
    """
    safe_max = limits.safe_chunk_chars
    raw_candidates = [1000, 1500, 2000, 2500, 3000, 4000]
    candidates = [c for c in raw_candidates if c <= safe_max]
    # Always include the safe_max value itself so we don't leave capacity on the floor
    if safe_max not in candidates and safe_max >= 512:
        candidates.append(safe_max)
    if not candidates:
        candidates = [max(256, safe_max)]
    # Sort ascending so the "prefer larger within tolerance" logic below works
    candidates = sorted(set(candidates))
    # On smaller GPUs, drop the largest options to avoid wasted probe time
    if device.kind == "cuda" and device.total_vram_gb < 12:
        candidates = [c for c in candidates if c <= min(2500, safe_max)] or [min(1500, safe_max)]

    best_window = candidates[0]
    best_ms_per_token = float("inf")

    # Probe with enough text to amortise kernel-launch + constant overhead.
    # Earlier (window * 4) was too small: at ~4 chunks the per-char measurement
    # was dominated by constant costs and looked flat across window sizes, which
    # caused the tolerance loop below to bias toward the largest window. With
    # ~16 chunks the O(n²) attention cost in deberta-v3 dominates and the
    # probe correctly favours smaller windows for this model family.
    PROBE_CHUNKS = 16

    for window in candidates:
        text = _make_probe_text(window * PROBE_CHUNKS)
        chunks = _chunk_for_probe(text, window=window, overlap=window // 20)
        if not chunks:
            continue
        try:
            _vram_reset()
            t = _time_call(model, chunks, base_labels[:25], batch_size=min(8, len(chunks)))
            total_chars = sum(len(c) for c in chunks)
            ms_per_token = (t * 1000) / max(1, total_chars)
            logger.info(
                f"[autotune] window={window:>4} chunks={len(chunks):>3} "
                f"t={t*1000:>6.0f}ms ms/char={ms_per_token:.4f}"
            )
            # Pick the SMALLEST window within 5% of the best so far. Smaller
            # windows are preferred when ties occur because (a) deberta-v3
            # attention is O(n²) so smaller windows expose more parallelism
            # under batched execution, and (b) smaller windows make the per-row
            # deadline accounting finer-grained, helping p99 control downstream.
            if ms_per_token < best_ms_per_token * (1 - THROUGHPUT_GAIN_THRESHOLD):
                # Strictly better — adopt unconditionally
                best_ms_per_token = ms_per_token
                best_window = window
            elif ms_per_token < best_ms_per_token * (1 + THROUGHPUT_GAIN_THRESHOLD):
                # Within tolerance — keep the smaller window (i.e. don't update)
                pass
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.info(f"[autotune] window={window} OOM; stopping window probe")
                _vram_reset()
                break
            raise
    return best_window


def _probe_batch_size(model, base_labels: List[str], window: int, device: DeviceInfo) -> tuple[int, float]:
    """Find the smallest batch size that achieves peak throughput.

    Returns (batch_size, peak_vram_gb).

    Why "smallest at peak throughput", not "largest under VRAM budget":
      For a large transformer (e.g. deberta-v3-large) on a strong GPU, the
      model saturates compute at small batch sizes (often bs=2..8). Larger
      batches add per-call latency and VRAM cost without raising throughput.
      Picking the smallest bs at peak throughput minimises p99 latency per
      forward pass, which directly helps the per-row deadline budget.

    Algorithm:
      1. Double bs from 1 upward until OOM or until we hit MAX_BS_TO_PROBE.
      2. Track peak throughput and the bs at which it was first achieved.
      3. Stop when we see two consecutive non-improvements (>5% drop or no gain)
         after having seen peak throughput.
      4. Return the smallest bs that was within 5% of peak throughput.
    """
    if device.kind != "cuda":
        # Non-CUDA: batch size is bounded by RAM, not VRAM. Pick a sensible default.
        return (8 if device.kind == "mps" else 4, 0.0)

    # Generate enough chunks to actually exercise the largest bs we want to probe.
    # Without enough chunks, the bs probe terminates early because
    # `bs <= len(chunks)` clips the loop. We probe up to bs=128, so we need ≥256
    # chunks (probe each bs against bs*2 chunks worth of work for stable timing).
    MAX_BS_TO_PROBE = 128
    n_chunks_needed = MAX_BS_TO_PROBE * 2
    text = _make_probe_text(window * (n_chunks_needed + 4))
    chunks = _chunk_for_probe(text, window=window, overlap=window // 20)
    if not chunks:
        return (8, 0.0)
    logger.debug(f"[autotune] batch probe has {len(chunks)} chunks available")

    ceiling_bytes = device.total_vram_bytes * (VRAM_TARGET_FRACTION + VRAM_SAFETY_MARGIN)

    # Record (bs, throughput, peak_vram_gb) for each step
    measurements: List[tuple[int, float, float]] = []
    bs = 1
    consecutive_non_improvements = 0
    peak_throughput = 0.0

    while bs <= MAX_BS_TO_PROBE and bs <= len(chunks):
        try:
            _vram_reset()
            # Time on a slice of chunks proportional to bs so each step is comparable.
            # Also cap upper-end probes — at bs=128 a single call can take 10+ seconds,
            # which inflates total probe time. Use bs*2 chunks (one full padded batch
            # plus headroom) instead of always using bs*2 for the smallest sizes too.
            n_for_step = min(len(chunks), max(bs * 2, 8))
            t = _time_call(model, chunks[:n_for_step], base_labels[:25], batch_size=bs, n_runs=2)
            peak = _vram_peak_gb()
            throughput = n_for_step / t
            measurements.append((bs, throughput, peak))
            logger.info(
                f"[autotune] bs={bs:>3} t={t*1000:>6.0f}ms peak_vram={peak:>5.2f}GB "
                f"throughput={throughput:>6.1f} ch/s"
            )

            if peak * 1e9 > ceiling_bytes:
                logger.info(f"[autotune] bs={bs} crossed VRAM ceiling; stopping")
                break

            # Track peak throughput and stop when we have two consecutive non-improvements
            if throughput > peak_throughput * (1 + THROUGHPUT_GAIN_THRESHOLD):
                peak_throughput = throughput
                consecutive_non_improvements = 0
            else:
                consecutive_non_improvements += 1
                if consecutive_non_improvements >= 2 and peak_throughput > 0:
                    logger.info(
                        f"[autotune] bs={bs} no improvement for 2 steps; throughput "
                        f"saturated at ~{peak_throughput:.1f} ch/s"
                    )
                    break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.info(f"[autotune] bs={bs} OOM; stopping batch probe")
                _vram_reset()
                break
            raise
        bs *= 2

    if not measurements:
        return (8, 0.0)

    # Pick the SMALLEST bs that's within 5% of peak throughput.
    # This minimises per-call latency (good for p99) and VRAM use without
    # giving up any throughput.
    peak_thr = max(m[1] for m in measurements)
    threshold_thr = peak_thr * (1 - THROUGHPUT_GAIN_THRESHOLD)
    chosen = next((m for m in measurements if m[1] >= threshold_thr), measurements[-1])
    chosen_bs, chosen_thr, chosen_peak = chosen
    logger.info(
        f"[autotune] selected bs={chosen_bs} (throughput={chosen_thr:.1f} ch/s, "
        f"within 5% of peak {peak_thr:.1f}, vram={chosen_peak:.2f}GB)"
    )
    return (chosen_bs, chosen_peak)


def _probe_labels_per_call(model, base_labels: List[str], window: int, batch_size: int,
                            limits: "ModelLimits") -> int:
    """Find the labels-per-call value with the lowest ms-per-(chunk × label) cost.

    Hard upper bound = limits.max_labels_per_call (the model's own cap from
    config.max_types). Probing above this is wasted — the model will internally
    truncate the label list.
    """
    hard_cap = limits.max_labels_per_call
    if torch is None or not torch.cuda.is_available():
        return min(15, hard_cap)

    raw_candidates = [10, 15, 20, 25, 30]
    candidates = [c for c in raw_candidates if c <= hard_cap]
    if not candidates:
        candidates = [hard_cap]
    text = _make_probe_text(window * 8)
    chunks = _chunk_for_probe(text, window=window, overlap=window // 20)
    if not chunks:
        return 25

    best_n = 25
    best_cost = float("inf")
    pool = base_labels * 10  # ensure we have enough to slice from

    for n in candidates:
        if n > len(pool):
            continue
        try:
            _vram_reset()
            t = _time_call(model, chunks, pool[:n], batch_size=min(batch_size, len(chunks)))
            cost = (t * 1000) / (len(chunks) * n)  # ms per (chunk × label)
            logger.debug(f"[autotune] labels={n} t={t*1000:.0f}ms cost={cost:.4f}")
            if cost < best_cost:
                best_cost = cost
                best_n = n
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.info(f"[autotune] labels={n} OOM; stopping label probe")
                _vram_reset()
                break
            raise
    return best_n


# ── Public API ──────────────────────────────────────────────────────────


# Default probe label set — generic PII labels that exercise the same code paths
# as the real taxonomy without requiring it. The probe is about *cost*, not accuracy.
_PROBE_LABELS = [
    "person", "first name", "last name", "email", "phone number",
    "date of birth", "social security number", "credit card number",
    "address", "street address", "city", "state", "zip code",
    "bank account number", "iban", "routing number", "ip address",
    "license plate", "driver license", "passport number", "company",
    "job title", "url", "date", "time", "currency amount",
    "username", "password", "api key", "medical record number",
    "diagnosis", "medication", "patient name", "doctor name",
]


def autotune(model, model_name: str, force_reprobe: bool = False) -> TuningResult:
    """Discover optimal hyperparameters for the given (model, device) combo.

    Reads from cache if available. Otherwise probes the live model and persists.

    Args:
        model: A loaded GLiNER model on its target device.
        model_name: Identifier of the model (for cache keying).
        force_reprobe: If True, ignore cache and re-probe.

    Returns:
        TuningResult with auto-discovered chunk_window, batch_size, max_labels_per_call.
    """
    dm = get_device_manager()
    cache_key = dm.cache_key(model_name)
    cache = _load_cache()

    if not force_reprobe and cache_key in cache:
        cached = cache[cache_key]
        try:
            result = TuningResult(**cached)
            if result.tuning_version != TUNING_VERSION:
                logger.info(
                    f"[autotune] cache entry for {model_name} is from probe v{result.tuning_version}, "
                    f"current is v{TUNING_VERSION}; re-probing"
                )
            else:
                logger.info(
                    f"[autotune] cache hit (v{TUNING_VERSION}) for {dm.primary.name} / {model_name}: "
                    f"window={result.chunk_window}, batch={result.batch_size}, "
                    f"labels/call={result.max_labels_per_call}"
                )
                return result
        except Exception as e:
            logger.warning(f"[autotune] cache entry malformed ({e}); re-probing")

    # CPU shortcut — probing is expensive and the answers are predictable
    if dm.primary.kind == "cpu":
        logger.warning("[autotune] CPU device; using conservative fallback values without probing")
        result = TuningResult(
            **CPU_FALLBACK,
            device_name=dm.primary.name,
            device_kind="cpu",
            model_name=model_name,
            torch_version=str(getattr(torch, "__version__", "unknown")) if torch else "none",
            probed_at=time.time(),
            tuning_version=TUNING_VERSION,
            notes=["cpu fallback; no probe performed"],
        )
        cache[cache_key] = result.to_dict()
        _save_cache(cache)
        return result

    logger.info(f"[autotune] probing {dm.primary.name} / {model_name} (one-time, ~30-90s)...")
    t_start = time.perf_counter()

    try:
        # Step 0: introspect hard model limits BEFORE probing — these bound everything else
        limits = _read_model_limits(model)

        window = _probe_chunk_window(model, _PROBE_LABELS, dm.primary, limits)
        batch_size, peak_vram = _probe_batch_size(model, _PROBE_LABELS, window, dm.primary)
        labels_per_call = _probe_labels_per_call(model, _PROBE_LABELS, window, batch_size, limits)

        # Quick post-measurement on the chosen config to record actual ms/chunk
        _vram_reset()
        text = _make_probe_text(window * 8)
        chunks = _chunk_for_probe(text, window=window, overlap=window // 20)
        t_call = _time_call(
            model, chunks, _PROBE_LABELS[:labels_per_call],
            batch_size=min(batch_size, len(chunks))
        )
        ms_per_chunk = (t_call * 1000) / max(1, len(chunks))

        result = TuningResult(
            chunk_window=window,
            chunk_overlap=max(50, window // 20),  # 5% overlap, derived from window
            batch_size=batch_size,
            max_labels_per_call=labels_per_call,
            device_name=dm.primary.name,
            device_kind=dm.primary.kind,
            model_name=model_name,
            torch_version=str(getattr(torch, "__version__", "unknown")) if torch else "none",
            probed_at=time.time(),
            probe_seconds=time.perf_counter() - t_start,
            measured_vram_peak_gb=peak_vram,
            measured_ms_per_chunk=ms_per_chunk,
            model_max_seq_tokens=limits.max_seq_tokens,
            model_max_labels=limits.max_labels_per_call,
            chars_per_token_p99=limits.chars_per_token_p99,
            tuning_version=TUNING_VERSION,
            notes=[],
        )
    except Exception as e:
        logger.error(f"[autotune] probe failed ({e}); using GPU fallback")
        result = TuningResult(
            **GPU_FALLBACK,
            device_name=dm.primary.name,
            device_kind=dm.primary.kind,
            model_name=model_name,
            torch_version=str(getattr(torch, "__version__", "unknown")) if torch else "none",
            probed_at=time.time(),
            probe_seconds=time.perf_counter() - t_start,
            tuning_version=TUNING_VERSION,
            notes=[f"probe failed: {e}"],
        )

    logger.info(
        f"[autotune] done in {result.probe_seconds:.1f}s — "
        f"window={result.chunk_window}, batch={result.batch_size}, "
        f"labels/call={result.max_labels_per_call}, "
        f"vram_peak={result.measured_vram_peak_gb:.2f}GB, "
        f"ms/chunk={result.measured_ms_per_chunk:.0f}"
    )

    cache[cache_key] = result.to_dict()
    _save_cache(cache)
    return result
