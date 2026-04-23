"""
Baseline benchmark: run the current HybridPIIEngine.detect() on each row
of tests/fake_transcripts.jsonl and report per-row latency and total time.

This is the "before" measurement. Run this BEFORE any GPU/optimization
changes so we have a baseline to compare against.

Usage:
    python tests/benchmark_baseline.py                    # canonical mode (default)
    python tests/benchmark_baseline.py --ultra-canonical  # ~30 labels, ~6x faster
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Surface autotune + GLiNER perf log lines (default Python logging is WARNING+)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Make `app` importable when running from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.engine import HybridPIIEngine  # noqa: E402

ROWS_PATH = ROOT / "tests" / "fake_transcripts.jsonl"


def load_rows():
    rows = []
    with ROWS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="PII detection benchmark")
    parser.add_argument(
        "--ultra-canonical",
        action="store_true",
        help="Enable ultra-canonical mode (~30 labels, ~6x faster, less label specificity)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-stage timing inside detect() and print a breakdown",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Load GLiNER as ONNX (default: onnx/model_quint8.onnx) via onnxruntime CUDA provider",
    )
    parser.add_argument(
        "--onnx-file",
        default="onnx/model_quint8.onnx",
        help="Which ONNX file inside the HF repo to load (e.g. 'onnx/model.onnx' for FP32)",
    )
    parser.add_argument(
        "--model",
        default="knowledgator/gliner-pii-large-v1.0",
        help="GLiNER model id (e.g. knowledgator/gliner-pii-base-v1.0 for the smaller variant)",
    )
    args = parser.parse_args()

    print("=" * 70)
    label = "ULTRA-CANONICAL" if args.ultra_canonical else "CANONICAL (default)"
    runtime = "ONNX-INT8" if args.onnx else "torch-fp16"
    print(f"BASELINE BENCHMARK — HybridPIIEngine, mode: {label}, runtime: {runtime}")
    print(f"  model: {args.model}")
    print("=" * 70)

    rows = load_rows()
    print(f"loaded {len(rows)} rows from {ROWS_PATH.name}")
    for r in rows:
        print(f"  row {r['id']}: {r['char_count']:,} chars")
    print()

    print("initializing HybridPIIEngine (this loads GLiNER + all detectors)...")
    t0 = time.perf_counter()
    engine = HybridPIIEngine(
        gliner_model_name=args.model,
        use_ultra_canonical=args.ultra_canonical,
        use_onnx=args.onnx,
        onnx_model_file=args.onnx_file,
    )
    if args.profile:
        engine.profile_stages = True
        # Reset so warmup row's timings don't pollute the timed runs
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"  init took {init_ms:,.0f} ms\n")

    # Report device GLiNER ended up on
    try:
        device = getattr(engine.gliner_detector.model, "device", "unknown")
        print(f"GLiNER device: {device}")
    except Exception:
        print("GLiNER device: (could not introspect)")
    print()

    # Warmup in bulk mode so the LSTM auto-retrain doesn't contaminate the timing.
    # The first detect() call also triggers GLiNER's CUDA kernel autotuning.
    print("warmup pass (row 1, bulk mode, not counted)...")
    t = time.perf_counter()
    engine._bulk_mode = True  # type: ignore[attr-defined]
    try:
        _ = engine.detect(rows[0]["text"])
    finally:
        engine._bulk_mode = False  # type: ignore[attr-defined]
    warm_ms = (time.perf_counter() - t) * 1000
    print(f"  warmup took {warm_ms:,.0f} ms\n")

    # Reset stage profile so warmup row's timings don't pollute the timed runs
    if args.profile:
        engine.reset_stage_profile()

    print("=" * 70)
    print("TIMED RUNS — bulk mode (detect_batch, defers LSTM retrain to end-of-batch)")
    print("=" * 70)

    detection_counts = []
    total_t = time.perf_counter()
    engine._bulk_mode = True  # type: ignore[attr-defined]
    engine._bulk_rows_seen = 0  # type: ignore[attr-defined]
    per_row = []
    per_row_meta = []  # (profile, char_count, detections, elapsed_ms)
    try:
        for r in rows:
            t = time.perf_counter()
            detections = engine.detect(r["text"])
            elapsed_ms = (time.perf_counter() - t) * 1000
            per_row.append(elapsed_ms)
            detection_counts.append(len(detections))
            profile = r.get("profile", "unknown")
            per_row_meta.append((profile, r["char_count"], len(detections), elapsed_ms))
            print(
                f"  row {r['id']:>2} [{profile:>6}]: {elapsed_ms:>9,.0f} ms   "
                f"({r['char_count']:,} chars, {len(detections)} detections)"
            )
    finally:
        t_fin = time.perf_counter()
        finalize_summary = engine.finalize_bulk()
        finalize_ms = (time.perf_counter() - t_fin) * 1000
        engine._bulk_mode = False  # type: ignore[attr-defined]
    total_ms = (time.perf_counter() - total_t) * 1000

    print()
    print(f"  finalize_bulk: {finalize_ms:>10,.0f} ms ({finalize_summary})")

    # Sort per_row for percentile calculations
    def pct_of(values, p):
        if not values:
            return 0.0
        s = sorted(values)
        idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
        return s[idx]

    print()
    print("=" * 70)
    print("SUMMARY — overall")
    print("=" * 70)
    print(f"  rows processed:     {len(rows)}")
    print(f"  total time:         {total_ms:>12,.0f} ms  ({total_ms / 1000:.2f} s)")
    print(f"  avg per row:        {sum(per_row) / len(per_row):>12,.0f} ms")
    print(f"  p50 per row:        {pct_of(per_row, 0.50):>12,.0f} ms")
    print(f"  p95 per row:        {pct_of(per_row, 0.95):>12,.0f} ms")
    print(f"  p99 per row:        {pct_of(per_row, 0.99):>12,.0f} ms")
    print(f"  min per row:        {min(per_row):>12,.0f} ms")
    print(f"  max per row:        {max(per_row):>12,.0f} ms")
    print(f"  avg detections:     {sum(detection_counts) / len(detection_counts):>12,.1f}")

    # Per-profile breakdown — this is where we see if candidate windows work
    profiles_seen = sorted(set(m[0] for m in per_row_meta))
    if len(profiles_seen) > 1:
        print()
        print("=" * 70)
        print("SUMMARY — per density profile")
        print("=" * 70)
        print(f"  {'profile':<8} {'rows':>5} {'avg ms':>10} {'p99 ms':>10} {'avg dets':>10} {'dets/1k chars':>14}")
        for prof in profiles_seen:
            sub = [m for m in per_row_meta if m[0] == prof]
            sub_times = [m[3] for m in sub]
            sub_dets = [m[2] for m in sub]
            sub_chars = sum(m[1] for m in sub)
            avg_ms = sum(sub_times) / len(sub_times)
            p99_ms = pct_of(sub_times, 0.99)
            avg_d = sum(sub_dets) / len(sub_dets)
            density = sum(sub_dets) / max(sub_chars, 1) * 1000
            print(f"  {prof:<8} {len(sub):>5} {avg_ms:>10,.0f} {p99_ms:>10,.0f} {avg_d:>10,.1f} {density:>14,.2f}")
    # GLiNER chunk-throughput diagnostics — this is the Phase 4 success metric
    try:
        gd = engine.gliner_detector
        chunks_total = getattr(gd, "_chunks_processed_total", 0)
        calls_total = getattr(gd, "_calls_total", 0)
        if calls_total:
            print(f"  GLiNER chunks/call: {chunks_total / calls_total:>12,.1f}")
            print(f"  GLiNER calls total: {calls_total:>12,d}")
            print(f"  GLiNER chunks tot:  {chunks_total:>12,d}")
    except Exception:
        pass

    # Per-stage profiling breakdown — only when --profile is set
    if args.profile:
        prof = engine.get_stage_profile()
        if prof["stages"]:
            n_rows = len(per_row)
            print()
            print("=" * 70)
            print("PER-STAGE PROFILE — average over timed runs")
            print("=" * 70)
            print(f"  {'stage':<24} {'total ms':>10} {'per row':>10} {'pct':>7}")
            print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*7}")
            for s in prof["stages"]:
                per_row_ms = s["total_ms"] / max(n_rows, 1)
                print(
                    f"  {s['stage']:<24} {s['total_ms']:>10,.0f} "
                    f"{per_row_ms:>10,.1f} {s['pct']:>6.1f}%"
                )
            print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*7}")
            print(f"  {'TOTAL':<24} {prof['total_ms']:>10,.0f} "
                  f"{prof['total_ms']/max(n_rows,1):>10,.1f}")
    print()
    # Projection: in a real 200k bulk run, finalize_bulk happens ONCE for the
    # whole batch, so its cost is amortised over 200k rows (not 5). The right
    # way to project: per-row average from the timed loop, plus a small amortised
    # finalize term. Note that finalize cost grows sub-linearly with batch size
    # (mainly one LSTM retrain pass + a few YAML writes), so this projection
    # is a lower bound; the true 200k finalize will be slightly larger but still
    # negligible amortised.
    per_row_avg_s = sum(per_row) / len(per_row) / 1000
    finalize_s = finalize_ms / 1000
    proj_per_row_s = per_row_avg_s + (finalize_s / 200_000)
    proj_total_s = proj_per_row_s * 200_000
    print("Projection to business workload (200,000 rows, single bulk batch):")
    print(f"  per-row work:        {per_row_avg_s:6.2f} s")
    print(f"  finalize amortised:  {finalize_s / 200_000 * 1000:6.2f} ms (over 200k rows)")
    print(f"  effective per-row:   {proj_per_row_s:6.2f} s")
    print(f"  total:               {proj_total_s:>12,.0f} s = {proj_total_s/3600:.1f} h")
    print()
    print("Target: avg < 1000 ms/row")
    cur = sum(per_row) / len(per_row)
    if cur < 1000:
        print(f"  STATUS: PASSING ({cur:,.0f} ms/row)")
    else:
        print(f"  STATUS: NOT YET — current {cur:,.0f} ms/row, need {cur / 1000:.1f}x speedup")


if __name__ == "__main__":
    main()
