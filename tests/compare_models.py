"""Equivalence harness: compare two GLiNER models on the same input.

Why this exists:
  Phase 8 swaps `gliner-pii-large-v1.0` for `gliner-pii-base-v1.0`. We need to
  know whether the smaller model produces "close enough" detections before
  shipping it. This script runs both models against the same fake transcripts
  and reports both a loose and a strict comparison.

Two views of the same diff:

  LOOSE — per-row, per-label count drift. One row of the table per label.
  Easy to read; tells you "the base model finds 12% fewer EMAIL_ADDRESS
  detections" without drowning you in offsets. Good for go/no-go calls.

  STRICT — per-detection (label, start, end) set diff. Computes Jaccard
  similarity, precision, recall, and F1 (treating model A as ground truth)
  per row and overall. Each per-row diff is dumped to a YAML file you can
  grep through later when investigating specific divergences.

Usage:
    python tests/compare_models.py \
        --model-a knowledgator/gliner-pii-large-v1.0 \
        --model-b knowledgator/gliner-pii-base-v1.0 \
        --diff-out tests/model_diff.yaml

Optional flags:
    --ultra-canonical    pass through to both engines (recommended for
                         apples-to-apples since the production runtime uses
                         ultra-canonical)
    --use-onnx           use the ONNX path for both models (off by default —
                         torch-fp16 is faster on this hardware)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.engine import HybridPIIEngine  # noqa: E402

ROWS_PATH = ROOT / "tests" / "fake_transcripts.jsonl"


def load_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with ROWS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def detect_with_model(
    model_name: str,
    rows: List[Dict[str, Any]],
    use_ultra_canonical: bool,
    use_onnx: bool,
) -> Tuple[List[List[Dict[str, Any]]], float]:
    """Load engine for one model and run all rows in bulk mode.

    Returns (per_row_detections, total_seconds). Each per-row detection is
    serialised to a plain dict so we don't keep the Detection objects across
    engine teardown — we may load a different model after this and want the
    first model's results to remain valid Python data.
    """
    logging.info(f"=== loading {model_name} ===")
    t0 = time.perf_counter()
    engine = HybridPIIEngine(
        gliner_model_name=model_name,
        use_ultra_canonical=use_ultra_canonical,
        use_onnx=use_onnx,
        # Inference-only: we don't want this comparison run to write into the
        # accumulator/trainer/active learner YAMLs. The benchmark already
        # exercises the learning pipeline; this script should be a pure read.
        inference_only=True,
    )
    init_s = time.perf_counter() - t0
    logging.info(f"init took {init_s:.1f}s")

    out: List[List[Dict[str, Any]]] = []
    t0 = time.perf_counter()
    engine._bulk_mode = True  # type: ignore[attr-defined]
    try:
        for r in rows:
            dets = engine.detect(r["text"])
            out.append(
                [
                    {
                        "label": d.label,
                        "start": d.start,
                        "end": d.end,
                        "text": d.text,
                        "score": round(float(d.score), 4),
                        "source": d.source,
                    }
                    for d in dets
                ]
            )
    finally:
        try:
            engine.finalize_bulk()
        finally:
            engine._bulk_mode = False  # type: ignore[attr-defined]
    total_s = time.perf_counter() - t0
    logging.info(f"detect total: {total_s:.1f}s ({total_s / len(rows) * 1000:.0f} ms/row)")
    return out, total_s


def loose_diff(
    a_rows: List[List[Dict[str, Any]]],
    b_rows: List[List[Dict[str, Any]]],
) -> Dict[str, Dict[str, int]]:
    """Per-label count totals across all rows for both models.

    Returns: {label: {"a": count, "b": count, "delta": b - a, "pct": ...}}
    """
    a_counts: Dict[str, int] = defaultdict(int)
    b_counts: Dict[str, int] = defaultdict(int)
    for row in a_rows:
        for d in row:
            a_counts[d["label"]] += 1
    for row in b_rows:
        for d in row:
            b_counts[d["label"]] += 1

    all_labels = sorted(set(a_counts) | set(b_counts))
    summary: Dict[str, Dict[str, Any]] = {}
    for lbl in all_labels:
        a = a_counts[lbl]
        b = b_counts[lbl]
        delta = b - a
        pct = (delta / a * 100) if a else float("inf") if b else 0.0
        summary[lbl] = {"a": a, "b": b, "delta": delta, "pct": pct}
    return summary


def _spans(row: List[Dict[str, Any]]) -> Set[Tuple[str, int, int]]:
    """Convert a row's detections to a set of (label, start, end) tuples."""
    return {(d["label"], d["start"], d["end"]) for d in row}


def strict_diff(
    a_rows: List[List[Dict[str, Any]]],
    b_rows: List[List[Dict[str, Any]]],
    rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Per-detection set diff treating A as the reference (ground truth).

    Returns:
      summary: aggregate precision/recall/F1/Jaccard across all rows.
      per_row: list of dicts with per-row tp/fp/fn/jaccard and the actual
               (label, start, end, text) tuples for missed and extra detections,
               suitable for dumping to YAML for later grep-based audit.
    """
    tp_total = 0
    fp_total = 0
    fn_total = 0
    union_total = 0

    per_row: List[Dict[str, Any]] = []
    for idx, (a_row, b_row, meta) in enumerate(zip(a_rows, b_rows, rows)):
        a_set = _spans(a_row)
        b_set = _spans(b_row)
        tp = len(a_set & b_set)
        fp = len(b_set - a_set)
        fn = len(a_set - b_set)
        union = len(a_set | b_set)
        jaccard = (tp / union) if union else 1.0

        tp_total += tp
        fp_total += fp
        fn_total += fn
        union_total += union

        a_index = {(d["label"], d["start"], d["end"]): d for d in a_row}
        b_index = {(d["label"], d["start"], d["end"]): d for d in b_row}

        missed = [
            {
                "label": k[0],
                "start": k[1],
                "end": k[2],
                "text": a_index[k]["text"],
                "score": a_index[k]["score"],
                "source": a_index[k]["source"],
            }
            for k in sorted(a_set - b_set)
        ]
        extra = [
            {
                "label": k[0],
                "start": k[1],
                "end": k[2],
                "text": b_index[k]["text"],
                "score": b_index[k]["score"],
                "source": b_index[k]["source"],
            }
            for k in sorted(b_set - a_set)
        ]

        per_row.append(
            {
                "row_id": meta.get("id", idx),
                "profile": meta.get("profile", "unknown"),
                "char_count": meta.get("char_count", len(meta.get("text", ""))),
                "a_count": len(a_set),
                "b_count": len(b_set),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "jaccard": round(jaccard, 4),
                "missed_by_b": missed,
                "extra_in_b": extra,
            }
        )

    precision = (tp_total / (tp_total + fp_total)) if (tp_total + fp_total) else 0.0
    recall = (tp_total / (tp_total + fn_total)) if (tp_total + fn_total) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    jaccard_all = (tp_total / union_total) if union_total else 1.0

    summary = {
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "jaccard": round(jaccard_all, 4),
    }
    return summary, per_row


def print_loose_table(summary: Dict[str, Dict[str, Any]], a_name: str, b_name: str) -> None:
    print("=" * 80)
    print(f"LOOSE DIFF — per-label count totals across all rows")
    print(f"  A = {a_name}")
    print(f"  B = {b_name}")
    print("=" * 80)
    print(f"  {'label':<32} {'A':>8} {'B':>8} {'Δ':>8} {'pct':>10}")
    print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    rows = sorted(summary.items(), key=lambda kv: -kv[1]["a"])
    for lbl, info in rows:
        pct = info["pct"]
        pct_str = f"{pct:+.1f}%" if pct != float("inf") else "+inf"
        print(f"  {lbl:<32} {info['a']:>8} {info['b']:>8} {info['delta']:>+8} {pct_str:>10}")
    total_a = sum(v["a"] for v in summary.values())
    total_b = sum(v["b"] for v in summary.values())
    total_delta = total_b - total_a
    total_pct = (total_delta / total_a * 100) if total_a else 0.0
    print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    print(f"  {'TOTAL':<32} {total_a:>8} {total_b:>8} {total_delta:>+8} {total_pct:>+9.1f}%")


def print_strict_summary(summary: Dict[str, Any], a_name: str, b_name: str) -> None:
    print()
    print("=" * 80)
    print(f"STRICT DIFF — per-detection (label, start, end) set comparison")
    print(f"  A treated as ground truth, B is being evaluated")
    print("=" * 80)
    print(f"  true positives  (in both)         : {summary['tp']:>6}")
    print(f"  false positives (in B, not in A)  : {summary['fp']:>6}")
    print(f"  false negatives (in A, not in B)  : {summary['fn']:>6}")
    print(f"  precision  (B's correctness)       : {summary['precision']:.4f}")
    print(f"  recall     (B's coverage)          : {summary['recall']:.4f}")
    print(f"  F1                                 : {summary['f1']:.4f}")
    print(f"  Jaccard    (overall set similarity): {summary['jaccard']:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two GLiNER models for equivalence")
    parser.add_argument(
        "--model-a",
        default="knowledgator/gliner-pii-large-v1.0",
        help="Reference model (treated as ground truth in the strict diff)",
    )
    parser.add_argument(
        "--model-b",
        default="knowledgator/gliner-pii-base-v1.0",
        help="Candidate model being evaluated",
    )
    parser.add_argument("--ultra-canonical", action="store_true")
    parser.add_argument("--use-onnx", action="store_true")
    parser.add_argument(
        "--diff-out",
        default=str(ROOT / "tests" / "model_diff.yaml"),
        help="Where to dump the per-row strict diff (YAML, grep-friendly)",
    )
    args = parser.parse_args()

    rows = load_rows()
    print(f"loaded {len(rows)} rows from {ROWS_PATH.name}")
    print()

    # ── Run model A first, then model B. Each load is independent. ──────
    a_dets, a_total_s = detect_with_model(
        args.model_a, rows, args.ultra_canonical, args.use_onnx
    )
    b_dets, b_total_s = detect_with_model(
        args.model_b, rows, args.ultra_canonical, args.use_onnx
    )

    # ── Loose: per-label count drift ────────────────────────────────────
    loose = loose_diff(a_dets, b_dets)
    print()
    print_loose_table(loose, args.model_a, args.model_b)

    # ── Strict: (label, start, end) set diff with P/R/F1/Jaccard ────────
    strict_summary, per_row = strict_diff(a_dets, b_dets, rows)
    print_strict_summary(strict_summary, args.model_a, args.model_b)

    # ── Latency side-by-side (cheap proxy; full benchmark is the real one) ─
    print()
    print("=" * 80)
    print("LATENCY (in this script — full benchmark gives p50/p95/p99)")
    print("=" * 80)
    print(f"  A: {a_total_s * 1000 / len(rows):>8.0f} ms/row total ({a_total_s:.1f}s for {len(rows)} rows)")
    print(f"  B: {b_total_s * 1000 / len(rows):>8.0f} ms/row total ({b_total_s:.1f}s for {len(rows)} rows)")
    speedup = a_total_s / b_total_s if b_total_s else float("inf")
    print(f"  speedup: B is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than A")

    # ── Dump strict per-row diff for later audit ────────────────────────
    diff_out = Path(args.diff_out)
    diff_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "ultra_canonical": args.ultra_canonical,
        "use_onnx": args.use_onnx,
        "loose": loose,
        "strict_summary": strict_summary,
        "per_row": per_row,
    }
    with diff_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True, width=120)
    print()
    print(f"per-row strict diff written to {diff_out}")
    print(f"  grep tip: yq '.per_row[] | select(.fn > 5)' {diff_out.name}   # rows where B missed >5 detections from A")

    # ── Verdict ─────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    f1 = strict_summary["f1"]
    jac = strict_summary["jaccard"]
    if f1 >= 0.95 and jac >= 0.90:
        print(f"VERDICT: SAFE TO SHIP — F1={f1:.3f}, Jaccard={jac:.3f}")
    elif f1 >= 0.85 and jac >= 0.75:
        print(f"VERDICT: REVIEW DIFF — F1={f1:.3f}, Jaccard={jac:.3f}")
        print("  Inspect tests/model_diff.yaml before shipping. Some divergence is expected")
        print("  for a smaller backbone, but check the missed_by_b lists for systemic gaps.")
    else:
        print(f"VERDICT: NOT SAFE — F1={f1:.3f}, Jaccard={jac:.3f}")
        print("  The candidate model diverges materially from the reference. Either keep")
        print("  the reference, retrain the candidate, or accept the precision/recall hit.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
