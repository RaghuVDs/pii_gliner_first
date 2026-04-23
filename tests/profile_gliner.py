"""
Profile GLiNER model directly under varying configurations to find the
optimal (batch_size, labels_per_call, chunk_window) for THIS GPU and THIS
model. Output drives the auto-tuner in gliner_detector.

Usage:
    python tests/profile_gliner.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch  # noqa: E402
from gliner import GLiNER  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ROWS_PATH = ROOT / "tests" / "fake_transcripts.jsonl"
MODEL_NAME = "knowledgator/gliner-pii-large-v1.0"


def load_one_row():
    with ROWS_PATH.open("r", encoding="utf-8") as f:
        return json.loads(f.readline())["text"]


def chunk_text(text: str, window: int, overlap: int):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + window, n)
        if end < n:
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        chunks.append(text[start:end])
        start = end - overlap
        if start <= 0:
            start = end
    return chunks


def time_call(model, texts, labels, batch_size, n_runs=3):
    """Run batch_predict_entities n_runs times, return median seconds."""
    # warmup
    with torch.inference_mode():
        _ = model.batch_predict_entities(texts, labels, threshold=0.2, batch_size=batch_size)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        with torch.inference_mode():
            _ = model.batch_predict_entities(texts, labels, threshold=0.2, batch_size=batch_size)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    times.sort()
    return times[len(times) // 2]


def main():
    print("=" * 78)
    print("GLiNER PROFILER — measuring real per-config latency on this GPU")
    print("=" * 78)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Model: {MODEL_NAME}")
    print()

    print("loading model...")
    t = time.perf_counter()
    model = GLiNER.from_pretrained(MODEL_NAME).to("cuda").eval().half()
    print(f"  loaded in {time.perf_counter() - t:.1f}s\n")

    text = load_one_row()
    print(f"sample row: {len(text):,} chars\n")

    # Synthetic but realistic label set (drawn from common PII)
    base_labels = [
        "person", "first name", "last name", "email", "phone number",
        "date of birth", "social security number", "credit card number",
        "address", "street address", "city", "state", "zip code",
        "bank account number", "iban", "routing number", "ip address",
        "license plate", "driver license", "passport number", "company",
        "job title", "url", "date", "time", "currency amount",
        "username", "password", "api key", "medical record number",
        "diagnosis", "medication", "patient name", "doctor name",
    ]

    # ── Test 1: chunk window sweep ──
    print("=" * 78)
    print("TEST 1: chunk window size (labels=25, batch_size=16)")
    print("=" * 78)
    print(f"{'window':>8} {'overlap':>8} {'#chunks':>8} {'time(s)':>10} {'ms/chunk':>10}")
    for window in [1500, 3000, 4500, 6000, 8000]:
        chunks = chunk_text(text, window=window, overlap=150)
        try:
            t_s = time_call(model, chunks, base_labels[:25], batch_size=16)
            print(f"{window:>8} {150:>8} {len(chunks):>8} {t_s:>10.3f} {t_s*1000/len(chunks):>10.1f}")
        except Exception as e:
            print(f"{window:>8} FAILED: {e}")
            torch.cuda.empty_cache()
    print()

    # Pick a reasonable window for the rest of the tests
    chunks = chunk_text(text, window=3000, overlap=150)
    print(f"using window=3000 ({len(chunks)} chunks) for remaining tests\n")

    # ── Test 2: labels-per-call sweep ──
    print("=" * 78)
    print("TEST 2: labels per call (window=3000, batch_size=16)")
    print("=" * 78)
    print(f"{'#labels':>8} {'time(s)':>10} {'ms/(chunk*label)':>18}")
    for n_labels in [5, 10, 15, 25, 35, 50, 75, 100, 150, 200]:
        # Repeat base labels if needed
        labels = (base_labels * 10)[:n_labels]
        try:
            t_s = time_call(model, chunks, labels, batch_size=16)
            print(f"{n_labels:>8} {t_s:>10.3f} {t_s*1000/(len(chunks)*n_labels):>18.3f}")
        except Exception as e:
            print(f"{n_labels:>8} FAILED: {e}")
            torch.cuda.empty_cache()
    print()

    # ── Test 3: batch_size sweep ──
    print("=" * 78)
    print("TEST 3: batch_size (window=3000, labels=25)")
    print("=" * 78)
    print(f"{'batch':>8} {'time(s)':>10} {'ms/chunk':>10}")
    for bs in [1, 4, 8, 16, 32, 64, 128]:
        try:
            t_s = time_call(model, chunks, base_labels[:25], batch_size=bs)
            print(f"{bs:>8} {t_s:>10.3f} {t_s*1000/len(chunks):>10.1f}")
        except Exception as e:
            print(f"{bs:>8} FAILED: {str(e)[:60]}")
            torch.cuda.empty_cache()
    print()

    # ── Test 4: full simulated row (best-case combined config) ──
    print("=" * 78)
    print("TEST 4: full row simulation (window=3000, batch=32, 25 labels × 25 calls)")
    print("=" * 78)
    chunks = chunk_text(text, window=3000, overlap=150)
    print(f"chunks: {len(chunks)}")
    n_label_groups = 25  # ~625 / 25
    t = time.perf_counter()
    with torch.inference_mode():
        for i in range(n_label_groups):
            _ = model.batch_predict_entities(
                chunks, base_labels[:25], threshold=0.2, batch_size=32
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t
    print(f"  total: {elapsed:.2f}s for {n_label_groups} label-groups")
    print(f"  per call: {elapsed * 1000 / n_label_groups:.0f} ms")
    print()

    print("=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
