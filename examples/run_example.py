"""
PII Detection Engine — Full Example
====================================
GLiNER-First Architecture with self-learning CNN-BiLSTM pattern classifier.

Pipeline (13 stages):
  1. GLiNER AI (primary)  →  2. Field Labels  →  3. Regex (fallback)
  →  4. Context promotion  →  5. LSTM pseudo-label  →  6. False positive filter
  →  7. Cross-validation  →  8. Name splitting  →  9. Overlap resolution
  → 10. Name propagation  → 11. Collect training data  → 12. Auto-retrain
  → 13. Auto-promote pending rules (no human reviewer)

Everything is fully automatic — no human intervention needed.
"""

import sys
import os
import json
import logging

import torch

# ─────────────────────────────────────────────────────────────────────
# 0. Setup — GPU check and logging
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(message)s",
)

print("=" * 60)
print("GPU STATUS")
print("=" * 60)
print(f"  CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device Name    : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"  VRAM           : {vram:.1f} GB")
print()

# ─────────────────────────────────────────────────────────────────────
# 1. Initialize Engine
# ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import HybridPIIEngine

print("=" * 60)
print("INITIALIZING ENGINE")
print("=" * 60)

engine = HybridPIIEngine(
    use_gliner=True,
    gliner_model_name="knowledgator/gliner-pii-large-v1.0",
    gliner_threshold=0.15,
)

# Check if LSTM model was loaded from a previous training run
print(f"  LSTM model ready : {engine.self_trainer.is_ready()}")
known = engine.lstm_known_labels()
if known:
    print(f"  LSTM knows {len(known)} labels: {known}")
else:
    print("  LSTM not yet trained — will learn from GLiNER detections automatically")
print()

# ─────────────────────────────────────────────────────────────────────
# 2. Run Detection & Redaction
# ─────────────────────────────────────────────────────────────────────
sample_text = """
[Call Date: 02/03/2026 | Duration: 22:15 | Agent: David Ramirez | Channel: Phone]
Agent: Thank you for calling United Family Insurance. This is David Ramirez, badge number UFI-33291. How may I assist you?
Customer: Hi David. My name is Dr. Priya Anand Chakraborty. I'm calling about a car accident I was in last week and I need to file claims under both my auto and health insurance.
Agent: I'm sorry to hear that, Dr. Chakraborty. Let me pull up your account. Can you verify your policy number and date of birth?
Customer: My auto policy is UFI-AUTO-5538291. My health insurance is through the same company, policy UFI-HEALTH-5538292. Group number GRP-BIOTECH-4421. My date of birth is November 3, 1985. SSN is 621-73-4489.
Agent: Verified. Can you walk me through what happened?
Customer: On January 28th at about 3:15 PM, I was rear-ended at the intersection of Camelback Road and 24th Street in Phoenix. The other driver's name was Robert James Martinez, and he gave me his license number — it was an Arizona license, D98321754. His insurance is State Farm, policy number SF-229-4817362-01. His plate number was Arizona FGR-8821. His vehicle was a 2019 Ford F-150, VIN 1FTEW1EP5KFA82943.
Agent: And your vehicle information?
Customer: 2024 Audi Q7, VIN WAUZZZ4M1RD019384, Arizona plate NLV-3392. It's leased through Audi Financial Services, account AFS-22918743. My lien holder address is on file.
Agent: Got it. Now for the medical portion — can you describe your injuries and treatment?
Customer: I went to the Scottsdale Osborn Medical Center ER that evening. My medical record number there is SMC-00482917. I was diagnosed with cervical whiplash, a mild concussion — ICD-10 code S13.4XXA and S06.0X0A. They did a CT scan and X-rays. The attending physician was Dr. Leonard Wu, NPI number 1497825310. I was prescribed Cyclobenzaprine 10mg, prescription number RX-2294817, filled at CVS pharmacy on Shea Boulevard, store number 6742.
Agent: Have you had any follow-up treatment?
Customer: Yes. I'm seeing Dr. Amanda Foster for physical therapy, NPI 1832749162, at Desert Spine & Rehabilitation. My patient ID there is DSR-P-08291. I've had three sessions so far. I also saw my primary care physician, Dr. Rajesh Gupta, NPI 1628374910, at Banner Health, patient ID BH-44829173. He referred me to a neurologist for the concussion follow-up. Oh, and my Medicare Beneficiary ID is 1EK4-TA2-HY74 since I also have Medicare Part B through my disability status.
Agent: Thank you. For the auto claim, can I get the police report number?
Customer: Phoenix PD report number 2026-PX-0028471. The responding officer was Officer Kevin Doyle, badge number PPD-4419. There were two witnesses — Maria Santos, phone 602-555-8837, and James Liu, phone 480-555-2291.
Agent: I need a few more details for your health claim. Can you verify your blood type and any existing conditions in your medical history that might be relevant?
Customer: Blood type is O-negative. I have a pre-existing condition of Type 2 diabetes, currently managed with Metformin. I also have a medical device — a continuous glucose monitor, device serial number DX-G6-482917384. My pharmacy benefit manager is Express Scripts, member ID ESI-77482913.
Agent: And your emergency contact on file?
Customer: My husband, Arjun Chakraborty. Cell phone 480-555-6614. He's my authorized representative on all accounts. His SSN is 621-73-4490 and date of birth is June 17, 1983.
Agent: Perfect. I've opened auto claim number UFI-AC-2026-019384 and health claim number UFI-HC-2026-019385. A claims adjuster will contact you within 48 hours. Your deductible on the auto is $500 and the health deductible has been met for the year. Is there anything else?
Customer: Can you send the claim documents to my work email? It's pchakraborty@genedyne-biotech.com. My employee ID there is GBT-E-4817. Actually, also copy my attorney — Patricia Huang, Esq., at phuang@desertlawgroup.com, phone 602-555-9918, bar number AZ-029481.
Agent: Done. Anything else, Dr. Chakraborty?
Customer: No, that covers it. Thank you, David.
"""

print("=" * 60)
print("RUNNING DETECTION & REDACTION")
print("=" * 60)

result = engine.redact(sample_text)

# ─────────────────────────────────────────────────────────────────────
# 3. Detection Results
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 130)
print("DETECTIONS")
print("=" * 130)
print(f"{'LABEL':<30} {'VALUE':<45} {'SCORE':>5} {'SOURCE':<16} INSTANCE")
print("-" * 130)

for d in result.detections:
    flags = ""
    if d.meta.get("cross_validated"):
        flags += " [XV]"
    if d.source == "pattern_lstm":
        flags += " [LSTM]"
    print(
        f"{d.label:<30} {d.text!r:<45} "
        f"{d.score:>5.2f} {d.source:<16} "
        f"{d.meta.get('instance_label', '')}{flags}"
    )

print("-" * 130)
total = len(result.detections)
xv_count = sum(1 for d in result.detections if d.meta.get("cross_validated"))
lstm_count = sum(1 for d in result.detections if d.source == "pattern_lstm")
print(f"Total: {total} | Cross-validated: {xv_count} | LSTM-upgraded: {lstm_count} | Unknown candidates: {len(result.unknown_candidates)}")

# Source breakdown
from collections import Counter
source_counts = Counter(d.source for d in result.detections)
print(f"By source: {dict(source_counts)}")

# ─────────────────────────────────────────────────────────────────────
# 4. Redacted Output
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("REDACTED TEXT")
print("=" * 60)
print(result.redacted_text)

# ─────────────────────────────────────────────────────────────────────
# 5. LSTM Pattern Classifier Status
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("LSTM PATTERN CLASSIFIER STATUS")
print("=" * 60)

known = engine.lstm_known_labels()
if known:
    print(f"LSTM can predict {len(known)} labels:")
    for i, label in enumerate(known, 1):
        print(f"  {i:2d}. {label}")
else:
    print("LSTM has no trained labels yet — keep processing transcripts!")

print()
print("LSTM Training Stats:")
print(json.dumps(engine.pattern_lstm_stats(), indent=2))

# ─────────────────────────────────────────────────────────────────────
# 6. Detection Statistics
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("DETECTION STATISTICS")
print("=" * 60)
print(json.dumps(engine.detection_stats(), indent=2))

# ─────────────────────────────────────────────────────────────────────
# 7. Adaptive Learning Status
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("ADAPTIVE LEARNING STATUS")
print("=" * 60)
print(json.dumps(engine.learning_stats(), indent=2))

pending = engine.review_pending()
if pending:
    print(f"\nPending rules ({len(pending)}):")
    for p in pending[:10]:
        print(f"  {p['group_key']}: {p['suggested_label']} (seen {p['seen_count']}x)")
else:
    print("\nNo pending rules")

# ─────────────────────────────────────────────────────────────────────
# 8. Training Metrics History
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TRAINING METRICS HISTORY")
print("=" * 60)

metrics = engine.metrics_history()
if metrics and metrics.get("runs"):
    print(f"Total retrains : {metrics['total_retrains']}")
    print(f"Last retrain   : {metrics['last_retrain']}")

    latest = metrics["runs"][-1]
    print(f"\nLatest training run:")
    print(f"  Examples         : {latest['num_examples']}")
    print(f"  Labels           : {latest['num_labels']}")
    print(f"  Val Weighted F1  : {latest['val_weighted_f1']:.3f}")
    print(f"  Val Macro F1     : {latest['val_macro_f1']:.3f}")
    print(f"  Best epoch       : {latest['best_epoch']}/{latest['stopped_epoch']}")
    print(f"  Early stopped    : {latest['early_stopped']}")
    print(f"  Device           : {latest['device']}")
    print(f"  Mixed precision  : {latest['mixed_precision']}")

    print(f"\n  Per-label validation metrics:")
    print(f"  {'LABEL':<35} {'PREC':>6} {'REC':>6} {'F1':>6} {'N':>4}")
    print(f"  {'-'*60}")
    for label, m in sorted(latest.get("val_per_label", {}).items()):
        print(f"  {label:<35} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>4}")
else:
    print("No training runs yet — model hasn't been trained")

# ─────────────────────────────────────────────────────────────────────
# 9. Manual Retrain (Optional — uncomment to force)
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("MANUAL RETRAIN")
print("=" * 60)
train_result = engine.retrain_pattern_model(epochs=50)
print(json.dumps(train_result, indent=2))
print("Known labels after retrain:", engine.lstm_known_labels())
