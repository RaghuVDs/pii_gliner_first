"""
Full 17-Stage Pipeline Test
============================
Tests every new component: bi-encoder, entropy detector, contextual anomaly,
LSTM, few-shot, active learning, confidence calibration, and synthetic data.

Run from project root:
    python examples/test_all_stages.py
"""
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter

# ── SECTION 1: PyTorch Diagnostics ────────────────────────────────────

def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

print_header("PYTORCH DIAGNOSTICS")
import torch
print(f"  PyTorch Version : {torch.__version__}")
print(f"  CUDA Available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU Device      : {torch.cuda.get_device_name(0)}")
print()

# ── SECTION 2: Engine Initialization ─────────────────────────────────

print_header("INITIALIZING ENGINE")

from app import HybridPIIEngine

start = time.time()
engine = HybridPIIEngine(
    use_gliner=True,
    gliner_model_name="knowledgator/gliner-pii-large-v1.0",
    gliner_threshold=0.15,
)
init_time = time.time() - start

print(f"  Init time          : {init_time:.1f}s")
print(f"  GLiNER loaded      : {engine.gliner_detector.enabled}")
print(f"  Bi-encoder cached  : {engine.gliner_detector._bi_encoder}")
print(f"  LSTM ready         : {engine.self_trainer.is_ready()}")
print(f"  Few-shot ready     : {engine.few_shot.is_ready()}")
known = engine.lstm_known_labels()
if known:
    print(f"  LSTM labels ({len(known)}): {known[:10]}{'...' if len(known) > 10 else ''}")
else:
    print("  LSTM not yet trained -- will learn from detections")
print()

# ── SECTION 3: Detection & Redaction ─────────────────────────────────

# This transcript contains diverse PII types including:
# - Person names (multiple people, titles, suffixes)
# - Government IDs (SSN, driver's license, passport)
# - Financial (credit cards, bank accounts, routing numbers)
# - Medical (MRN, NPI, prescriptions, Medicare)
# - Vehicle (VIN, plates)
# - Digital (emails, phones, API keys, tokens)
# - Unknown fields (CustomTrackingField, BiometricRef)
sample_text = """
[Call Date: 02/03/2026 | Duration: 22:15 | Agent: David Ramirez | Channel: Phone]

Agent: Thank you for calling United Family Insurance. This is David Ramirez, badge number UFI-33291. How may I assist you?

Customer: Hi David. My name is Dr. Priya Anand Chakraborty. I'm calling about a car accident I was in last week and I need to file claims under both my auto and health insurance.

Agent: I'm sorry to hear that, Dr. Chakraborty. Let me pull up your account. Can you verify your policy number and date of birth?

Customer: My auto policy is UFI-AUTO-5538291. My health insurance is through the same company, policy UFI-HEALTH-5538292. Group number GRP-BIOTECH-4421. My date of birth is November 3, 1985. SSN is 621-73-4489.

Agent: Verified. Can you walk me through what happened?

Customer: On January 28th at about 3:15 PM, I was rear-ended at the intersection of Camelback Road and 24th Street in Phoenix. The other driver's name was Robert James Martinez, and he gave me his license number -- it was an Arizona license, D98321754. His insurance is State Farm, policy number SF-229-4817362-01. His plate number was Arizona FGR-8821. His vehicle was a 2019 Ford F-150, VIN 1FTEW1EP5KFA82943.

Agent: And your vehicle information?

Customer: 2024 Audi Q7, VIN WAUZZZ4M1RD019384, Arizona plate NLV-3392. It's leased through Audi Financial Services, account AFS-22918743. My lien holder address is on file.

Agent: Got it. Now for the medical portion -- can you describe your injuries and treatment?

Customer: I went to the Scottsdale Osborn Medical Center ER that evening. My medical record number there is SMC-00482917. I was diagnosed with cervical whiplash, a mild concussion -- ICD-10 code S13.4XXA and S06.0X0A. They did a CT scan and X-rays. The attending physician was Dr. Leonard Wu, NPI number 1497825310. I was prescribed Cyclobenzaprine 10mg, prescription number RX-2294817, filled at CVS pharmacy on Shea Boulevard, store number 6742.

Agent: Have you had any follow-up treatment?

Customer: Yes. I'm seeing Dr. Amanda Foster for physical therapy, NPI 1832749162, at Desert Spine & Rehabilitation. My patient ID there is DSR-P-08291. I've had three sessions so far. I also saw my primary care physician, Dr. Rajesh Gupta, NPI 1628374910, at Banner Health, patient ID BH-44829173. He referred me to a neurologist for the concussion follow-up. Oh, and my Medicare Beneficiary ID is 1EK4-TA2-HY74 since I also have Medicare Part B through my disability status.

Agent: Thank you. For the auto claim, can I get the police report number?

Customer: Phoenix PD report number 2026-PX-0028471. The responding officer was Officer Kevin Doyle, badge number PPD-4419. There were two witnesses -- Maria Santos, phone 602-555-8837, and James Liu, phone 480-555-2291.

Agent: I need a few more details for your health claim. Can you verify your blood type and any existing conditions in your medical history that might be relevant?

Customer: Blood type is O-negative. I have a pre-existing condition of Type 2 diabetes, currently managed with Metformin. I also have a medical device -- a continuous glucose monitor, device serial number DX-G6-482917384. My pharmacy benefit manager is Express Scripts, member ID ESI-77482913.

Agent: And your emergency contact on file?

Customer: My husband, Arjun Chakraborty. Cell phone 480-555-6614. He's my authorized representative on all accounts. His SSN is 621-73-4490 and date of birth is June 17, 1983.

Agent: Perfect. I've opened auto claim number UFI-AC-2026-019384 and health claim number UFI-HC-2026-019385. A claims adjuster will contact you within 48 hours. Your deductible on the auto is $500 and the health deductible has been met for the year. Is there anything else?

Customer: Can you send the claim documents to my work email? It's pchakraborty@genedyne-biotech.com. My employee ID there is GBT-E-4817. Actually, also copy my attorney -- Patricia Huang, Esq., at phuang@desertlawgroup.com, phone 602-555-9918, bar number AZ-029481.

Customer: Oh, one more thing -- here's my developer portal API key for the insurance app: sk_live_aR7x9P2mQ4vL8nR3bF5jY1wE6hT0uC9dA8sF4gHzK and the JWT from my last session: eyJhbGciOiJIUzI1NiJ9.dGVzdHBheWxvYWRoZXJlYWJjZGVm.xYz123AbC456dEf789

Customer: Also, our lab uses a custom biometric system. BiometricRef: BIO-7X92-KM41-QP38

Agent: Done. Anything else, Dr. Chakraborty?

Customer: No, that covers it. Thank you, David.
"""

print_header("RUNNING 17-STAGE DETECTION")

start = time.time()
result = engine.redact(sample_text)
detect_time = time.time() - start

print(f"  Detection time     : {detect_time:.2f}s")
print(f"  Total detections   : {len(result.detections)}")
print(f"  Unknown candidates : {len(result.unknown_candidates)}")
print()

# ── SECTION 4: Detection Table ───────────────────────────────────────

print_header("DETECTION TABLE")

header = f"{'#':<4} {'LABEL':<35} {'VALUE':<40} {'SCORE':>6} {'SOURCE':<22} {'FLAGS'}"
print(header)
print("-" * len(header) + "-" * 20)

for i, d in enumerate(result.detections, 1):
    flags = []
    if d.meta.get("cross_validated"):
        flags.append("XV")
    if d.source == "pattern_lstm":
        flags.append("LSTM")
    if d.source == "few_shot":
        flags.append("FS")
    if d.source == "entropy_anomaly":
        flags.append("ENT")
    if d.source == "contextual_anomaly":
        flags.append("CTX")

    flag_str = " ".join(f"[{f}]" for f in flags)
    val = d.text[:38] + ".." if len(d.text) > 40 else d.text
    print(f"{i:<4} {d.label:<35} {val:<40} {d.score:>6.3f} {d.source:<22} {flag_str}")

# ── SECTION 5: Statistics ────────────────────────────────────────────

print_header("STATISTICS")

# Source breakdown
source_counts = Counter(d.source for d in result.detections)
print("  Detections by source:")
for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
    pct = 100 * count / len(result.detections)
    bar = "#" * int(pct / 2)
    print(f"    {source:<22} {count:>3} ({pct:>5.1f}%) {bar}")

# Cross-validated
xv = sum(1 for d in result.detections if d.meta.get("cross_validated"))
print(f"\n  Cross-validated    : {xv}")

# LSTM-upgraded
lstm = sum(1 for d in result.detections if d.source == "pattern_lstm")
print(f"  LSTM-upgraded      : {lstm}")

# Entropy detections
ent = [d for d in result.detections if d.source == "entropy_anomaly"]
print(f"  Entropy anomalies  : {len(ent)}")
for d in ent:
    print(f"    -> {d.text[:50]} (entropy={d.meta.get('entropy', '?')})")

# Contextual anomalies
ctx = [d for d in result.detections if d.source == "contextual_anomaly"]
print(f"  Contextual anomalies: {len(ctx)}")
for d in ctx:
    field = d.meta.get("discovered_field", "")
    kws = d.meta.get("nearby_keywords", [])
    info = f"field={field}" if field else f"keywords={kws[:3]}"
    print(f"    -> {d.text[:40]} ({info})")

# ── SECTION 6: Redacted Text ────────────────────────────────────────

print_header("REDACTED TEXT (first 2000 chars)")
print(result.redacted_text[:2000])
if len(result.redacted_text) > 2000:
    print(f"\n  ... ({len(result.redacted_text)} total chars)")

# ── SECTION 7: Component Status ─────────────────────────────────────

print_header("COMPONENT STATUS")

print("\n  --- LSTM Pattern Classifier ---")
print(f"  Ready        : {engine.self_trainer.is_ready()}")
labels = engine.lstm_known_labels()
print(f"  Known labels : {len(labels)}")
if labels:
    print(f"  Labels       : {labels[:15]}{'...' if len(labels) > 15 else ''}")

print("\n  --- Few-Shot Classifier ---")
fs = engine.few_shot_stats()
print(f"  Labels       : {fs['total_labels']}")
print(f"  Prototypes   : {fs['total_prototypes']}")

print("\n  --- Active Learning ---")
al = engine.active_learning_stats()
print(f"  Queue size   : {al['total_items']}")
print(f"  Pending      : {al['pending_review']}")
if al['total_items'] > 0:
    print(f"  Avg info     : {al['avg_informativeness']:.4f}")

print("\n  --- Active Learning Review Queue ---")
queue = engine.get_review_queue(top_k=5)
if queue:
    for item in queue:
        print(f"    [{item['informativeness']:.2f}] {item['current_label']}: "
              f"{item['structure'][:30]} (seen {item['seen_count']}x)")
else:
    print("  Queue empty -- no candidates need review")

print("\n  --- Adaptive Learning (Pending Rules) ---")
pending = engine.review_pending()
if pending:
    for p in pending[:5]:
        print(f"    [{p.get('seen_count', 0)}x] {p.get('suggested_label', '?')}: "
              f"keywords={p.get('suggested_keywords', [])[:4]}")
else:
    print("  No pending rules")

# ── SECTION 8: Synthetic Data Generation ─────────────────────────────

print_header("SYNTHETIC DATA GENERATION")

try:
    stats = engine.generate_synthetic_data(n_per_label=5, n_documents=10)
    print(f"  Total examples  : {stats['total_examples']}")
    print(f"  Isolated        : {stats['isolated_examples']}")
    print(f"  From documents  : {stats['document_examples']}")
    print(f"  Labels covered  : {len(stats['labels_covered'])}")
    print(f"  Labels          : {stats['labels_covered'][:10]}...")
except ImportError:
    print("  Skipped -- faker not installed (pip install faker)")

# ── SECTION 9: Performance Summary ───────────────────────────────────

print_header("PERFORMANCE SUMMARY")
print(f"  Engine init    : {init_time:.1f}s")
print(f"  Detection      : {detect_time:.2f}s")
print(f"  Text length    : {len(sample_text)} chars")
print(f"  Chars/sec      : {len(sample_text)/detect_time:.0f}")
print(f"  Detections     : {len(result.detections)}")
bi_enc = "YES" if engine.gliner_detector._bi_encoder else "NO"
print(f"  Bi-encoder     : {bi_enc}")
print()
print("Done!")
