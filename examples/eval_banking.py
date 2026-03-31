"""
Banking Conversation Accuracy Evaluation
==========================================
Tests the 17-stage pipeline against a banking transcript with known ground truth.
Measures precision, recall, and F1 per label and overall.

Run from project root:
    python examples/eval_banking.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import defaultdict

# ── Ground Truth: Banking Wire Transfer Conversation ──────────────────

BANKING_TEXT = """
[Call Date: 03/28/2026 | Duration: 18:42 | Agent: Sarah Mitchell | Dept: International Wire Transfers]

Agent: Good afternoon, this is Sarah Mitchell with First National Bank, employee ID FNB-E-88241. How can I help you today?

Customer: Hi Sarah. My name is Michael Anthony Torres. I need to initiate an international wire transfer from my business checking account to a supplier in Germany.

Agent: I'd be happy to assist with that, Mr. Torres. For verification, can I get your date of birth and the last four digits of your Social Security Number?

Customer: Sure. Date of birth is March 12, 1978. Last four of my SSN is 4489. Actually, let me give you the full number for the new account verification — it's 287-65-4489.

Agent: Thank you. And can you confirm the account you're sending from?

Customer: The business checking account number is 7291048356. The routing number for that account is 021000021. The account is registered under Torres International Trading LLC.

Agent: Verified. Now for the receiving account details?

Customer: The beneficiary is Hans-Peter Braun at Braun Maschinenbau GmbH. Their IBAN is DE89370400440532013000. The SWIFT code is COBADEFFXXX. Their address is Industriestrasse 47, 80339 Munich, Germany.

Agent: What's the transfer amount and purpose?

Customer: $47,250.00 USD for invoice INV-2026-03847 — industrial equipment parts. The purchase order number is PO-TIT-2026-00219.

Agent: I'll need a few more details. What's your contact information for the confirmation?

Customer: My cell is 415-555-7823. Email is mtorres@torrestrading.com. My business EIN is 84-2917453. Oh, and my driver's license for the additional ID check — it's a California license, C7829415.

Agent: And your business address on file?

Customer: 2847 Market Street, Suite 400, San Francisco, CA 94105.

Agent: Perfect. For international transfers over $10,000, I need to verify your IP address for the online banking session that initiated this request. I show it as 198.51.100.42. Does that look correct?

Customer: Yes, that's my office connection.

Agent: Great. I'm also seeing that you have a credit card on file ending in 8834, card number 4532-7891-2345-8834, expiration 09/27, for any fees. Should I charge the wire fee of $45 to that card?

Customer: Yes, that's fine. The CVV is 847.

Agent: Let me also update your records. Your wife Lisa Torres is still listed as the authorized signer on the business account, correct? Her SSN is 312-89-7756 and date of birth is July 22, 1980?

Customer: That's correct. Her phone number is 415-555-2198. Actually, also add my accountant as a view-only contact — David Park, phone 415-555-6641, email dpark@bayareacpa.com.

Agent: Done. Your wire transfer reference number is WT-FNB-2026-0284719. The cut-off for same-day processing is 2:00 PM Pacific. Estimated arrival is 1-2 business days. You'll receive a confirmation at your email. The OFAC screening has cleared. Is there anything else?

Customer: Actually, can you also check the balance on my savings account? Account number 7291048412, routing 021000021. And my money market account 8834-2917-0042?

Agent: Your savings shows $124,847.23 and the money market is at $312,456.78. Anything else?

Customer: No, that's everything. Thanks, Sarah.

Agent: You're welcome, Mr. Torres. Have a great day.
"""

# Ground truth: (label, text) pairs — every PII in the text
GROUND_TRUTH = [
    # Persons
    ("PERSON_FIRST_NAME", "Sarah"),
    ("PERSON_LAST_NAME", "Mitchell"),
    ("PERSON_FIRST_NAME", "Michael"),
    ("PERSON_MIDDLE_NAME", "Anthony"),
    ("PERSON_LAST_NAME", "Torres"),
    ("PERSON_FIRST_NAME", "Hans-Peter"),
    ("PERSON_LAST_NAME", "Braun"),
    ("PERSON_FIRST_NAME", "Lisa"),
    ("PERSON_LAST_NAME", "Torres"),
    ("PERSON_FIRST_NAME", "David"),
    ("PERSON_LAST_NAME", "Park"),

    # Employee/Badge
    ("EMPLOYEE_ID", "FNB-E-88241"),

    # DOB
    ("DATE_OF_BIRTH", "March 12, 1978"),
    ("DATE_OF_BIRTH", "July 22, 1980"),

    # SSN
    ("SSN", "287-65-4489"),
    ("SSN", "312-89-7756"),

    # Bank Accounts & Routing
    ("BANK_ACCOUNT_NUMBER", "7291048356"),
    ("BANK_ACCOUNT_NUMBER", "7291048412"),
    ("ROUTING_NUMBER", "021000021"),
    ("IBAN", "DE89370400440532013000"),
    ("SWIFT_BIC_CODE", "COBADEFFXXX"),

    # Credit Card
    ("CREDIT_CARD_NUMBER", "4532-7891-2345-8834"),
    ("CARD_EXPIRATION_DATE", "09/27"),
    ("CARD_SECURITY_CODE", "847"),

    # Contact
    ("PHONE_NUMBER", "415-555-7823"),
    ("PHONE_NUMBER", "415-555-2198"),
    ("PHONE_NUMBER", "415-555-6641"),
    ("EMAIL_ADDRESS", "mtorres@torrestrading.com"),
    ("EMAIL_ADDRESS", "dpark@bayareacpa.com"),
    ("STREET_ADDRESS", "2847 Market Street, Suite 400, San Francisco, CA 94105"),

    # IDs
    ("DRIVERS_LICENSE_NUMBER", "C7829415"),
    ("IP_ADDRESS", "198.51.100.42"),
    ("TAX_ID", "84-2917453"),

    # Vehicle/Other accounts
    ("BANK_ACCOUNT_NUMBER", "8834-2917-0042"),
]

# ── Evaluation Logic ─────────────────────────────────────────────────

def normalize(text):
    """Normalize text for fuzzy matching."""
    return text.strip().lower().replace(" ", "").replace("-", "").replace(".", "")

def evaluate(detections, ground_truth):
    """Compute precision, recall, F1 per label and overall."""

    # Build ground truth lookup: normalized_value -> set of acceptable labels
    gt_by_value = defaultdict(set)
    gt_by_label = defaultdict(list)
    for label, text in ground_truth:
        gt_by_value[normalize(text)].add(label)
        gt_by_label[label].append(text)

    # Compatible label groups (any in group counts as correct)
    COMPAT = [
        {"PERSON_FULL_NAME", "PERSON_FIRST_NAME", "PERSON_LAST_NAME", "PERSON_MIDDLE_NAME"},
        {"SSN", "TAX_ID", "NATIONAL_ID"},
        {"CREDIT_CARD_NUMBER", "ACCOUNT_NUMBER_AMEX"},
        {"BANK_ACCOUNT_NUMBER", "ACCOUNT_LAST4", "BANK_ACCOUNT_LAST4", "GENERIC_LAST4"},
        {"ROUTING_NUMBER", "BANK_ACCOUNT_NUMBER"},
        {"PROFESSIONAL_LICENSE_NUMBER", "NATIONAL_ID"},
        {"STREET_ADDRESS", "PLACE_OF_BIRTH"},
        {"EMPLOYEE_ID", "MEDICAL_RECORD_NUMBER"},
        {"ACCOUNT_LAST4", "GENERIC_LAST4", "BANK_ACCOUNT_LAST4", "CARD_SECURITY_CODE"},
        {"INSURANCE_POLICY", "HEALTH_INSURANCE_ID"},
    ]

    def labels_match(det_label, gt_labels):
        """Check if detected label matches any ground truth label (with compatibility)."""
        if det_label in gt_labels:
            return True
        for group in COMPAT:
            if det_label in group and gt_labels & group:
                return True
        return False

    # Match detections to ground truth
    matched_gt = set()   # indices of matched ground truth items
    matched_det = set()  # indices of matched detections
    tp_by_label = defaultdict(int)
    fp_by_label = defaultdict(int)

    for di, d in enumerate(detections):
        det_norm = normalize(d.text)
        # Try exact match first
        if det_norm in gt_by_value:
            gt_labels = gt_by_value[det_norm]
            if labels_match(d.label, gt_labels):
                matched_det.add(di)
                # Find and mark the GT entry
                for gi, (gl, gt) in enumerate(ground_truth):
                    if gi not in matched_gt and normalize(gt) == det_norm:
                        if labels_match(d.label, {gl}):
                            matched_gt.add(gi)
                            tp_by_label[d.label] += 1
                            break
                else:
                    tp_by_label[d.label] += 1  # Match by value even if GT index used
                continue

        # Try substring match (for partial detections like address)
        found = False
        for gi, (gl, gt) in enumerate(ground_truth):
            if gi in matched_gt:
                continue
            gt_norm = normalize(gt)
            if det_norm in gt_norm or gt_norm in det_norm:
                if labels_match(d.label, {gl}):
                    matched_det.add(di)
                    matched_gt.add(gi)
                    tp_by_label[d.label] += 1
                    found = True
                    break

        if not found:
            fp_by_label[d.label] += 1

    # Compute FN (ground truth items not matched)
    fn_by_label = defaultdict(int)
    for gi, (gl, gt) in enumerate(ground_truth):
        if gi not in matched_gt:
            fn_by_label[gl] += 1

    # Aggregate metrics
    all_labels = sorted(set(list(tp_by_label.keys()) + list(fp_by_label.keys()) + list(fn_by_label.keys())))

    total_tp = sum(tp_by_label.values())
    total_fp = sum(fp_by_label.values())
    total_fn = sum(fn_by_label.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp_by_label": dict(tp_by_label),
        "fp_by_label": dict(fp_by_label),
        "fn_by_label": dict(fn_by_label),
        "all_labels": all_labels,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_gt": matched_gt,
        "ground_truth": ground_truth,
        "detections": detections,
    }


# ── Main ─────────────────────────────────────────────────────────────

def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

print_header("BANKING CONVERSATION ACCURACY EVALUATION")

from app import HybridPIIEngine

print("Loading engine...")
engine = HybridPIIEngine(
    use_gliner=True,
    gliner_model_name="knowledgator/gliner-pii-large-v1.0",
    gliner_threshold=0.15,
)
print(f"  GLiNER: {'ON' if engine.gliner_detector.enabled else 'OFF'}")
print(f"  LSTM: {'ON' if engine.self_trainer.is_ready() else 'OFF'}")

print_header("RUNNING DETECTION")

start = time.time()
result = engine.redact(BANKING_TEXT)
elapsed = time.time() - start
print(f"  Time: {elapsed:.2f}s | Detections: {len(result.detections)} | Ground truth: {len(GROUND_TRUTH)}")

# ── Detection Table ──────────────────────────────────────────────────

print_header("ALL DETECTIONS")
print(f"{'#':<4} {'LABEL':<35} {'VALUE':<40} {'SCORE':>6} {'SOURCE':<18}")
print("-" * 110)
for i, d in enumerate(result.detections, 1):
    val = d.text[:38] + ".." if len(d.text) > 40 else d.text
    print(f"{i:<4} {d.label:<35} {val:<40} {d.score:>6.3f} {d.source:<18}")

# ── Evaluate ─────────────────────────────────────────────────────────

metrics = evaluate(result.detections, GROUND_TRUTH)

print_header("PER-LABEL METRICS")
print(f"{'LABEL':<35} {'TP':>4} {'FP':>4} {'FN':>4} {'PREC':>7} {'REC':>7} {'F1':>7}")
print("-" * 75)

for label in metrics["all_labels"]:
    tp = metrics["tp_by_label"].get(label, 0)
    fp = metrics["fp_by_label"].get(label, 0)
    fn = metrics["fn_by_label"].get(label, 0)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"{label:<35} {tp:>4} {fp:>4} {fn:>4} {p:>7.1%} {r:>7.1%} {f:>7.1%}")

print_header("OVERALL METRICS")
print(f"  True Positives  : {metrics['total_tp']}")
print(f"  False Positives : {metrics['total_fp']}")
print(f"  False Negatives : {metrics['total_fn']}")
print(f"  Precision       : {metrics['precision']:.1%}")
print(f"  Recall          : {metrics['recall']:.1%}")
print(f"  F1 Score        : {metrics['f1']:.1%}")

# ── Missed PII (False Negatives) ─────────────────────────────────────

print_header("MISSED PII (FALSE NEGATIVES)")
missed = [(GROUND_TRUTH[i][0], GROUND_TRUTH[i][1]) for i in range(len(GROUND_TRUTH)) if i not in metrics["matched_gt"]]
if missed:
    for label, text in missed:
        print(f"  MISSED: {label:<35} {text}")
else:
    print("  None! All ground truth PII detected.")

# ── False Positives ──────────────────────────────────────────────────

print_header("FALSE POSITIVES")
fp_labels = {l for l, c in metrics["fp_by_label"].items() if c > 0}
if fp_labels:
    for d in result.detections:
        if d.label in fp_labels:
            # Check if this specific detection is a FP
            det_norm = normalize(d.text)
            is_tp = any(det_norm == normalize(gt) or det_norm in normalize(gt) or normalize(gt) in det_norm
                       for _, gt in GROUND_TRUTH)
            if not is_tp:
                print(f"  FP: {d.label:<35} \"{d.text[:50]}\" (score={d.score:.3f}, source={d.source})")
else:
    print("  None! Zero false positives.")

# ── Redacted Text Preview ────────────────────────────────────────────

print_header("REDACTED TEXT")
print(result.redacted_text[:2500])
if len(result.redacted_text) > 2500:
    print(f"\n  ... ({len(result.redacted_text)} total chars)")

print()
print("Done!")
