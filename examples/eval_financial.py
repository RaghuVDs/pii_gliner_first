"""
Financial PII Accuracy Evaluation
===================================
Tests the pipeline against a complex financial advisor conversation
with diverse PII types including some novel/uncommon ones.

Run from project root:
    python examples/eval_financial.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import defaultdict

# ── Ground Truth: Financial Advisor Conversation ──────────────────────

FINANCIAL_TEXT = """
Amex agent - Rd. #29 Milwaukee Oregon 97267; this statement "Rd. #29" gets redacted successfully. Customer - Southeast Vineyard Rd. spelled Vineyard; Amex agent - OK; Customer - #29
"""

# Ground truth: (label, text) pairs
GROUND_TRUTH = [
    # Persons
    ("PERSON_FIRST_NAME", "Rebecca"),
    ("PERSON_LAST_NAME", "Lawson"),
    ("PERSON_FIRST_NAME", "Jonathan"),
    ("PERSON_MIDDLE_NAME", "Michael"),
    ("PERSON_LAST_NAME", "Chen"),
    ("PERSON_FIRST_NAME", "Mei-Lin"),
    ("PERSON_LAST_NAME", "Chen"),
    ("PERSON_FIRST_NAME", "Alexander"),
    ("PERSON_LAST_NAME", "Chen"),
    ("PERSON_FIRST_NAME", "Sophia"),
    ("PERSON_MIDDLE_NAME", "Grace"),
    ("PERSON_LAST_NAME", "Chen"),
    ("PERSON_FIRST_NAME", "David"),
    ("PERSON_LAST_NAME", "Blackwell"),
    ("PERSON_FIRST_NAME", "Sandra"),
    ("PERSON_LAST_NAME", "Reeves"),

    # Employee IDs
    ("EMPLOYEE_ID", "MFA-E-55207"),
    ("EMPLOYEE_ID", "QDI-E-7291"),

    # DOB
    ("DATE_OF_BIRTH", "September 8, 1971"),
    ("DATE_OF_BIRTH", "February 14, 1974"),
    ("DATE_OF_BIRTH", "May 3, 2005"),
    ("DATE_OF_BIRTH", "November 21, 2008"),

    # SSN
    ("SSN", "498-37-2216"),
    ("SSN", "501-28-7743"),
    ("SSN", "498-37-5518"),
    ("SSN", "498-37-6629"),

    # Driver's License
    ("DRIVERS_LICENSE_NUMBER", "E4827391"),

    # Investment Accounts
    ("INVESTMENT_ACCOUNT_NUMBER", "8291-4738-2019"),
    ("INVESTMENT_ACCOUNT_NUMBER", "VG-82947103-IRA"),
    ("INVESTMENT_ACCOUNT_NUMBER", "401K-F-2847391"),
    ("INVESTMENT_ACCOUNT_NUMBER", "MWM-0092847-JC"),
    ("CUSIP_NUMBER", "13063DAC3"),
    ("CRYPTO_WALLET", "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD08"),

    # Bank Accounts
    ("BANK_ACCOUNT_NUMBER", "9284710356"),
    ("ROUTING_NUMBER", "021000021"),
    ("ROUTING_NUMBER", "121202211"),

    # Insurance
    ("INSURANCE_POLICY", "NWM-LI-4829174"),
    ("HEALTH_INSURANCE_ID", "AET-82917453"),
    ("INSURANCE_POLICY", "SF-HO-4829174-01"),

    # Tax
    ("TAX_ID", "92-4817253"),
    ("FICO_SCORE", "812"),
    ("INCOME", "$847,000"),

    # Contact
    ("PHONE_NUMBER", "650-555-4829"),
    ("PHONE_NUMBER", "650-555-9182"),
    ("PHONE_NUMBER", "650-555-3347"),
    ("EMAIL_ADDRESS", "dblackwell@pacifictrustlaw.com"),
    ("EMAIL_ADDRESS", "jchen.private@protonmail.com"),
    ("EMAIL_ADDRESS", "jonathan.chen@quantumdynamics.io"),
    ("EMAIL_ADDRESS", "sreeves@reevescpa.com"),
    ("STREET_ADDRESS", "1847 Eucalyptus Drive, Atherton, CA 94027"),

    # Professional Licenses
    ("PROFESSIONAL_LICENSE_NUMBER", "CA-SBN-284719"),
    ("PROFESSIONAL_LICENSE_NUMBER", "CA-CPA-91827"),

    # Other IDs
    ("PASSPORT_NUMBER", "C48291573"),
    ("USERNAME", "jmchen_trust"),
    ("MOTHERS_MAIDEN_NAME", "Nakamura"),
    ("PROPERTY_TAX_PARCEL", "061-281-470"),
]

# ── Evaluation Logic ─────────────────────────────────────────────────

def normalize(text):
    return text.strip().lower().replace(" ", "").replace("-", "").replace(".", "").replace("$", "").replace(",", "")

def evaluate(detections, ground_truth):
    gt_by_value = defaultdict(set)
    for label, text in ground_truth:
        gt_by_value[normalize(text)].add(label)

    COMPAT = [
        {"PERSON_FULL_NAME", "PERSON_FIRST_NAME", "PERSON_LAST_NAME", "PERSON_MIDDLE_NAME"},
        {"SSN", "TAX_ID", "NATIONAL_ID"},
        {"CREDIT_CARD_NUMBER", "ACCOUNT_NUMBER_AMEX"},
        {"BANK_ACCOUNT_NUMBER", "ACCOUNT_LAST4", "BANK_ACCOUNT_LAST4", "GENERIC_LAST4"},
        {"ROUTING_NUMBER", "BANK_ACCOUNT_NUMBER"},
        {"INVESTMENT_ACCOUNT_NUMBER", "BANK_ACCOUNT_NUMBER", "MR_NUMBER"},
        {"EMPLOYEE_ID", "MEDICAL_RECORD_NUMBER"},
        {"INSURANCE_POLICY", "HEALTH_INSURANCE_ID"},
        {"PROFESSIONAL_LICENSE_NUMBER", "NATIONAL_ID"},
        {"PROPERTY_TAX_PARCEL", "COURT_CASE_NUMBER"},
        {"CRYPTO_WALLET", "UNKNOWN_SECRET", "UNKNOWN_PII"},
        {"FICO_SCORE", "AGE"},
        {"INCOME", "UNKNOWN_PII"},
        {"USERNAME", "UNKNOWN_PII"},
        {"MOTHERS_MAIDEN_NAME", "PERSON_LAST_NAME", "SECURITY_ANSWER"},
    ]

    def labels_match(det_label, gt_labels):
        if det_label in gt_labels:
            return True
        for group in COMPAT:
            if det_label in group and gt_labels & group:
                return True
        return False

    matched_gt = set()
    tp_by_label = defaultdict(int)
    fp_by_label = defaultdict(int)

    for di, d in enumerate(detections):
        det_norm = normalize(d.text)
        found = False

        # Exact match
        if det_norm in gt_by_value:
            gt_labels = gt_by_value[det_norm]
            if labels_match(d.label, gt_labels):
                for gi, (gl, gt) in enumerate(ground_truth):
                    if gi not in matched_gt and normalize(gt) == det_norm and labels_match(d.label, {gl}):
                        matched_gt.add(gi)
                        tp_by_label[d.label] += 1
                        found = True
                        break
                if not found:
                    tp_by_label[d.label] += 1
                    found = True

        # Substring match
        if not found:
            for gi, (gl, gt) in enumerate(ground_truth):
                if gi in matched_gt:
                    continue
                gt_norm = normalize(gt)
                if (det_norm and gt_norm and (det_norm in gt_norm or gt_norm in det_norm)):
                    if labels_match(d.label, {gl}):
                        matched_gt.add(gi)
                        tp_by_label[d.label] += 1
                        found = True
                        break

        if not found:
            fp_by_label[d.label] += 1

    fn_by_label = defaultdict(int)
    for gi, (gl, gt) in enumerate(ground_truth):
        if gi not in matched_gt:
            fn_by_label[gl] += 1

    all_labels = sorted(set(list(tp_by_label.keys()) + list(fp_by_label.keys()) + list(fn_by_label.keys())))
    total_tp = sum(tp_by_label.values())
    total_fp = sum(fp_by_label.values())
    total_fn = sum(fn_by_label.values())
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp_by_label": dict(tp_by_label), "fp_by_label": dict(fp_by_label),
        "fn_by_label": dict(fn_by_label), "all_labels": all_labels,
        "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
        "precision": precision, "recall": recall, "f1": f1,
        "matched_gt": matched_gt, "ground_truth": ground_truth,
    }


# ── Main ─────────────────────────────────────────────────────────────

def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

print_header("FINANCIAL PII ACCURACY EVALUATION")

from app import HybridPIIEngine

print("Loading engine...")
engine = HybridPIIEngine(
    use_gliner=True,
    gliner_model_name="knowledgator/gliner-pii-large-v1.0",
    gliner_threshold=0.15,
)
print(f"  GLiNER: {'ON' if engine.gliner_detector.enabled else 'OFF'}")
print(f"  LSTM: {'ON (' + str(len(engine.lstm_known_labels())) + ' labels)' if engine.self_trainer.is_ready() else 'OFF'}")

print_header("RUNNING DETECTION")

start = time.time()
result = engine.redact(FINANCIAL_TEXT)
elapsed = time.time() - start
print(f"  Time: {elapsed:.2f}s | Detections: {len(result.detections)} | Ground truth: {len(GROUND_TRUTH)}")

# ── Detection Table ──────────────────────────────────────────────────

print_header("ALL DETECTIONS")
print(f"{'#':<4} {'LABEL':<35} {'VALUE':<45} {'SCORE':>6} {'SOURCE':<18}")
print("-" * 115)
for i, d in enumerate(result.detections, 1):
    val = d.text[:43] + ".." if len(d.text) > 45 else d.text
    print(f"{i:<4} {d.label:<35} {val:<45} {d.score:>6.3f} {d.source:<18}")

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

# ── Missed PII ───────────────────────────────────────────────────────

print_header("MISSED PII (FALSE NEGATIVES)")
missed = [(GROUND_TRUTH[i][0], GROUND_TRUTH[i][1]) for i in range(len(GROUND_TRUTH)) if i not in metrics["matched_gt"]]
if missed:
    for label, text in missed:
        print(f"  MISSED: {label:<35} \"{text}\"")
else:
    print("  None! All ground truth PII detected.")

# ── False Positives ──────────────────────────────────────────────────

print_header("FALSE POSITIVES (sample)")
fp_count = 0
for d in result.detections:
    det_norm = normalize(d.text)
    is_tp = any(det_norm == normalize(gt) or det_norm in normalize(gt) or normalize(gt) in det_norm
               for _, gt in GROUND_TRUTH)
    if not is_tp:
        print(f"  FP: {d.label:<35} \"{d.text[:50]}\" (score={d.score:.3f}, source={d.source})")
        fp_count += 1
        if fp_count >= 15:
            print("  ... (truncated)")
            break

if fp_count == 0:
    print("  None! Zero false positives.")

print()
print("Done!")
