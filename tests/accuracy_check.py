"""Accuracy check: feed a controlled text with known PII, verify detections.

This script builds a ~25k char document with embedded PII of known types
and positions — including intentionally tricky / unknown PII that the model
must discover dynamically. After running the full pipeline it checks whether
each planted PII was detected with the correct label, and prints:

  1. Per-item FOUND / WRONG_LABEL / MISSED table
  2. Confusion matrix (expected label × detected label)
  3. Per-label precision / recall / F1
  4. Overall accuracy summary

Usage:
    python tests/accuracy_check.py
    python tests/accuracy_check.py --model knowledgator/gliner-pii-base-v1.0
    python tests/accuracy_check.py --ultra-canonical
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.engine import HybridPIIEngine  # noqa: E402
from app.models import Detection  # noqa: E402


# ── Known PII items planted in the document ─────────────────────────────
# Each entry: (pii_value, expected_label_or_labels, description)
# expected_label_or_labels can be a string or a set of acceptable labels.
PLANTED_PII: List[Tuple[str, Any, str]] = [
    # ── Person names ──
    ("Jonathan Michael Harrison", {"PERSON_FULL_NAME", "PERSON_FIRST_NAME", "PERSON_LAST_NAME"}, "full name"),
    ("Emily Chen", {"PERSON_FULL_NAME", "PERSON_FIRST_NAME", "PERSON_LAST_NAME"}, "full name"),
    ("Robert Williams", {"PERSON_FULL_NAME", "PERSON_FIRST_NAME", "PERSON_LAST_NAME"}, "full name"),
    ("Dr. Priya Sharma", {"PERSON_FULL_NAME", "PERSON_FIRST_NAME", "PERSON_LAST_NAME"}, "full name (titled)"),

    # ── Email addresses ──
    ("jonathan.harrison@globalfinance.com", "EMAIL_ADDRESS", "email"),
    ("emily.chen@techcorp.io", "EMAIL_ADDRESS", "email"),
    ("support@acme-services.org", "EMAIL_ADDRESS", "email"),

    # ── Phone numbers ──
    ("(555) 867-5309", "PHONE_NUMBER", "phone"),
    ("1-800-555-0199", "PHONE_NUMBER", "phone"),
    ("+44 20 7946 0958", "PHONE_NUMBER", "phone"),
    ("+91 98765 43210", "PHONE_NUMBER", "intl phone"),

    # ── SSN ──
    ("483-29-7165", "SSN", "SSN"),
    ("612-44-8901", "SSN", "SSN"),

    # ── Credit card numbers ──
    ("4532-0151-2345-6789", "CREDIT_CARD_NUMBER", "credit card"),
    ("5425-2334-1122-9988", "CREDIT_CARD_NUMBER", "credit card"),

    # ── Street addresses ──
    ("742 Evergreen Terrace, Springfield, IL 62704", "STREET_ADDRESS", "street address"),
    ("1600 Pennsylvania Avenue NW, Washington, DC 20500", "STREET_ADDRESS", "street address"),

    # ── Date of birth ──
    ("March 15, 1987", "DATE_OF_BIRTH", "DOB"),
    ("07/22/1994", "DATE_OF_BIRTH", "DOB"),

    # ── IP addresses ──
    ("192.168.14.231", "IP_ADDRESS", "IP address"),
    ("10.0.0.42", "IP_ADDRESS", "IP address"),

    # ── Bank account / routing ──
    ("8827461039", "BANK_ACCOUNT_NUMBER", "bank account"),
    ("021000021", "ROUTING_NUMBER", "routing number"),

    # ── Driver's license ──
    ("D120-4589-7823", "DRIVERS_LICENSE_NUMBER", "driver license"),

    # ── Passport ──
    ("X12345678", "PASSPORT_NUMBER", "passport"),

    # ── IBAN ──
    ("GB29 NWBK 6016 1331 9268 19", "IBAN", "IBAN"),

    # ── Medical ──
    ("MRN-2024-00847", "MEDICAL_RECORD_NUMBER", "medical record"),

    # ── Username ──
    ("jharrison_admin", "USERNAME", "username"),

    # ══════════════════════════════════════════════════════════════════
    # UNKNOWN / TRICKY PII — tests the model's ability to dynamically
    # identify PII types it hasn't been explicitly trained on, or types
    # that appear in unusual contexts.
    # ══════════════════════════════════════════════════════════════════

    # ── Vehicle Identification Number (VIN) ──
    ("1HGBH41JXMN109186", "VIN", "VIN"),

    # ── License plate ──
    ("7ABC123", "LICENSE_PLATE_NUMBER", "license plate"),

    # ── Employee ID (unusual format) ──
    ("EMP-2024-88431", {"EMPLOYEE_ID", "UNKNOWN_PII"}, "employee ID"),

    # ── Tax ID / EIN ──
    ("84-2913756", {"TAX_ID", "SSN", "UNKNOWN_PII"}, "tax ID / EIN"),

    # ── Biometric / unusual identifiers ──
    ("BIO-HASH-9f8e7d6c5b4a3210", {"UNKNOWN_PII", "API_KEY", "PASSWORD"}, "biometric hash"),

    # ── Crypto wallet address ──
    ("bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq", {"UNKNOWN_PII", "CRYPTO_WALLET"}, "crypto wallet"),

    # ── Insurance policy number ──
    ("POL-HLT-2024-991847", {"UNKNOWN_PII", "INSURANCE_POLICY"}, "insurance policy"),

    # ── Mother's maiden name (security question answer) ──
    ("Fitzpatrick", {"PERSON_FULL_NAME", "PERSON_LAST_NAME", "MOTHERS_MAIDEN_NAME", "UNKNOWN_PII"}, "maiden name"),

    # ── Court case number ──
    ("2024-CV-04821", {"COURT_CASE_NUMBER", "UNKNOWN_PII"}, "court case"),

    # ── Student ID ──
    ("STU-20241587", {"STUDENT_ID", "UNKNOWN_PII"}, "student ID"),

    # ── PIN (spoken in conversation) ──
    ("7392", {"PIN", "CARD_SECURITY_CODE", "UNKNOWN_PII"}, "PIN"),

    # ── Membership / loyalty number ──
    ("GOLD-MEM-8847291", {"UNKNOWN_PII"}, "loyalty number"),
]


def build_test_document() -> str:
    """Build a ~25k char document with known PII embedded in realistic context."""
    sections = []

    sections.append("""
CUSTOMER SERVICE TRANSCRIPT — CONFIDENTIAL
Date: 2024-11-15 | Duration: 47 minutes | Channel: Phone
Agent: Marcus Thompson | Department: Financial Services

--- BEGIN TRANSCRIPT ---

[00:00:12] Agent: Thank you for calling Global Financial Services. My name is Marcus. How can I help you today?

[00:00:18] Customer: Hi Marcus, my name is Jonathan Michael Harrison. I'm calling about my account — I think there may have been some unauthorized activity.

[00:00:25] Agent: I'm sorry to hear that, Mr. Harrison. Let me pull up your account. Can you verify your email address on file?

[00:00:31] Customer: Sure, it's jonathan.harrison@globalfinance.com.

[00:00:35] Agent: Thank you. And your phone number?

[00:00:38] Customer: (555) 867-5309.

[00:00:42] Agent: Perfect. For security verification, can you provide the last four digits of your Social Security Number?

[00:00:48] Customer: My full SSN is 483-29-7165. Wait, you only need the last four? Sorry about that.

[00:00:55] Agent: No worries, I'll note that for the security team. I can see your account now. Can you tell me what suspicious activity you noticed?

[00:01:02] Customer: There's a charge on my credit card ending in 6789. The full number is 4532-0151-2345-6789. I didn't make a purchase at "TechGadgets Online" for $847.99 on November 12th.

[00:01:15] Agent: I see that transaction. Let me check the details. While I look into this, can you confirm your mailing address?

[00:01:22] Customer: It's 742 Evergreen Terrace, Springfield, IL 62704.

[00:01:28] Agent: Thank you. And your date of birth for additional verification?

[00:01:32] Customer: March 15, 1987.

[00:01:36] Agent: Everything checks out. I'm going to flag this transaction and initiate a dispute. You'll receive a temporary credit within 3-5 business days.

[00:02:00] Customer: Also, I need to update my vehicle insurance. My VIN is 1HGBH41JXMN109186 and my license plate is 7ABC123.

[00:02:10] Agent: I've noted that. Let me also verify your employee ID for your corporate discount.

[00:02:15] Customer: It's EMP-2024-88431.

[00:02:20] Agent: And your company's tax ID?

[00:02:24] Customer: 84-2913756.

[00:02:30] Agent: I see you also have a health insurance policy on file. Can you confirm the policy number?

[00:02:35] Customer: Yes, it's POL-HLT-2024-991847.

[00:02:40] Agent: For the security question on your account — what is your mother's maiden name?

[00:02:45] Customer: Fitzpatrick.

[00:02:48] Agent: Perfect. And I notice there's a pending court case reference on your account?

[00:02:52] Customer: Yes, case number 2024-CV-04821. That's for the insurance dispute.

[00:02:58] Agent: One more thing — could you verify the PIN associated with your debit card?

[00:03:02] Customer: It's 7392.

[00:03:05] Customer: Oh, and I wanted to check on my son's student account. His student ID is STU-20241587.

[00:03:12] Customer: Also, can you check my loyalty program status? My membership number is GOLD-MEM-8847291.

[00:03:18] Customer: And I need to update my crypto wallet for the rewards program. It's bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq.

[00:03:28] Agent: I've updated everything. Is there anything else?

[00:03:32] Customer: My biometric authentication hash shows BIO-HASH-9f8e7d6c5b4a3210 — can you verify that's current?

[00:03:40] Agent: Yes, that matches our records. Your international phone number is +91 98765 43210, correct?

[00:03:45] Customer: That's right.
""")

    # Filler dialogue
    filler = """
[00:04:15] Customer: That sounds good. I also wanted to ask about my savings account.

[00:04:20] Agent: Of course. What would you like to know?

[00:04:25] Customer: I've been thinking about setting up automatic transfers. Can you walk me through the process?

[00:04:32] Agent: Absolutely. We offer several options for automatic transfers. You can set them up daily, weekly, bi-weekly, or monthly. The minimum transfer amount is $25. Would you like me to go through each option?

[00:04:45] Customer: Monthly would be fine. Let's start with $500 per month.

[00:04:50] Agent: Great choice. I'll set that up for you. The transfer will occur on the first business day of each month. Is there anything else I can help with regarding your accounts?

[00:05:00] Customer: Actually, I had another question about your investment products. I've been looking at mutual funds and I'm not sure which ones would be right for my situation.

[00:05:12] Agent: I'd be happy to provide some general information, but for specific investment advice, I'd recommend speaking with one of our certified financial advisors. Would you like me to schedule an appointment?

[00:05:25] Customer: That would be great, yes.

[00:05:28] Agent: I'll transfer you to our scheduling department after we finish here. Is there anything else?

[00:05:35] Customer: No, I think that covers everything for now. Thank you for your help with the disputed charge.

[00:05:40] Agent: You're welcome, Mr. Harrison. The dispute has been filed and you should receive confirmation via email within 24 hours. Is there anything else I can assist you with today?

[00:05:50] Customer: No, that's all. Thank you.

[00:05:53] Agent: Thank you for calling Global Financial Services. Have a great day.

--- END OF SECTION 1 ---
"""
    sections.append(filler)

    sections.append("""
--- SECTION 2: INTERNAL CASE NOTES ---

Case #: FD-2024-11847
Opened: 2024-11-15 by Agent Marcus Thompson
Customer: Jonathan Michael Harrison
Status: Under Investigation

CUSTOMER PROFILE:
  Full Name: Jonathan Michael Harrison
  Email: jonathan.harrison@globalfinance.com
  Phone: (555) 867-5309
  Address: 742 Evergreen Terrace, Springfield, IL 62704
  DOB: March 15, 1987
  SSN: 483-29-7165
  Primary Card: 4532-0151-2345-6789
  Employee ID: EMP-2024-88431
  Tax ID: 84-2913756
  VIN: 1HGBH41JXMN109186
  Plate: 7ABC123

NOTES:
- Customer reported unauthorized charge of $847.99 at TechGadgets Online on 2024-11-12
- Card has been flagged for monitoring
- Temporary credit to be issued within 3-5 business days
- Customer also inquired about automatic transfers and investment products
- Scheduled follow-up with financial advisor
- Dr. Priya Sharma from the compliance team will review this case

--- SECTION 3: RELATED CUSTOMER RECORD ---

A separate customer, Emily Chen, called in on the same day regarding a similar issue.

  Name: Emily Chen
  Email: emily.chen@techcorp.io
  Phone: 1-800-555-0199
  SSN: 612-44-8901
  Card: 5425-2334-1122-9988
  Address: 1600 Pennsylvania Avenue NW, Washington, DC 20500
  DOB: 07/22/1994
  IP Address (last login): 192.168.14.231

Emily reported a $1,200 charge from the same merchant. This suggests a potential data breach at TechGadgets Online. Escalating to the fraud investigation team.

--- SECTION 4: TECHNICAL DETAILS ---

System logs from the investigation:

  Source IP: 10.0.0.42
  Bank Account: 8827461039
  Routing Number: 021000021
  Driver License: D120-4589-7823
  Passport: X12345678
  IBAN: GB29 NWBK 6016 1331 9268 19
  Medical Record: MRN-2024-00847
  International contact: +44 20 7946 0958
  Support email: support@acme-services.org
  Admin username: jharrison_admin

The third-party vendor, Robert Williams, was contacted for additional information about the transactions.
""")

    # Padding to reach ~25k
    padding = """
--- SECTION 5: POLICY REFERENCE ---

Per company policy FP-2024-003, all suspected fraud cases must be:
1. Documented within 24 hours of customer report
2. Escalated to the fraud investigation team if the amount exceeds $500
3. Resolved within 10 business days
4. Followed up with the customer within 48 hours of resolution

The customer has been informed of the dispute process and the expected timeline for resolution. All communications will be logged in the case management system.

Additional notes on company procedures:
- All agents must verify customer identity using at least two of the following: full name, date of birth, last four digits of SSN, email address, or phone number
- Disputed transactions over $1,000 require supervisor approval
- Cards with multiple disputes within 90 days should be flagged for enhanced monitoring
- Customer satisfaction surveys should be sent within 7 days of case resolution

The quality assurance team reviews a random sample of 15% of all fraud cases monthly to ensure compliance with these procedures. Agents who consistently follow proper verification protocols receive performance bonuses.

Training materials for fraud detection are updated quarterly. The latest update includes new patterns for online merchant fraud, which has increased by 23% in the current fiscal year.

Regional managers should review escalated cases weekly and provide feedback to agents on areas for improvement. Cross-department collaboration between fraud detection, customer service, and IT security teams is essential for maintaining a robust defense against financial crimes.

Our partnership with law enforcement agencies ensures that cases involving amounts over $5,000 are reported to the appropriate authorities within the required timeframe. The legal team coordinates with external counsel when necessary.

Customer data protection remains our highest priority. All case files are encrypted at rest and in transit. Access is restricted to authorized personnel only, with full audit logging enabled.
"""
    current_len = sum(len(s) for s in sections)
    while current_len < 24000:
        sections.append(padding)
        current_len += len(padding)

    return "\n".join(sections)


def find_matches(
    text: str,
    detections: List[Detection],
    pii_value: str,
) -> List[Detection]:
    """Find detections that overlap with a planted PII value."""
    matching: List[Detection] = []

    # Find all positions of the planted value in the text
    pii_positions: List[Tuple[int, int]] = []
    pos = 0
    while True:
        idx = text.find(pii_value, pos)
        if idx == -1:
            break
        pii_positions.append((idx, idx + len(pii_value)))
        pos = idx + 1

    for d in detections:
        # Text containment check (handles partial matches)
        if pii_value in d.text or d.text in pii_value:
            if d not in matching:
                matching.append(d)
            continue
        # Normalized match (ignore spaces/dashes)
        d_norm = d.text.replace(" ", "").replace("-", "").lower()
        p_norm = pii_value.replace(" ", "").replace("-", "").lower()
        if d_norm == p_norm:
            if d not in matching:
                matching.append(d)
            continue
        # Positional overlap check
        for pii_start, pii_end in pii_positions:
            overlap = max(0, min(d.end, pii_end) - max(d.start, pii_start))
            min_span = min(pii_end - pii_start, d.end - d.start)
            if min_span > 0 and overlap / min_span > 0.5:
                if d not in matching:
                    matching.append(d)

    return matching


def check_detections(
    text: str,
    detections: List[Detection],
    planted: List[Tuple[str, Any, str]],
) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    """Check each planted PII against the detection list.

    Returns (found, wrong_label, missed, details_list).
    """
    found = 0
    wrong_label = 0
    missed = 0
    details: List[Dict[str, Any]] = []

    for pii_value, expected_labels, desc in planted:
        if isinstance(expected_labels, str):
            expected_labels = {expected_labels}

        matching_dets = find_matches(text, detections, pii_value)
        correct_label_dets = [d for d in matching_dets if d.label in expected_labels]
        wrong_label_dets = [d for d in matching_dets if d.label not in expected_labels]

        if correct_label_dets:
            status = "FOUND"
            found += 1
            d = correct_label_dets[0]
        elif wrong_label_dets:
            status = "WRONG_LABEL"
            wrong_label += 1
            d = wrong_label_dets[0]
        else:
            status = "MISSED"
            missed += 1
            d = None

        detail: Dict[str, Any] = {
            "pii_value": pii_value,
            "expected_labels": sorted(expected_labels),
            "description": desc,
            "status": status,
        }
        if d is not None:
            detail["detected_as"] = d.label
            detail["detected_text"] = d.text
            detail["score"] = round(d.score, 3)
            detail["source"] = d.source

        details.append(detail)

    return found, wrong_label, missed, details


def build_confusion_matrix(
    details: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """Build a confusion matrix: expected_label × detected_label.

    For items with multiple acceptable labels, we use the first expected label
    as the canonical expected label (the primary one).
    """
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_labels: Set[str] = set()

    for d in details:
        expected = d["expected_labels"][0]  # primary expected label
        all_labels.add(expected)

        if d["status"] == "FOUND":
            detected = d["detected_as"]
            all_labels.add(detected)
            matrix[expected][detected] += 1
        elif d["status"] == "WRONG_LABEL":
            detected = d["detected_as"]
            all_labels.add(detected)
            matrix[expected][detected] += 1
        else:
            matrix[expected]["<MISSED>"] += 1
            all_labels.add("<MISSED>")

    sorted_labels = sorted(all_labels - {"<MISSED>"}) + (["<MISSED>"] if "<MISSED>" in all_labels else [])
    return dict(matrix), sorted_labels


def print_confusion_matrix(
    matrix: Dict[str, Dict[str, int]],
    all_labels: List[str],
) -> None:
    """Print a compact confusion matrix."""
    # Build label abbreviations for column headers
    expected_labels = sorted(set(matrix.keys()))

    # Only show columns that have at least one entry
    active_cols: List[str] = []
    for col in all_labels:
        has_entry = any(matrix.get(row, {}).get(col, 0) > 0 for row in expected_labels)
        if has_entry:
            active_cols.append(col)

    # Abbreviate long labels for column headers
    def abbrev(label: str, max_len: int = 12) -> str:
        if len(label) <= max_len:
            return label
        return label[:max_len - 1] + "."

    print("=" * 80)
    print("CONFUSION MATRIX (rows = expected, columns = detected)")
    print("=" * 80)

    row_label_width = max(len(abbrev(l, 28)) for l in expected_labels) + 2
    col_width = max(max(len(abbrev(l)) for l in active_cols), 5) + 1

    # Header
    header = " " * row_label_width
    for col in active_cols:
        header += f"{abbrev(col):>{col_width}}"
    header += f"{'Total':>{col_width}}"
    print(header)
    print("-" * len(header))

    # Rows
    for row_label in expected_labels:
        row_data = matrix.get(row_label, {})
        line = f"{abbrev(row_label, 28):<{row_label_width}}"
        row_total = 0
        for col in active_cols:
            val = row_data.get(col, 0)
            row_total += val
            cell = str(val) if val > 0 else "."
            line += f"{cell:>{col_width}}"
        line += f"{row_total:>{col_width}}"
        print(line)

    # Totals row
    print("-" * len(header))
    totals_line = f"{'Total':<{row_label_width}}"
    grand_total = 0
    for col in active_cols:
        col_total = sum(matrix.get(row, {}).get(col, 0) for row in expected_labels)
        grand_total += col_total
        totals_line += f"{col_total:>{col_width}}"
    totals_line += f"{grand_total:>{col_width}}"
    print(totals_line)


def compute_per_label_metrics(
    details: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute precision, recall, F1 per expected label category."""
    # Group by primary expected label
    by_label: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fn": 0, "wrong": 0})

    for d in details:
        label = d["expected_labels"][0]
        if d["status"] == "FOUND":
            by_label[label]["tp"] += 1
        elif d["status"] == "WRONG_LABEL":
            by_label[label]["wrong"] += 1
        else:
            by_label[label]["fn"] += 1

    metrics: List[Dict[str, Any]] = []
    for label in sorted(by_label.keys()):
        tp = by_label[label]["tp"]
        fn = by_label[label]["fn"]
        wrong = by_label[label]["wrong"]
        total = tp + fn + wrong
        recall = tp / total if total else 0.0
        # Precision: of the items we detected for this type, how many were correct?
        # This is approximate since we're only looking at planted items
        precision = tp / (tp + wrong) if (tp + wrong) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics.append({
            "label": label,
            "planted": total,
            "found": tp,
            "wrong_label": wrong,
            "missed": fn,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        })

    return metrics


def main():
    parser = argparse.ArgumentParser(description="PII accuracy check with known planted data")
    parser.add_argument(
        "--model",
        default="knowledgator/gliner-pii-base-v1.0",
        help="GLiNER model to test",
    )
    parser.add_argument("--ultra-canonical", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 80)
    print(f"ACCURACY CHECK — planted PII in ~25k char document")
    print(f"  model: {args.model}")
    print(f"  ultra-canonical: {args.ultra_canonical}")
    print("=" * 80)

    text = build_test_document()
    print(f"  document length: {len(text):,} chars")
    print(f"  planted PII items: {len(PLANTED_PII)}")
    known_count = sum(1 for _, l, _ in PLANTED_PII if isinstance(l, str) or "UNKNOWN_PII" not in l)
    unknown_count = len(PLANTED_PII) - known_count
    print(f"    known types: {known_count}")
    print(f"    unknown / tricky: {unknown_count}")
    print()

    print("loading engine...")
    t0 = time.perf_counter()
    engine = HybridPIIEngine(
        gliner_model_name=args.model,
        use_ultra_canonical=args.ultra_canonical,
        inference_only=True,
    )
    print(f"  init: {(time.perf_counter() - t0) * 1000:.0f} ms")

    # Warmup: first call triggers NVRTC kernel compilation (~2s one-time cost)
    print("warmup (first call, not timed)...")
    _ = engine.detect(text[:2000])
    print("  warmup done")

    print("running detect()...")
    t0 = time.perf_counter()
    detections = engine.detect(text)
    detect_ms = (time.perf_counter() - t0) * 1000
    print(f"  detect: {detect_ms:.0f} ms  (steady-state, post-warmup)")
    print(f"  total detections: {len(detections)}")
    print()

    # ── Per-item results ────────────────────────────────────────────────
    found, wrong_label, missed, details = check_detections(text, detections, PLANTED_PII)

    print("=" * 80)
    print("PER-ITEM RESULTS")
    print("=" * 80)
    print(f"  {'Status':<12} {'Type':<18} {'Expected':<28} {'Value':<38} {'Detected As'}")
    print(f"  {'-'*12} {'-'*18} {'-'*28} {'-'*38} {'-'*35}")

    for d in details:
        status_icon = {
            "FOUND": "FOUND",
            "WRONG_LABEL": "WRONG_LBL",
            "MISSED": "MISSED",
        }[d["status"]]
        expected = d["expected_labels"][0] if len(d["expected_labels"]) == 1 else "/".join(d["expected_labels"][:2])
        value = d["pii_value"][:36]
        detected = ""
        if d.get("detected_as"):
            detected = f"{d['detected_as']} (src={d['source']}, {d['score']:.2f})"

        color = ""
        reset = ""
        if d["status"] == "FOUND":
            color, reset = "\033[92m", "\033[0m"
        elif d["status"] == "WRONG_LABEL":
            color, reset = "\033[93m", "\033[0m"
        elif d["status"] == "MISSED":
            color, reset = "\033[91m", "\033[0m"

        print(f"{color}  {status_icon:<12} {d['description']:<18} {expected:<28} {value:<38} {detected}{reset}")

    # ── Confusion matrix ────────────────────────────────────────────────
    print()
    matrix, all_labels = build_confusion_matrix(details)
    print_confusion_matrix(matrix, all_labels)

    # ── Per-label precision / recall / F1 ───────────────────────────────
    print()
    print("=" * 80)
    print("PER-LABEL METRICS (on planted items only)")
    print("=" * 80)
    per_label = compute_per_label_metrics(details)
    print(f"  {'Label':<30} {'Planted':>8} {'Found':>6} {'Wrong':>6} {'Miss':>6} {'Recall':>8} {'Prec':>8} {'F1':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for m in per_label:
        print(
            f"  {m['label']:<30} {m['planted']:>8} {m['found']:>6} {m['wrong_label']:>6} "
            f"{m['missed']:>6} {m['recall']:>7.0%} {m['precision']:>7.0%} {m['f1']:>7.2f}"
        )
    # Totals
    total_planted = sum(m["planted"] for m in per_label)
    total_found = sum(m["found"] for m in per_label)
    total_wrong = sum(m["wrong_label"] for m in per_label)
    total_missed = sum(m["missed"] for m in per_label)
    overall_recall = total_found / total_planted if total_planted else 0
    overall_prec = total_found / (total_found + total_wrong) if (total_found + total_wrong) else 0
    overall_f1 = 2 * overall_prec * overall_recall / (overall_prec + overall_recall) if (overall_prec + overall_recall) else 0
    print(f"  {'-'*30} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    print(
        f"  {'OVERALL':<30} {total_planted:>8} {total_found:>6} {total_wrong:>6} "
        f"{total_missed:>6} {overall_recall:>7.0%} {overall_prec:>7.0%} {overall_f1:>7.2f}"
    )

    # ── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"SUMMARY: {total_found}/{total_planted} FOUND, "
          f"{total_wrong} WRONG LABEL, {total_missed} MISSED")
    print(f"  Overall recall:    {overall_recall:.1%}")
    print(f"  Overall precision: {overall_prec:.1%}")
    print(f"  Overall F1:        {overall_f1:.3f}")
    print("=" * 80)

    # ── All detection labels found (including extras not in planted list) ──
    print()
    print("ALL DETECTION LABELS IN OUTPUT:")
    label_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    for d in detections:
        label_counts[d.label] = label_counts.get(d.label, 0) + 1
        source_counts[d.source] = source_counts.get(d.source, 0) + 1
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<35} {count:>4}")
    print()
    print("DETECTIONS BY SOURCE:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source:<20} {count:>4}")

    return 0 if (total_missed == 0 and total_wrong == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
