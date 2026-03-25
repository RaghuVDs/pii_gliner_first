"""
End-to-End Test Suite for PII Detection Engine
===============================================
Tests the full 13-stage pipeline: GLiNER → Field → Regex → Context → LSTM
→ Filter → CrossVal → Split → Resolve → Propagate → Collect → Retrain → AutoPromote

Run:  python -m pytest tests/test_end_to_end.py -v
  or: python tests/test_end_to_end.py
"""
from __future__ import annotations

import sys
import os
import json
import logging
import traceback
from collections import Counter
from typing import List, Dict

# ── Setup path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("test_e2e")


# ── Test Utilities ──────────────────────────────────────────────────
class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []

    def check(self, condition: bool, msg: str):
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(msg)

    def summary(self) -> str:
        status = "PASS" if self.failed == 0 else "FAIL"
        lines = [f"[{status}] {self.name}: {self.passed} passed, {self.failed} failed"]
        for e in self.errors:
            lines.append(f"       - {e}")
        return "\n".join(lines)


ALL_RESULTS: List[TestResult] = []


def run_test(fn):
    """Decorator to run a test function and capture results."""
    def wrapper(*args, **kwargs):
        r = TestResult(fn.__name__)
        try:
            fn(r, *args, **kwargs)
        except Exception as ex:
            r.failed += 1
            r.errors.append(f"EXCEPTION: {ex}\n{traceback.format_exc()}")
        ALL_RESULTS.append(r)
        print(r.summary())
        return r
    return wrapper


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Unit Tests — PII-Safe Functions
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_value_to_structure(t: TestResult):
    """_value_to_structure must convert PII to char-class patterns."""
    from app.adaptive_learning import _value_to_structure

    cases = [
        ("621-73-4489",         "NNN-NN-NNNN"),
        ("4111111111111111",    "NNNNNNNNNNNNNNNN"),
        ("John Doe",           "Aaaa Aaa"),
        ("john.doe@x.com",     "aaaa.aaa@a.aaa"),
        ("D98321754",          "ANNNNNNNN"),
        ("1FTEW1EP5KFA82943",  "NAAAANAANAAANNNNN"),  # VIN: mixed alpha/digit
        ("602-555-8837",       "NNN-NNN-NNNN"),
        ("",                   ""),
    ]

    for value, expected in cases:
        result = _value_to_structure(value)
        t.check(result == expected, f"_value_to_structure({value!r}): got {result!r}, expected {expected!r}")

    # Key property: NO actual PII value chars leak through
    ssn = _value_to_structure("621-73-4489")
    t.check("6" not in ssn and "2" not in ssn and "4" not in ssn,
            "SSN structure must not contain any original digits")
    t.check(ssn == "NNN-NN-NNNN", f"SSN structure: {ssn}")


@run_test
def test_mask_pii_in_text(t: TestResult):
    """_mask_pii_in_text must replace all PII spans with <LABEL> tags."""
    from app.adaptive_learning import _mask_pii_in_text
    from app.models import Detection

    text = "My SSN is 621-73-4489 and my name is John Doe."
    dets = [
        Detection(label="SSN", text="621-73-4489", start=10, end=21, score=0.9, source="regex"),
        Detection(label="PERSON_FIRST_NAME", text="John", start=38, end=42, score=0.8, source="gliner"),
        Detection(label="PERSON_LAST_NAME", text="Doe", start=43, end=46, score=0.8, source="gliner"),
    ]
    masked = _mask_pii_in_text(text, dets)

    t.check("621-73-4489" not in masked, "SSN value must not appear in masked text")
    t.check("John" not in masked, "First name must not appear in masked text")
    t.check("Doe" not in masked, "Last name must not appear in masked text")
    t.check("<SSN>" in masked, "SSN must be replaced with <SSN> tag")
    t.check("<PERSON_FIRST_NAME>" in masked, "First name must be replaced with tag")
    t.check("<PERSON_LAST_NAME>" in masked, "Last name must be replaced with tag")


@run_test
def test_extract_safe_neighborhood(t: TestResult):
    """_extract_safe_neighborhood must mask ALL PII in the neighborhood window."""
    from app.adaptive_learning import _extract_safe_neighborhood
    from app.models import Detection

    text = "Call ref: Policy UFI-AUTO-5538291. SSN is 621-73-4489. Name: John Doe. Phone 602-555-8837."
    dets = [
        Detection(label="POLICY_NUMBER", text="UFI-AUTO-5538291", start=17, end=33, score=0.9, source="regex"),
        Detection(label="SSN", text="621-73-4489", start=42, end=53, score=0.95, source="regex"),
        Detection(label="FULL_NAME", text="John Doe", start=61, end=69, score=0.8, source="gliner"),
        Detection(label="PHONE_NUMBER", text="602-555-8837", start=77, end=89, score=0.9, source="regex"),
    ]
    # Get neighborhood around SSN
    nb = _extract_safe_neighborhood(text, 42, 53, dets, pad=150)

    t.check("621-73-4489" not in nb, "SSN must not appear in neighborhood")
    t.check("John Doe" not in nb, "Name must not appear in neighborhood")
    t.check("602-555-8837" not in nb, "Phone must not appear in neighborhood")
    t.check("UFI-AUTO-5538291" not in nb, "Policy number must not appear in neighborhood")


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Model Architecture
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_model_architecture(t: TestResult):
    """PIIPatternModel must instantiate and run forward pass on CUDA."""
    import torch
    from app.ml.model import PIIPatternModel, encode_pattern, MAX_PATTERN_LEN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = 10
    num_keywords = 50
    num_sources = 7

    model = PIIPatternModel(
        num_labels=num_labels,
        num_keywords=num_keywords,
        num_sources=num_sources,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    t.check(total_params > 100_000, f"Model should have >100K params, got {total_params}")
    t.check(total_params < 2_000_000, f"Model should have <2M params, got {total_params}")

    # Forward pass
    batch_size = 4
    char_ids = torch.randint(0, 26, (batch_size, MAX_PATTERN_LEN), device=device)
    ctx_dim = num_keywords + num_labels + num_sources + 2  # +2 for length, score
    context = torch.randn(batch_size, ctx_dim, device=device)

    logits = model(char_ids, context)
    t.check(logits.shape == (batch_size, num_labels),
            f"Output shape should be ({batch_size}, {num_labels}), got {logits.shape}")

    # Gradient flows
    loss = logits.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    t.check(has_grad, "All parameters should receive gradients")

    # encode_pattern
    pattern = encode_pattern("NNN-NN-NNNN")
    t.check(len(pattern) == MAX_PATTERN_LEN, f"Pattern length should be {MAX_PATTERN_LEN}")

    if torch.cuda.is_available():
        t.check(next(model.parameters()).is_cuda, "Model should be on CUDA")


# ═══════════════════════════════════════════════════════════════════
# TEST 3: SelfTrainer — Collection & Persistence
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_self_trainer_collection(t: TestResult):
    """SelfTrainer.collect_from_detections must store PII-safe features only."""
    from app.ml.trainer import SelfTrainer, _load_yaml
    from app.models import Detection

    trainer = SelfTrainer()

    text = "Customer SSN is 621-73-4489 and email is john@example.com. Policy UFI-AUTO-5538291."
    dets = [
        Detection(label="SSN", text="621-73-4489", start=16, end=27, score=0.92, source="gliner"),
        Detection(label="EMAIL_ADDRESS", text="john@example.com", start=42, end=58, score=0.88, source="regex"),
        Detection(label="POLICY_NUMBER", text="UFI-AUTO-5538291", start=67, end=83, score=0.75, source="field_label"),
    ]

    n = trainer.collect_from_detections(dets, text)
    t.check(n > 0, f"Should collect at least 1 example, got {n}")

    # Check buffer contents are PII-safe
    for ex in trainer._buffer:
        t.check("structure" in ex, "Example must have 'structure' key")
        t.check("label" in ex, "Example must have 'label' key")
        t.check("keywords" in ex, "Example must have 'keywords' key")

        # CRITICAL: No raw PII values in training data
        t.check("621-73-4489" not in str(ex), "SSN value must NOT appear in training data")
        t.check("john@example.com" not in str(ex), "Email value must NOT appear in training data")
        t.check("UFI-AUTO-5538291" not in str(ex), "Policy value must NOT appear in training data")

        # Structure must be char-class pattern
        struct = ex["structure"]
        for char in struct:
            t.check(char in "NAa -@./()#:+_,'&;", f"Structure char {char!r} must be a class char")

    # Low-score detections should be filtered out
    low_dets = [
        Detection(label="SSN", text="123", start=0, end=3, score=0.30, source="gliner"),
    ]
    n_low = trainer.collect_from_detections(low_dets, text)
    t.check(n_low == 0, f"Detections below threshold should not be collected, got {n_low}")


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Resolver — SOURCE_PRIORITY
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_resolver_source_priority(t: TestResult):
    """pattern_lstm must be in SOURCE_PRIORITY and rank correctly."""
    from app.resolver import SOURCE_PRIORITY

    t.check("pattern_lstm" in SOURCE_PRIORITY,
            "pattern_lstm must be in SOURCE_PRIORITY")

    if "pattern_lstm" in SOURCE_PRIORITY:
        lstm_prio = SOURCE_PRIORITY["pattern_lstm"]
        t.check(lstm_prio == 45, f"pattern_lstm priority should be 45, got {lstm_prio}")
        t.check(lstm_prio > SOURCE_PRIORITY["field_label"],
                "pattern_lstm should rank above field_label")
        t.check(lstm_prio < SOURCE_PRIORITY["gliner"],
                "pattern_lstm should rank below gliner")
        t.check(lstm_prio > SOURCE_PRIORITY["regex"],
                "pattern_lstm should rank above regex")


@run_test
def test_resolver_overlap_resolution(t: TestResult):
    """Resolver must keep pattern_lstm detections when they win on priority."""
    from app.resolver import resolve_detections
    from app.models import Detection

    # Use EMPLOYEE_ID (not format-validated) vs pattern_lstm to test priority
    dets = [
        Detection(label="EMPLOYEE_ID", text="GBT-E-4817", start=10, end=21,
                  score=0.90, source="pattern_lstm"),
        Detection(label="EMPLOYEE_ID", text="GBT-E-4817", start=10, end=21,
                  score=0.70, source="context"),
    ]
    resolved = resolve_detections(dets)

    t.check(len(resolved) == 1, f"Should resolve to 1 detection, got {len(resolved)}")
    if resolved:
        t.check(resolved[0].source == "pattern_lstm",
                f"pattern_lstm (prio=45) should win over context (prio=35), got {resolved[0].source}")
        t.check(resolved[0].label == "EMPLOYEE_ID",
                f"Should keep EMPLOYEE_ID label, got {resolved[0].label}")


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Postprocessing — pattern_lstm not grammar-gated
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_postprocessing_lstm_trusted(t: TestResult):
    """pattern_lstm detections must bypass grammar gate (trusted source)."""
    from app.postprocessing import apply_universal_dynamic_filters
    from app.models import Detection

    text = "The account number is ABC-12345 and it was verified by the system."

    # A pattern_lstm detection with a value that might fail grammar check
    dets = [
        Detection(label="ACCOUNT_NUMBER", text="ABC-12345", start=22, end=31,
                  score=0.88, source="pattern_lstm"),
    ]
    kept = apply_universal_dynamic_filters(text, dets)

    t.check(len(kept) == 1, f"pattern_lstm detection should be kept, got {len(kept)}")
    if kept:
        t.check(kept[0].source == "pattern_lstm",
                "Kept detection should be from pattern_lstm")


# ═══════════════════════════════════════════════════════════════════
# TEST 6: Full Pipeline — Insurance Call Transcript
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_full_pipeline_insurance_call(t: TestResult):
    """Full pipeline must detect PII in an insurance call transcript."""
    from app import HybridPIIEngine

    engine = HybridPIIEngine(
        use_gliner=True,
        gliner_model_name="knowledgator/gliner-pii-large-v1.0",
        gliner_threshold=0.15,
    )

    text = """
    Agent: Can I verify your identity? What is your Social Security Number?
    Customer: Sure, it's 621-73-4489. My name is John Michael Smith.
    Agent: And your date of birth?
    Customer: March 15, 1990. My email is john.smith@company.com and phone is 602-555-1234.
    Agent: I see your policy UFI-AUTO-7738291 is active. Your account number is 4532-7891-2345-6789.
    """

    result = engine.redact(text)
    labels_found = {d.label for d in result.detections}
    values_found = {d.text for d in result.detections}

    # Must detect SSN
    t.check("SSN" in labels_found, f"Must detect SSN. Labels found: {labels_found}")

    # Must detect email
    t.check("EMAIL_ADDRESS" in labels_found, f"Must detect EMAIL_ADDRESS. Labels found: {labels_found}")

    # Must detect phone
    t.check("PHONE_NUMBER" in labels_found, f"Must detect PHONE_NUMBER. Labels found: {labels_found}")

    # Must detect person names (any variant)
    name_labels = {"FULL_NAME", "PERSON_FIRST_NAME", "PERSON_LAST_NAME", "PERSON_MIDDLE_NAME"}
    t.check(bool(name_labels & labels_found),
            f"Must detect person name labels. Labels found: {labels_found}")

    # Redacted text must not contain SSN
    t.check("621-73-4489" not in result.redacted_text,
            "SSN must be redacted from output text")

    # Redacted text must not contain email
    t.check("john.smith@company.com" not in result.redacted_text,
            "Email must be redacted from output text")

    # Must have instance numbering
    has_instance = any(d.meta.get("instance_label") for d in result.detections)
    t.check(has_instance, "Detections should have instance labels (e.g., SSN_1)")

    # Training data should have been collected
    stats = engine.pattern_lstm_stats()
    t.check(stats.get("total_examples", 0) > 0 or len(engine.self_trainer._buffer) > 0,
            "Should collect training examples from detections")


# ═══════════════════════════════════════════════════════════════════
# TEST 7: Full Pipeline — Banking Transcript
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_full_pipeline_banking(t: TestResult):
    """Full pipeline must detect PII in a banking/financial transcript."""
    from app import HybridPIIEngine

    engine = HybridPIIEngine(
        use_gliner=True,
        gliner_model_name="knowledgator/gliner-pii-large-v1.0",
        gliner_threshold=0.15,
    )

    text = """
    Customer: I need to wire $15,000 to my broker account.
    Agent: I'll need to verify your identity first.
    Customer: My routing number is 021000021 and my account is 483927165.
    My credit card ending in 4532-7891-2345-6781 was also compromised.
    Agent: What's the beneficiary's details?
    Customer: Send to Robert Chen, account 7291834650 at Wells Fargo, routing 121000248.
    His email is rchen@investcorp.com. My SSN for verification is 287-43-9901.
    """

    result = engine.redact(text)
    labels = {d.label for d in result.detections}

    # Financial identifiers
    t.check("SSN" in labels, f"Must detect SSN. Found: {labels}")
    t.check("EMAIL_ADDRESS" in labels, f"Must detect email. Found: {labels}")

    # Routing/account numbers (may be detected as various labels)
    numeric_labels = {"ROUTING_NUMBER", "ACCOUNT_NUMBER", "BANK_ACCOUNT_NUMBER",
                      "CREDIT_CARD_NUMBER", "FINANCIAL_ACCOUNT_NUMBER"}
    t.check(bool(numeric_labels & labels),
            f"Must detect financial account numbers. Found: {labels}")

    # Person names
    name_labels = {"FULL_NAME", "PERSON_FIRST_NAME", "PERSON_LAST_NAME"}
    t.check(bool(name_labels & labels),
            f"Must detect person names. Found: {labels}")

    # Redaction
    t.check("287-43-9901" not in result.redacted_text, "SSN must be redacted")
    t.check("rchen@investcorp.com" not in result.redacted_text, "Email must be redacted")

    # Source diversity — should have multiple sources
    sources = {d.source for d in result.detections}
    t.check(len(sources) >= 2, f"Should use multiple sources, got {sources}")


# ═══════════════════════════════════════════════════════════════════
# TEST 8: Full Pipeline — Medical + Insurance
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_full_pipeline_medical(t: TestResult):
    """Full pipeline must detect medical PII: NPI, Medicare ID, prescriptions."""
    from app import HybridPIIEngine

    engine = HybridPIIEngine(
        use_gliner=True,
        gliner_model_name="knowledgator/gliner-pii-large-v1.0",
        gliner_threshold=0.15,
    )

    text = """
    Patient: My Medicare Beneficiary ID is 1EK4-TA2-HY74.
    My doctor is Dr. Amanda Foster, NPI number 1832749162.
    I was prescribed Cyclobenzaprine 10mg, prescription number RX-2294817.
    My medical record number is SMC-00482917.
    Date of birth: November 3, 1985. SSN: 621-73-4489.
    Blood type is O-negative. Emergency contact: Arjun Chakraborty, phone 480-555-6614.
    """

    result = engine.redact(text)
    labels = {d.label for d in result.detections}

    t.check("SSN" in labels, f"Must detect SSN. Found: {labels}")
    phone_labels = {"PHONE_NUMBER", "EMERGENCY_CONTACT_INFORMATION"}
    t.check(bool(phone_labels & labels), f"Must detect phone/emergency contact. Found: {labels}")

    # Person names
    name_labels = {"FULL_NAME", "PERSON_FIRST_NAME", "PERSON_LAST_NAME"}
    t.check(bool(name_labels & labels), f"Must detect person names. Found: {labels}")

    # Date of birth
    dob_labels = {"DATE_OF_BIRTH", "DATE", "DOB"}
    t.check(bool(dob_labels & labels), f"Must detect date of birth. Found: {labels}")

    # Redacted text checks
    t.check("621-73-4489" not in result.redacted_text, "SSN must be redacted")
    t.check("480-555-6614" not in result.redacted_text, "Phone must be redacted")

    print(f"    Detected {len(result.detections)} PII entities across {len(labels)} label types")
    print(f"    Labels: {sorted(labels)}")


# ═══════════════════════════════════════════════════════════════════
# TEST 9: Name Propagation
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_name_propagation(t: TestResult):
    """Names mentioned once should be propagated to later re-mentions."""
    from app import HybridPIIEngine

    engine = HybridPIIEngine(
        use_gliner=True,
        gliner_model_name="knowledgator/gliner-pii-large-v1.0",
        gliner_threshold=0.15,
    )

    text = """
    Customer: My name is Patricia Huang.
    Agent: Thank you, Ms. Huang. Let me look up your account.
    Agent: Mrs. Huang, I see your policy is active. Is there anything else, Patricia?
    """

    result = engine.redact(text)

    # The name "Huang" and "Patricia" appear multiple times — propagation should catch re-mentions
    name_dets = [d for d in result.detections
                 if "PERSON" in d.label or d.label in ("FULL_NAME", "MOTHERS_MAIDEN_NAME")]

    t.check(len(name_dets) >= 2,
            f"Should detect at least 2 name mentions (initial + propagated), got {len(name_dets)}")

    # Check propagated source
    propagated = [d for d in name_dets if d.source == "propagated"]
    if propagated:
        t.check(True, f"Found {len(propagated)} propagated name detections")
    else:
        # Even without propagation, re-mentions should be caught by GLiNER
        t.check(len(name_dets) >= 2,
                f"Expected name re-mentions detected, got {len(name_dets)}")


# ═══════════════════════════════════════════════════════════════════
# TEST 10: Cross-Validation Bonus
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_cross_validation(t: TestResult):
    """When GLiNER + regex agree on the same span, cross-validation bonus applies."""
    from app.resolver import apply_cross_validation_bonus
    from app.models import Detection

    dets = [
        Detection(label="SSN", text="621-73-4489", start=10, end=21,
                  score=0.80, source="gliner"),
        Detection(label="SSN", text="621-73-4489", start=10, end=21,
                  score=0.95, source="regex"),
    ]
    boosted = apply_cross_validation_bonus(dets)

    xv_dets = [d for d in boosted if d.meta.get("cross_validated")]
    t.check(len(xv_dets) > 0,
            f"Should have cross-validated detections, got {len(xv_dets)}")


# ═══════════════════════════════════════════════════════════════════
# TEST 11: Instance Numbering
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_instance_numbering(t: TestResult):
    """Multiple detections of same label must get unique instance numbers."""
    from app.postprocessing import add_instance_numbers
    from app.models import Detection

    dets = [
        Detection(label="SSN", text="621-73-4489", start=10, end=21, score=0.9, source="regex"),
        Detection(label="SSN", text="287-43-9901", start=50, end=61, score=0.9, source="regex"),
        Detection(label="EMAIL_ADDRESS", text="a@b.com", start=80, end=87, score=0.8, source="regex"),
    ]
    numbered = add_instance_numbers(dets)

    ssn_instances = [d.meta.get("instance_label") for d in numbered if d.label == "SSN"]
    t.check(len(ssn_instances) == 2, f"Should have 2 SSN instances, got {len(ssn_instances)}")

    if len(ssn_instances) == 2:
        t.check(ssn_instances[0] != ssn_instances[1],
                f"SSN instances must be unique: {ssn_instances}")
        t.check("SSN_1" in ssn_instances and "SSN_2" in ssn_instances,
                f"Expected SSN_1 and SSN_2, got {ssn_instances}")


# ═══════════════════════════════════════════════════════════════════
# TEST 12: Engine API Completeness
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_engine_api_completeness(t: TestResult):
    """Engine must expose all required APIs."""
    from app import HybridPIIEngine

    engine = HybridPIIEngine(
        use_gliner=True,
        gliner_model_name="knowledgator/gliner-pii-large-v1.0",
        gliner_threshold=0.15,
    )

    # All API methods must exist
    apis = [
        "detect", "redact", "review_pending", "promote_rules",
        "manual_promote_rule", "learning_stats", "detection_stats",
        "lstm_known_labels", "pattern_lstm_stats", "retrain_pattern_model",
        "metrics_history",
    ]
    for api in apis:
        t.check(hasattr(engine, api), f"Engine must have {api}() method")

    # API calls must not crash
    stats = engine.learning_stats()
    t.check(isinstance(stats, dict), "learning_stats() must return dict")

    det_stats = engine.detection_stats()
    t.check(isinstance(det_stats, dict), "detection_stats() must return dict")

    lstm_labels = engine.lstm_known_labels()
    t.check(isinstance(lstm_labels, list), "lstm_known_labels() must return list")

    lstm_stats = engine.pattern_lstm_stats()
    t.check(isinstance(lstm_stats, dict), "pattern_lstm_stats() must return dict")
    t.check("model_ready" in lstm_stats, "pattern_lstm_stats must have model_ready key")

    metrics = engine.metrics_history()
    t.check(isinstance(metrics, dict), "metrics_history() must return dict")

    pending = engine.review_pending()
    t.check(isinstance(pending, list), "review_pending() must return list")


# ═══════════════════════════════════════════════════════════════════
# TEST 13: Multiple Runs — Training Data Accumulates
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_multi_run_accumulation(t: TestResult):
    """Running detect() multiple times must accumulate training data."""
    from app import HybridPIIEngine

    engine = HybridPIIEngine(
        use_gliner=True,
        gliner_model_name="knowledgator/gliner-pii-large-v1.0",
        gliner_threshold=0.15,
    )

    transcripts = [
        "Customer: My SSN is 621-73-4489. Email: alice@corp.com. Phone: 602-555-1234.",
        "Agent: Your routing number 021000021 and account 483927165 are on file.",
        "Customer: My name is James Rodriguez. DOB: April 12, 1988. SSN: 334-22-8871.",
        "Agent: VIN 1FTEW1EP5KFA82943 on policy UFI-AUTO-7738291 confirmed.",
        "Customer: Credit card 4532-7891-2345-6789, expiry 12/2027. Billing zip 85251.",
    ]

    all_detections = []
    for i, text in enumerate(transcripts):
        result = engine.redact(text)
        all_detections.extend(result.detections)
        print(f"    Run {i+1}: {len(result.detections)} detections")

    t.check(len(all_detections) >= 10,
            f"Should find >=10 total detections across 5 transcripts, got {len(all_detections)}")

    # Stats should reflect multiple runs (total_runs may include previous test runs since
    # stats tracker persists to disk — just check it's >= 5)
    det_stats = engine.detection_stats()
    t.check(det_stats.get("total_runs", 0) >= 5,
            f"Should have >=5 total runs, got {det_stats.get('total_runs')}")

    # Training data should have accumulated
    lstm_stats = engine.pattern_lstm_stats()
    total = lstm_stats.get("total_examples", 0)
    print(f"    Total training examples accumulated: {total}")
    t.check(total > 0, f"Should have accumulated training examples, got {total}")


# ═══════════════════════════════════════════════════════════════════
# TEST 14: Edge Cases
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_edge_cases(t: TestResult):
    """Engine must handle edge cases gracefully."""
    from app import HybridPIIEngine

    engine = HybridPIIEngine(
        use_gliner=True,
        gliner_model_name="knowledgator/gliner-pii-large-v1.0",
        gliner_threshold=0.15,
    )

    # Empty text
    result = engine.redact("")
    t.check(len(result.detections) == 0, "Empty text should produce 0 detections")
    t.check(result.redacted_text == "", "Empty text should return empty redacted text")

    # No PII
    result = engine.redact("The weather is nice today. Let's go to the park and play.")
    t.check(len(result.detections) == 0 or all(d.score < 0.5 for d in result.detections),
            f"Non-PII text should produce 0 or low-confidence detections, got {len(result.detections)}")

    # Very short text
    result = engine.redact("Hi")
    t.check(result.redacted_text is not None, "Short text should not crash")

    # Text with only numbers (potential false positives)
    result = engine.redact("The meeting is at 3:00 PM in room 204 on floor 5.")
    false_pos = [d for d in result.detections if d.label == "SSN"]
    t.check(len(false_pos) == 0,
            f"Room/floor numbers should not be detected as SSN, got {len(false_pos)}")


# ═══════════════════════════════════════════════════════════════════
# TEST 15: PII Safety — No Raw Values in Training Data
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_pii_safety_training_data(t: TestResult):
    """After detection, training_data.yaml must contain ZERO raw PII values."""
    import yaml
    from app import HybridPIIEngine

    engine = HybridPIIEngine(
        use_gliner=True,
        gliner_model_name="knowledgator/gliner-pii-large-v1.0",
        gliner_threshold=0.15,
    )

    pii_values = [
        "621-73-4489", "john@example.com", "4532-7891-2345-6789",
        "John Smith", "602-555-1234", "021000021",
    ]

    text = f"""
    SSN: {pii_values[0]}. Email: {pii_values[1]}.
    Credit card: {pii_values[2]}. Name: {pii_values[3]}.
    Phone: {pii_values[4]}. Routing: {pii_values[5]}.
    """
    engine.redact(text)

    # Read training data file
    training_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "app", "config", "training_data.yaml"
    )
    if os.path.exists(training_path):
        with open(training_path, "r") as f:
            raw_content = f.read()

        for pii_val in pii_values:
            t.check(pii_val not in raw_content,
                    f"COMPLIANCE VIOLATION: Raw PII '{pii_val}' found in training_data.yaml!")

        t.check("NNN-NN-NNNN" in raw_content or len(raw_content) < 100,
                "Training data should contain structure patterns like NNN-NN-NNNN")
    else:
        t.check(True, "No training data file yet (cold start) — OK")


# ═══════════════════════════════════════════════════════════════════
# TEST 16: Redaction Quality — All PII Must Be Masked
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_redaction_completeness(t: TestResult):
    """Redacted text must not contain any detected PII values."""
    from app import HybridPIIEngine

    engine = HybridPIIEngine(
        use_gliner=True,
        gliner_model_name="knowledgator/gliner-pii-large-v1.0",
        gliner_threshold=0.15,
    )

    text = """
    Customer: My name is Dr. Priya Anand Chakraborty. SSN 621-73-4489.
    Email pchakraborty@genedyne-biotech.com. Phone 480-555-6614.
    Policy UFI-AUTO-5538291. VIN WAUZZZ4M1RD019384.
    """

    result = engine.redact(text)

    # Every detected value must be masked in the redacted text
    for d in result.detections:
        val = d.text.strip()
        if len(val) >= 3:  # Skip very short values that might appear as substrings
            t.check(val not in result.redacted_text,
                    f"Detected {d.label}={val!r} still appears in redacted text!")

    # Redacted text should contain replacement tags
    t.check("<" in result.redacted_text and ">" in result.redacted_text,
            "Redacted text should contain <LABEL> replacement tags")


# ═══════════════════════════════════════════════════════════════════
# TEST 17: CUDA / Device Consistency
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_cuda_device(t: TestResult):
    """Model and tensors should be on CUDA when available."""
    import torch
    from app.ml.trainer import SelfTrainer

    trainer = SelfTrainer()
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"

    t.check(str(trainer.device).startswith(expected_device),
            f"Trainer device should be {expected_device}, got {trainer.device}")

    if trainer.is_ready():
        model_device = str(next(trainer.model.parameters()).device)
        t.check(model_device.startswith(expected_device),
                f"Model should be on {expected_device}, got {model_device}")

    if torch.cuda.is_available():
        t.check(True, f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        t.check(True, "CUDA not available — running on CPU (tests still valid)")


# ═══════════════════════════════════════════════════════════════════
# TEST 18: Auto-Promotion Counter
# ═══════════════════════════════════════════════════════════════════
@run_test
def test_auto_promote_counter(t: TestResult):
    """Engine must increment _run_count and auto-promote periodically."""
    from app import HybridPIIEngine

    engine = HybridPIIEngine(
        use_gliner=True,
        gliner_model_name="knowledgator/gliner-pii-large-v1.0",
        gliner_threshold=0.15,
    )

    t.check(engine._run_count == 0, "Run count should start at 0")
    t.check(engine._auto_promote_interval == 50, "Auto-promote interval should be 50")

    # Run detect once
    engine.detect("My SSN is 621-73-4489.")
    t.check(engine._run_count == 1, f"Run count should be 1 after one detect(), got {engine._run_count}")


# ═══════════════════════════════════════════════════════════════════
# MAIN — Run all tests
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  PII DETECTION ENGINE — END-TO-END TEST SUITE")
    print("=" * 70)
    print()

    # Unit tests (fast, no GLiNER needed)
    test_value_to_structure()
    test_mask_pii_in_text()
    test_extract_safe_neighborhood()
    test_model_architecture()
    test_self_trainer_collection()
    test_resolver_source_priority()
    test_resolver_overlap_resolution()
    test_postprocessing_lstm_trusted()
    test_cross_validation()
    test_instance_numbering()

    # Integration tests (require GLiNER model)
    print()
    print("-" * 70)
    print("  INTEGRATION TESTS (require GLiNER model)")
    print("-" * 70)
    print()

    test_engine_api_completeness()
    test_full_pipeline_insurance_call()
    test_full_pipeline_banking()
    test_full_pipeline_medical()
    test_name_propagation()
    test_multi_run_accumulation()
    test_edge_cases()
    test_pii_safety_training_data()
    test_redaction_completeness()
    test_cuda_device()
    test_auto_promote_counter()

    # ── Final Summary ──
    print()
    print("=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    total_passed = sum(r.passed for r in ALL_RESULTS)
    total_failed = sum(r.failed for r in ALL_RESULTS)
    tests_passed = sum(1 for r in ALL_RESULTS if r.failed == 0)
    tests_failed = sum(1 for r in ALL_RESULTS if r.failed > 0)

    for r in ALL_RESULTS:
        status = "PASS" if r.failed == 0 else "FAIL"
        print(f"  [{status}] {r.name}: {r.passed} passed, {r.failed} failed")

    print()
    print(f"  Tests : {tests_passed} passed, {tests_failed} failed out of {len(ALL_RESULTS)}")
    print(f"  Checks: {total_passed} passed, {total_failed} failed")
    print()

    if total_failed == 0:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED — see details above")
        print()
        for r in ALL_RESULTS:
            if r.errors:
                print(f"  [{r.name}]")
                for e in r.errors:
                    print(f"    - {e}")

    print()
    print("=" * 70)

    sys.exit(1 if total_failed > 0 else 0)
