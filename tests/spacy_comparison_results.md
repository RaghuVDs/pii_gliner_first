# spaCy vs No-spaCy Comparison Results

**Date:** 2026-03-24
**Test runner:** `tests/spacy_comparison_single.py`
**Engine:** GLiNER (gliner-pii-large-v1.0, threshold=0.15) + Regex + Field + Context
**Method:** Both phases ran in single process (GLiNER loaded once), spaCy monkey-patched out in Phase 2

## Summary

| Metric | With spaCy | Without spaCy | Delta |
|--------|-----------|--------------|-------|
| Total detections | 173 | 173 | +0 |
| Cross-validated | 37 | 37 | +0 |
| Unknown candidates | 0 | 0 | +0 |

## Per-Test Breakdown

| Test Case | With spaCy | Without spaCy | Delta | Cross-Validated |
|-----------|-----------|--------------|-------|-----------------|
| 01_full_transcript | 66 | 66 | +0 | 14 |
| 02_unusual_names | 21 | 21 | +0 | 3 |
| 03_false_positive_stress | 1 | 1 | +0 | 0 |
| 04_ambiguous_names | 20 | 20 | +0 | 4 |
| 05_financial_heavy | 13 | 13 | +0 | 3 |
| 06_medical_heavy | 19 | 19 | +0 | 7 |
| 07_minimal_context | 10 | 10 | +0 | 2 |
| 08_informal_chat | 10 | 10 | +0 | 3 |
| 09_repeated_values | 13 | 13 | +0 | 1 |
| 10_non_pii | 0 | 0 | +0 | 0 |

## Detailed Differences

**NO DIFFERENCES FOUND — results are identical across all 10 tests.**

## False Positive Analysis

- **03_false_positive_stress:** 1 detection with spaCy, 1 without — no increase
- **10_non_pii:** 0 detections with spaCy, 0 without — no increase

## Test Descriptions

1. **01_full_transcript** — Full insurance call transcript (baseline, 12 people, NPI, Medicare, police report, badges, plates, medical records)
2. **02_unusual_names** — Non-Western names: Xiu Ling Zhang, Oluwaseun Adebayo, Svetlana Petrova-Nakamura, Bhupinder Singh Khalsa, Chimamanda Okafor, Tran Duc Nguyen, Aarav Mehta
3. **03_false_positive_stress** — Text with common words that could false-positive (active, premium, system, manager, supervisor, merchant, gold, silver, etc.)
4. **04_ambiguous_names** — Names that are also common English words: Grace, Summer, Chase, Mark, Grant, Joy, Dawn, Sage, Rose, Pearl
5. **05_financial_heavy** — Financial PII: account numbers, credit cards, routing numbers, IBAN, CVV, SSNs
6. **06_medical_heavy** — Medical PII: diagnoses, NPI numbers, prescriptions, Medicare ID, medical records, blood type
7. **07_minimal_context** — Bare PII with minimal surrounding context
8. **08_informal_chat** — Informal/conversational language with slang
9. **09_repeated_values** — Same names and values appearing multiple times
10. **10_non_pii** — Business text with no PII (should produce zero detections)

## Why spaCy Has Zero Impact

1. **Grammar gate bypass** — GLiNER detections with score >= 0.65 skip POS check entirely (most exceed this)
2. **Person NER cross-validation** — Only fires for multi-word GLiNER PERSON detections with score < 0.45 (essentially never triggers)
3. **Regex/field/context** — These sources skip the grammar gate entirely
4. **Regex person patterns** — New name patterns + name splitting + propagation handle names without spaCy

## Conclusion

spaCy (`en_core_web_sm` model) can be safely removed with zero impact on detection quality. The heuristic replacement (uppercase first letter = proper noun) produces identical results across all test scenarios.

### Benefits of removing spaCy:
- Eliminates ~12MB `en_core_web_sm` model download
- Removes spaCy import overhead and lazy-load complexity
- Reduces dependency footprint
- Faster cold start (no model loading)
