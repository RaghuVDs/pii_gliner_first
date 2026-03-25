"""
Single-process comparison: spaCy enabled vs disabled.
Runs both phases in one process so GLiNER loads once.
"""
import sys, os, re
sys.path.insert(0, r"g:\Project_01_pii_gliner_first")

from tests.spacy_comparison_test import TEXTS, run_all_tests, compare_results
from app import HybridPIIEngine
import app.postprocessing as pp

# ============================================================
# Build the engine ONCE (GLiNER loads once)
# ============================================================
engine = HybridPIIEngine(
    use_gliner=True,
    gliner_model_name="knowledgator/gliner-pii-large-v1.0",
    gliner_threshold=0.15,
)

# ============================================================
# PHASE 1: WITH spaCy (current behavior)
# ============================================================
print("=" * 120)
print("PHASE 1: WITH spaCy (current behavior)")
print("=" * 120)
results_with = run_all_tests(engine, "WITH spaCy")

# ============================================================
# PHASE 2: Monkey-patch spaCy out, re-run
# ============================================================

# Save originals
_orig_check_grammar = pp.check_grammar
_orig_spacy_person = pp.spacy_recognizes_as_person

# Clear caches from phase 1
if hasattr(pp.check_grammar, "cache_clear"):
    pp.check_grammar.cache_clear()
if hasattr(pp.spacy_recognizes_as_person, "cache_clear"):
    pp.spacy_recognizes_as_person.cache_clear()

def check_grammar_no_spacy(val):
    """Heuristic replacement: uppercase first letter = PROPN, digit token = NUM."""
    tokens = val.split()
    has_propn = any(t[0].isupper() for t in tokens if t)
    has_num = any(t.isdigit() or t.replace(".", "", 1).isdigit() for t in tokens)
    return has_propn, has_num

def spacy_recognizes_as_person_no_spacy(val):
    """No spaCy NER cross-validation - always return False."""
    return False

pp.check_grammar = check_grammar_no_spacy
pp.spacy_recognizes_as_person = spacy_recognizes_as_person_no_spacy

print("\n" + "=" * 120)
print("PHASE 2: WITHOUT spaCy (heuristic replacement)")
print("=" * 120)
results_without = run_all_tests(engine, "WITHOUT spaCy")

# ============================================================
# COMPARE
# ============================================================
compare_results(results_with, results_without)

# ============================================================
# Restore originals
# ============================================================
pp.check_grammar = _orig_check_grammar
pp.spacy_recognizes_as_person = _orig_spacy_person
