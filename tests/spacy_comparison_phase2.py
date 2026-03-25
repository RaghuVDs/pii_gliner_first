"""
Phase 2: Run the same tests with spaCy functions replaced by lightweight heuristics.
Then compare against Phase 1 results.
"""
import sys, os, re, pickle, tempfile
sys.path.insert(0, r"g:\Project_01_pii_gliner_first")

# ============================================================
# MONKEY-PATCH: Replace spaCy with lightweight heuristics
# ============================================================
import app.postprocessing as pp

# Replace check_grammar: Instead of spaCy POS tagging,
# use simple heuristics: uppercase first letter = PROPN, contains digit = NUM
def check_grammar_no_spacy(val: str):
    """Heuristic replacement for spaCy POS-based grammar check."""
    tokens = val.split()
    has_propn = any(t[0].isupper() for t in tokens if t)
    has_num = any(t.isdigit() or t.replace(".", "", 1).isdigit() for t in tokens)
    return has_propn, has_num

# Replace spacy_recognizes_as_person: always return False
# (effectively removes the spaCy NER cross-validation gate)
def spacy_recognizes_as_person_no_spacy(val: str):
    """No spaCy NER — always return False (most permissive)."""
    return False

# Apply monkey patches
pp.check_grammar = check_grammar_no_spacy
pp.spacy_recognizes_as_person = spacy_recognizes_as_person_no_spacy

# Also clear any lru_cache from phase 1 if present
if hasattr(pp.check_grammar, "cache_clear"):
    pp.check_grammar.cache_clear()
if hasattr(pp.spacy_recognizes_as_person, "cache_clear"):
    pp.spacy_recognizes_as_person.cache_clear()

print("spaCy functions replaced with lightweight heuristics.")

# ============================================================
# Import test texts and engine
# ============================================================
from tests.spacy_comparison_test import TEXTS, run_all_tests, compare_results
from app import HybridPIIEngine

# Phase 2: WITHOUT spaCy
print("=" * 120)
print("PHASE 2: WITHOUT spaCy (heuristic replacement)")
print("=" * 120)
engine = HybridPIIEngine(
    use_gliner=True,
    gliner_model_name="knowledgator/gliner-pii-large-v1.0",
    gliner_threshold=0.15,
)
results_without = run_all_tests(engine, "WITHOUT spaCy")

# Load Phase 1 results
tmpdir = tempfile.gettempdir()
with open(os.path.join(tmpdir, "spacy_with_results.pkl"), "rb") as f:
    results_with = pickle.load(f)

# Compare
compare_results(results_with, results_without)
