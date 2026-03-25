"""
Rigorous comparison test: spaCy enabled vs disabled.
Tests 10 diverse transcripts to measure impact of removing spaCy.
"""
import sys, os, re, pickle
sys.path.insert(0, r"g:\Project_01_pii_gliner_first")

TEXTS = {}

# 1. Original full transcript (baseline)
TEXTS["01_full_transcript"] = (
    "[Call Date: 02/03/2026 | Duration: 22:15 | Agent: David Ramirez | Channel: Phone]\n"
    "Agent: Thank you for calling United Family Insurance. This is David Ramirez, badge number UFI-33291. How may I assist you?\n"
    "Customer: Hi David. My name is Dr. Priya Anand Chakraborty. I am calling about a car accident I was in last week and I need to file claims under both my auto and health insurance.\n"
    "Agent: I am sorry to hear that, Dr. Chakraborty. Let me pull up your account. Can you verify your policy number and date of birth?\n"
    "Customer: My auto policy is UFI-AUTO-5538291. My health insurance is through the same company, policy UFI-HEALTH-5538292. Group number GRP-BIOTECH-4421. My date of birth is November 3, 1985. SSN is 621-73-4489.\n"
    "Agent: Verified. Can you walk me through what happened?\n"
    "Customer: On January 28th at about 3:15 PM, I was rear-ended at the intersection of Camelback Road and 24th Street in Phoenix. The other driver name was Robert James Martinez, and he gave me his license number it was an Arizona license, D98321754. His insurance is State Farm, policy number SF-229-4817362-01. His plate number was Arizona FGR-8821. His vehicle was a 2019 Ford F-150, VIN 1FTEW1EP5KFA82943.\n"
    "Agent: And your vehicle information?\n"
    "Customer: 2024 Audi Q7, VIN WAUZZZ4M1RD019384, Arizona plate NLV-3392. It is leased through Audi Financial Services, account AFS-22918743. My lien holder address is on file.\n"
    "Agent: Got it. Now for the medical portion can you describe your injuries and treatment?\n"
    "Customer: I went to the Scottsdale Osborn Medical Center ER that evening. My medical record number there is SMC-00482917. I was diagnosed with cervical whiplash, a mild concussion ICD-10 code S13.4XXA and S06.0X0A. They did a CT scan and X-rays. The attending physician was Dr. Leonard Wu, NPI number 1497825310. I was prescribed Cyclobenzaprine 10mg, prescription number RX-2294817, filled at CVS pharmacy on Shea Boulevard, store number 6742.\n"
    "Agent: Have you had any follow-up treatment?\n"
    "Customer: Yes. I am seeing Dr. Amanda Foster for physical therapy, NPI 1832749162, at Desert Spine and Rehabilitation. My patient ID there is DSR-P-08291. I have had three sessions so far. I also saw my primary care physician, Dr. Rajesh Gupta, NPI 1628374910, at Banner Health, patient ID BH-44829173. He referred me to a neurologist for the concussion follow-up. Oh, and my Medicare Beneficiary ID is 1EK4-TA2-HY74 since I also have Medicare Part B through my disability status.\n"
    "Agent: Thank you. For the auto claim, can I get the police report number?\n"
    "Customer: Phoenix PD report number 2026-PX-0028471. The responding officer was Officer Kevin Doyle, badge number PPD-4419. There were two witnesses Maria Santos, phone 602-555-8837, and James Liu, phone 480-555-2291.\n"
    "Agent: I need a few more details for your health claim. Can you verify your blood type and any existing conditions in your medical history that might be relevant?\n"
    "Customer: Blood type is O-negative. I have a pre-existing condition of Type 2 diabetes, currently managed with Metformin. I also have a medical device a continuous glucose monitor, device serial number DX-G6-482917384. My pharmacy benefit manager is Express Scripts, member ID ESI-77482913.\n"
    "Agent: And your emergency contact on file?\n"
    "Customer: My husband, Arjun Chakraborty. Cell phone 480-555-6614. He is my authorized representative on all accounts. His SSN is 621-73-4490 and date of birth is June 17, 1983.\n"
    "Agent: Perfect. I have opened auto claim number UFI-AC-2026-019384 and health claim number UFI-HC-2026-019385. A claims adjuster will contact you within 48 hours. Your deductible on the auto is 500 dollars and the health deductible has been met for the year. Is there anything else?\n"
    "Customer: Can you send the claim documents to my work email? It is pchakraborty@genedyne-biotech.com. My employee ID there is GBT-E-4817. Actually, also copy my attorney Patricia Huang, Esq., at phuang@desertlawgroup.com, phone 602-555-9918, bar number AZ-029481.\n"
    "Agent: Done. Anything else, Dr. Chakraborty?\n"
    "Customer: No, that covers it. Thank you, David.\n"
)

# 2. Unusual names that spaCy might not recognize
TEXTS["02_unusual_names"] = (
    "Customer: My name is Xiu Ling Zhang. My husband is Oluwaseun Adebayo.\n"
    "Our daughter Svetlana Petrova-Nakamura is also on the policy.\n"
    "The witness was Bhupinder Singh Khalsa, phone 555-123-4567.\n"
    "My attorney is Chimamanda Okafor, Esq., email cokafor@lawfirm.com.\n"
    "The other driver said his name was Tran Duc Nguyen, license TX-8827341.\n"
    "My doctor is Dr. Aarav Mehta, NPI 1234567890.\n"
)

# 3. Common words that should NOT be detected as PII
TEXTS["03_false_positive_stress"] = (
    "The customer account is active and premium status was approved.\n"
    "The agent verified the system was operational on Monday.\n"
    "The service support team handled the default configuration.\n"
    "Account manager reviewed the pending request on Tuesday.\n"
    "The unknown caller was transferred to the supervisor.\n"
    "The merchant transaction was processed on Wednesday for the gold member.\n"
    "Silver tier benefits include basic coverage and standard deductible.\n"
    "The analyst coordinator specialist consultant confirmed the demo sample.\n"
)

# 4. Ambiguous names (Grace, Summer, Chase, Mark, etc.)
TEXTS["04_ambiguous_names"] = (
    "My name is Grace Park. My daughter Summer Lee is 14 years old.\n"
    "My son Chase Hunter, DOB March 5, 2010, SSN 432-11-8876.\n"
    "My husband Mark Grant, SSN 432-11-8877, works at Chase bank.\n"
    "Our address is 123 Rose Lane, Pearl City, HI 96782.\n"
    "My phone is 808-555-1234 and email is gpark@email.com.\n"
    "The doctor was Dr. Joy Dawn, NPI 9876543210, at Sage Medical.\n"
    "Policy number is HI-AUTO-7782910. License plate ABC-1234.\n"
)

# 5. Financial heavy
TEXTS["05_financial_heavy"] = (
    "Agent: I see your checking account 4829173841 and savings 2918374610.\n"
    "Customer: Yes, and my credit card ending in 4532 was compromised.\n"
    "Full card number is 4532-8812-7791-0034, expiration 09/2027, CVV 481.\n"
    "My routing number is 122105155 and the wire was sent to IBAN GB29NWBK60161331926819.\n"
    "The fraud amount was charged to merchant ID MRC-44829. My claim reference is FRD-2026-08817.\n"
    "My SSN is 287-44-9913 and my wife Emily Chen, SSN 287-44-9914.\n"
    "We live at 4401 Birch Street, Apt 12B, Denver CO 80202.\n"
)

# 6. Medical heavy
TEXTS["06_medical_heavy"] = (
    "Patient: Robert O Brien, DOB 04/15/1978, MRN MED-00294817.\n"
    "Diagnosis: Type 1 diabetes (E10.9), hypertension (I10), chronic kidney disease stage 3 (N18.3).\n"
    "Attending: Dr. Sanjay Patel, NPI 1122334455.\n"
    "Prescribed: Insulin Glargine 30 units, Lisinopril 20mg, Rx numbers RX-112948 and RX-112949.\n"
    "Patient Medicare ID is 4AB2-CD3-EF56. Blood type A-positive.\n"
    "Emergency contact: wife, Colleen O Brien, phone 303-555-8812.\n"
    "Next appointment with Dr. Wei Zhang at Banner Health, patient ID BH-99182734.\n"
    "Referred to nephrologist Dr. Fatima Al-Rashid, NPI 5566778899.\n"
    "Insurance: Aetna policy AET-HMO-9928174, group GRP-METRO-5521.\n"
)

# 7. Minimal context
TEXTS["07_minimal_context"] = (
    "SSN 123-45-6789. DOB 01/01/1990. Phone 555-000-1111.\n"
    "Email test@example.com. Name: John Smith. License D12345678.\n"
    "VIN 1HGCM82633A004352. Plate XYZ-1234. Card 4111111111111111.\n"
)

# 8. Informal chat
TEXTS["08_informal_chat"] = (
    "yeah so my name is Mike Johnson and my buddy Alex Rivera was in the car too\n"
    "his phone is 480-555-3321 and mine is 480-555-3322\n"
    "the cop who showed up was Officer Rodriguez badge 1847\n"
    "my insurance is through GEICO policy GK-29481773\n"
    "my SSN is 555-12-3456 and alex SSN is 555-12-3457\n"
    "the other dude plate was california 7ABC123\n"
)

# 9. Repeated values
TEXTS["09_repeated_values"] = (
    "Customer name: Sarah Williams. Policyholder: Sarah Williams.\n"
    "Emergency contact: Tom Williams, phone 555-222-3333.\n"
    "Secondary contact: also Tom Williams at 555-222-3333.\n"
    "Policy UFI-AUTO-1234567 and UFI-HEALTH-1234568.\n"
    "SSN for Sarah: 111-22-3333. SSN for Tom: 111-22-3334.\n"
    "Both live at 789 Oak Drive, Phoenix AZ 85001.\n"
)

# 10. Non-PII — should produce near-zero detections
TEXTS["10_non_pii"] = (
    "The quarterly report shows a 15 percent increase in customer satisfaction.\n"
    "Our new product launch is scheduled for next quarter. The marketing\n"
    "team has prepared the campaign materials. The engineering department\n"
    "completed the infrastructure upgrade last week. Revenue targets for\n"
    "Q2 are set at 2.5 million dollars. The board meeting is Thursday at 3 PM.\n"
    "Please review the attached spreadsheet for the full breakdown.\n"
)


def run_all_tests(engine, label):
    """Run all texts through an engine and return structured results."""
    results = {}
    for name, text in TEXTS.items():
        result = engine.redact(text)
        dets = []
        for d in sorted(result.detections, key=lambda x: x.start):
            dets.append({
                "label": d.label,
                "text": d.text,
                "score": round(d.score, 3),
                "source": d.source,
                "xv": bool(d.meta.get("cross_validated")),
            })
        results[name] = {
            "total": len(result.detections),
            "xv": sum(1 for d in result.detections if d.meta.get("cross_validated")),
            "unk": len(result.unknown_candidates),
            "detections": dets,
        }
        print(f"  [{name}] {results[name]['total']:3d} detections, {results[name]['xv']:2d} XV, {results[name]['unk']:2d} unknown")
    return results


def compare_results(with_spacy, without_spacy):
    """Compare two result sets and report differences."""
    print("\n" + "=" * 120)
    print("COMPARISON: WITH spaCy vs WITHOUT spaCy")
    print("=" * 120)

    total_with = sum(r["total"] for r in with_spacy.values())
    total_without = sum(r["total"] for r in without_spacy.values())
    total_xv_with = sum(r["xv"] for r in with_spacy.values())
    total_xv_without = sum(r["xv"] for r in without_spacy.values())
    total_unk_with = sum(r["unk"] for r in with_spacy.values())
    total_unk_without = sum(r["unk"] for r in without_spacy.values())

    print(f"\n{'Metric':<35} {'With spaCy':>12} {'Without spaCy':>15} {'Delta':>8}")
    print("-" * 75)
    print(f"{'Total detections':<35} {total_with:>12} {total_without:>15} {total_without - total_with:>+8}")
    print(f"{'Cross-validated':<35} {total_xv_with:>12} {total_xv_without:>15} {total_xv_without - total_xv_with:>+8}")
    print(f"{'Unknown candidates':<35} {total_unk_with:>12} {total_unk_without:>15} {total_unk_without - total_unk_with:>+8}")

    print(f"\n{'Test Case':<30} {'With':>6} {'Without':>8} {'Delta':>7} {'Notes'}")
    print("-" * 90)
    for name in TEXTS:
        w = with_spacy[name]["total"]
        wo = without_spacy[name]["total"]
        delta = wo - w
        note = ""
        if delta > 0:
            note = "MORE detections without spaCy"
        elif delta < 0:
            note = "FEWER detections without spaCy"
        else:
            note = "identical count"
        print(f"{name:<30} {w:>6} {wo:>8} {delta:>+7} {note}")

    # Detailed diff per test
    print("\n" + "=" * 120)
    print("DETAILED DIFFERENCES")
    print("=" * 120)

    any_diff = False
    for name in TEXTS:
        w_dets = with_spacy[name]["detections"]
        wo_dets = without_spacy[name]["detections"]

        # Convert to comparable tuples
        w_set = set((d["label"], d["text"], d["source"]) for d in w_dets)
        wo_set = set((d["label"], d["text"], d["source"]) for d in wo_dets)

        only_with = w_set - wo_set
        only_without = wo_set - w_set

        if only_with or only_without:
            any_diff = True
            print(f"\n--- {name} ---")
            if only_with:
                print(f"  LOST without spaCy ({len(only_with)}):")
                for label, text, source in sorted(only_with):
                    print(f"    - {label:30} {text!r:40} ({source})")
            if only_without:
                print(f"  GAINED without spaCy ({len(only_without)}):")
                for label, text, source in sorted(only_without):
                    print(f"    + {label:30} {text!r:40} ({source})")

    if not any_diff:
        print("\n  *** NO DIFFERENCES FOUND — results are identical ***")

    # False positive analysis for test 03 and 10
    print("\n" + "=" * 120)
    print("FALSE POSITIVE ANALYSIS (tests 03_false_positive_stress and 10_non_pii)")
    print("=" * 120)
    for name in ["03_false_positive_stress", "10_non_pii"]:
        print(f"\n--- {name} ---")
        print(f"  With spaCy:    {with_spacy[name]['total']} detections")
        print(f"  Without spaCy: {without_spacy[name]['total']} detections")
        if without_spacy[name]["total"] > with_spacy[name]["total"]:
            wo_dets = without_spacy[name]["detections"]
            w_set = set((d["label"], d["text"]) for d in with_spacy[name]["detections"])
            new_fps = [d for d in wo_dets if (d["label"], d["text"]) not in w_set]
            if new_fps:
                print(f"  NEW FALSE POSITIVES:")
                for d in new_fps:
                    print(f"    ! {d['label']:30} {d['text']!r:40} score={d['score']:.3f} ({d['source']})")
        else:
            print(f"  No increase in false positives.")

    return total_with, total_without


if __name__ == "__main__":
    from app import HybridPIIEngine

    # Phase 1: WITH spaCy
    print("=" * 120)
    print("PHASE 1: WITH spaCy (current behavior)")
    print("=" * 120)
    engine = HybridPIIEngine(
        use_gliner=True,
        gliner_model_name="knowledgator/gliner-pii-large-v1.0",
        gliner_threshold=0.15,
    )
    results_with = run_all_tests(engine, "WITH spaCy")

    # Save for phase 2
    import tempfile
    tmpdir = tempfile.gettempdir()
    with open(os.path.join(tmpdir, "spacy_with_results.pkl"), "wb") as f:
        pickle.dump(results_with, f)

    print("\n\nPhase 1 complete. Now run phase 2 with spaCy disabled.\n")
