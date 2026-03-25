"""
Fresh LSTM Training — Zero Knowledge Start
=============================================
Feeds diverse financial/insurance/medical transcripts through GLiNER,
collects all detections as PII-safe training data, then trains the
CNN-BiLSTM pattern classifier from scratch.

The LSTM learns structural patterns (NNN-NN-NNNN, aaaa@aaa.aaa, etc.)
combined with context keywords — distilling GLiNER's semantic knowledge
into a fast, deterministic pattern classifier.

Run:  python examples/train_lstm_fresh.py
"""
import sys
import os
import json
import logging
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("train_fresh")

# ─────────────────────────────────────────────────────────────────────
# Diverse Financial Transcripts for Training
# ─────────────────────────────────────────────────────────────────────
TRANSCRIPTS = [
    # 1. Insurance claim call — auto + health
    """
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
    """,

    # 2. Banking — wire transfer and account verification
    """
    [Call Date: 01/15/2026 | Agent: Sarah Chen | Channel: Phone]
    Agent: Welcome to Pacific National Bank. This is Sarah Chen, employee ID PNB-44821. How may I help you today?
    Customer: Hi Sarah. I'm Marcus Anthony Williams. I need to wire $45,000 to an international account and also report a suspicious charge on my debit card.
    Agent: I can help with both. Let me verify your identity first. Can you provide your account number and the last four of your Social?
    Customer: Sure, my checking account is 7829-4461-3350. SSN last four is 7723. Full SSN is 445-82-7723. Date of birth March 22, 1978.
    Agent: And your mother's maiden name?
    Customer: Richardson.
    Agent: Verified. Now for the wire transfer — what are the recipient details?
    Customer: Send to Hiroshi Tanaka at Mizuho Bank in Tokyo. His IBAN is JP92 0009 0101 0000 1234 5678. SWIFT code is MHCBJPJT. Routing number for intermediary is 021000021.
    Agent: And the purpose?
    Customer: Real estate investment. The property is at 1847 Sakura Lane, Shibuya-ku, Tokyo. My mortgage broker is Amanda Liu, license number ML-8827419, at Pacific Lending Group.
    Agent: Got it. Now about the suspicious charge — what card was affected?
    Customer: My Visa debit ending in 8834. Full number is 4532-8891-2345-8834, expiry 09/2028, CVV 742. There was a $3,200 charge from "TECHWORLD ELECTRONICS" in Newark that I didn't make.
    Agent: I see that charge. I'll flag it. Your card will be replaced. Anything else?
    Customer: Yes, my wife Jennifer Williams, SSN 445-82-7724, needs to be added as an authorized user on our joint savings, account 7829-4461-3351. Her email is jwilliams@pacificcoast-realty.com, phone 415-555-3392.
    Agent: Done. The new card will arrive at your address, 2847 Lighthouse Drive, Apt 12B, San Francisco, CA 94116?
    Customer: Yes, that's correct. Also, can you update my employer to Quantum Analytics Inc, employee ID QA-E-9281?
    Agent: Updated. Anything else?
    Customer: No, thank you Sarah.
    """,

    # 3. Credit card dispute — AMEX specific
    """
    [Call Date: 02/10/2026 | Agent: Michael Torres | Channel: Phone]
    Agent: American Express, this is Michael Torres, badge MT-55192. How can I assist you?
    Customer: Hi Michael. My name is Elizabeth Grace Montgomery. Cardmember since 2014. I need to dispute three charges on my Platinum card.
    Agent: Of course. Can I get your card number and verification details?
    Customer: Card number 3782-822463-10058. My membership rewards number is MR-448291736. Cardmember ID CM-88291453.
    Agent: Date of birth for verification?
    Customer: July 8, 1972. SSN 287-56-3318.
    Agent: Verified. What are the disputed charges?
    Customer: First, $8,500 at "LUXE TRAVEL AGENCY" on January 22nd — I never booked that trip. Second, $1,200 at "DIAMOND JEWELERS NYC" on January 25th. Third, $450 at "PREMIUM WINE CLUB" on January 28th. None of these are mine.
    Agent: I'll open disputes on all three. Your billing address is 445 Park Avenue, Penthouse 3, New York, NY 10022?
    Customer: Yes. Also, I noticed someone may have my supplementary card details. The supplementary card is under my son, Alexander Montgomery, card ending 0062. His date of birth is December 14, 2001. SSN 287-56-3319.
    Agent: I'll flag that card too. Your authorized representative for this account?
    Customer: My husband, Dr. Richard Montgomery. His phone is 212-555-8847, email rmontgomery@nyc-cardiology.org. He's also my emergency contact.
    Agent: I've opened dispute case numbers AX-2026-D-001847, AX-2026-D-001848, and AX-2026-D-001849. Temporary credits will appear within 48 hours.
    Customer: Thank you, Michael.
    """,

    # 4. Mortgage and loan refinancing
    """
    [Call Date: 01/28/2026 | Agent: Lisa Park | Channel: Phone]
    Agent: HomeFirst Mortgage, Lisa Park speaking, NMLS number 1284759. How can I help?
    Customer: Hi Lisa. I'm calling about refinancing my mortgage. My name is Christopher Daniel O'Brien. My wife is Mei Lin Chen-O'Brien.
    Agent: Happy to help. Let me pull up your loan.
    Customer: Current mortgage number is HFM-2019-4482917. Property at 892 Maple Creek Court, Denver, CO 80220. Parcel number 0171234-00-042.
    Agent: Got it. For the refinancing application, I'll need some details.
    Customer: Sure. My SSN is 512-34-6678. DOB August 19, 1980. My wife's SSN is 512-34-6679, DOB February 3, 1982. We both work — I'm at Frontier Energy Solutions, employee ID FES-2291, salary $142,000. She's at Denver General Hospital, employee ID DGH-N-4817, salary $98,000.
    Agent: Your current bank for auto-pay?
    Customer: First National Bank, routing 102003154, checking account 44829173650. My credit score was 782 last time I checked.
    Agent: Any other debts I should know about for the DTI calculation?
    Customer: My auto loan with Capital One, account CPA-882917436, balance $18,200. Student loan with FedLoan, account FL-4829173, balance $42,000. And a Home Equity Line, HELOC number HFM-HE-2021-88291.
    Agent: Perfect. What about insurance on the property?
    Customer: Homeowners through State Farm, policy SF-HO-882917-CO. Flood insurance with NFIP, policy NFIP-0822917. My insurance agent is Robert Kim, phone 303-555-7291, email rkim@statefarm-denver.com.
    Agent: I'll start the application. You'll get loan estimate within three business days. Reference number HFM-REFI-2026-00482.
    Customer: Great. You can reach me at chris.obrien@frontier-energy.com or 303-555-4418.
    Agent: Got it. Thank you, Mr. O'Brien.
    """,

    # 5. Investment and brokerage account
    """
    [Call Date: 02/05/2026 | Agent: David Kim | Channel: Phone]
    Agent: Meridian Wealth Management, David Kim, Series 7 license number 5528194. How may I assist you?
    Customer: David, it's Sophia Marie Andersen. I need to rebalance my portfolio and also set up a trust account.
    Agent: Good morning, Ms. Andersen. Let me verify your identity.
    Customer: Account number MWM-IRA-7729184. SSN 334-21-8845. Date of birth April 15, 1965. Security question answer is "Lake Tahoe."
    Agent: Verified. What changes would you like?
    Customer: First, sell 500 shares of AAPL from my brokerage account MWM-BROK-7729185, CUSIP 037833100. Move the proceeds to my money market account MWM-MM-7729186.
    Agent: I'll process that. And the trust?
    Customer: Set up an irrevocable trust for my granddaughter, Emma Rose Andersen. Her SSN is 334-21-8850, DOB January 7, 2020. The trust should be funded with $500,000 from my savings at Chase Bank, account 4482-9173-6500, routing 021000021.
    Agent: Attorney for the trust documents?
    Customer: James Harrison, Esq., bar number CA-448291. His firm is Harrison & Associates, 1200 Financial District Blvd, Suite 800, San Francisco, CA 94111. Phone 415-555-9284, email jharrison@harrison-law.com.
    Agent: Beneficiary designations on your IRA?
    Customer: Primary: my daughter, Dr. Claire Andersen-Nakamura, SSN 334-21-8847. Secondary: my son, Erik Andersen, SSN 334-21-8848. Both at 50%.
    Agent: And your emergency contact?
    Customer: My husband, Lars Andersen. Phone 415-555-7738. His passport number is 687429183, issued in Norway. Norwegian national ID 15046512345.
    Agent: Everything is documented. Reference number MWM-2026-TX-00891. Anything else?
    Customer: Send the trust documents to my home at 3847 Pacific Heights Lane, San Francisco, CA 94115. And copy my CPA, Michelle Wong, at mwong@wong-accounting.com, PTIN P01234567.
    Agent: Done. Thank you, Ms. Andersen.
    """,

    # 6. Health insurance enrollment
    """
    [Call Date: 02/12/2026 | Agent: Karen Mitchell | Channel: Phone]
    Agent: Blue Shield Health, this is Karen Mitchell, agent ID BSH-7728. How can I help?
    Customer: Hi Karen. I need to enroll in a family health plan. My name is Ahmed Hassan Al-Rashidi. I'm a naturalized US citizen.
    Agent: Welcome, Mr. Al-Rashidi. Let me start the application.
    Customer: My SSN is 498-73-2251. Date of birth September 28, 1975. I live at 1592 Cedar Ridge Drive, Houston, TX 77024. Email ahmed.rashidi@petrochem-global.com. Phone 713-555-4428.
    Agent: Employer information?
    Customer: PetroChem Global Solutions, employee ID PCG-E-8829. My salary is $165,000. I need coverage for myself, my wife Fatima Al-Rashidi, SSN 498-73-2252, DOB March 14, 1978, and our three children.
    Agent: Children's details?
    Customer: Omar Al-Rashidi, DOB May 5, 2008, SSN 498-73-2253. Layla Al-Rashidi, DOB November 20, 2010, SSN 498-73-2254. Yasmin Al-Rashidi, DOB July 3, 2015, SSN 498-73-2255. Omar has Type 1 diabetes and uses an insulin pump, device serial number MP-670G-48291738.
    Agent: Previous insurance?
    Customer: We had Aetna through my previous employer, group number AET-GRP-88291, member ID AET-M-7729184. My wife's primary care doctor is Dr. Susan Park, NPI 1234567890, at Houston Medical Associates, patient ID HMA-P-44829.
    Agent: Any prescriptions to transfer?
    Customer: Omar's insulin, Humalog, prescription RX-7729184 at Walgreens on Westheimer Road. Fatima takes Levothyroxine, prescription RX-7729185 at the same pharmacy. My prescription for Lisinopril is RX-7729186.
    Agent: Medicare or Medicaid for anyone?
    Customer: No, but my mother lives with us — Amira Al-Rashidi, Medicare ID 2FK7-TB3-JZ89. She's not on this plan but I'm her emergency contact.
    Agent: I've created application BSH-APP-2026-04829. You'll get plan options within 24 hours. Enrollment effective March 1st.
    Customer: Great. Can you also send documents to my attorney, Michael Chen, at mchen@houston-legal.com, phone 713-555-8891?
    Agent: Done. Thank you, Mr. Al-Rashidi.
    """,

    # 7. Auto insurance claim — hit and run
    """
    [Call Date: 01/20/2026 | Agent: Rachel Green | Channel: Phone]
    Agent: SafeDrive Auto Insurance, Rachel Green, badge SD-44291. How may I help?
    Customer: Hi Rachel. I'm calling to report a hit-and-run. My name is James Theodore Wilson. Policy number SD-AUTO-2025-77291.
    Agent: I'm sorry to hear that, Mr. Wilson. Let me verify your account. Date of birth and last four of Social?
    Customer: February 14, 1992. Last four 6612. Full is 189-44-6612.
    Agent: Verified. Tell me what happened.
    Customer: Last night around 11 PM, my car was hit while parked on 4th Street and Main in downtown Portland. My 2023 Tesla Model 3, VIN 5YJ3E1EA8PF482917, Oregon plate XKR-7291. The car has significant damage to the rear quarter panel and bumper.
    Agent: Did you get any information on the other vehicle?
    Customer: A witness, Dorothy Mae Patterson, phone 503-555-8829, said she saw a dark blue pickup truck with Washington plates starting with "BKL." She took a photo. Her email is dpatterson@portland-art.edu.
    Agent: Police report?
    Customer: Portland PD report number 2026-PPD-0017829. Officer Maria Gonzalez, badge PPD-8821.
    Agent: I'll need your repair shop preference.
    Customer: Tesla-authorized: Precision Auto Body, 8829 Industrial Parkway, Portland, OR 97201. My contact there is Tony Russo, phone 503-555-2291.
    Agent: Your deductible is $250. I've opened claim SD-CLM-2026-008291. Adjuster will be assigned within 24 hours.
    Customer: Also, I need a rental car. My credit card on file is Visa 4916-7382-9174-5508, exp 03/2028.
    Agent: Rental approved through Enterprise. Anything else?
    Customer: My dashcam footage is uploading to the portal. My login email is jwilson@techstart-pdx.com, employee ID TS-E-2291.
    Agent: Great. Thank you, Mr. Wilson.
    """,

    # 8. Fraud investigation — identity theft
    """
    [Call Date: 02/18/2026 | Agent: Daniel Cruz | Channel: Phone]
    Agent: National Identity Protection Services, Daniel Cruz, investigator ID NIPS-D-8829. This call is recorded. How can I help?
    Customer: Someone stole my identity and opened accounts in my name. I'm Victoria Suzanne Blackwell.
    Agent: I'm very sorry, Ms. Blackwell. Let me take your details to start a case.
    Customer: SSN 556-23-9918. Date of birth October 30, 1988. Current address 1247 Birchwood Lane, Apt 4C, Charlotte, NC 28202. Phone 704-555-3318. Email vblackwell@duke-energy.com. Employee ID there is DE-V-88291.
    Agent: What accounts were fraudulently opened?
    Customer: A Capital One credit card, account number 5412-7829-1734-6601, was opened January 5th. A Chase checking account, number 8829-1734-5500, opened January 8th. And a T-Mobile phone line, account TM-882917345, opened January 10th.
    Agent: Did you file a police report?
    Customer: Yes, Charlotte-Mecklenburg PD, report number 2026-CMPD-002891. Detective Assigned is Lt. William Foster, badge CMPD-6629.
    Agent: FTC Identity Theft Report?
    Customer: FTC reference number FTC-2026-ITR-4482917. I also placed fraud alerts with all three bureaus. Equifax PIN 882917, Experian PIN 334829, TransUnion PIN 661748.
    Agent: Do you suspect how the breach occurred?
    Customer: I think it was a data breach at my gym, FitLife Charlotte. My membership number there was FL-M-44829. They had my driver's license — North Carolina, number B882917345 — my DOB, and SSN on file.
    Agent: I've opened investigation case NIPS-2026-INV-008829. Your identity restoration specialist will be Linda Park, ext 4429, direct line 704-555-9918.
    Customer: Should I contact anyone else?
    Agent: I recommend notifying your bank, your employer's HR department, and the IRS Identity Protection PIN program. Your IRS IP PIN for this year is 448291.
    Customer: My husband Robert Blackwell should also be monitored. His SSN is 556-23-9919, DOB December 5, 1986.
    Agent: I'll add him. Case reference for your records: NIPS-2026-INV-008829. Anything else?
    Customer: No, thank you Daniel.
    """,

    # 9. Life insurance application
    """
    [Call Date: 02/20/2026 | Agent: Jennifer Lee | Channel: Phone]
    Agent: Guardian Life Insurance, Jennifer Lee, license number GA-INS-77829. How can I help?
    Customer: I'd like to apply for a term life insurance policy. My name is Dr. Nathaniel Joseph Harrington.
    Agent: Wonderful. Let me start the application.
    Customer: I'm a cardiologist at Boston General Hospital. Employee ID BGH-D-4829. NPI number 1987654321. Board certified, license MA-MD-448291. I'm 52 years old, born January 25, 1974. SSN 023-45-6789.
    Agent: Marital status and beneficiaries?
    Customer: Married to Catherine Anne Harrington, née Sullivan. Her SSN is 023-45-6790, DOB September 12, 1976. She's my primary beneficiary at 60%. Secondary beneficiaries are our children — William Harrington, age 22, SSN 023-45-6791, and Grace Harrington, age 19, SSN 023-45-6792.
    Agent: Your address?
    Customer: 847 Beacon Hill Drive, Boston, MA 02108. Phone 617-555-7729. Email nharrington@bostongeneral.org.
    Agent: Medical history for underwriting?
    Customer: I had a cardiac catheterization in 2022 at Mass General, medical record number MGH-MR-882917. Cholesterol is managed with Atorvastatin 20mg, prescription RX-8829174. Blood pressure is controlled, 130/80. I'm a non-smoker. My PCP is Dr. Sarah Kim, NPI 1876543210, at Beacon Hill Medical, patient ID BHM-P-7729.
    Agent: Family medical history?
    Customer: Father had a heart attack at 65, deceased. Mother is 78, has Type 2 diabetes. No cancer history in immediate family.
    Agent: Any existing life insurance?
    Customer: Yes, a $500,000 group policy through the hospital, certificate number BGH-GRP-LI-44829. I'm looking for an additional $2 million term policy.
    Agent: I've started application GL-APP-2026-007729. We'll need a paramedical exam. Your preferred time?
    Customer: Weekday mornings work best. Send everything to my home address or email. My wife's email is charrington@sullivan-designs.com, phone 617-555-8834.
    Agent: Application submitted. Reference GL-APP-2026-007729. You'll hear from underwriting within a week.
    Customer: Thank you, Jennifer.
    """,

    # 10. Tax preparation and IRS issues
    """
    [Call Date: 02/22/2026 | Agent: Robert Chang | Channel: Phone]
    Agent: Premier Tax Services, Robert Chang, PTIN P00887291. How can I help?
    Customer: Hi Robert. I'm Olivia Marie Fernandez. I need help with my 2025 tax return and I have an IRS notice to deal with.
    Agent: Let me get your information.
    Customer: SSN 287-91-4456. Date of birth June 3, 1983. Address is 3291 Sunset Boulevard, Apt 7F, Los Angeles, CA 90028. Phone 323-555-4429. Email ofernandez@creative-media-la.com.
    Agent: Employment details?
    Customer: I'm a freelance film editor. I received W-2 from Creative Media Inc, EIN 95-4482917, wages $87,000. Also 1099-NEC from Independent Productions LLC, EIN 95-5529183, payment $34,000. And 1099-INT from Pacific Savings Bank, account 882917345, interest $1,240.
    Agent: What's the IRS notice about?
    Customer: Notice CP2000 for tax year 2024, notice number LTR-4210C-2025-882917. They're saying I didn't report $12,000 in 1099 income from Sunset Studios, EIN 95-6618294.
    Agent: Deductions and credits?
    Customer: Home office — I rent, so I'll take the simplified deduction. Student loan interest from FedLoan, account FL-OF-7729, paid $2,800. Health insurance premiums through Covered California, policy CC-2025-882917, premiums $9,600 for the year. I also contributed $6,500 to my Roth IRA at Vanguard, account VG-ROTH-4482917.
    Agent: Dependents?
    Customer: My daughter, Isabella Fernandez, age 8, SSN 287-91-4457. She's at Sunshine Elementary. After-school care expenses were $4,200 from Bright Futures Academy, EIN 95-7729184.
    Agent: I'll prepare the return and respond to the CP2000. Filing reference PTS-2026-TAX-00829.
    Customer: My ex-husband, Carlos Fernandez, also claims Isabella in alternating years. His SSN is 287-91-4458. Just want to make sure there's no conflict.
    Agent: Noted. I'll verify the claiming schedule. Your estimated refund should be around $4,800. Anything else?
    Customer: Direct deposit to my checking at Pacific Savings, routing 122000661, account 882917345. Thank you, Robert.
    """,
]


def main():
    print("=" * 70)
    print("  LSTM FRESH TRAINING — Zero Knowledge Start")
    print("=" * 70)
    print()

    # GPU status
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("  Running on CPU")
    print()

    # ── Initialize Engine ──
    from app import HybridPIIEngine

    print("Initializing engine (LSTM should have zero knowledge)...")
    engine = HybridPIIEngine(
        use_gliner=True,
        gliner_model_name="knowledgator/gliner-pii-large-v1.0",
        gliner_threshold=0.15,
    )

    print(f"  LSTM ready: {engine.self_trainer.is_ready()}")
    print(f"  LSTM known labels: {engine.lstm_known_labels()}")
    print()

    # ── Phase 1: Feed transcripts through GLiNER to collect training data ──
    print("=" * 70)
    print("  PHASE 1: Collecting training data from GLiNER detections")
    print("=" * 70)
    print()

    total_detections = 0
    all_labels = set()

    for i, text in enumerate(TRANSCRIPTS):
        t0 = time.time()
        result = engine.redact(text)
        elapsed = time.time() - t0

        labels = {d.label for d in result.detections}
        sources = {}
        for d in result.detections:
            sources[d.source] = sources.get(d.source, 0) + 1
        all_labels.update(labels)
        total_detections += len(result.detections)

        print(
            f"  Transcript {i+1:2d}/{len(TRANSCRIPTS)}: "
            f"{len(result.detections):3d} detections, "
            f"{len(labels):2d} labels, "
            f"sources={sources}, "
            f"{elapsed:.1f}s"
        )

    print()
    print(f"  Total detections: {total_detections}")
    print(f"  Unique labels seen: {len(all_labels)}")
    print(f"  Labels: {sorted(all_labels)}")

    # Check training data accumulated
    lstm_stats = engine.pattern_lstm_stats()
    print()
    print(f"  Training examples collected: {lstm_stats.get('total_examples', 0)}")
    print(f"  Label distribution: {json.dumps(lstm_stats.get('top_labels', {}), indent=4)}")

    # ── Phase 2: Force retrain ──
    print()
    print("=" * 70)
    print("  PHASE 2: Training LSTM from scratch")
    print("=" * 70)
    print()

    t0 = time.time()
    result = engine.retrain_pattern_model(epochs=80)
    elapsed = time.time() - t0

    print(f"  Status          : {result.get('status')}")
    print(f"  Training time   : {elapsed:.1f}s")
    print(f"  Device          : {result.get('device')}")
    print(f"  Mixed precision : {result.get('mixed_precision')}")
    print(f"  Examples        : {result.get('num_examples')}")
    print(f"  Train/Val split : {result.get('train_size')}/{result.get('val_size')}")
    print(f"  Labels          : {result.get('num_labels')}")
    print(f"  Keywords        : {result.get('num_keywords')}")
    print(f"  Batch size      : {result.get('batch_size')}")
    print(f"  Learning rate   : {result.get('learning_rate')}")
    print(f"  Best epoch      : {result.get('best_epoch')}/{result.get('stopped_epoch')}")
    print(f"  Early stopped   : {result.get('early_stopped')}")
    print(f"  Best val loss   : {result.get('best_val_loss', 0):.4f}")
    print(f"  Train accuracy  : {result.get('train_accuracy', 0):.4f}")
    print(f"  Val accuracy    : {result.get('val_accuracy', 0):.4f}")
    print(f"  Val Macro F1    : {result.get('val_macro_f1', 0):.4f}")
    print(f"  Val Weighted F1 : {result.get('val_weighted_f1', 0):.4f}")

    # ── Per-label metrics ──
    print()
    print("  Per-label validation metrics:")
    print(f"  {'LABEL':<40} {'PREC':>6} {'REC':>6} {'F1':>6} {'N':>4}")
    print(f"  {'-'*62}")

    per_label = result.get("val_per_label", {})
    for label, m in sorted(per_label.items()):
        print(f"  {label:<40} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>4}")

    # ── Phase 3: Verify LSTM knowledge ──
    print()
    print("=" * 70)
    print("  PHASE 3: LSTM Knowledge After Training")
    print("=" * 70)
    print()

    known = engine.lstm_known_labels()
    print(f"  LSTM can now predict {len(known)} labels:")
    for i, label in enumerate(known, 1):
        print(f"    {i:2d}. {label}")

    # ── Phase 4: Test LSTM on a new unseen transcript ──
    print()
    print("=" * 70)
    print("  PHASE 4: Testing on UNSEEN transcript")
    print("=" * 70)
    print()

    test_text = """
    Customer: My name is Raymond Joseph Delgado. I need to file a homeowners claim.
    Agent: Can you verify your identity?
    Customer: SSN 773-29-1184. Date of birth May 8, 1970. Policy number HF-HOME-2024-99281.
    My property address is 4521 Oak Terrace Way, Austin, TX 78704. Phone 512-555-8827.
    Email rdelgado@austintech.com. Employee ID AT-E-3391.
    My mortgage is with Wells Fargo, loan number WF-MTG-882917, monthly payment $2,450.
    The damage was caused by a burst pipe. Plumber was Tony Vasquez, license TX-PLB-44829.
    My emergency contact is my sister, Maria Delgado, phone 512-555-4429.
    I also have a home warranty with American Home Shield, contract AHS-2024-882917.
    VIN for the car in the garage that was damaged: 3VWDP7AJ5MM482917.
    """

    test_result = engine.redact(test_text)

    print(f"  {'LABEL':<35} {'VALUE':<40} {'SCORE':>5} {'SOURCE':<16}")
    print(f"  {'-'*100}")
    for d in test_result.detections:
        flag = " [LSTM]" if d.source == "pattern_lstm" else ""
        print(f"  {d.label:<35} {d.text!r:<40} {d.score:>5.2f} {d.source:<16}{flag}")

    lstm_count = sum(1 for d in test_result.detections if d.source == "pattern_lstm")
    total = len(test_result.detections)
    print()
    print(f"  Total: {total} detections, {lstm_count} from LSTM")

    # ── Final summary ──
    print()
    print("=" * 70)
    print("  TRAINING COMPLETE — SUMMARY")
    print("=" * 70)
    print()
    print(f"  Training examples : {result.get('num_examples')}")
    print(f"  Labels learned    : {result.get('num_labels')}")
    print(f"  Val Weighted F1   : {result.get('val_weighted_f1', 0):.4f}")
    print(f"  Val Macro F1      : {result.get('val_macro_f1', 0):.4f}")
    print(f"  Model saved to    : app/ml/saved/pattern_model.pt")
    print(f"  Vocab saved to    : app/ml/saved/vocab.yaml")
    print(f"  Metrics log       : app/ml/saved/metrics_log.yaml")
    print()
    print("  The LSTM will now automatically:")
    print("    1. Upgrade UNKNOWN_PII detections using learned patterns")
    print("    2. Fix low-confidence GLiNER predictions (<0.45)")
    print("    3. Collect new training data from future detections")
    print("    4. Auto-retrain every 100 new examples")
    print("    5. Auto-promote pending rules every 50 runs")
    print()


if __name__ == "__main__":
    main()
