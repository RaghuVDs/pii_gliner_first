"""Generate realistic call-center transcripts at multiple density profiles.

Why this exists:
  Real call transcripts spend 70-90% of their text on small talk, problem
  description, hold music, and pleasantries — and only 10-30% on verification
  segments where PII is actually exchanged. The previous version of this
  generator embedded PII in nearly every line, producing a worst-case
  density of ~1 detection per 100 chars that defeats candidate-window
  optimisations and is unrepresentative of real workloads.

Output (tests/fake_transcripts.jsonl):
  Three density profiles, NUM_PER_PROFILE rows each:

    sparse  — ~1 PII per ~1500 chars   (typical clean conversation)
    medium  — ~1 PII per ~500 chars    (heavy verification call)
    dense   — ~1 PII per ~120 chars    (worst case, adversarial)

  Each row: {"id": int, "profile": str, "char_count": int, "text": str}

Usage:
    pip install faker
    python tests/generate_fake_transcripts.py
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

from faker import Faker

NUM_PER_PROFILE = 5
TARGET_CHARS = 80_000
OUT_PATH = Path(__file__).parent / "fake_transcripts.jsonl"

fake = Faker("en_US")
Faker.seed(42)
random.seed(42)


@dataclass
class DensityProfile:
    """Per-line probabilities that produce a target detection density."""
    name: str
    pii_line_prob: float       # chance a turn contains a PII-bearing template
    plain_line_prob: float     # chance a turn is plain dialogue
    filler_prob: float         # chance of a filler/non-speech token
    description_burst_prob: float  # chance of a long PII-free problem-description run

    def assert_valid(self) -> None:
        total = self.pii_line_prob + self.plain_line_prob + self.filler_prob + self.description_burst_prob
        assert abs(total - 1.0) < 1e-6, f"{self.name} probabilities must sum to 1.0 (got {total})"


PROFILES = [
    DensityProfile(
        name="sparse",
        pii_line_prob=0.06,
        plain_line_prob=0.55,
        filler_prob=0.27,
        description_burst_prob=0.12,
    ),
    DensityProfile(
        name="medium",
        pii_line_prob=0.20,
        plain_line_prob=0.45,
        filler_prob=0.27,
        description_burst_prob=0.08,
    ),
    DensityProfile(
        name="dense",
        pii_line_prob=0.55,
        plain_line_prob=0.30,
        filler_prob=0.10,
        description_burst_prob=0.05,
    ),
]
for p in PROFILES:
    p.assert_valid()


def make_pii_pool() -> List[dict]:
    """Pre-generate a small pool of customer records so the same person
    is referenced multiple times throughout the call (realistic re-mention
    patterns)."""
    customers = []
    for _ in range(6):
        customers.append({
            "name": fake.name(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "ssn": fake.ssn(),
            "dob": fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat(),
            "address": fake.address().replace("\n", ", "),
            "card": fake.credit_card_number(),
            "card_exp": fake.credit_card_expire(),
            "card_cvv": fake.credit_card_security_code(),
            "iban": fake.iban(),
            "routing": str(fake.random_number(digits=9, fix_len=True)),
            "account": str(fake.random_number(digits=10, fix_len=True)),
            "ip": fake.ipv4(),
            "license": fake.license_plate(),
            "company": fake.company(),
            "job": fake.job(),
        })
    return customers


# Lines that DO contain PII placeholders — used sparingly in sparse mode
PII_LINES = [
    "AGENT: For verification, can you confirm your full name and date of birth?",
    "CUSTOMER: Sure, my name is {cust} and my date of birth is {dob}.",
    "CUSTOMER: Yes, my social security number is {ssn}.",
    "AGENT: And the email address we have on file?",
    "CUSTOMER: That's {email}.",
    "CUSTOMER: My phone number is {phone}.",
    "CUSTOMER: I live at {address}.",
    "CUSTOMER: My account number is {acct}.",
    "AGENT: Reading the card back: {card}, expires {exp}, CVV {cvv}.",
    "CUSTOMER: I logged in from {ip} yesterday.",
    "CUSTOMER: Could you also update my routing number to {routing}?",
    "CUSTOMER: My driver's license plate is {plate}.",
    "AGENT: I'll send a confirmation email to {email}.",
    "AGENT: Just to confirm, the charge of ${amt} on {date}.",
    "CUSTOMER: My IBAN is {iban}.",
]

# Plain dialogue with NO PII placeholders — the bulk of a real conversation
PLAIN_LINES = [
    "AGENT: Thank you for calling, how can I help you today?",
    "AGENT: I appreciate your patience while I look into this.",
    "AGENT: I understand your frustration and I want to make this right.",
    "AGENT: Let me pull up your account, one moment.",
    "AGENT: Could you please describe what happened?",
    "AGENT: I am noting that in the account record now.",
    "AGENT: I'll need to escalate this to a supervisor.",
    "AGENT: Please hold for just a moment while I check on that.",
    "AGENT: Is there anything else I can help you with today?",
    "AGENT: I will note this in your file for our records.",
    "AGENT: That is a great question, let me find out for you.",
    "AGENT: We do see this from time to time and there is a fix.",
    "AGENT: I am sorry for the inconvenience this has caused.",
    "AGENT: It looks like the issue is on our end.",
    "AGENT: Have you tried clearing your browser cache?",
    "AGENT: I will follow up with you within two business days.",
    "AGENT: Thank you for being a valued customer.",
    "AGENT: We do appreciate the feedback.",
    "CUSTOMER: I am calling because I have an issue with my account.",
    "CUSTOMER: This has been going on for a while and it is very frustrating.",
    "CUSTOMER: I have been a customer for many years.",
    "CUSTOMER: Yes I will hold.",
    "CUSTOMER: Thank you for looking into this.",
    "CUSTOMER: That is helpful, thank you.",
    "CUSTOMER: Okay that makes sense.",
    "CUSTOMER: I appreciate your help.",
    "CUSTOMER: Yes please go ahead.",
    "CUSTOMER: That sounds reasonable.",
    "CUSTOMER: I have tried that already and it did not work.",
    "CUSTOMER: Could you explain that one more time?",
    "CUSTOMER: I am not sure what to do at this point.",
    "CUSTOMER: I will think about it and call back.",
    "CUSTOMER: Thank you, that resolves my issue.",
    "CUSTOMER: I will wait for the email.",
    "CUSTOMER: How long should this take?",
    "CUSTOMER: Will I get a confirmation?",
    "CUSTOMER: I am at home right now if you need to call back.",
    "CUSTOMER: Yes I am still here.",
]

FILLER = [
    "[pause]", "[hold music]", "[background noise]", "[customer typing]",
    "[agent reviewing account]", "Mm-hmm.", "Right.", "Okay.", "Yes.",
    "Got it.", "Sure, no problem.", "One moment please.",
    "Let me check on that for you.", "Could you repeat that please?",
    "Sorry, you cut out for a second.", "Thank you.",
]

# Long PII-free description segments — these inflate row size without adding PII.
# Real customers narrate problems at length without mentioning identifying info.
DESCRIPTION_FRAGMENTS = [
    "So I tried to log in earlier today and it kept giving me an error message saying that my session had expired even though I had just typed in my password and clicked the button.",
    "Then I waited a few minutes and tried again but the same thing happened, and now I am worried that maybe my account has been compromised somehow.",
    "I have been using this service for a long time and I have never had this kind of problem before, so I really do not know what changed or what I might have done differently.",
    "When I called yesterday the person I spoke with told me that there was nothing wrong on their end and suggested that I try restarting my computer and clearing my browser cache.",
    "I did all of those things but it still did not fix anything, and at this point I am starting to feel like nobody really knows what is going on.",
    "The reason I am following up today is because I never received a callback or any kind of confirmation that the issue was being looked into.",
    "I understand that you are busy and I am not trying to be difficult, but this has been dragging on for over a week now and I just want it resolved.",
    "If there is a manager available who can authorize a different approach I would really appreciate the opportunity to speak with them directly.",
    "Last time I had a similar issue it turned out to be a problem with the system on your end and not anything that I had done wrong on my own device.",
    "I am hoping that we can figure out what is going on today because I really need to be able to access my information by the end of the week at the latest.",
]


def make_pii_turn(line_template: str, customer: dict, agent_name: str) -> str:
    """Render a PII-bearing turn with all placeholders filled."""
    return line_template.format(
        agent=agent_name,
        cust=customer["name"],
        cust_first=customer["name"].split()[0],
        dob=customer["dob"],
        ssn=customer["ssn"],
        email=customer["email"],
        phone=customer["phone"],
        address=customer["address"],
        amt=f"{random.uniform(10, 5000):.2f}",
        date=fake.date_this_year().isoformat(),
        card=customer["card"],
        exp=customer["card_exp"],
        cvv=customer["card_cvv"],
        ip=customer["ip"],
        acct=customer["account"],
        routing=customer["routing"],
        plate=customer["license"],
        iban=customer["iban"],
    )


def build_transcript(target_chars: int, profile: DensityProfile) -> str:
    customers = make_pii_pool()
    agent_name = fake.name()
    parts: List[str] = [
        "=== CALL TRANSCRIPT ===",
        f"Date: {fake.date_time_this_year().isoformat()}",
        f"Agent: {agent_name}",
        f"Channel: Voice",
        "",
    ]
    customer = random.choice(customers)
    total = sum(len(p) for p in parts) + len(parts)

    while total < target_chars:
        roll = random.random()
        if roll < profile.pii_line_prob:
            line = random.choice(PII_LINES)
            parts.append(make_pii_turn(line, customer, agent_name))
        elif roll < profile.pii_line_prob + profile.plain_line_prob:
            parts.append(random.choice(PLAIN_LINES))
        elif roll < profile.pii_line_prob + profile.plain_line_prob + profile.filler_prob:
            parts.append(random.choice(FILLER))
        else:
            # Description burst — stitch 2-4 fragments together with no PII
            n_frags = random.randint(2, 4)
            burst = " ".join(random.choice(DESCRIPTION_FRAGMENTS) for _ in range(n_frags))
            parts.append(f"CUSTOMER: {burst}")

        total = sum(len(p) for p in parts) + len(parts)

    text = "\n".join(parts)
    if len(text) > target_chars:
        cut = text.rfind(" ", 0, target_chars)
        text = text[: cut if cut > 0 else target_chars]
    return text


def main():
    rows = []
    row_id = 0
    for profile in PROFILES:
        for _ in range(NUM_PER_PROFILE):
            row_id += 1
            text = build_transcript(TARGET_CHARS, profile)
            rows.append({
                "id": row_id,
                "profile": profile.name,
                "char_count": len(text),
                "text": text,
            })
            print(f"row {row_id} [{profile.name:>6}]: {len(text):,} chars")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} rows to {OUT_PATH}")
    print(f"profiles: {NUM_PER_PROFILE} sparse + {NUM_PER_PROFILE} medium + {NUM_PER_PROFILE} dense")


if __name__ == "__main__":
    main()
