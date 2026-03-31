"""Synthetic PII data generator for LSTM training augmentation.

Generates realistic but FAKE PII values embedded in diverse document contexts.
Uses the faker library for structured data generation.

Key constraints:
  - All data is synthetic — no real PII is ever stored
  - Person name labels are EXCLUDED (too generic for pattern learning)
  - Synthetic examples get confidence_type "synthetic" and 0.5x training weight
"""
from __future__ import annotations

import random
import string
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pii_engine.synthetic")

try:
    from faker import Faker
    _HAS_FAKER = True
except ImportError:
    _HAS_FAKER = False
    logger.warning("faker not installed — synthetic data generation unavailable. Install with: pip install faker")

from app.adaptive_learning import _value_to_structure

# Labels that should NOT be synthetically generated (too generic for pattern learning)
EXCLUDED_LABELS = frozenset({
    "PERSON_FULL_NAME", "PERSON_FIRST_NAME", "PERSON_LAST_NAME",
    "PERSON_MIDDLE_NAME", "CARDMEMBER_NAME",
    # Also exclude contextual labels that have no structural pattern
    "OPINIONS_CAPTURED", "PERFORMANCE_RATING", "MARITAL_STATUS",
    "CHILDBEARING_STATUS", "ETHNICITY", "RELIGION", "GENDER",
})


class SyntheticPIIGenerator:
    """Generates fake PII data for training augmentation."""

    def __init__(self, locale: str = "en_US", seed: int = 42):
        if not _HAS_FAKER:
            raise ImportError("faker library required: pip install faker")
        self.fake = Faker(locale)
        Faker.seed(seed)
        random.seed(seed)

        # Registry of generators per PII label
        self._generators: Dict[str, callable] = {
            "SSN": self._gen_ssn,
            "TAX_ID": self._gen_tax_id,
            "EMAIL_ADDRESS": self._gen_email,
            "PHONE_NUMBER": self._gen_phone,
            "STREET_ADDRESS": self._gen_address,
            "DATE_OF_BIRTH": self._gen_dob,
            "AGE": self._gen_age,
            "CREDIT_CARD_NUMBER": self._gen_credit_card,
            "BANK_ACCOUNT_NUMBER": self._gen_bank_account,
            "ROUTING_NUMBER": self._gen_routing_number,
            "IP_ADDRESS": self._gen_ip,
            "URL": self._gen_url,
            "DRIVERS_LICENSE_NUMBER": self._gen_drivers_license,
            "PASSPORT_NUMBER": self._gen_passport,
            "EMPLOYEE_ID": self._gen_employee_id,
            "MEDICAL_RECORD_NUMBER": self._gen_medical_record,
            "INSURANCE_POLICY": self._gen_insurance_policy,
            "VIN": self._gen_vin,
            "LICENSE_PLATE_NUMBER": self._gen_license_plate,
            "API_KEY": self._gen_api_key,
            "USERNAME": self._gen_username,
            "PASSWORD": self._gen_password,
            "CARD_EXPIRATION_DATE": self._gen_card_exp,
            "CARD_SECURITY_CODE": self._gen_csc,
            "SWIFT_BIC_CODE": self._gen_swift,
            "IBAN": self._gen_iban,
            "ACCOUNT_NUMBER_AMEX": self._gen_amex_account,
            "STUDENT_ID": self._gen_student_id,
            "PAYROLL_NUMBER": self._gen_payroll,
            "PRESCRIPTION_NUMBER": self._gen_prescription,
            "PIN": self._gen_pin,
            "ONE_TIME_CODE": self._gen_otp,
        }

    # ── Document Templates ────────────────────────────────────────────

    TEMPLATES = [
        # Insurance call transcript
        "Agent: I'm pulling up your policy now. Your {INSURANCE_POLICY_label} is {INSURANCE_POLICY}. "
        "Can you confirm your {SSN_label}? Customer: Yes, it's {SSN}. "
        "Agent: And the {EMAIL_ADDRESS_label} on file? Customer: {EMAIL_ADDRESS}. "
        "Agent: Your {DATE_OF_BIRTH_label} shows as {DATE_OF_BIRTH}. Is that correct? "
        "Customer: That's right. My {PHONE_NUMBER_label} is {PHONE_NUMBER}.",

        # Medical intake
        "Patient Registration Form\n"
        "{MEDICAL_RECORD_NUMBER_label}: {MEDICAL_RECORD_NUMBER}\n"
        "{SSN_label}: {SSN}\n"
        "{DATE_OF_BIRTH_label}: {DATE_OF_BIRTH}\n"
        "{PHONE_NUMBER_label}: {PHONE_NUMBER}\n"
        "{EMAIL_ADDRESS_label}: {EMAIL_ADDRESS}\n"
        "{STREET_ADDRESS_label}: {STREET_ADDRESS}\n"
        "{INSURANCE_POLICY_label}: {INSURANCE_POLICY}",

        # Banking transaction
        "Wire Transfer Request\n"
        "From {BANK_ACCOUNT_NUMBER_label}: {BANK_ACCOUNT_NUMBER}\n"
        "{ROUTING_NUMBER_label}: {ROUTING_NUMBER}\n"
        "Amount: $4,250.00\n"
        "Beneficiary {BANK_ACCOUNT_NUMBER_label}: {BANK_ACCOUNT_NUMBER}\n"
        "{SWIFT_BIC_CODE_label}: {SWIFT_BIC_CODE}\n"
        "Reference: {EMPLOYEE_ID}",

        # Credit card dispute
        "Card ending in {CREDIT_CARD_NUMBER} was charged $189.99 on 03/15.\n"
        "Card {CARD_EXPIRATION_DATE_label}: {CARD_EXPIRATION_DATE}\n"
        "{CARD_SECURITY_CODE_label}: {CARD_SECURITY_CODE}\n"
        "Cardholder {EMAIL_ADDRESS_label}: {EMAIL_ADDRESS}\n"
        "Please verify your {SSN_label}: {SSN}",

        # HR/Employment
        "Employee Onboarding\n"
        "{EMPLOYEE_ID_label}: {EMPLOYEE_ID}\n"
        "{SSN_label}: {SSN}\n"
        "{DRIVERS_LICENSE_NUMBER_label}: {DRIVERS_LICENSE_NUMBER}\n"
        "{DATE_OF_BIRTH_label}: {DATE_OF_BIRTH}\n"
        "{STREET_ADDRESS_label}: {STREET_ADDRESS}\n"
        "{BANK_ACCOUNT_NUMBER_label} for direct deposit: {BANK_ACCOUNT_NUMBER}\n"
        "{ROUTING_NUMBER_label}: {ROUTING_NUMBER}",

        # E-commerce order
        "Order Confirmation\n"
        "Ship to: {STREET_ADDRESS}\n"
        "Contact: {PHONE_NUMBER}\n"
        "Email: {EMAIL_ADDRESS}\n"
        "Payment: Card ending {CREDIT_CARD_NUMBER}\n"
        "IP Address: {IP_ADDRESS}",

        # Vehicle registration
        "Vehicle Registration\n"
        "{VIN_label}: {VIN}\n"
        "{LICENSE_PLATE_NUMBER_label}: {LICENSE_PLATE_NUMBER}\n"
        "Owner {DRIVERS_LICENSE_NUMBER_label}: {DRIVERS_LICENSE_NUMBER}\n"
        "{SSN_label}: {SSN}\n"
        "{STREET_ADDRESS_label}: {STREET_ADDRESS}",

        # Technical/API
        "Service Configuration\n"
        "API Key: {API_KEY}\n"
        "Endpoint: {URL}\n"
        "Server IP: {IP_ADDRESS}\n"
        "Admin: {USERNAME}\n"
        "Passphrase: {PASSWORD}",
    ]

    # Field label synonyms for template variation
    LABEL_SYNONYMS = {
        "SSN_label": ["SSN", "Social Security Number", "Social", "SS#", "SSN#"],
        "EMAIL_ADDRESS_label": ["Email", "E-mail", "Email Address", "Contact Email"],
        "PHONE_NUMBER_label": ["Phone", "Phone Number", "Contact", "Mobile", "Tel"],
        "DATE_OF_BIRTH_label": ["DOB", "Date of Birth", "Birth Date", "D.O.B."],
        "STREET_ADDRESS_label": ["Address", "Mailing Address", "Home Address", "Street"],
        "INSURANCE_POLICY_label": ["Policy Number", "Policy", "Policy #", "Insurance ID"],
        "MEDICAL_RECORD_NUMBER_label": ["MRN", "Medical Record", "Chart Number", "Patient ID"],
        "BANK_ACCOUNT_NUMBER_label": ["Account", "Account Number", "Acct #", "Bank Account"],
        "ROUTING_NUMBER_label": ["Routing", "Routing Number", "ABA", "Transit Number"],
        "CREDIT_CARD_NUMBER_label": ["Card Number", "CC#", "Credit Card", "Card"],
        "CARD_EXPIRATION_DATE_label": ["Exp", "Expiration", "Exp Date", "Valid Thru"],
        "CARD_SECURITY_CODE_label": ["CVV", "CVC", "CSC", "Security Code"],
        "EMPLOYEE_ID_label": ["Employee ID", "EID", "Staff ID", "Badge Number"],
        "DRIVERS_LICENSE_NUMBER_label": ["DL", "Driver's License", "License Number", "DL#"],
        "VIN_label": ["VIN", "Vehicle ID", "Vehicle Identification Number"],
        "LICENSE_PLATE_NUMBER_label": ["Plate", "License Plate", "Tag Number"],
        "SWIFT_BIC_CODE_label": ["SWIFT", "BIC", "SWIFT/BIC", "Bank Code"],
    }

    def generate(self, n_per_label: int = 20) -> List[Dict[str, Any]]:
        """Generate synthetic training examples.

        Args:
            n_per_label: Number of examples to generate per PII label.

        Returns:
            List of training example dicts compatible with trainer.py format.
        """
        examples = []

        for label, gen_fn in self._generators.items():
            if label in EXCLUDED_LABELS:
                continue

            for _ in range(n_per_label):
                value = gen_fn()
                if not value:
                    continue

                structure = _value_to_structure(value)
                # Pick a random template and fill it
                keywords = self._extract_template_keywords(label)

                examples.append({
                    "structure": structure,
                    "label": label,
                    "length": len(value),
                    "keywords": keywords[:12],
                    "co_labels": self._random_co_labels(label),
                    "source": "synthetic",
                    "score": 0.90,
                    "confidence_type": "synthetic",
                    "timestamp": datetime.now().isoformat(),
                })

        random.shuffle(examples)
        logger.info(f"[SYNTHETIC] Generated {len(examples)} synthetic training examples across {len(self._generators)} labels")
        return examples

    def generate_documents(self, n_documents: int = 50) -> List[Dict[str, Any]]:
        """Generate full synthetic documents with embedded PII.

        Returns training examples extracted from realistic document contexts,
        providing richer keyword and co-label features.
        """
        examples = []

        for _ in range(n_documents):
            template = random.choice(self.TEMPLATES)
            filled_values: Dict[str, str] = {}

            # Fill label synonyms
            for key, synonyms in self.LABEL_SYNONYMS.items():
                template = template.replace("{" + key + "}", random.choice(synonyms))

            # Fill PII values
            for label, gen_fn in self._generators.items():
                placeholder = "{" + label + "}"
                while placeholder in template:
                    value = gen_fn()
                    template = template.replace(placeholder, value, 1)
                    filled_values[label] = value

            # Extract examples from the filled document
            for label, value in filled_values.items():
                if label in EXCLUDED_LABELS or not value:
                    continue
                structure = _value_to_structure(value)
                co_labels = [l for l in filled_values.keys() if l != label and l not in EXCLUDED_LABELS]

                # Extract keywords from surrounding context
                idx = template.find(value)
                if idx >= 0:
                    neighborhood = template[max(0, idx - 100):min(len(template), idx + len(value) + 100)]
                    keywords = self._extract_keywords_from_text(neighborhood)
                else:
                    keywords = self._extract_template_keywords(label)

                examples.append({
                    "structure": structure,
                    "label": label,
                    "length": len(value),
                    "keywords": keywords[:12],
                    "co_labels": co_labels[:10],
                    "source": "synthetic",
                    "score": 0.90,
                    "confidence_type": "synthetic",
                    "timestamp": datetime.now().isoformat(),
                })

        logger.info(f"[SYNTHETIC] Generated {len(examples)} document-based examples from {n_documents} documents")
        return examples

    # ── Value Generators ──────────────────────────────────────────────

    def _gen_ssn(self) -> str:
        fmt = random.choice(["###-##-####", "### ## ####"])
        return self.fake.bothify(fmt, letters="")

    def _gen_tax_id(self) -> str:
        return self.fake.bothify("9##-##-####", letters="")

    def _gen_email(self) -> str:
        return self.fake.email()

    def _gen_phone(self) -> str:
        fmt = random.choice([
            "(###) ###-####", "###-###-####", "+1 ###-###-####",
            "### ### ####", "###.###.####",
        ])
        return self.fake.bothify(fmt, letters="")

    def _gen_address(self) -> str:
        return self.fake.address().replace("\n", ", ")

    def _gen_dob(self) -> str:
        fmt = random.choice(["%m/%d/%Y", "%Y-%m-%d", "%d %B %Y", "%B %d, %Y"])
        return self.fake.date_of_birth(minimum_age=18, maximum_age=90).strftime(fmt)

    def _gen_age(self) -> str:
        return str(random.randint(18, 90))

    def _gen_credit_card(self) -> str:
        return self.fake.credit_card_number()

    def _gen_bank_account(self) -> str:
        length = random.randint(8, 17)
        return "".join(random.choices(string.digits, k=length))

    def _gen_routing_number(self) -> str:
        # Generate valid-looking 9-digit routing number
        return "".join(random.choices(string.digits, k=9))

    def _gen_ip(self) -> str:
        return random.choice([self.fake.ipv4(), self.fake.ipv6()])

    def _gen_url(self) -> str:
        return self.fake.url()

    def _gen_drivers_license(self) -> str:
        state = random.choice(["CA", "NY", "TX", "FL", "IL", "PA", "OH"])
        num = "".join(random.choices(string.digits, k=random.randint(7, 12)))
        return f"{state}{num}"

    def _gen_passport(self) -> str:
        letter = random.choice(string.ascii_uppercase)
        num = "".join(random.choices(string.digits, k=random.randint(7, 8)))
        return f"{letter}{num}"

    def _gen_employee_id(self) -> str:
        prefix = random.choice(["EMP", "E", "ID", ""])
        num = "".join(random.choices(string.digits, k=random.randint(5, 8)))
        sep = random.choice(["-", "", "#"])
        return f"{prefix}{sep}{num}"

    def _gen_medical_record(self) -> str:
        prefix = random.choice(["MRN", "MR", ""])
        num = "".join(random.choices(string.digits, k=random.randint(6, 10)))
        sep = random.choice(["-", "", "#"])
        return f"{prefix}{sep}{num}"

    def _gen_insurance_policy(self) -> str:
        prefix = random.choice(["POL", "INS", "P", "HIC"])
        num = "".join(random.choices(string.digits, k=random.randint(6, 10)))
        return f"{prefix}-{num}"

    def _gen_vin(self) -> str:
        chars = string.ascii_uppercase.replace("I", "").replace("O", "").replace("Q", "") + string.digits
        return "".join(random.choices(chars, k=17))

    def _gen_license_plate(self) -> str:
        fmt = random.choice(["???-####", "### ???", "?? ####", "????###"])
        return self.fake.bothify(fmt)

    def _gen_api_key(self) -> str:
        prefix = random.choice(["sk-", "pk-", "api_", "key_", ""])
        length = random.randint(24, 48)
        chars = string.ascii_letters + string.digits
        return prefix + "".join(random.choices(chars, k=length))

    def _gen_username(self) -> str:
        return self.fake.user_name()

    def _gen_password(self) -> str:
        return self.fake.password(length=random.randint(8, 16))

    def _gen_card_exp(self) -> str:
        month = f"{random.randint(1, 12):02d}"
        year = str(random.randint(25, 30))
        return f"{month}/{year}"

    def _gen_csc(self) -> str:
        return "".join(random.choices(string.digits, k=random.choice([3, 4])))

    def _gen_swift(self) -> str:
        bank = "".join(random.choices(string.ascii_uppercase, k=4))
        country = random.choice(["US", "GB", "DE", "FR", "JP", "AU"])
        loc = "".join(random.choices(string.ascii_uppercase + string.digits, k=2))
        branch = "".join(random.choices(string.ascii_uppercase + string.digits, k=3))
        return f"{bank}{country}{loc}{branch}"

    def _gen_iban(self) -> str:
        country = random.choice(["GB", "DE", "FR", "NL", "ES"])
        check = f"{random.randint(10, 99)}"
        bank = "".join(random.choices(string.ascii_uppercase, k=4))
        account = "".join(random.choices(string.digits, k=random.randint(10, 18)))
        return f"{country}{check}{bank}{account}"

    def _gen_amex_account(self) -> str:
        return "3" + random.choice(["4", "7"]) + "".join(random.choices(string.digits, k=13))

    def _gen_student_id(self) -> str:
        prefix = random.choice(["STU", "S", ""])
        num = "".join(random.choices(string.digits, k=random.randint(6, 9)))
        return f"{prefix}{num}"

    def _gen_payroll(self) -> str:
        prefix = random.choice(["PR", "PAY", ""])
        num = "".join(random.choices(string.digits, k=random.randint(5, 8)))
        return f"{prefix}-{num}"

    def _gen_prescription(self) -> str:
        prefix = random.choice(["RX", "Rx", ""])
        num = "".join(random.choices(string.digits, k=random.randint(7, 12)))
        return f"{prefix}{num}"

    def _gen_pin(self) -> str:
        return "".join(random.choices(string.digits, k=random.choice([4, 6])))

    def _gen_otp(self) -> str:
        return "".join(random.choices(string.digits, k=6))

    # ── Helper Methods ────────────────────────────────────────────────

    def _extract_template_keywords(self, label: str) -> List[str]:
        """Extract likely context keywords for a label from template synonyms."""
        keywords = []
        for key, synonyms in self.LABEL_SYNONYMS.items():
            if label in key.replace("_label", ""):
                keywords.extend(w.lower() for s in synonyms for w in s.split())
        # Add generic keywords from the label name itself
        keywords.extend(w.lower() for w in label.split("_") if len(w) > 2)
        return list(set(keywords))[:12]

    def _random_co_labels(self, exclude_label: str) -> List[str]:
        """Pick random co-occurring labels (realistic PII co-occurrence)."""
        available = [l for l in self._generators if l != exclude_label and l not in EXCLUDED_LABELS]
        n = random.randint(2, min(6, len(available)))
        return sorted(random.sample(available, n))

    @staticmethod
    def _extract_keywords_from_text(text: str) -> List[str]:
        """Extract meaningful keywords from a text snippet."""
        import re
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "to", "of", "in", "for", "on", "at", "by", "and", "or", "it",
                     "that", "this", "with", "from", "as", "your", "you", "my", "i",
                     "can", "yes", "no", "not"}
        words = re.findall(r'[a-z]{3,}', text.lower())
        return [w for w in words if w not in stopwords][:12]
