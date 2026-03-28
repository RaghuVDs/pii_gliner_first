---
license: apache-2.0
language:
- en
library_name: gliner
pipeline_tag: token-classification
tags:
- NER
- GLiNER
- information extraction
- PII
- PHI
- PCI
- entity recognition
- multilingual
---


# GLiNER-PII: Zero-shot PII model

A production-grade open-source model for privacy-focused PII, PHI, and PCI detection with zero-shot entity recognition capabilities. 
This model was developed in collaboration between [Wordcab](https://wordcab.com/) and [Knowledgator](https://www.knowledgator.com/). For enterprise-ready, specialized PII/PHI/PCI models, contact us at info@wordcab.com.

## 🧠 What is GLiNER?

GLiNER (Generalist and Lightweight Named Entity Recognition) is a bidirectional transformer model that can identify **any entity type** without predefined categories. Unlike traditional NER models that are limited to specific entity classes, GLiNER allows you to specify exactly what entities you want to extract at runtime.

### Key Advantages

- **Zero-shot recognition**: Extract any entity type without retraining
- **Privacy-first**: Process sensitive data locally without API calls
- **Lightweight**: Much faster than large language models for NER tasks
- **Production-ready**: Quantization-aware training with FP16 and UINT8 ONNX models
- **Comprehensive**: 60+ predefined PII categories with custom entity support

### How GLiNER Works

Instead of predicting from a fixed set of entity classes, GLiNER takes both text and a list of desired entity types as input, then identifies spans that match those categories:

```python
text = "John Smith called from 415-555-1234 to discuss his account."
entities = ["name", "phone number", "account number"]
# GLiNER finds: "John Smith" → name, "415-555-1234" → phone number
```

## 🐍 Python Implementation

The primary GLiNER implementation provides comprehensive PII detection with 60+ entity categories, fine-tuned specifically for privacy and compliance use cases.

### Installation

```bash
pip install gliner
```

### Quick Start

```python
from gliner import GLiNER

# Load the model (downloads automatically on first use)
model = GLiNER.from_pretrained("knowledgator/gliner-pii-base-v1.0")

text = "John Smith called from 415-555-1234 to discuss his account number 12345678."
labels = ["name", "phone number", "account number"]

entities = model.predict_entities(text, labels, threshold=0.3)

for entity in entities:
    print(f"{entity['text']} => {entity['label']} (confidence: {entity['score']:.2f})")
```

Output:
```
John Smith => name (confidence: 0.95)
415-555-1234 => phone number (confidence: 0.92)
12345678 => account number (confidence: 0.88)
```

### Comprehensive PII Detection

The model was specifically optimized for 60+ predefined PII categories organized by domain, but it can work in zero-shot as well, meaning you can put any labels you need:

#### Personal Identifiers

```python
personal_labels = [
    "name",                       # Full names
    "first name",                 # First names  
    "last name",                  # Last names
    "name medical professional",  # Healthcare provider names
    "dob",                        # Date of birth
    "age",                        # Age information
    "gender",                     # Gender identifiers
    "marital status"              # Marital status
]
```

#### Contact Information

```python
contact_labels = [
    "email address",          # Email addresses
    "phone number",           # Phone numbers
    "ip address",             # IP addresses
    "url",                    # URLs
    "location address",       # Street addresses
    "location street",        # Street names
    "location city",          # City names
    "location state",         # State/province names
    "location country",       # Country names
    "location zip"            # ZIP/postal codes
]
```

#### Financial Information

```python
financial_labels = [
    "account number",         # Account numbers
    "bank account",           # Bank account numbers
    "routing number",         # Routing numbers
    "credit card",            # Credit card numbers
    "credit card expiration", # Card expiration dates  
    "cvv",                    # CVV/security codes
    "ssn",                    # Social Security Numbers
    "money"                   # Monetary amounts
]
```

#### Healthcare Information

```python
healthcare_labels = [
    "condition",                    # Medical conditions
    "medical process",              # Medical procedures
    "drug",                         # Drugs
    "dose",                         # Dosage information
    "blood type",                   # Blood types
    "injury",                       # Injuries
    "organization medical facility",# Healthcare facility names
    "healthcare number",            # Healthcare numbers
    "medical code"                  # Medical codes
]
```

#### Identification Documents

```python
id_labels = [
    "passport number",       # Passport numbers
    "driver license",        # Driver's license numbers
    "username",              # Usernames
    "password",              # Passwords
    "vehicle id"             # Vehicle IDs
]
```

### Advanced Usage Examples

#### Multi-Category Detection
```python
text = """
Patient Mary Johnson, DOB 01/15/1980, was discharged on March 10, 2024 
from St. Mary's Hospital. Contact: mary.j@email.com, (555) 123-4567.
Insurance policy: POL-789456123.
"""

labels = [
    "name", "dob", "discharge date", "organization medical facility",
    "email address", "phone number", "policy number"
]

entities = model.predict_entities(text, labels, threshold=0.3)

for entity in entities:
    print(f"Found '{entity['text']}' as {entity['label']}")
```

#### Batch Processing for High Throughput
```python
documents = [
    "Customer John called about his credit card ending in 4532.",
    "Sarah's SSN 123-45-6789 needs verification.",
    "Email support@company.com for account 987654321 issues."
]

labels = ["name", "credit card", "ssn", "email address", "account number"]

# Process multiple documents efficiently
results = model.run(documents, labels, threshold=0.3, batch_size=8)

for doc_idx, entities in enumerate(results):
    print(f"\nDocument {doc_idx + 1}:")
    for entity in entities:
        print(f"  {entity['text']} => {entity['label']}")
```

#### Custom Entity Detection
```python
# GLiNER isn't limited to PII - you can detect any entities
text = "The MacBook Pro with M2 chip costs $1,999 at the Apple Store in Manhattan."
custom_labels = ["product", "processor", "price", "store", "location"]

entities = model.predict_entities(text, custom_labels, threshold=0.3)
```

#### Threshold Optimization
```python
# Lower threshold: Higher recall, more false positives
high_recall = model.predict_entities(text, labels, threshold=0.2)

# Higher threshold: Higher precision, fewer false positives
high_precision = model.predict_entities(text, labels, threshold=0.6)

# Recommended starting point for production
balanced = model.predict_entities(text, labels, threshold=0.3)
```

## 💡 Use Cases

GLiNER excels in privacy-focused applications where traditional cloud-based NER services pose compliance risks.

### 🎯 **Primary Applications**

#### Privacy-First Voice & Transcription
```python
# Automatically redact PII from voice transcriptions
transcription = "Hi, my name is Sarah Johnson and my phone number is 415-555-0123"
pii_labels = ["name", "phone number", "email address", "ssn"]

entities = model.predict_entities(transcription, pii_labels)
# Redact or anonymize detected PII before storage
```

#### Compliance-Ready Document Processing  
```python
# Healthcare: HIPAA-compliant note processing
medical_note = "Patient John Doe, MRN 123456, diagnosed with diabetes..."
phi_labels = ["name", "medical record number", "condition", "dob"]

# Finance: PCI-DSS compliant transaction logs
transaction_log = "Card ****4532 charged $299.99 to John Smith"
pci_labels = ["credit card", "money", "name"]

# Legal: Attorney-client privilege protection
legal_doc = "Client Jane Doe vs. Corporation ABC, case #2024-CV-001"
legal_labels = ["name", "organization", "case number"]
```

#### Real-Time Data Anonymization
```python
def anonymize_text(text, entity_types):
    """Anonymize PII in real-time"""
    entities = model.predict_entities(text, entity_types)
    
    # Sort by position to replace from end to start
    entities.sort(key=lambda x: x['start'], reverse=True)
    
    anonymized = text
    for entity in entities:
        placeholder = f"<{entity['label'].upper()}>"
        anonymized = anonymized[:entity['start']] + placeholder + anonymized[entity['end']:]
    
    return anonymized

original = "John Smith's SSN is 123-45-6789"
anonymized = anonymize_text(original, ["name", "ssn"])
print(anonymized)  # "<NAME>'s SSN is <SSN>"
```

### 🌟 **Extended Applications**

#### Enhanced Search & Content Understanding
```python
# Extract key entities from user queries for better search
query = "Find restaurants near Stanford University in Palo Alto"
search_entities = ["organization", "location city", "business type"]

# Intelligent document tagging
document = "This quarterly report discusses Microsoft's Azure growth..."
doc_entities = ["organization", "product", "time period"]
```

#### GDPR-Compliant Chatbot Logs
```python
def sanitize_chat_log(message):
    """Remove PII from chat logs per GDPR requirements"""
    sensitive_types = [
        "name", "email address", "phone number", "location address",
        "credit card", "ssn", "passport number"
    ]
    
    entities = model.predict_entities(message, sensitive_types)
    if entities:
        # Log anonymized version, alert compliance team
        return anonymize_text(message, sensitive_types)
    return message
```

#### Secure Mobile & Edge Processing
```python
# Process sensitive data entirely on-device
def process_locally(user_input):
    """Process PII detection without cloud APIs"""
    pii_types = ["name", "phone number", "email address", "ssn", "credit card"]
    
    # All processing happens locally - no data leaves device
    detected_pii = model.predict_entities(user_input, pii_types)
    
    if detected_pii:
        return "⚠️ Sensitive information detected - proceed with caution"
    return "✅ No PII detected - safe to share"
```

## 📊 Performance Benchmarks

### Accuracy Evaluation

The following benchmarks were run on the **synthetic-multi-pii-ner-v1** dataset.
We compare multiple GLiNER-based PII models, including our new **Knowledgator GLiNER PII Edge v1.0**.

| Model Path                                                             | Precision | Recall | F1 Score   |
| ---------------------------------------------------------------------- | --------- | ------ | ---------- |
| **knowledgator/gliner-pii-edge-v1.0** | 78.96%    | 72.34% | **75.50%** |
| **knowledgator/gliner-pii-small-v1.0**                                             | 78.99%    | 74.80% | **76.84%** |
| **knowledgator/gliner-pii-base-v1.0**                                              | 79.28%    | 82.78% | **80.99%** |
| **knowledgator/gliner-pii-large-v1.0**             | 87.42% | 79.4% | **83.25%** |
| **urchade/gliner\_multi\_pii-v1**                                      | 79.19%    | 74.67% | **76.86%** |
| **E3-JSI/gliner-multi-pii-domains-v1**                                 | 78.35%    | 74.46% | **76.36%** |
| **gravitee-io/gliner-pii-detection**                                   | 81.27%    | 56.76% | **66.84%** |

### Key Takeaways

* **Base Post Model** (`knowledgator/gliner-pii-base-v1.0`) achieves the **highest F1 score (80.99%)**, indicating the strongest overall performance.
* **Knowledgator Edge Model** (`knowledgator/gliner-pii-edge-v1.0`) is optimized for **edge environments**, trading a slight decrease in recall for lower latency and footprint.
* **Gravitee-io Model** shows strong precision but lower recall, indicating it is tuned for high confidence but misses more entities.


### Comparison with Alternatives

| Solution              | Speed | Privacy | Accuracy | Flexibility | Cost     |
| --------------------- | ----- | ------- | -------- | ----------- | -------- |
| **GLiNER**            | ⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐        | Free     |
| Cloud NER APIs        | ⭐⭐⭐   | ⭐⭐⭐     | ⭐⭐⭐⭐⭐    | ⭐⭐⭐         | \$\$\$   |
| Large Language Models | ⭐⭐    | ⭐⭐      | ⭐⭐⭐⭐     | ⭐⭐⭐⭐        | \$\$\$\$ |
| Traditional NER       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐   | ⭐⭐⭐⭐     | ⭐           | Free     |


## 🚀 Alternative Implementations

While Python provides the most comprehensive PII detection capabilities, GLiNER is available across multiple languages for different deployment scenarios.

### 🦀 Rust Implementation (gline-rs)

**Best for**: High-performance backend services, microservices

```toml
[dependencies]
"gline-rs" = "1"
```

```rust
use gline_rs::{GLiNER, TextInput, Parameters, RuntimeParameters};

let model = GLiNER::<TokenMode>::new(
    Parameters::default(),
    RuntimeParameters::default(),
    "tokenizer.json",
    "model.onnx",
)?;

let input = TextInput::from_str(
    &["My name is James Bond."],
    &["person"],
)?;

let output = model.inference(input)?;
```

**Performance**: 4x faster than Python on CPU, 37x faster with GPU acceleration.

### ⚡ C++ Implementation (GLiNER.cpp)

**Best for**: Embedded systems, mobile apps, edge devices

```cpp
#include "GLiNER/model.hpp"

gliner::Config config{12, 512};
gliner::Model model("./model.onnx", "./tokenizer.json", config);

std::vector<std::string> texts = {"John works at Microsoft"};
std::vector<std::string> entities = {"person", "organization"};

auto output = model.inference(texts, entities);
```

### 🌐 JavaScript Implementation (GLiNER.js) 

**Best for**: Web applications, browser-based processing

```bash
npm install gliner
```

```javascript
import { Gliner } from 'gliner';

const gliner = new Gliner({
  tokenizerPath: "onnx-community/gliner_small-v2",
  onnxSettings: {
    modelPath: "public/model.onnx",
    executionProvider: "webgpu",
  }
});

await gliner.initialize();

const results = await gliner.inference({
  texts: ["John Smith works at Microsoft"],
  entities: ["person", "organization"],
  threshold: 0.1,
});
```

## 🏗️ Model Architecture & Training

### Quantization-Aware Pretraining

GLiNER models use quantization-aware pretraining, which optimizes performance while maintaining accuracy. This allows efficient inference even with quantized models.

### Available ONNX Formats

| Format | Size | Use Case |
|--------|------|----------|
| **FP16** | 330MB | Balanced performance/accuracy |
| **UINT8** | 197MB | Maximum efficiency |

### Model Conversion

```bash
python convert_to_onnx.py \
  --model_path knowledgator/gliner-pii-base-v1.0 \
  --save_path ./model \
  --quantize True  # For UINT8 quantization
```


## 📄 References

- [GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer](https://arxiv.org/abs/2311.08526)
- [GLiNER multi-task: Generalist Lightweight Model for Various Information Extraction Tasks](https://arxiv.org/abs/2406.12925) 
- [Named Entity Recognition as Structured Span Prediction](https://arxiv.org/abs/2212.13415)

## 🙏 Acknowledgments

Special thanks to the all GLiNER contributors, the Wordcab team and additional thanks to maintainers of the Rust, C++, and JavaScript implementations.

## 📞 Support

- **Hugging Face**: [Ihor/gliner-pii-small](https://huggingface.co/Ihor/gliner-pii-small)
- **GitHub Issues**: [Report bugs and request features](https://github.com/info-wordcab/wordcab-pii)
- **Discord**: [Join community discussions](https://discord.gg/wRF7tuY9)

---

*GLiNER: Open-source privacy-first entity recognition for production applications.*