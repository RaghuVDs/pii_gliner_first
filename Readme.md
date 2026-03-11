---
title: "Enterprise Hybrid PII Redaction Engine: Architecture & Documentation"
author: "Raghuveera Narasimha"
date: "March 11th 2026"
output:
  html_document:
    toc: true
    toc_depth: 3
    theme: readable
---

# 1. Executive Summary

### What I Built
I engineered a, Data Loss Prevention (DLP) engine capable of detecting and redacting Personally Identifiable Information (PII) from highly ambiguous, unstructured text (such as raw chat logs, legal documents, and corporate emails). 

### Why I Built It
Traditional PII scanners rely purely on Regex (Mathematical patterns) and large dictionaries of "bad words." This fails in the modern enterprise because unstructured data doesn't follow strict rules. If a customer is named "Chase Hunter," traditional systems might ignore the name (thinking they are verbs/nouns) or falsely flag the word "Chase" as a bank. I needed an engine that actually *understands context* while running fast enough to be deployed on standard CPU servers.

### The Hybrid Approach
Instead of relying on a single tool, I chose a **Hybrid Architecture**:
1. **Regex Engine:** Handles the absolute math (Credit Cards, Routing Numbers, IP Addresses).
2. **AI Semantic Engine (GLiNER):** Handles the human context (Names, Employer Names, Medical conditions and other specific SDEs).
3. **Grammatical Engine (spaCy):** Acts as the referee to stop the AI from hallucinating.

---

# 2. The Core AI: What is GLiNER?

### What is GLiNER?
GLiNER (Generalist Model for Named Entity Recognition) is a modern, lightweight NLP (Natural Language Processing) model. Traditional models (like standard spaCy or standard BERT) are hardcoded to find specific things (eg:, `PERSON`, `ORG`, `LOCATION`). If you want them to find a new entity, like a "Crypto Wallet" or "Performance Rating", you have to manually label thousands of documents and retrain the AI.

GLiNER is a **Zero-Shot** model. This means you can simply type the name of the entity you want to find into a config file (e.g., "Medical Condition" or "Late Payment Clause"), and the AI will dynamically figure out what that looks like based on its general understanding of the English language. 

### Why I Chose GLiNER
* **Future-Proofing:** I never have to retrain the model. When a new compliance law is passed, I simply added a new line to the `pii_taxonomy.yaml` file, and GLiNER instantly knows how to find it.
* **Context Awareness:** It reads the whole sentence, not just the word. It knows that "April" can be a month or a person depending on the words around it.

### The Architecture of GLiNER
GLiNER is built on a bidirectional transformer architecture, but it works differently than typical models:
1. **The Text Encoder:** It reads the input document and turns every word into a mathematical vector (understanding its context).
2. **The Prompt Encoder:** It takes our list of PII labels (from the YAML file) and turns those into mathematical vectors.
3. **Span Representation:** Instead of looking at individual words, it groups words into "spans" (e.g., "John", "John Doe", "John Doe Smith").
4. **The Matcher:** It computes the mathematical similarity between the Document Spans and the Prompt Labels. If the similarity crosses our confidence threshold (e.g., `0.35`), it flags it as a match.

---

# 3. The False Positive Problem & Introducing spaCy

### The Problem: AI Hallucinations
Because GLiNER is trying to guess meaning, it "occasionally panics". If it sees a highly legal document, it might aggressively flag the word "Salary" as an `INCOME` variable or flag the generic phrase "late payment" as a `PAYMENT_HISTORY` event. 

Historically, developers fix this by hardcoding "Stop Words" (eg: `if word == "salary": ignore()`). This creates a "whack-a-mole" problem. It is impossible to predict and hardcode every generic word the AI might mess up in the future.

### The Solution: Grammatical Cross-Validation with spaCy
I have also introduced **spaCy**, an ultra-fast, offline natural language processing library. I did not use spaCy to find PII; I used spaCy strictly to analyze the **Grammar** of what GLiNER found. 

By introducing spaCy, I have completely eliminated hardcoded stop-word lists and replaced them with **The 3 Universal Laws of Information Theory**.

### The 3 Universal Dynamic Laws (How spaCy works in our engine):

* **LAW 1: Morphological Integrity (Shape over Meaning)**
  * *The Rule:* If the AI flags something as an "ID", "Code", or "Number", my code mathematically checks if it contains at least one digit. 
  * *Why:* If the AI thinks the word "Illinois" is an ID card, Law 1 sees zero digits and instantly kills the detection.
  
* **LAW 2: The Universal Grammar Gate (Powered by spaCy)**
  * *The Rule:* I will pass the extracted text to spaCy to get its Part-of-Speech (POS). Actual PII is inherently composed of Proper Nouns (`PROPN`) or Numbers (`NUM`). 
  * *Why:* If the AI flags "late payment" as PII, spaCy looks at it, says "late" is an Adjective and "payment" is a common Noun. Because it lacks a Proper Noun or Number, Law 2 dynamically drops it. 

* **LAW 3: The Sparsity Principle**
  * *The Rule:* Real PII is rare (a specific SSN or Credit Card only appears a few times). If a non-numeric string repeats 4 or more times in a document, it is mathematically proven to be the subject matter of the document, not an isolated piece of PII.
  * *Why:* This stops the engine from redacting standard boilerplate legal terms that appear throughout a 50-page contract.

---

# 4. Performance & CPU Optimization

### The Bottleneck
The engine taken 10 to 30 seconds to process a document depending on the size. This occurs because I am running a 350-million parameter neural network on a standard CPU and I accidentally caused "Thread Thrashing" (Python threads fighting with C++ PyTorch threads for CPU core dominance).

# 5. THE OLDER SYSTEM

### Advantages of the Older System:
1. Blazing Fast
2. Highly Predictable
3. Cheap to Run ( does not require AI)

### Disadvantages of the Older System:
1. It has absolutely no understanding of context
2. If a user makes a typo or a new form is introduced the system completely fails to detect it because it wasn't explicitly programmed to look for it.
3. Massive False Positives

# 6. "MY MODEL" (The Hybrid GLiNER + Regex + spaCy Engine)

# Advantages of My Model:
1. True Contextual Awareness
2. Zero Hardcoding Required
3. Future-Proof (Zero-Shot)
4. By layering AI and Regex together, the engine catches edge cases (**Still research and testing is needed)
5. Highest Accuracy for a light weight model

# Disadvantages of My Model:
1. Slower Processing Time
2. AI Hallucinations Occasionally

# 7. FUTURE IMPROVEMENT 2: MODEL FINE-TUNING VIA REINFORCEMENT LEARNING

Reinforcement Learning from Human Feedback (RLHF): Build a continuous-learning pipeline. When the engine flags a document, human  officers will review it. If the model made a mistake (a false positive), the human clicks "Reject." Feed this feedback back into the model, punishing it mathematically for that specific hallucination. Over time, the AI learns the exact boundaries of corporate data.

By continually rewarding the model for correctly identifying complex corporate structures the model will transition from a "Better Performer" into a highly specialized Enterprise Auditor, naturally increasing its base confidence scores and reducing reliance on the Python safety nets.

# 5. Conclusion

By combining the determinism of Regex, the semantic adaptability of GLiNER and the grammatical discipline of spaCy I have created an engine that does not rely on guessing the future. 

It does not require endless maintenance of hardcoded dictionaries. It dynamically adapts to new data structures, aggressively protects financial data and self-corrects its own AI hallucinations in real-time.