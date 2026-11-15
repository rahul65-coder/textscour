#!/usr/bin/env python3
"""
Deterministic test corpus generator for TEXTSCOUR.
Generates diverse, realistic test datasets across formats (JSONL, JSON, Parquet)
covering clean text, promotional noise, URLs, PII, boilerplate, deduplication candidates,
and multilingual samples (English, Hindi, Hinglish).

Supports scaling up to 200,000 diverse documents (~3 KB average, ~5% duplicates,
nested schemas) with memory-efficient chunked streaming.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Generator

# Extensive thematic sentence and paragraph pools for combinatoric generation

SCIENCE_EN = [
    "Machine learning algorithms construct mathematical models based on sample training data to make predictions or decisions.",
    "Supervised learning algorithms infer a mapping function from input features to known target outputs across high-dimensional manifolds.",
    "Photosynthesis is a fundamental biochemical pathway in which chlorophyll pigments capture solar photons to synthesize hexose sugars.",
    "The laws of thermodynamics state that the total energy of an isolated physical system remains constant over time.",
    "Quantum mechanics describes microscopic physical phenomena utilizing probability wavefunctions and non-commuting operators.",
    "Plate tectonics explains seismic fault line dynamics and continental drift driven by mantle convection currents.",
    "Mitochondrial cellular respiration generates adenosine triphosphate via proton gradients across inner cristae membranes.",
    "Stellar nucleosynthesis inside supernova explosions produces heavy atomic elements distributed into the interstellar medium.",
    "DNA polymerase enzymes replicate complementary nucleic acid strands with high fidelity through proofreading mechanisms.",
    "General relativity characterizes gravitational attraction as the geometric curvature of four-dimensional spacetime.",
]

TECH_EN = [
    "Distributed consensus protocols such as Raft and Paxos ensure state machine replication across unreliable networks.",
    "Relational database management systems employ Write-Ahead Logging to guarantee transaction atomicity and durability.",
    "Modern operating system kernels implement virtual memory subsystems with multi-level page table paging hierarchies.",
    "Compiler optimization pipelines perform static single assignment transformations, constant folding, and dead code elimination.",
    "Cloud storage architectures partition objects across erasure-coded storage nodes to achieve high durability at low cost.",
    "Asynchronous event loops multiplex concurrent socket input and output operations using epoll and kqueue system calls.",
    "Hardware security modules safeguard cryptographic root keys against physical side-channel and timing attacks.",
    "Software container runtimes isolate process namespaces and cgroups to enforce strict CPU and memory resource quotas.",
    "Microservice message brokers utilize partitioned commit logs to facilitate high-throughput decoupled event streaming.",
    "Garbage collectors in managed runtimes balance generational allocation throughput against stop-the-world pause latency.",
]

HISTORY_EN = [
    "The Indus Valley Civilisation flourished in the fertile river basins of northwest South Asia from 3300 BCE onwards.",
    "Archaeological excavations at Mohenjo-daro revealed sophisticated civic drainage systems and brick masonry architecture.",
    "Ancient maritime trade routes connected ports along the Arabian Sea with Mesopotamian merchant settlements.",
    "The Mauryan Empire established centralized administrative structures across the subcontinent under Chandragupta and Ashoka.",
    "Renaissance humanism catalyzed scientific inquiries and artistic innovations throughout fifteenth-century Europe.",
    "The construction of the Grand Canal during the Sui Dynasty facilitated large-scale inland grain transportation in China.",
    "Medieval guilds regulated commercial apprenticeship standards and manufacturing quality across European towns.",
    "The Industrial Revolution transformed manufacturing processes through steam power mechanization and metallurgical advances.",
    "Classical antiquity witnessed significant developments in axiomatic geometry, rhetoric, and political philosophy.",
    "Maritime navigational advancements during the Age of Discovery enabled transatlantic commerce and cartographic mapping.",
]

ECONOMICS_EN = [
    "Central banks calibrate benchmark monetary interest rates to maintain price stability and sustainable macroeconomic growth.",
    "Fiscal policy interventions stimulate aggregate demand during economic contractions through targeted public capital expenditures.",
    "Global supply chains optimize inventory logistics utilizing just-in-time manufacturing and multimodal shipping corridors.",
    "Financial market equilibria reflect the aggregation of dispersed market expectations under rational expectations theory.",
    "Labor market mobility and specialized human capital investments drive long-term productivity growth across metropolitan regions.",
    "Econometric time-series models evaluate structural shifts and volatility clustering in sovereign bond yield curves.",
    "Microeconomic consumer choice theory models utility maximization subject to budgetary constraint frontiers.",
    "Regulatory frameworks mandate prudential capital adequacy ratios to mitigate systemic liquidity contagion in commercial banking.",
]

CULTURE_EN = [
    "Traditional culinary practices across diverse cultural regions emphasize indigenous spices, grains, and preservation techniques.",
    "Folk music traditions pass down narrative ballads and distinct rhythmic structures through generations of oral performance.",
    "Architectural heritage sites reflect regional climatic adaptations, native materials, and historical aesthetic sensibilities.",
    "Linguistic diversity enriches literature through unique idiomatic expressions, syntactic structures, and cultural metaphors.",
    "Community agricultural harvest festivals celebrate seasonal transitions with communal feasts, dances, and symbolic rituals.",
    "Textile weaving traditions incorporate intricate geometric patterns that signify ancestral lineage and regional identity.",
]

# Hindi sentence pools (Devanagari script)
SCIENCE_HI = [
    "कंप्यूटर विज्ञान सूचना और अभिकलन के सैद्धांतिक आधारों और प्रणालियों में उनके अनुप्रयोग का वैज्ञानिक अध्ययन है।",
    "सौर मंडल सूर्य और गुरुत्वाकर्षण द्वारा उससे बंधे सभी आकाशीय पिंडों जैसे ग्रहों, उपग्रहों और क्षुद्रग्रहों से निर्मित है।",
    "प्रकाश संश्लेषण द्वारा हरे पौधे सूर्य के प्रकाश की ऊर्जा को रासायनिक ऊर्जा में परिवर्तित करते हैं।",
    "ऊर्जा संरक्षण का नियम यह प्रतिपादित करता है कि ऊर्जा को न तो उत्पन्न किया जा सकता है और न ही नष्ट किया जा सकता है।",
    "क्वांटम भौतिकी अत्यंत सूक्ष्म परमाण्विक स्तर पर कणों के व्यवहार और उनके तरंग गुणों की व्याख्या करती है।",
    "पृथ्वी की आंतरिक संरचना क्रस्ट, मेंटल और कोर नामक विभिन्न घनत्व वाली परतों में विभाजित है।",
    "कोशिका विभाजन की प्रक्रिया जीवों की वृद्धि, विकास और क्षतिग्रस्त ऊतकों के पुनर्निर्माण के लिए अनिवार्य है।",
    "खगोलविदों ने दूरबीनों की सहायता से दूरस्थ आकाशगंगाओं और कृष्ण विवरों के गुरुत्वाकर्षण प्रभावों का अध्ययन किया है।",
]

HISTORY_HI = [
    "सिंधु घाटी सभ्यता प्राचीन विश्व की प्रमुख नदी घाटी सभ्यताओं में से एक थी जो नगर नियोजन के लिए विख्यात है।",
    "हड़प्पा और मोहनजोदड़ो के उत्खनन से सुव्यवस्थित जल निकासी प्रणाली और पक्की ईंटों के निर्माण के साक्ष्य मिले हैं।",
    "प्राचीन भारत में विज्ञान, गणित और आयुर्वेद के क्षेत्र में चरक, सुश्रुत और आर्यभट्ट ने युगांतरकारी शोध किए थे।",
    "मौर्य साम्राज्य के शासनकाल में संपूर्ण भारतीय उपमहाद्वीप में सुदृढ़ प्रशासनिक व्यवस्था और व्यापार का विस्तार हुआ।",
    "भारतीय दर्शन की विभिन्न शाखाओं ने ज्ञान मीमांसा, तत्वमीमांसा और मानवीय मूल्यों पर गहन विचार प्रस्तुत किए हैं।",
    "मध्यकालीन भारत में स्थापत्य कला, संगीत और हस्तशिल्प ने सांस्कृतिक समन्वय के नए आयाम स्थापित किए।",
]

TECH_HI = [
    "कृत्रिम बुद्धिमत्ता और मशीन लर्निंग तकनीकें वर्तमान में उद्योगों, स्वास्थ्य सेवाओं और शिक्षा को नई दिशा दे रही हैं।",
    "क्लाउड कंप्यूटिंग प्रणालियां विशाल डेटा संग्रहण और तेज प्रोसेसिंग क्षमता न्यूनतम लागत पर उपलब्ध कराती हैं।",
    "साइबर सुरक्षा उपाय डिजिटल लेनदेन और व्यक्तिगत डेटा को अनधिकृत पहुंच और हमलों से सुरक्षित रखते हैं।",
    "स्मार्टफोन और हाई-स्पीड इंटरनेट के प्रसार ने ग्रामीण और शहरी क्षेत्रों में डिजिटल सेवाओं को सर्वसुलभ बनाया है।",
]

# Hinglish sentence pools (Romanized Hindi-English mix)
TECH_HINGLISH = [
    "Aaj kal machine learning aur deep learning algorithms tech industry me bohot widespread use ho rahe hain.",
    "Agar aapko large scale software systems build karne hain to data structures aur algorithms achhe se samajhne honge.",
    "Cloud infrastructure par microservices deploy karne ke liye Docker containers aur Kubernetes best practices maane jaate hain.",
    "System design me database sharding aur caching implement karke high traffic efficiently handle kiya jaa sakta hai.",
    "Naye software developers ke liye code review process aur clean code principles seekhna career growth me help karta hai.",
    "Asynchronous programming aur message queues ka use karke backend services ka response time kafi improve hota hai.",
]

DAILY_HINGLISH = [
    "Daily routine me health aur work-life balance maintain karna productivity ke liye bohot zaroori hai.",
    "Online shopping aur digital payments ne daily essentials aur groceries order karna kafi convenient bana diya hai.",
    "Weekend par family aur friends ke sath outdoor travel explore karna sabhi ko refreshing experience deta hai.",
    "Education sector me digital tools aur online courses ki help se students naye skills aasani se seekh sakte hain.",
    "Social media platforms par informative content aur educational discussions se awareness bohot tezi se badhti hai.",
    "Customer requirements samajhkar unke queries ka timely resolution dena business reputation ke liye vital hota hai.",
]

# Noise injection components
PROMO_PHRASES = [
    "Click here to buy now and get an exclusive 50% discount on all premium courses!",
    "Limited time special promotional offer! Subscribe to our newsletter today for voucher rewards.",
    "Follow our official channel @TechDeals for daily cashback coupons and promo discount codes.",
    "Use promo coupon code SAVE50 at checkout to claim your instant cashback reward.",
    "Call us at +91 9876543210 for free consultation, trial demo, and technical customer support.",
    "Abhi visit karein hamari website aur paayein special discounts aur exclusive offers!",
]

URL_SPAM = [
    "Visit http://spam-tracker.example.com/click?aff=9942 for instant promotional access.",
    "Check out https://discount-zone.xyz/offer?ref=promo for seasonal clearance prices.",
    "Join our telegram group t.me/free_crypto_signals today for immediate trading updates!",
]

BOILERPLATE_FOOTERS = [
    "All rights reserved. Copyright (c) 2026 Acme Corp. Terms of service and privacy conditions apply.",
    "This website uses cookies to enhance user experience. By continuing to browse you accept our privacy policy.",
    "Home | About Us | Privacy Policy | Terms of Service | Contact Us | Careers | Site Map",
]

PII_STRINGS = [
    "My registered contact email address is developer.test@example.org.",
    "Official Indian PAN taxation identifier on file: ABCDE1234F.",
    "Server network configuration: 192.168.1.105 with secret credentials AKIAIOSFODNN7EXAMPLE.",
]


def make_diverse_paragraph(lang: str, rng: random.Random) -> str:
    """Compose a realistic paragraph of 3-5 sentences in chosen language."""
    if lang == "en":
        pool = rng.choice([SCIENCE_EN, TECH_EN, HISTORY_EN, ECONOMICS_EN, CULTURE_EN])
        k_val = min(len(pool), rng.randint(3, 5))
        sents = rng.sample(pool, k=k_val)
        return " ".join(sents)
    elif lang == "hi":
        pool = rng.choice([SCIENCE_HI, HISTORY_HI, TECH_HI])
        k_val = min(len(pool), rng.randint(2, 4))
        sents = rng.sample(pool, k=k_val)
        return " ".join(sents)
    else:
        pool = rng.choice([TECH_HINGLISH, DAILY_HINGLISH])
        k_val = min(len(pool), rng.randint(3, 5))
        sents = rng.sample(pool, k=k_val)
        return " ".join(sents)


def make_realistic_doc_text(lang: str, rng: random.Random, target_bytes: int = 3000) -> str:
    """Generate realistic diverse document text averaging roughly target_bytes (~3 KB)."""
    paragraphs = []
    current_bytes = 0

    while current_bytes < target_bytes:
        para = make_diverse_paragraph(lang, rng)
        paragraphs.append(para)
        current_bytes += len(para.encode("utf-8")) + 2
        if len(paragraphs) >= 5:
            break

    return "\n\n".join(paragraphs)


def create_clean_doc(doc_id: str, rng: random.Random) -> dict[str, Any]:
    """Generate a clean document exceeding min_doc_chars with zero noise."""
    lang = rng.choices(["en", "hi", "hinglish"], weights=[0.50, 0.25, 0.25])[0]
    text = make_realistic_doc_text(lang, rng, target_bytes=3000)

    return {
        "id": doc_id,
        "text": text,
        "domain": f"domain_{rng.randint(1, 20)}",
        "lang": lang,
        "meta": {"verified": True, "source": "clean_seed"},
    }


def create_noisy_doc(doc_id: str, rng: random.Random) -> dict[str, Any]:
    """Generate a document injected with promo phrases, spam URLs, or boilerplate."""
    lang = rng.choices(["en", "hi", "hinglish"], weights=[0.60, 0.20, 0.20])[0]
    base = make_realistic_doc_text(lang, rng, target_bytes=2500)
    noise_type = rng.choice(["promo", "url", "boilerplate", "pii", "short", "repetition"])

    if noise_type == "promo":
        promo = rng.choice(PROMO_PHRASES)
        text = f"{base}\n\n{promo}"
    elif noise_type == "url":
        url = rng.choice(URL_SPAM)
        text = f"{base}\n\nFor more details {url}"
    elif noise_type == "boilerplate":
        footer = rng.choice(BOILERPLATE_FOOTERS)
        text = f"{base}\n\n{footer}"
    elif noise_type == "pii":
        pii = rng.choice(PII_STRINGS)
        text = f"{base}\n\nPersonal records: {pii}"
    elif noise_type == "short":
        text = "Too short snippet."
    else:  # repetition
        rep_line = "This is a repetitive line for heuristic testing."
        text = "\n".join([rep_line] * 12)

    return {
        "id": doc_id,
        "text": text,
        "domain": "test_noise",
        "noise_type": noise_type,
    }


def create_nested_doc(doc_id: str, rng: random.Random) -> dict[str, Any]:
    """Generate nested schema documents (e.g. messages format or nested dict)."""
    lang = rng.choices(["en", "hi", "hinglish"], weights=[0.50, 0.25, 0.25])[0]
    clean_body = make_realistic_doc_text(lang, rng, target_bytes=3000)
    format_type = rng.choice(["messages", "path", "template"])

    if format_type == "messages":
        return {
            "id": doc_id,
            "conversations": [
                {"role": "user", "content": "Please explain the fundamental principles in detail."},
                {"role": "assistant", "content": clean_body},
            ],
            "domain": "chat",
        }
    elif format_type == "path":
        return {
            "id": doc_id,
            "data": {
                "article": {
                    "body": clean_body,
                    "tags": ["science", "research"],
                }
            },
            "domain": "nested_path",
        }
    else:  # template
        return {
            "id": doc_id,
            "title": f"Scientific Overview Volume {rng.randint(1, 500)}",
            "body": clean_body,
            "domain": "template",
        }


def stream_200k_corpus(
    out_path: Path,
    total_docs: int = 200000,
    dup_rate: float = 0.05,
    seed: int = 42,
) -> None:
    """Stream out realistic, diverse 200k documents with ~5% duplicate rate and ~3KB avg."""
    rng = random.Random(seed)
    recent_history: list[dict[str, Any]] = []
    max_history = 2000

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(total_docs):
            doc_id = f"doc_{i:06d}"

            # Controlled duplicate rate (~5%)
            if recent_history and rng.random() < dup_rate:
                donor = rng.choice(recent_history)
                # 50% exact duplicate, 50% near duplicate
                if rng.random() < 0.50:
                    dup_doc = {
                        "id": doc_id,
                        "text": donor["text"],
                        "domain": donor.get("domain", "default"),
                    }
                else:
                    dup_doc = {
                        "id": doc_id,
                        "text": donor["text"] + " Furthermore, supplementary evidence aligns with this summary.",
                        "domain": donor.get("domain", "default"),
                    }
                f.write(json.dumps(dup_doc, ensure_ascii=False) + "\n")
                continue

            # Schema distribution: 70% flat, 15% nested path, 15% conversations
            schema_roll = rng.random()
            if schema_roll < 0.70:
                doc = create_clean_doc(doc_id, rng)
            else:
                doc = create_nested_doc(doc_id, rng)

            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

            # Maintain sliding history window for duplicate injection
            if "text" in doc:
                recent_history.append(doc)
            elif "data" in doc and "article" in doc["data"]:
                recent_history.append({"text": doc["data"]["article"]["body"], "domain": "nested"})
            elif "conversations" in doc:
                recent_history.append({"text": doc["conversations"][-1]["content"], "domain": "chat"})

            if len(recent_history) > max_history:
                recent_history.pop(0)

            if (i + 1) % 25000 == 0 or (i + 1) == total_docs:
                print(f"[make_corpus] Progress: {i + 1}/{total_docs} documents written...")


def generate_corpus(
    out_dir: str | Path = "tests/data",
    n_docs: int = 200,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate deterministic test files into out_dir."""
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # 1. Clean JSONL corpus
    clean_path = target_dir / "sample_clean.jsonl"
    clean_docs = [create_clean_doc(f"clean_{i:04d}", rng) for i in range(n_docs)]
    with open(clean_path, "w", encoding="utf-8") as f:
        for doc in clean_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # 2. Noisy JSONL corpus
    noisy_path = target_dir / "sample_noisy.jsonl"
    noisy_docs = [create_noisy_doc(f"noisy_{i:04d}", rng) for i in range(n_docs)]
    with open(noisy_path, "w", encoding="utf-8") as f:
        for doc in noisy_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # 3. Multilingual JSONL corpus
    multi_path = target_dir / "sample_multilingual.jsonl"
    multi_docs = [create_clean_doc(f"multi_{i:04d}", rng) for i in range(n_docs)]
    with open(multi_path, "w", encoding="utf-8") as f:
        for doc in multi_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # 4. Nested schema JSONL corpus
    nested_path = target_dir / "sample_nested.jsonl"
    nested_docs = [create_nested_doc(f"nested_{i:04d}", rng) for i in range(n_docs)]
    with open(nested_path, "w", encoding="utf-8") as f:
        for doc in nested_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # 5. Deduplication test corpus
    dedup_path = target_dir / "sample_dedup.jsonl"
    dedup_docs = []
    for i in range(n_docs // 2):
        base_doc = create_clean_doc(f"dedup_orig_{i:04d}", rng)
        dedup_docs.append(base_doc)
        dedup_docs.append({
            "id": f"dedup_exact_{i:04d}",
            "text": base_doc["text"],
            "domain": base_doc["domain"],
        })
        dedup_docs.append({
            "id": f"dedup_near_{i:04d}",
            "text": base_doc["text"] + " Indeed, this conclusion holds true.",
            "domain": base_doc["domain"],
        })
    with open(dedup_path, "w", encoding="utf-8") as f:
        for doc in dedup_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # 6. JSON array file
    json_path = target_dir / "sample_records.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_docs[:50], f, ensure_ascii=False, indent=2)

    # 7. Parquet file
    parquet_path = target_dir / "sample.parquet"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table_data = {
            "id": [d["id"] for d in clean_docs],
            "text": [d["text"] for d in clean_docs],
            "domain": [d["domain"] for d in clean_docs],
        }
        table = pa.Table.from_pydict(table_data)
        pq.write_table(table, parquet_path)
    except Exception as exc:
        print(f"[make_corpus] Warning: could not write Parquet file: {exc}")

    results = {
        "clean": clean_path,
        "noisy": noisy_path,
        "multilingual": multi_path,
        "nested": nested_path,
        "dedup": dedup_path,
        "json": json_path,
        "parquet": parquet_path,
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic test corpus")
    parser.add_argument("--out-dir", type=str, default="tests/data", help="Output directory")
    parser.add_argument("--docs", "--n-docs", dest="docs", type=int, default=200, help="Total documents to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--file", type=str, default=None, help="Explicit single output JSONL file path")
    args = parser.parse_args()

    if args.file:
        out_file = Path(args.file)
        print(f"Generating single corpus file: {args.docs} docs -> {out_file}")
        stream_200k_corpus(out_file, total_docs=args.docs, seed=args.seed)
        size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"Done. Wrote {args.docs} documents ({size_mb:.2f} MB).")
        return 0

    if args.docs >= 5000:
        corpus_path = Path(args.out_dir) / f"corpus_{args.docs}.jsonl"
        print(f"Generating large corpus: {args.docs} docs -> {corpus_path}")
        stream_200k_corpus(corpus_path, total_docs=args.docs, seed=args.seed)
        size_mb = corpus_path.stat().st_size / (1024 * 1024)
        print(f"Done. Wrote {args.docs} documents ({size_mb:.2f} MB).")
        return 0

    print(f"Generating test corpus: n_docs={args.docs}, seed={args.seed}, dir={args.out_dir}")
    files = generate_corpus(out_dir=args.out_dir, n_docs=args.docs, seed=args.seed)
    for name, path in files.items():
        if path.exists():
            print(f"  [{name}] {path} ({path.stat().st_size} bytes)")
    print("Test corpus generation complete.")
    return 0


if __name__ == "__main__":
    main()
