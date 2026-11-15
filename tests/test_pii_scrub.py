#!/usr/bin/env python3
"""Standalone measured benchmark evaluator for F12: PII Scrub.

Contract:
- PII recall >= 0.95 on synthetic PII set (email, phone IN/intl, Aadhaar w/ Verhoeff, PAN, IPv4, key patterns, Luhn card)
- False positives on clean corpus <= 0.2% of docs
- Saves measured numbers to evidence/f12_pii.json
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import scour_core
from textscour.config import load_config, to_engine_dict

# Verhoeff check-digit implementation for synthetic Aadhaar generation
D_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

P_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

INV_TABLE = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def compute_verhoeff(num_str: str) -> int:
    """Compute Verhoeff check digit for an 11-digit string."""
    c = 0
    digits = [int(ch) for ch in num_str if ch.isdigit()]
    for i, d in enumerate(reversed(digits)):
        p_val = P_TABLE[(i + 1) % 8][d]
        c = D_TABLE[c][p_val]
    return INV_TABLE[c]


def generate_valid_aadhaar(rng: random.Random) -> str:
    """Generate a valid 12-digit Aadhaar number with correct Verhoeff check digit."""
    base = "".join(str(rng.randint(1 if i == 0 else 0, 9)) for i in range(11))
    check = compute_verhoeff(base)
    full = f"{base}{check}"
    return f"{full[:4]} {full[4:8]} {full[8:]}"


def generate_valid_luhn_card(rng: random.Random) -> str:
    """Generate a valid 16-digit credit card number using Luhn algorithm."""
    prefix = rng.choice(["4", "51", "52", "53", "54", "55"])
    rem_len = 15 - len(prefix)
    digits = [int(ch) for ch in prefix] + [rng.randint(0, 9) for _ in range(rem_len)]

    # Compute check digit
    sum_val = 0
    alt = True
    for d in reversed(digits):
        if alt:
            doubled = d * 2
            if doubled > 9:
                doubled -= 9
            sum_val += doubled
        else:
            sum_val += d
        alt = not alt

    check_digit = (10 - (sum_val % 10)) % 10
    digits.append(check_digit)
    return "".join(str(d) for d in digits)


CLEAN_SENTENCES = [
    "The software architecture follows a modular decoupled microservice design pattern.",
    "Data analysis revealed statistically significant improvements in processing latency.",
    "Researchers developed a novel computational framework for atmospheric pressure modeling.",
    "The historical monument was constructed using carved sandstone blocks in the fifteenth century.",
    "Mathematical theorems establish rigorous bounds on communication complexity in distributed clusters.",
    "Renewable energy investments expanded significantly across wind, solar, and hydroelectric sectors.",
    "Biological enzymes accelerate metabolic chemical reactions without being consumed in the process.",
    "Operating system schedulers prioritize real-time interactive tasks over low-priority background jobs.",
    "Geological formations in the Himalayan ranges provide insights into continental plate collision dynamics.",
    "Modern compilers execute static code optimization passes including loop unrolling and dead code elimination.",
]


def generate_pii_item(kind: str, rng: random.Random) -> tuple[str, str]:
    """Generate synthetic PII string and expected redaction token."""
    if kind == "email":
        user = f"user_{rng.randint(100, 9999)}"
        domain = rng.choice(["example.org", "company.test", "university.edu", "service.net"])
        return f"{user}@{domain}", "<EMAIL>"
    elif kind == "phone_in":
        first_digit = rng.choice([7, 8, 9])
        rem = "".join(str(rng.randint(0, 9)) for _ in range(9))
        prefix = rng.choice(["+91 ", "+91-", ""])
        return f"{prefix}{first_digit}{rem}", "<PHONE>"
    elif kind == "phone_intl":
        country_code = rng.randint(1, 44)
        area = rng.randint(200, 999)
        num = rng.randint(100000, 999999)
        return f"+{country_code}-{area}-{num}", "<PHONE>"
    elif kind == "aadhaar":
        return generate_valid_aadhaar(rng), "<AADHAAR>"
    elif kind == "pan":
        letters1 = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
        nums = "".join(str(rng.randint(0, 9)) for _ in range(4))
        letter2 = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return f"{letters1}{nums}{letter2}", "<PAN>"
    elif kind == "ipv4":
        ip = ".".join(str(rng.randint(10, 250)) for _ in range(4))
        return ip, "<IP>"
    elif kind == "key":
        prefix = rng.choice(["AKIA", "ghp_"])
        if prefix == "AKIA":
            chars = "".join(rng.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(16))
            return f"AKIA{chars}", "<KEY>"
        else:
            chars = "".join(rng.choice("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(36))
            return f"ghp_{chars}", "<KEY>"
    elif kind == "card":
        card = generate_valid_luhn_card(rng)
        return card, "<CARD>"
    else:
        raise ValueError(f"Unknown kind {kind}")


def run_benchmark(n_pii_docs: int = 1000, n_clean_docs: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Execute F12 PII Scrub benchmark measuring recall and false positive rate."""
    rng = random.Random(seed)
    cfg = load_config("configs/local_gate.yaml")
    engine = scour_core.Engine(to_engine_dict(cfg))

    pii_kinds = ["email", "phone_in", "phone_intl", "aadhaar", "pan", "ipv4", "key", "card"]
    pii_stats: dict[str, dict[str, int]] = {k: {"total": 0, "recalled": 0} for k in pii_kinds}

    # 1. PII Recall Evaluation
    for i in range(n_pii_docs):
        kind = pii_kinds[i % len(pii_kinds)]
        pii_str, expected_token = generate_pii_item(kind, rng)

        context_before = rng.choice(CLEAN_SENTENCES)
        context_after = rng.choice(CLEAN_SENTENCES)
        text = f"{context_before} Please note the personal record: {pii_str}. {context_after}"

        batch = engine.stage1([text])
        results = batch.finalize([], "rules_only")
        assert len(results) == 1
        res = results[0]

        pii_stats[kind]["total"] += 1
        if expected_token in res.text and pii_str not in res.text:
            pii_stats[kind]["recalled"] += 1

    per_kind_recall = {
        k: round(pii_stats[k]["recalled"] / max(1, pii_stats[k]["total"]), 4)
        for k in pii_kinds
    }
    total_pii = sum(pii_stats[k]["total"] for k in pii_kinds)
    total_recalled = sum(pii_stats[k]["recalled"] for k in pii_kinds)
    overall_recall = round(total_recalled / total_pii, 4)

    # 2. False Positive Evaluation on Pure Clean Corpus
    all_tokens = ["<EMAIL>", "<PHONE>", "<AADHAAR>", "<PAN>", "<IP>", "<KEY>", "<CARD>"]
    fp_docs = 0

    for i in range(n_clean_docs):
        paras = rng.sample(CLEAN_SENTENCES, k=rng.randint(3, 5))
        text = " ".join(paras)

        batch = engine.stage1([text])
        results = batch.finalize([], "rules_only")
        assert len(results) == 1
        res = results[0]

        if any(tok in res.text for tok in all_tokens):
            fp_docs += 1

    false_positive_rate = round(fp_docs / n_clean_docs, 6)

    # Invariants
    recall_target_met = overall_recall >= 0.95 and all(r >= 0.90 for r in per_kind_recall.values())
    fp_target_met = false_positive_rate <= 0.002  # <= 0.2%
    gate_passed = recall_target_met and fp_target_met

    report = {
        "benchmark": "F12_pii_scrub",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "total_pii_docs": total_pii,
        "total_clean_docs": n_clean_docs,
        "overall_recall": overall_recall,
        "per_kind_recall": per_kind_recall,
        "pii_stats": pii_stats,
        "clean_corpus_false_positives": fp_docs,
        "false_positive_rate": false_positive_rate,
        "recall_gte_95_pct": recall_target_met,
        "fp_lte_0_2_pct": fp_target_met,
        "gate_passed": gate_passed,
    }

    return report


def test_pii_scrub() -> None:
    """Verify PII detection and redaction recall and false positive rate."""
    results = run_benchmark(n_pii_docs=50, n_clean_docs=50, seed=42)
    assert results["overall_recall"] >= 0.90
    assert results["false_positive_rate"] <= 0.05


def main() -> None:
    evidence_dir = Path("evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    out_file = evidence_dir / "f12_pii.json"

    print("Running F12 PII Scrub Benchmark (1000 PII docs, 1000 Clean docs)...")
    results = run_benchmark(n_pii_docs=1000, n_clean_docs=1000)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {out_file}")
    print(f"Overall Recall: {results['overall_recall'] * 100:.2f}%")
    for kind, rec in results["per_kind_recall"].items():
        print(f"  - {kind}: {rec * 100:.2f}%")
    print(f"Clean FP Rate: {results['false_positive_rate'] * 100:.4f}% ({results['clean_corpus_false_positives']}/{results['total_clean_docs']})")
    print(f"F12 Gate Passed: {results['gate_passed']}")

    if not results["gate_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
