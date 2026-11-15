#!/usr/bin/env python3
"""Standalone measured benchmark evaluator for F11: Language ID (en/hi/Hinglish).

Contract:
- >= 0.95 accuracy on synthetic 3-language test set (>= 300 docs each)
- Hinglish never flagged wrong_lang when allowed_langs contains hinglish
- Saves measured numbers to evidence/f11_lang_id.json
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

# Vocabulary and phrase seeds for rich synthetic corpus generation
EN_SNIPPETS = [
    "Machine learning algorithms build mathematical models based on sample training data to make predictions.",
    "Photosynthesis is a biological process through which green plants convert light energy into chemical energy.",
    "Quantum computing leverages superposition and entanglement to perform complex computational simulations.",
    "The Indus Valley Civilization flourished around the Indus river basin in northwest South Asia.",
    "Computer networks enable digital communication across global telecommunication infrastructure and cloud clusters.",
    "The thermodynamics principles govern the transfer of heat and mechanical work between physical systems.",
    "Plate tectonics describes the large-scale motion of seven large plates composing Earth's lithosphere.",
    "Cellular respiration converts biochemical energy from nutrients into adenosine triphosphate inside living cells.",
    "Distributed database systems provide high availability, partition tolerance, and eventual consistency at scale.",
    "Operating system kernels manage hardware resources, process scheduling, virtual memory, and device drivers.",
]

HI_SNIPPETS = [
    "कंप्यूटर विज्ञान सूचना और अभिकलन के सैद्धांतिक आधारों और प्रणालियों में उनके अनुप्रयोग का अध्ययन है।",
    "सौर मंडल सूर्य और गुरुत्वाकर्षण से बंधे सभी आकाशीय पिंडों जैसे ग्रहों, उपग्रहों और क्षुद्रग्रहों से बना है।",
    "भारतीय उपमहाद्वीप का इतिहास अत्यंत प्राचीन और समृद्ध है जिसमें विज्ञान, कला और दर्शन का विकास हुआ।",
    "प्रकाश संश्लेषण द्वारा हरे पौधे सूर्य के प्रकाश का उपयोग करके जल और कार्बन डाइऑक्साइड से भोजन बनाते हैं।",
    "कृत्रिम बुद्धिमत्ता आधुनिक युग में विभिन्न उद्योगों, शिक्षा और स्वास्थ्य सेवाओं में क्रांतिकारी बदलाव ला रही है।",
    "पर्यावरण संरक्षण के लिए जल संसाधनों, वनों और वन्यजीवों का संवर्धन अत्यंत आवश्यक हो गया है।",
    "प्राचीन भारत में चरक और सुश्रुत ने आयुर्वेद और शल्य चिकित्सा में महत्वपूर्ण योगदान दिया था।",
    "गणित और खगोल विज्ञान में आर्यभट्ट और वराहमिहिर के शोध ने खगोलीय गणनाओं को नई दिशा प्रदान की।",
    "जलवायु परिवर्तन के प्रभावों को कम करने के लिए नवीकरणीय ऊर्जा स्रोतों को अपनाना आवश्यक है।",
    "भारतीय संविधान देश के नागरिकों को मौलिक अधिकार, समानता और लोकतांत्रिक मूल्य प्रदान करता है।",
]

HINGLISH_SNIPPETS = [
    "Aaj kal artificial intelligence aur machine learning bohot fast grow kar rahe hain sabhi tech companies me.",
    "Yeh online learning platform students ke liye bohot helpful aur useful ban gaya hai concept samajhne ke liye.",
    "Humne meeting me discussion kiya ki naya software release agle hafte deploy karna best option hoga.",
    "Aapko python programming aur data structures daily practice karni chahiye achhi coding skills ke liye.",
    "Aaj kal smartphones aur fast internet connectivity ki wajah se communication bohot aasan ho gaya hai.",
    "Hume apne personal data aur privacy ka dhyan rakhna chahiye jab hum internet use karte hain.",
    "Cloud computing aur remote working tools ne companies ka productivity level bohot improve kiya hai.",
    "Digital marketing aur social media campaigns brand growth ke liye bohot effective sabit ho rahe hain.",
    "Bhai project deadline pass aa rahi hai isliye team ko coordination improve karke deliver karna padega.",
    "Customer feedback sunna aur system performance monitor karna software maintenance ka zaruri hissa hai.",
]


def generate_doc(lang: str, doc_id: str, rng: random.Random) -> str:
    """Generate a realistic document of roughly 300-800 characters."""
    if lang == "en":
        pool = EN_SNIPPETS
    elif lang == "hi":
        pool = HI_SNIPPETS
    else:
        pool = HINGLISH_SNIPPETS

    selected = rng.sample(pool, k=rng.randint(3, 5))
    return " ".join(selected)


def run_benchmark(docs_per_lang: int = 350, seed: int = 12345) -> dict[str, Any]:
    """Execute F11 Language ID benchmark across en, hi, and hinglish."""
    rng = random.Random(seed)
    cfg = load_config("configs/local_gate.yaml")
    engine_dict = to_engine_dict(cfg)
    engine = scour_core.Engine(engine_dict)

    langs = ["en", "hi", "hinglish"]
    stats_by_lang: dict[str, dict[str, Any]] = {
        l: {"total": 0, "correct": 0, "wrong_lang_flagged": 0} for l in langs
    }

    confusion_matrix: dict[str, dict[str, int]] = {
        true_l: {pred_l: 0 for pred_l in langs + ["other"]} for true_l in langs
    }

    for lang in langs:
        for idx in range(docs_per_lang):
            doc_id = f"{lang}_{idx:04d}"
            text = generate_doc(lang, doc_id, rng)

            batch = engine.stage1([text])
            results = batch.finalize([], "rules_only")
            assert len(results) == 1, "Expected 1 result from batch"
            res = results[0]

            predicted_lang = res.lang
            stats_by_lang[lang]["total"] += 1
            if predicted_lang in confusion_matrix[lang]:
                confusion_matrix[lang][predicted_lang] += 1
            else:
                confusion_matrix[lang]["other"] += 1

            if predicted_lang == lang:
                stats_by_lang[lang]["correct"] += 1

            if "wrong_lang" in res.reasons:
                stats_by_lang[lang]["wrong_lang_flagged"] += 1

    accuracies = {
        l: round(stats_by_lang[l]["correct"] / max(1, stats_by_lang[l]["total"]), 4)
        for l in langs
    }
    total_docs = sum(stats_by_lang[l]["total"] for l in langs)
    total_correct = sum(stats_by_lang[l]["correct"] for l in langs)
    overall_accuracy = round(total_correct / total_docs, 4)

    # Invariant checks
    accuracy_target_met = all(acc >= 0.95 for acc in accuracies.values())
    hinglish_not_flagged = stats_by_lang["hinglish"]["wrong_lang_flagged"] == 0
    gate_passed = accuracy_target_met and hinglish_not_flagged and overall_accuracy >= 0.95

    report = {
        "benchmark": "F11_language_id",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "docs_per_lang": docs_per_lang,
        "total_docs": total_docs,
        "overall_accuracy": overall_accuracy,
        "accuracies_by_lang": accuracies,
        "stats_by_lang": stats_by_lang,
        "confusion_matrix": confusion_matrix,
        "hinglish_never_flagged_wrong_lang": hinglish_not_flagged,
        "accuracy_gte_95_pct": accuracy_target_met,
        "gate_passed": gate_passed,
    }

    return report


def test_language_id() -> None:
    """Verify Language ID classification accuracy and Hinglish retention."""
    results = run_benchmark(docs_per_lang=30, seed=42)
    assert results["overall_accuracy"] >= 0.90
    assert results["hinglish_never_flagged_wrong_lang"] is True


def main() -> None:
    evidence_dir = Path("evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    out_file = evidence_dir / "f11_lang_id.json"

    print("Running F11 Language ID Benchmark (350 docs per language)...")
    results = run_benchmark(docs_per_lang=350)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {out_file}")
    print(f"Overall Accuracy: {results['overall_accuracy'] * 100:.2f}%")
    for lang, acc in results["accuracies_by_lang"].items():
        print(f"  - {lang}: {acc * 100:.2f}%")
    print(f"Hinglish wrong_lang flags: {results['stats_by_lang']['hinglish']['wrong_lang_flagged']}")
    print(f"F11 Gate Passed: {results['gate_passed']}")

    if not results["gate_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
