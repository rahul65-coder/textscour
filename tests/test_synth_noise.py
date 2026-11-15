"""
Unit tests for textscour.train.synth_noise: Synthetic template counts,
train/test trigram overlap <= 0.5, slot substitution, span alignment, benchmark generation.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from textscour.train.synth_noise import (
    BenchDoc,
    Template,
    fill_template,
    inject,
    load_templates,
    make_benchmark,
)


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
LEXICON_DIR = WORKSPACE_ROOT / "configs" / "lexicons"


def compute_char_trigrams(text: str) -> set[str]:
    """Extract character trigrams for overlap checking."""
    text_clean = "".join(c.lower() for c in text if c.isalnum() or c.isspace())
    if len(text_clean) < 3:
        return {text_clean}
    return {text_clean[i:i + 3] for i in range(len(text_clean) - 2)}


def test_template_loading_counts():
    """Verify template files exist and meet minimum count gates (>=150 train, >=50 test)."""
    train_file = LEXICON_DIR / "synth_templates_train.tsv"
    test_file = LEXICON_DIR / "synth_templates_test.tsv"

    assert train_file.is_file(), f"Missing {train_file}"
    assert test_file.is_file(), f"Missing {test_file}"

    train_templates = load_templates(train_file)
    test_templates = load_templates(test_file)

    assert len(train_templates) >= 150, f"Expected >= 150 train templates, found {len(train_templates)}"
    assert len(test_templates) >= 50, f"Expected >= 50 test templates, found {len(test_templates)}"


def test_train_test_template_trigram_overlap():
    """Verify held-out test templates have trigram Jaccard overlap <= 0.5 with train templates."""
    train_templates = load_templates(LEXICON_DIR / "synth_templates_train.tsv")
    test_templates = load_templates(LEXICON_DIR / "synth_templates_test.tsv")

    train_trigrams: set[str] = set()
    for tmpl in train_templates:
        train_trigrams.update(compute_char_trigrams(tmpl.template))

    test_trigrams: set[str] = set()
    for tmpl in test_templates:
        test_trigrams.update(compute_char_trigrams(tmpl.template))

    intersection = train_trigrams.intersection(test_trigrams)
    union = train_trigrams.union(test_trigrams)
    jaccard = len(intersection) / len(union) if union else 0.0

    assert jaccard <= 0.50, f"Trigram overlap between train and test templates ({jaccard:.3f}) exceeds 0.50 limit"


def test_fill_template_slot_replacement():
    """Verify fill_template replaces all placeholder slots without residual braces."""
    raw_template = "Buy {BRAND} on {URL} with promo {CODE} for {PCT} discount! Follow {HANDLE} or call {PHONE}."
    rng = random.Random(42)

    filled = fill_template(raw_template, rng)
    assert "{BRAND}" not in filled
    assert "{URL}" not in filled
    assert "{CODE}" not in filled
    assert "{PCT}" not in filled
    assert "{HANDLE}" not in filled
    assert "{PHONE}" not in filled
    assert len(filled) > len(raw_template)


def test_noise_injection_and_span_alignment():
    """Verify inject accurately tracks (start, end, kind) character spans in new text."""
    clean_text = (
        "Supervised learning algorithms build a mathematical model of sample data. "
        "The model is trained to minimize prediction error on known examples. "
        "Energy is a conserved quantity that cannot be created or destroyed."
    )

    templates = [
        Template(lang="en", kind="promo", template="Get 50% discount at {BRAND} today!"),
        Template(lang="en", kind="cta", template="Click here to subscribe immediately."),
    ]

    rng = random.Random(123)
    noisy_text, spans = inject(text=clean_text, rng=rng, rate=0.6, templates=templates)

    assert len(spans) > 0
    assert len(noisy_text) > len(clean_text)

    # Verify that each span slice matches the expected kind
    for start, end, kind in spans:
        span_slice = noisy_text[start:end]
        assert len(span_slice) > 0
        assert span_slice in noisy_text
        assert kind in ("promo", "cta")


def test_make_benchmark_determinism():
    """Verify make_benchmark produces reproducible BenchDocs given the same random seed."""
    docs = [
        "First clean sample document for testing benchmark generation.",
        "Second clean sample document discussing mathematical principles.",
    ]
    test_templates = load_templates(LEXICON_DIR / "synth_templates_test.tsv")

    bench1 = make_benchmark(docs, test_templates, seed=999)
    bench2 = make_benchmark(docs, test_templates, seed=999)

    assert len(bench1) == len(docs)
    for b1, b2 in zip(bench1, bench2):
        assert b1.original == b2.original
        assert b1.injected == b2.injected
        assert b1.spans == b2.spans
