"""
Unit tests for textscour.train.bench_accuracy: Accuracy benchmark execution,
Section A.4 metrics computation, and accuracy.json / accuracy.md artifact generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from textscour.train.bench_accuracy import run_benchmark


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
LEXICON_DIR = WORKSPACE_ROOT / "configs" / "lexicons"


def test_run_benchmark_metrics_and_artifacts(tmp_path: Path):
    """Verify run_benchmark executes evaluation on clean sample docs and creates reports."""
    clean_sample_docs = [
        (
            "Supervised learning algorithms build a mathematical model of sample data. "
            "The model is trained on known inputs to minimize prediction error. "
            "In physics, energy is a conserved scalar quantity that cannot be destroyed."
        ),
        (
            "The Indus Valley Civilisation was a Bronze Age civilisation in South Asia. "
            "Together with ancient Egypt and Mesopotamia, it was one of three early civilisations. "
            "Ancient cities exhibited urban planning, brick houses, and water supply systems."
        ),
    ]

    test_templates_tsv = str(LEXICON_DIR / "synth_templates_test.tsv")
    train_templates_tsv = str(LEXICON_DIR / "synth_templates_train.tsv")

    report_data = run_benchmark(
        clean_docs=clean_sample_docs,
        test_templates_path=test_templates_tsv,
        train_templates_path=train_templates_tsv,
        out_dir=str(tmp_path),
        seed=101,
    )

    assert isinstance(report_data, dict)
    assert "heldout_promo_span_recall" in report_data
    assert "mean_clean_preservation" in report_data
    assert "false_drop_rate" in report_data
    assert "total_benchmark_docs" in report_data
    assert report_data["total_benchmark_docs"] == len(clean_sample_docs)

    # Check that accuracy.json was created and is valid
    json_path = tmp_path / "accuracy.json"
    assert json_path.is_file()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["total_benchmark_docs"] == len(clean_sample_docs)

    # Check that accuracy.md was created
    md_path = tmp_path / "accuracy.md"
    assert md_path.is_file()
    md_text = md_path.read_text(encoding="utf-8")
    assert "# TEXTSCOUR Benchmark Accuracy Evaluation Report" in md_text
    assert "Promo-Span Recall" in md_text
