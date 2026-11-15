"""
Unit tests for textscour.metrics: Dashboard telemetry, metrics.jsonl snapshots,
and final report generation (REPORT.md / REPORT.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from textscour.metrics import Dashboard, write_report


def test_dashboard_snapshot_capture(tmp_path: Path):
    """Verify Dashboard captures metrics snapshots and writes to metrics.jsonl."""
    counters = {
        "docs_in": 1500,
        "docs_clean": 1200,
        "docs_garbage": 300,
        "quarantined": 0,
        "bytes_in": 1048576,
        "reasons": {"promo_phrase": 200, "url_spam": 100},
    }

    dash = Dashboard(output_dir=tmp_path, counters=counters, interval_sec=1.0)
    dash.start()

    snapshot = dash.capture_snapshot(event="TEST_SAMPLE")
    dash.stop()

    assert snapshot.docs_in == 1500
    assert snapshot.docs_clean == 1200
    assert snapshot.docs_garbage == 300
    assert snapshot.event == "TEST_SAMPLE"

    # Verify metrics.jsonl written
    jsonl_file = tmp_path / "metrics.jsonl"
    assert jsonl_file.is_file()

    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) >= 1
    assert lines[0]["docs_in"] == 1500


def test_write_report_markdown_and_json(tmp_path: Path):
    """Verify write_report creates formatted REPORT.md and machine-readable REPORT.json."""
    summary_data = {
        "docs_in": 5000,
        "docs_clean": 4100,
        "docs_garbage": 900,
        "quarantined": 5,
        "elapsed_sec": 25.0,
        "reasons": {
            "promo_phrase": 600,
            "url_spam": 200,
            "too_short": 100,
        },
        "autotune": {
            "workers": 4,
            "rust_threads": 2,
            "batch_docs": 1024,
        },
        "errors_by_code": {
            "E-SCHEMA-002": 5,
        },
        "per_source": {
            "test_source": {"in": 5000, "clean": 4100, "garbage": 900},
        },
    }

    md_path, json_path = write_report(tmp_path, summary_data)

    assert md_path.is_file()
    assert json_path.is_file()

    # Check REPORT.json
    with open(json_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["docs_clean"] == 4100
    assert loaded_json["autotune"]["workers"] == 4

    # Check REPORT.md content
    md_text = md_path.read_text(encoding="utf-8")
    assert "# TEXTSCOUR Pipeline Execution Report" in md_text
    assert "| Documents Clean | 4,100 | 82.00% |" in md_text
    assert "`promo_phrase`" in md_text
    assert "`E-SCHEMA-002`" in md_text
    assert "`test_source`" in md_text
