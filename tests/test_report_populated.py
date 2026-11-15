"""
Integration test verifying REPORT.json fields are fully populated with
non-zero bytes_in, non-zero mb_per_sec, non-empty stage_timings,
non-empty dedup_stats, and non-empty autotune parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from textscour.cli import cmd_run
from tests.make_corpus import generate_corpus


def test_report_populated_fields(tmp_path: Path):
    """Verify REPORT.json contains all required telemetry, timing, and dedup fields."""
    data_dir = tmp_path / "data"
    generate_corpus(out_dir=data_dir, n_docs=50, seed=123)

    output_dir = tmp_path / "outputs"

    cfg_dict = {
        "sources": [
            {
                "type": "jsonl",
                "paths": [str(data_dir / "sample_clean.jsonl")],
                "text": {"mode": "field", "field": "text"},
                "domain": "wiki",
            },
            {
                "type": "jsonl",
                "paths": [str(data_dir / "sample_noisy.jsonl")],
                "text": {"mode": "field", "field": "text"},
                "domain": "noise",
            },
        ],
        "output": {
            "dir": str(output_dir),
            "clean_name": "clean_rep",
            "garbage_name": "garbage_rep",
            "format": "jsonl",
            "shard_mb": 5,
        },
        "cascade": {
            "gpu_mode": "off",
            "min_doc_chars": 50,
            "clean_threshold": 5.0,
        },
        "runtime": {
            "readers": 1,
            "workers": 1,
            "rust_threads": 1,
            "batch_docs": 16,
            "queue_depth": 2,
        },
        "dedup": {
            "mode": "exact",
        },
    }

    config_path = tmp_path / "test_report.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f)

    class Args:
        config = str(config_path)
        set = None
        clean_name = None
        garbage_name = None
        fresh = True
        resume = False
        accept_config_change = False
        dry_run = None

    ret = cmd_run(Args())
    assert ret == 0, "Pipeline run failed"

    report_path = output_dir / "REPORT.json"
    assert report_path.is_file(), f"Missing {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Non-zero bytes_in
    assert "bytes_in" in data, "bytes_in missing from REPORT.json"
    assert isinstance(data["bytes_in"], (int, float)), "bytes_in must be numeric"
    assert data["bytes_in"] > 0, f"bytes_in must be non-zero, got {data['bytes_in']}"

    # 2. Non-zero mb_per_sec
    assert "mb_per_sec" in data, "mb_per_sec missing from REPORT.json"
    assert isinstance(data["mb_per_sec"], (int, float)), "mb_per_sec must be numeric"
    assert data["mb_per_sec"] > 0, f"mb_per_sec must be non-zero, got {data['mb_per_sec']}"

    # 3. Non-empty stage_timings
    assert "stage_timings" in data, "stage_timings missing from REPORT.json"
    assert isinstance(data["stage_timings"], dict), "stage_timings must be a dict"
    assert len(data["stage_timings"]) > 0, "stage_timings must be non-empty"
    for stage in ["reader", "cascade", "writer"]:
        assert stage in data["stage_timings"], f"Stage '{stage}' missing from stage_timings"
        assert data["stage_timings"][stage] > 0, f"Stage timing for '{stage}' must be > 0"

    # 4. Non-empty dedup_stats
    assert "dedup_stats" in data, "dedup_stats missing from REPORT.json"
    assert isinstance(data["dedup_stats"], dict), "dedup_stats must be a dict"
    assert len(data["dedup_stats"]) > 0, "dedup_stats must be non-empty"
    for field_name in ["exact_duplicates", "near_duplicates", "unique_seen", "index_entries"]:
        assert field_name in data["dedup_stats"], f"'{field_name}' missing from dedup_stats"

    # 5. Non-empty autotune
    assert "autotune" in data, "autotune missing from REPORT.json"
    assert isinstance(data["autotune"], dict), "autotune must be a dict"
    assert len(data["autotune"]) > 0, "autotune must be non-empty"
    for param in [
        "workers",
        "rust_threads",
        "readers",
        "batch_docs",
        "queue_depth",
        "gpu_batch_tokens",
        "chunk_target_chars",
    ]:
        assert param in data["autotune"], f"'{param}' missing from autotune parameters"
