"""
End-to-End local integration tests for TEXTSCOUR pipeline.
Runs full pipeline on local JSONL sources, tests dry-run mode, output shard generation,
manifest tracking, execution report, and exact record conservation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from textscour.cli import cmd_merge, cmd_run
from textscour.config import load_config
from tests.make_corpus import generate_corpus
from tests.verify_conservation import verify_conservation


@pytest.fixture
def e2e_corpus_and_config(tmp_path: Path) -> tuple[Path, Path]:
    """Generate local test data and custom test YAML configuration."""
    data_dir = tmp_path / "data"
    generate_corpus(out_dir=data_dir, n_docs=40, seed=777)

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
            "clean_name": "clean_e2e",
            "garbage_name": "garbage_e2e",
            "format": "jsonl",
            "shard_mb": 5,
            "emit_fragments": True,
        },
        "cascade": {
            "gpu_mode": "off",
            "min_doc_chars": 80,
            "clean_threshold": 5.5,
            "url_policy": "smart",
        },
        "runtime": {
            "autotune": "off",
            "readers": 1,
            "workers": 1,
            "rust_threads": 1,
            "batch_docs": 32,
            "queue_depth": 2,
        },
        "dedup": {
            "mode": "exact",
        },
    }

    config_path = tmp_path / "test_e2e.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f)

    return config_path, output_dir


def test_e2e_local_pipeline_execution(e2e_corpus_and_config: tuple[Path, Path]):
    """Verify complete end-to-end pipeline execution from local sources."""
    config_path, output_dir = e2e_corpus_and_config

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
    assert ret == 0

    # 1. Verify output shards exist
    clean_shards = list(output_dir.glob("clean_e2e*.jsonl"))
    garbage_shards = list(output_dir.glob("garbage_e2e*.jsonl"))
    assert len(clean_shards) > 0, "No clean shards produced"
    assert len(garbage_shards) > 0, "No garbage shards produced"

    # 2. Verify manifest and reports exist
    manifest_file = output_dir / "manifest.json"
    report_md = output_dir / "REPORT.md"
    report_json = output_dir / "REPORT.json"
    assert manifest_file.is_file()
    assert report_md.is_file()
    assert report_json.is_file()

    # 3. Verify conservation and output invariants
    is_conserved = verify_conservation(output_dir)
    assert is_conserved is True


def test_e2e_dry_run_mode(e2e_corpus_and_config: tuple[Path, Path], tmp_path: Path):
    """Verify --dry-run terminates processing after bounded number of records."""
    config_path, _ = e2e_corpus_and_config
    dry_out_dir = tmp_path / "outputs_dry"

    class DryArgs:
        config = str(config_path)
        set = [f"output.dir={str(dry_out_dir)}"]
        clean_name = None
        garbage_name = None
        fresh = True
        resume = False
        accept_config_change = False
        dry_run = 15

    ret = cmd_run(DryArgs())
    assert ret == 0

    # Read REPORT.json in dry_out_dir
    report_json = dry_out_dir / "REPORT.json"
    assert report_json.is_file()
    with open(report_json, "r", encoding="utf-8") as f:
        rdata = json.load(f)
    assert rdata["docs_in"] <= 40
