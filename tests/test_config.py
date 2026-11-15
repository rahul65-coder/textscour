"""
Unit tests for textscour.config: Dataclass schemas, validations, YAML parsing,
dotted overrides, error reporting, and Rust engine dictionary mapping.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from textscour.config import (
    CascadeCfg,
    Config,
    DedupCfg,
    GopherCfg,
    OutputCfg,
    RuntimeCfg,
    SourceCfg,
    TextSpec,
    UploadCfg,
    critical_config_hash,
    load_config,
    to_engine_dict,
)
from textscour.errors import ScourError


def test_default_config_creation():
    """Verify default configurations have expected types and defaults."""
    cfg = Config(
        sources=[SourceCfg(type="jsonl", paths=["data.jsonl"])],
    )
    assert cfg.output.clean_name == "clean"
    assert cfg.output.garbage_name == "garbage"
    assert cfg.cascade.gpu_mode == "auto"
    assert cfg.cascade.promo_lo == 0.30
    assert cfg.cascade.promo_hi == 0.70
    assert cfg.runtime.autotune == "auto"
    assert cfg.dedup.mode == "minhash"


def test_load_valid_config_from_dict():
    """Verify loading and validating a complete valid configuration dict."""
    valid_dict = {
        "sources": [
            {
                "type": "jsonl",
                "paths": ["tests/data/sample_clean.jsonl"],
                "text": {"mode": "field", "field": "text"},
            }
        ],
        "output": {
            "dir": "outputs/test",
            "clean_name": "clean_docs",
            "garbage_name": "garbage_docs",
            "format": "jsonl",
        },
        "cascade": {
            "promo_lo": 0.25,
            "promo_hi": 0.75,
            "min_doc_chars": 150,
            "clean_threshold": 5.5,
        },
        "runtime": {
            "workers": 2,
            "rust_threads": 2,
        },
        "dedup": {
            "mode": "exact",
        },
    }

    cfg = load_config(valid_dict)
    assert cfg.cascade.promo_lo == 0.25
    assert cfg.cascade.promo_hi == 0.75
    assert cfg.output.clean_name == "clean_docs"
    assert cfg.runtime.workers == 2
    assert cfg.dedup.mode == "exact"


def test_dotted_key_overrides():
    """Verify dotted key overrides mutate loaded configuration values."""
    base_dict = {
        "sources": [{"type": "jsonl", "paths": ["data.jsonl"]}],
        "cascade": {"promo_lo": 0.30, "promo_hi": 0.70},
    }
    overrides = [
        "cascade.promo_lo=0.35",
        "output.shard_mb=250",
        "dedup.mode=off",
    ]
    cfg = load_config(base_dict, overrides=overrides)
    assert cfg.cascade.promo_lo == 0.35
    assert cfg.output.shard_mb == 250
    assert cfg.dedup.mode == "off"


def test_validation_promo_threshold_order():
    """Ensure promo_lo >= promo_hi raises E-CFG-004."""
    invalid = {
        "sources": [{"type": "jsonl", "paths": ["data.jsonl"]}],
        "cascade": {"promo_lo": 0.80, "promo_hi": 0.60},
    }
    with pytest.raises(ScourError) as exc_info:
        load_config(invalid)
    assert exc_info.value.code == "E-CFG-004"


def test_validation_gpu_threshold_order():
    """Ensure gpu_strip_p > gpu_drop_p raises E-CFG-004."""
    invalid = {
        "sources": [{"type": "jsonl", "paths": ["data.jsonl"]}],
        "cascade": {"gpu_strip_p": 0.95, "gpu_drop_p": 0.70},
    }
    with pytest.raises(ScourError) as exc_info:
        load_config(invalid)
    assert exc_info.value.code == "E-CFG-004"


def test_validation_chunk_sizing():
    """Ensure chunk_min > chunk_target raises E-CFG-005."""
    invalid = {
        "sources": [{"type": "jsonl", "paths": ["data.jsonl"]}],
        "cascade": {
            "chunk_min_chars": 900,
            "chunk_target_chars": 800,
            "chunk_max_chars": 2000,
        },
    }
    with pytest.raises(ScourError) as exc_info:
        load_config(invalid)
    assert exc_info.value.code == "E-CFG-005"


def test_validation_identical_output_names():
    """Ensure clean_name == garbage_name raises E-CFG-007."""
    invalid = {
        "sources": [{"type": "jsonl", "paths": ["data.jsonl"]}],
        "output": {"clean_name": "dump", "garbage_name": "dump"},
    }
    with pytest.raises(ScourError) as exc_info:
        load_config(invalid)
    assert exc_info.value.code == "E-CFG-007"


def test_validation_upload_without_repo():
    """Ensure upload.enabled=true without repo raises E-CFG-008."""
    invalid = {
        "sources": [{"type": "jsonl", "paths": ["data.jsonl"]}],
        "upload": {"enabled": True, "repo": None},
    }
    with pytest.raises(ScourError) as exc_info:
        load_config(invalid)
    assert exc_info.value.code == "E-CFG-008"


def test_validation_unknown_key():
    """Ensure unknown configuration key raises E-CFG-002."""
    invalid = {
        "sources": [{"type": "jsonl", "paths": ["data.jsonl"]}],
        "non_existent_section": {"foo": "bar"},
    }
    with pytest.raises(ScourError) as exc_info:
        load_config(invalid)
    assert exc_info.value.code == "E-CFG-002"


def test_validation_invalid_enum():
    """Ensure invalid enum value raises E-CFG-003."""
    invalid = {
        "sources": [{"type": "jsonl", "paths": ["data.jsonl"]}],
        "cascade": {"url_policy": "unsupported_policy"},
    }
    with pytest.raises(ScourError) as exc_info:
        load_config(invalid)
    assert exc_info.value.code == "E-CFG-003"


def test_to_engine_dict_conversion():
    """Verify conversion of Config to Rust scour_core engine dictionary."""
    cfg = Config(
        sources=[SourceCfg(type="jsonl", paths=["data.jsonl"])],
        cascade=CascadeCfg(
            promo_lo=0.32,
            promo_hi=0.72,
            min_doc_chars=250,
            url_policy="smart",
        ),
    )
    engine_dict = to_engine_dict(cfg)
    assert engine_dict["promo_lo"] == 0.32
    assert engine_dict["promo_hi"] == 0.72
    assert engine_dict["min_doc_chars"] == 250
    assert engine_dict["url_policy"] == "smart"
    assert "gopher" in engine_dict
    assert isinstance(engine_dict["gopher"], dict)


def test_config_hash_stability_and_sensitivity():
    """Verify compute_config_hash produces identical hashes for same critical settings."""
    cfg1 = Config(
        sources=[SourceCfg(type="jsonl", paths=["data.jsonl"])],
        cascade=CascadeCfg(promo_lo=0.30, clean_threshold=6.0),
    )
    cfg2 = Config(
        sources=[SourceCfg(type="jsonl", paths=["other.jsonl"])],
        cascade=CascadeCfg(promo_lo=0.30, clean_threshold=6.0),
        runtime=RuntimeCfg(workers=8),  # Non-critical difference
    )
    cfg3 = Config(
        sources=[SourceCfg(type="jsonl", paths=["data.jsonl"])],
        cascade=CascadeCfg(promo_lo=0.35, clean_threshold=6.0),  # Critical difference
    )

    hash1 = critical_config_hash(cfg1)
    hash2 = critical_config_hash(cfg2)
    hash3 = critical_config_hash(cfg3)

    assert hash1 == hash2
    assert hash1 != hash3
