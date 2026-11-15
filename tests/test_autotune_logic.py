"""
Unit tests for textscour.autotune: Coordinate search constraints,
hardware fingerprint caching, error rejection, and build target selection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from textscour.autotune import (
    TrialResult,
    choose_build,
    coordinate_search,
    load_cache,
    run_trial,
    save_cache,
)
from textscour.config import Config, SourceCfg
from textscour.env_probe import EnvInfo


@pytest.fixture
def dummy_env() -> EnvInfo:
    return EnvInfo(
        cpu_cores_effective=8,
        cpu_cores_host=8,
        ram_total_gb=16.0,
        ram_avail_gb=12.0,
        disk_free_gb=50.0,
        workdir="/tmp",
        gpus=[],
        cuda=False,
        torch_version=None,
        python="3.13.0",
        rustc="1.99.0",
        cpu_flags=["avx2", "fma", "sse4_2"],
        is_kaggle=False,
        hf_reachable=True,
        fingerprint="dummy_fp_8core_avx2",
    )


def test_autotune_cache_lifecycle(tmp_path: Path):
    """Verify save_cache persists parameters and load_cache restores matching fingerprint."""
    cache_file = tmp_path / "autotune_cache.json"
    fingerprint = "fp_test_123"

    params = {"workers": 4, "rust_threads": 2, "batch_docs": 1024}
    results = {"docs_per_s": 5000.0, "errors": 0}

    assert load_cache(cache_file, fingerprint) is None
    save_cache(cache_file, fingerprint, params, results)

    loaded = load_cache(cache_file, fingerprint)
    assert loaded is not None
    assert loaded["workers"] == 4
    assert loaded["batch_docs"] == 1024

    # Different fingerprint returns None
    assert load_cache(cache_file, "fp_different") is None


def test_autotune_coordinate_search_constraints(dummy_env: EnvInfo):
    """Verify coordinate_search respects concurrency constraint: workers * rust_threads <= cores + 1."""
    cfg = Config(sources=[SourceCfg(type="jsonl", paths=["test.jsonl"])])
    sample_texts = ["A sample clean document for throughput benchmarking." * 10] * 50

    best_params = coordinate_search(
        env=dummy_env,
        cfg=cfg,
        budget_trials=5,
        sample_texts=sample_texts,
    )

    cores = dummy_env.cpu_cores_effective
    concurrency = best_params["workers"] * best_params["rust_threads"]
    assert concurrency <= (cores + 1), f"Concurrency {concurrency} exceeded {cores + 1}"
    assert best_params["batch_docs"] in (256, 512, 1024, 2048)


def test_choose_build_target():
    """Verify choose_build selects native for avx2/fma and generic otherwise."""
    env_native = EnvInfo(
        cpu_cores_effective=4,
        cpu_cores_host=4,
        ram_total_gb=8.0,
        ram_avail_gb=6.0,
        disk_free_gb=20.0,
        workdir="/tmp",
        gpus=[],
        cuda=False,
        torch_version=None,
        python="3.13.0",
        rustc="1.99.0",
        cpu_flags=["avx2", "fma"],
        is_kaggle=False,
        hf_reachable=True,
        fingerprint="fp_native",
    )
    assert choose_build(env_native) == "native"

    env_generic = EnvInfo(
        cpu_cores_effective=2,
        cpu_cores_host=2,
        ram_total_gb=4.0,
        ram_avail_gb=2.0,
        disk_free_gb=10.0,
        workdir="/tmp",
        gpus=[],
        cuda=False,
        torch_version=None,
        python="3.13.0",
        rustc="1.99.0",
        cpu_flags=["sse2"],
        is_kaggle=False,
        hf_reachable=True,
        fingerprint="fp_generic",
    )
    assert choose_build(env_generic) == "generic"
