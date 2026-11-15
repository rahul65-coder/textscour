"""
Unit tests for textscour.supervisor.ControlLoop: AIMD resource adaptation,
memory throttle, disk safety pause, and GPU backlog adjustment.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from textscour.config import Config, RuntimeCfg, SourceCfg
from textscour.env_probe import EnvInfo
from textscour.supervisor import ControlLoop


@pytest.fixture(autouse=True)
def mock_probe_in_supervisor(monkeypatch: pytest.MonkeyPatch, dummy_env: EnvInfo):
    """Mock probe to avoid slow network timeout calls during control loop testing."""
    monkeypatch.setattr("textscour.supervisor.probe", lambda d: dummy_env)


@pytest.fixture
def dummy_env() -> EnvInfo:
    return EnvInfo(
        cpu_cores_effective=8,
        cpu_cores_host=8,
        ram_total_gb=10.0,
        ram_avail_gb=8.0,
        disk_free_gb=50.0,
        workdir="/tmp",
        gpus=[],
        cuda=False,
        torch_version=None,
        python="3.13.0",
        rustc="1.99.0",
        cpu_flags=[],
        is_kaggle=False,
        hf_reachable=True,
        fingerprint="test_fp",
    )


class DummySupervisor:
    """Lightweight dummy supervisor for isolated ControlLoop AIMD testing."""

    def __init__(self, tmp_path: Path):
        self.cfg = Config(
            sources=[SourceCfg(type="jsonl", paths=["test.jsonl"])],
            runtime=RuntimeCfg(
                ram_budget_frac=0.80,
                disk_budget_gb=100.0,
                gpu_inflight=2,
            ),
        )
        self.env = EnvInfo(
            cpu_cores_effective=8,
            cpu_cores_host=8,
            ram_total_gb=10.0,
            ram_avail_gb=8.0,
            disk_free_gb=50.0,
            workdir=str(tmp_path),
            gpus=[],
            cuda=False,
            torch_version=None,
            python="3.13.0",
            rustc="1.99.0",
            cpu_flags=[],
            is_kaggle=False,
            hf_reachable=True,
            fingerprint="test_fp",
        )
        self.counters: dict[str, int] = {}
        self.in_q = MagicMock()
        self.in_q.qsize.return_value = 0
        self.out_q = MagicMock()
        self.out_q.qsize.return_value = 0
        self.gpu_q = MagicMock()
        self.gpu_q.qsize.return_value = 0

        self.batch_docs_hint = MagicMock()
        self.batch_docs_hint.value = 1024
        self.gpu_batch_tokens_hint = MagicMock()
        self.gpu_batch_tokens_hint.value = 16384
        self.paused_flag = MagicMock()
        self.paused_flag.value = 0

        self.dashboard = MagicMock()
        self.dashboard._get_rss_gb.return_value = 4.0


def test_control_loop_rss_overcommit_halves_batch(tmp_path: Path):
    """Verify high memory (RSS > 85% budget) triggers multiplicative decrease on batch_docs_hint."""
    sup = DummySupervisor(tmp_path)
    # RAM budget is 10.0 * 0.80 = 8.0 GB. 85% is 6.8 GB.
    # Set RSS = 7.5 GB > 6.8 GB
    sup.dashboard._get_rss_gb.return_value = 7.5

    loop = ControlLoop(supervisor=sup, interval_sec=1.0)
    loop.step()

    # Old hint 1024 halved -> 512
    assert sup.batch_docs_hint.value == 512
    sup.dashboard.log_control_event.assert_called_with(
        "RSS_OVERCOMMIT_HALVE_BATCH",
        "RSS=7.5GB, new_batch=512",
    )


def test_control_loop_low_rss_streak_increases_batch(tmp_path: Path):
    """Verify sustained low memory (< 60% budget for 5 intervals) triggers additive increase."""
    sup = DummySupervisor(tmp_path)
    # RAM budget = 8.0 GB. 60% is 4.8 GB. Set RSS = 2.0 GB.
    sup.dashboard._get_rss_gb.return_value = 2.0
    sup.batch_docs_hint.value = 512

    loop = ControlLoop(supervisor=sup, interval_sec=1.0)
    for _ in range(4):
        loop.step()
        assert sup.batch_docs_hint.value == 512  # Not yet 5 streaks

    # 5th step triggers +25%
    loop.step()
    assert sup.batch_docs_hint.value == int(512 * 1.25)


def test_control_loop_gpu_backlog_increases_tokens(tmp_path: Path):
    """Verify GPU queue backlog > 2*gpu_inflight increases token limit."""
    sup = DummySupervisor(tmp_path)
    # gpu_inflight = 2, so backlog threshold is > 4
    sup.gpu_q.qsize.return_value = 10
    sup.gpu_batch_tokens_hint.value = 16384

    loop = ControlLoop(supervisor=sup, interval_sec=1.0)
    loop.step()

    assert sup.gpu_batch_tokens_hint.value == int(16384 * 1.5)
