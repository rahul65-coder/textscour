"""
Unit tests for textscour.checkpoint: Two-phase commit, config hash validation, failover.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from textscour.checkpoint import Checkpoint
from textscour.errors import ScourError


def test_checkpoint_commit_and_load(tmp_path: Path):
    """Verify committing a checkpoint writes atomically and can be reloaded."""
    ckpt = Checkpoint(state_dir=tmp_path)
    source_key = "local_test_source"

    payload = {
        "source": source_key,
        "next_batch_id": 42,
        "records_in": 1000,
        "records_out": {"clean": 850, "garbage": 150, "quarantine": 0},
        "shard_idx": {"clean": 1, "garbage": 1},
        "config_hash": "a1b2c3d4e5f6",
    }

    assert not ckpt.has_checkpoint(source_key)
    saved_path = ckpt.commit(source_key, payload)
    assert saved_path.is_file()
    assert ckpt.has_checkpoint(source_key)

    loaded = ckpt.load(source_key, current_config_hash="a1b2c3d4e5f6")
    assert loaded is not None
    assert loaded["next_batch_id"] == 42
    assert loaded["records_in"] == 1000
    assert loaded["records_out"]["clean"] == 850


def test_checkpoint_rotation_to_prev(tmp_path: Path):
    """Verify that committing a second generation creates a .prev backup."""
    ckpt = Checkpoint(state_dir=tmp_path)
    source_key = "source_rot"

    payload1 = {"batch_id": 1, "config_hash": "hash1"}
    ckpt.commit(source_key, payload1)

    prev_file = tmp_path / "ckpt-source_rot.json.prev"
    assert not prev_file.is_file()

    payload2 = {"batch_id": 2, "config_hash": "hash1"}
    ckpt.commit(source_key, payload2)
    assert prev_file.is_file()

    with open(prev_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["batch_id"] == 1


def test_checkpoint_config_hash_mismatch(tmp_path: Path):
    """Verify that config hash mismatch raises E-CKPT-001 unless accepted."""
    ckpt = Checkpoint(state_dir=tmp_path)
    source_key = "mismatch_source"

    payload = {"source": source_key, "config_hash": "old_hash_123"}
    ckpt.commit(source_key, payload)

    with pytest.raises(ScourError) as exc_info:
        ckpt.load(source_key, current_config_hash="new_hash_456")
    assert exc_info.value.code == "E-CKPT-001"

    # With accept_config_change=True, load succeeds
    loaded = ckpt.load(
        source_key,
        current_config_hash="new_hash_456",
        accept_config_change=True,
    )
    assert loaded is not None
    assert loaded["config_hash"] == "old_hash_123"


def test_checkpoint_corrupt_fallback_to_prev(tmp_path: Path):
    """Verify failover to .prev when main checkpoint file is corrupt."""
    ckpt = Checkpoint(state_dir=tmp_path)
    source_key = "corrupt_test"

    payload1 = {"source": source_key, "generation": 1}
    ckpt.commit(source_key, payload1)

    payload2 = {"source": source_key, "generation": 2}
    ckpt.commit(source_key, payload2)

    # Corrupt the main checkpoint file
    main_file = tmp_path / "ckpt-corrupt_test.json"
    with open(main_file, "w", encoding="utf-8") as f:
        f.write("{ invalid json corrupt")

    loaded = ckpt.load(source_key)
    assert loaded is not None
    assert loaded["generation"] == 1
    assert loaded.get("_loaded_from_prev") is True


def test_checkpoint_force_restart_and_clear(tmp_path: Path):
    """Verify force_restart ignores existing checkpoint and clear removes files."""
    ckpt = Checkpoint(state_dir=tmp_path)
    source_key = "restart_source"

    ckpt.commit(source_key, {"source": source_key})
    assert ckpt.has_checkpoint(source_key)

    # force_restart returns None
    assert ckpt.load(source_key, force_restart=True) is None

    # clear removes files
    ckpt.clear(source_key)
    assert not ckpt.has_checkpoint(source_key)
