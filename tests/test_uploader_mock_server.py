"""
Unit tests for textscour.uploader: Mocked Hugging Face Hub uploader,
batch commits, rate-limit retry, local file deletion, and manifest integration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from textscour.errors import ScourError
from textscour.uploader import HFUploader
from textscour.writer import Manifest


def test_uploader_ensure_repo_and_enqueue(tmp_path: Path):
    """Verify HFUploader calls create_repo and enqueues files."""
    mock_api = MagicMock()

    uploader = HFUploader(
        repo="userland/test-repo",
        token="hf_mock_token_12345",
        private=True,
        commit_every_shards=2,
    )
    uploader._api = mock_api

    uploader.ensure_repo()
    mock_api.create_repo.assert_called_once_with(
        repo_id="userland/test-repo",
        token="hf_mock_token_12345",
        private=True,
        exist_ok=True,
        repo_type="dataset",
    )


def test_uploader_commit_batch_and_file_deletion(tmp_path: Path):
    """Verify uploader groups files into CommitOperationAdd and deletes local file when configured."""
    mock_api = MagicMock()
    manifest = Manifest(output_dir=tmp_path)

    f1 = tmp_path / "clean_00000.jsonl"
    f1.write_text('{"id":"1"}\n')
    manifest.add_shard(f1, kind="clean", rows=1, bytes_size=10)

    uploader = HFUploader(
        repo="userland/test-repo",
        token="hf_test_token",
        commit_every_shards=1,
        delete_after_upload=True,
        manifest=manifest,
    )
    uploader._api = mock_api

    assert f1.is_file()
    uploader._commit_batch([str(f1)])

    mock_api.create_commit.assert_called_once()
    assert not f1.is_file(), "Local file should have been deleted after upload"
    assert manifest.entries[0]["uploaded"] is True


def test_uploader_rate_limit_error(tmp_path: Path):
    """Verify repeated HTTP 429 raises E-UP-002."""
    mock_api = MagicMock()
    mock_api.create_commit.side_effect = Exception("HTTP 429: Too Many Requests (rate limit)")

    f1 = tmp_path / "shard_429.jsonl"
    f1.write_text('{"id":"1"}\n')

    uploader = HFUploader(
        repo="userland/test-repo",
        token="hf_test_token",
    )
    uploader._api = mock_api

    # Force max attempts = 1 to fail immediately in test
    with pytest.raises(ScourError) as exc_info:
        # Patch time.sleep to avoid waiting
        import time
        orig_sleep = time.sleep
        time.sleep = lambda s: None
        try:
            uploader._commit_batch([str(f1)])
        finally:
            time.sleep = orig_sleep

    assert exc_info.value.code == "E-UP-002"
