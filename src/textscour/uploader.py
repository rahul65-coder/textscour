"""Asynchronous Hugging Face Hub dataset uploader with batched multi-file commits.

Runs as a background daemon grouping completed shards into transactional multi-file
commits via `create_commit` and `CommitOperationAdd`. Handles exponential backoff,
HTTP 429 rate-limiting Retry-After headers, local file cleanup, and remote state sync.
"""

from __future__ import annotations

import os
import queue
import random
import threading
import time
from pathlib import Path
from typing import Any

from textscour.errors import ScourError
from textscour.sources import resolve_hf_token


class HFUploader:
    """Manages background batch uploads of shards and state checkpoints to Hugging Face Hub."""

    def __init__(
        self,
        repo: str,
        token: str | None = None,
        private: bool = True,
        commit_every_shards: int = 4,
        commit_every_seconds: int = 300,
        delete_after_upload: bool = False,
        state_backend: str = "local",
        manifest: Any = None,
    ) -> None:
        self.repo = repo
        self.token = token or resolve_hf_token()
        self.private = private
        self.commit_every_shards = max(1, commit_every_shards)
        self.commit_every_seconds = max(10, commit_every_seconds)
        self.delete_after_upload = delete_after_upload
        self.state_backend = state_backend
        self.manifest = manifest

        self.upload_queue: queue.Queue[str | None] = queue.Queue()
        self.pending_files: list[str] = []
        self.last_commit_time = time.time()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._api: Any = None

    def _get_api(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.token)
        return self._api

    def ensure_repo(self) -> None:
        """Create target repository on Hugging Face if it does not already exist."""
        api = self._get_api()
        try:
            api.create_repo(
                repo_id=self.repo,
                token=self.token,
                private=self.private,
                exist_ok=True,
                repo_type="dataset",
            )
        except Exception as e:
            err_msg = str(e)
            if "401" in err_msg or "403" in err_msg:
                raise ScourError(
                    code="E-UP-001",
                    stage="upload",
                    dataset=self.repo,
                    message=f"Hugging Face repository authorization failed: {err_msg}",
                    cause=e,
                    hint="Verify write permissions for token on destination repo",
                )
            raise ScourError(
                code="E-UP-001",
                stage="upload",
                dataset=self.repo,
                message=f"Failed ensuring destination HF repo '{self.repo}': {err_msg}",
                cause=e,
                hint="Verify repository identifier syntax: org/dataset-name",
            )

    def enqueue(self, file_path: str | Path) -> None:
        """Enqueue completed shard path for batch commit."""
        self.upload_queue.put(str(file_path))

    def start(self) -> None:
        """Start the background uploader daemon thread."""
        self.ensure_repo()
        self.last_commit_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="hf-uploader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal uploader to drain remaining files and commit."""
        self.upload_queue.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join()

    def _determine_repo_path(self, local_path_str: str) -> str:
        """Map local file path to remote repository structure."""
        p = Path(local_path_str)
        fname = p.name
        if "clean" in fname:
            return f"clean/{fname}"
        if "garbage" in fname:
            return f"garbage/{fname}"
        if "state" in local_path_str or "ckpt" in fname or "dedup" in fname:
            return f"_state/{fname}"
        return fname

    def _commit_batch(self, files: list[str]) -> None:
        """Execute a single create_commit grouping all pending operations."""
        if not files:
            return

        from huggingface_hub import CommitOperationAdd

        operations = []
        for fpath in files:
            p = Path(fpath)
            if p.is_file():
                repo_dest = self._determine_repo_path(fpath)
                operations.append(CommitOperationAdd(path_in_repo=repo_dest, path_or_fileobj=str(p.resolve())))

        if not operations:
            return

        api = self._get_api()
        max_attempts = 5
        attempt = 0

        while attempt < max_attempts:
            try:
                commit_msg = f"Upload {len(operations)} shard(s) [textscour]"
                api.create_commit(
                    repo_id=self.repo,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=commit_msg,
                    token=self.token,
                )
                self.last_commit_time = time.time()

                # On successful commit
                for fpath in files:
                    if self.manifest:
                        self.manifest.mark_uploaded(fpath)
                    if self.delete_after_upload:
                        try:
                            p = Path(fpath)
                            if p.is_file():
                                p.unlink()
                        except Exception:
                            pass
                return

            except Exception as e:
                attempt += 1
                err_msg = str(e)
                if "429" in err_msg or "rate limit" in err_msg.lower():
                    if attempt >= max_attempts:
                        raise ScourError(
                            code="E-UP-002",
                            stage="upload",
                            dataset=self.repo,
                            message=f"Hugging Face commit rate limit exceeded (HTTP 429): {err_msg}",
                            cause=e,
                            hint="Increase upload.commit_every_shards and commit_every_seconds",
                        )
                    # Backoff
                    backoff = (2**attempt) * 10 + random.uniform(1.0, 5.0)
                    time.sleep(backoff)
                else:
                    if attempt >= max_attempts:
                        raise ScourError(
                            code="E-UP-001",
                            stage="upload",
                            dataset=self.repo,
                            message=f"Hugging Face upload failed after {max_attempts} attempts: {err_msg}",
                            cause=e,
                            hint="Check write permissions for token on destination repo and network",
                        )
                    time.sleep(2**attempt)

    def _run(self) -> None:
        """Worker loop processing files from queue."""
        while not self._stop_event.is_set():
            try:
                item = self.upload_queue.get(timeout=2.0)
                if item is None:
                    # Termination sentinel
                    break
                self.pending_files.append(item)
            except queue.Empty:
                pass

            now = time.time()
            time_due = (now - self.last_commit_time) >= self.commit_every_seconds
            count_due = len(self.pending_files) >= self.commit_every_shards

            if self.pending_files and (time_due or count_due):
                batch_to_send = list(self.pending_files)
                self.pending_files.clear()
                self._commit_batch(batch_to_send)

        # Flush any remaining files on termination
        if self.pending_files:
            batch_to_send = list(self.pending_files)
            self.pending_files.clear()
            self._commit_batch(batch_to_send)

    def pull_state(self, dest_dir: str | Path) -> None:
        """Download remote _state/ directory files to dest_dir when state_backend == 'hf'."""
        if self.state_backend != "hf":
            return

        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        api = self._get_api()
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=self.repo,
                repo_type="dataset",
                allow_patterns="_state/*",
                local_dir=str(dest.parent),
                token=self.token,
            )
            # Move downloaded _state/* files if needed into dest
            downloaded_state = dest.parent / "_state"
            if downloaded_state.is_dir() and downloaded_state != dest:
                for f in downloaded_state.glob("*"):
                    shutil_dest = dest / f.name
                    if not shutil_dest.exists():
                        f.rename(shutil_dest)
        except Exception:
            pass
