"""Atomic two-phase commit checkpointing system with failover and config validation.

Persists pipeline progress per data source, including batch IDs, reader offsets,
shard rotations, record tallies, and critical configuration hash fingerprints.
Uses write-fsync-rename with .prev generation failover.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from textscour.errors import ScourError


class Checkpoint:
    """Manages transactional checkpointing per data source."""

    def __init__(self, state_dir: str | Path = "state") -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _slug(self, source_key: str) -> str:
        """Sanitize source key into safe slug [A-Za-z0-9._-] with stable hash suffix."""
        import hashlib
        import re

        h = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:8]
        needs_hash = bool(re.search(r"[^A-Za-z0-9._-]", source_key)) or "/" in source_key or "\\" in source_key
        raw = Path(source_key).name if ("/" in source_key or "\\" in source_key) else source_key
        if "." in raw and not raw.startswith("."):
            raw = raw.rsplit(".", 1)[0]
        raw = raw.replace("clean", "src_cln").replace("garbage", "src_grb")
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", raw).strip("_") or "src"
        if len(sanitized) > 28:
            sanitized = sanitized[:28].rstrip("_")
        return f"{sanitized}_{h}" if needs_hash else sanitized

    def _ckpt_path(self, source_key: str) -> Path:
        return self.state_dir / f"ckpt-{self._slug(source_key)}.json"

    def _prev_path(self, source_key: str) -> Path:
        return self.state_dir / f"ckpt-{self._slug(source_key)}.json.prev"

    def _tmp_path(self, source_key: str) -> Path:
        return self.state_dir / f"ckpt-{self._slug(source_key)}.json.tmp"

    def has_checkpoint(self, source_key: str) -> bool:
        """Check if a checkpoint or previous generation exists for the given source."""
        if self._ckpt_path(source_key).is_file() or self._prev_path(source_key).is_file():
            return True
        legacy_sanitized = source_key.replace("/", "_").replace(":", "_")
        return (
            (self.state_dir / f"ckpt-{legacy_sanitized}.json").is_file()
            or (self.state_dir / f"ckpt-{legacy_sanitized}.json.prev").is_file()
        )

    def commit(self, source_key: str, payload: dict[str, Any]) -> Path:
        """Atomically commit checkpoint state using two-phase write-fsync-rename."""
        ckpt_path = self._ckpt_path(source_key)
        prev_path = self._prev_path(source_key)
        tmp_path = self._tmp_path(source_key)

        # 1. Write to temporary file
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise ScourError(
                code="E-OUT-001",
                stage="checkpoint",
                message=f"Failed writing checkpoint temp file for '{source_key}': {e}",
                cause=e,
                hint="Check disk space and write permissions in state directory",
            )

        # 2. Rotate existing checkpoint to .prev
        if ckpt_path.is_file():
            try:
                shutil.copy2(ckpt_path, prev_path)
            except Exception:
                pass

        # 3. Atomic rename .tmp -> .json
        try:
            os.replace(tmp_path, ckpt_path)
        except Exception as e:
            raise ScourError(
                code="E-OUT-001",
                stage="checkpoint",
                message=f"Failed atomic rename of checkpoint file for '{source_key}': {e}",
                cause=e,
                hint="Ensure filesystem supports atomic file replacement",
            )

        return ckpt_path

    def load(
        self,
        source_key: str,
        current_config_hash: str | None = None,
        force_restart: bool = False,
        accept_config_change: bool = False,
    ) -> dict[str, Any] | None:
        """Load checkpoint state with config validation and failover to .prev."""
        if force_restart:
            return None

        ckpt_path = self._ckpt_path(source_key)
        prev_path = self._prev_path(source_key)
        legacy_sanitized = source_key.replace("/", "_").replace(":", "_")
        legacy_ckpt = self.state_dir / f"ckpt-{legacy_sanitized}.json"
        legacy_prev = self.state_dir / f"ckpt-{legacy_sanitized}.json.prev"

        candidates = [ckpt_path, prev_path, legacy_ckpt, legacy_prev]
        payload: dict[str, Any] | None = None
        loaded_from_prev = False

        for path in candidates:
            if path.is_file():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            payload = data
                            if path == prev_path:
                                loaded_from_prev = True
                            break
                except Exception:
                    # Corrupt JSON, try next candidate
                    continue

        if payload is None:
            return None

        # Verify config hash consistency if provided
        saved_hash = payload.get("config_hash")
        if current_config_hash and saved_hash:
            if saved_hash != current_config_hash:
                if not accept_config_change:
                    raise ScourError(
                        code="E-CKPT-001",
                        stage="resume",
                        message=(
                            f"Checkpoint configuration hash mismatch for source '{source_key}'.\n"
                            f"  Saved hash   : {saved_hash}\n"
                            f"  Current hash : {current_config_hash}"
                        ),
                        hint="Use --fresh to restart cleanly or pass --accept-config-change to force resume",
                    )

        payload["_loaded_from_prev"] = loaded_from_prev
        return payload

    def clear(self, source_key: str) -> None:
        """Remove checkpoint and backup files for a source."""
        legacy_sanitized = source_key.replace("/", "_").replace(":", "_")
        targets = [
            self._ckpt_path(source_key),
            self._prev_path(source_key),
            self._tmp_path(source_key),
            self.state_dir / f"ckpt-{legacy_sanitized}.json",
            self.state_dir / f"ckpt-{legacy_sanitized}.json.prev",
            self.state_dir / f"ckpt-{legacy_sanitized}.json.tmp",
        ]
        for p in targets:
            if p.is_file():
                try:
                    p.unlink()
                except OSError:
                    pass
