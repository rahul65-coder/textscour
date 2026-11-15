"""Atomic shard writing, reorder buffering, deduplication integration, and quarantine management.

Enforces strict 5-field schema for clean records, doc/fragment schemas for garbage,
atomic write-fsync-rename shard lifecycle (.part -> rename), shard rotation,
and transactional checkpoint commit order.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textscour.checkpoint import Checkpoint
from textscour.config import Config
from textscour.errors import ScourError


class AtomicShard:
    """Atomic shard writer using `.part` temporary file and fsync before rename."""

    def __init__(self, final_path: str | Path, fmt: str = "jsonl") -> None:
        self.final_path = Path(final_path)
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.part_path = Path(str(self.final_path) + ".part")
        self.fmt = fmt
        self.bytes_written = 0
        self.records_written = 0
        self.is_closed = False

        self._file: Any = None
        self._zstd_writer: Any = None
        self._parquet_records: list[dict[str, Any]] = []

        self._open()

    def _open(self) -> None:
        if self.fmt == "jsonl.zst":
            try:
                import zstandard as zstd

                self._file = open(self.part_path, "wb")
                cctx = zstd.ZstdCompressor(level=3)
                self._zstd_writer = cctx.stream_writer(self._file)
            except ImportError as e:
                raise ScourError(
                    code="E-OUT-001",
                    stage="writer",
                    message="zstandard package required for jsonl.zst output format",
                    cause=e,
                    hint="Install zstandard: pip install zstandard",
                )
        elif self.fmt == "parquet":
            # Parquet records collected in memory per shard and written on commit
            self._parquet_records = []
        else:
            # Standard jsonl
            self._file = open(self.part_path, "w", encoding="utf-8")

    def write_record(self, record: dict[str, Any]) -> None:
        """Write a single record to active shard."""
        if self.is_closed:
            raise ScourError(
                code="E-OUT-001",
                stage="writer",
                message=f"Attempted write to closed shard {self.final_path}",
                hint="Verify shard rotation lifecycle",
            )

        if self.fmt == "parquet":
            self._parquet_records.append(record)
            # Estimate bytes
            self.bytes_written += len(str(record))
            self.records_written += 1
            return

        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        if self.fmt == "jsonl.zst":
            line_bytes = line.encode("utf-8")
            self._zstd_writer.write(line_bytes)
            self.bytes_written += len(line_bytes)
        else:
            self._file.write(line)
            self.bytes_written += len(line.encode("utf-8"))

        self.records_written += 1

    def commit(self) -> Path:
        """Flush, fsync, close, and atomically rename `.part` -> final target."""
        if self.is_closed:
            return self.final_path

        if self.records_written == 0:
            self.close()
            return self.final_path

        if self.fmt == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            if self._parquet_records:
                table = pa.Table.from_pylist(self._parquet_records)
                pq.write_table(table, self.part_path, compression="zstd")
            self.is_closed = True
        elif self.fmt == "jsonl.zst":
            self._zstd_writer.flush()
            self._zstd_writer.close()
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except OSError:
                pass
            self._file.close()
            self.is_closed = True
        else:
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except OSError:
                pass
            self._file.close()
            self.is_closed = True

        # Atomic rename
        try:
            os.replace(self.part_path, self.final_path)
        except Exception as e:
            raise ScourError(
                code="E-OUT-001",
                stage="writer",
                message=f"Failed atomic rename of shard to {self.final_path}: {e}",
                cause=e,
                hint="Check filesystem write permissions and disk quota.",
            )

        return self.final_path

    def close(self) -> None:
        """Close without rename if not yet committed (cleans up .part)."""
        if not self.is_closed:
            if self._zstd_writer:
                try:
                    self._zstd_writer.close()
                except Exception:
                    pass
            if self._file:
                try:
                    self._file.close()
                except Exception:
                    pass
            self.is_closed = True
            if self.part_path.exists():
                try:
                    self.part_path.unlink()
                except OSError:
                    pass


class Manifest:
    """Manages manifest.json tracking all committed shards."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.json"
        self.entries: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.manifest_path.is_file():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.entries = data
            except Exception:
                pass

    def add_shard(self, path: Path, kind: str, rows: int, bytes_size: int) -> None:
        """Record completed shard in manifest with relative path."""
        sha256 = ""
        try:
            with open(path, "rb") as f:
                sha256 = hashlib.file_digest(f, "sha256").hexdigest()
        except Exception:
            pass

        rel_path = str(path.relative_to(self.output_dir)) if path.is_relative_to(self.output_dir) else path.name
        entry = {
            "path": rel_path,
            "full_path": str(path.resolve()),
            "kind": kind,
            "rows": rows,
            "bytes": bytes_size,
            "sha256": sha256,
            "uploaded": False,
        }
        self.entries.append(entry)
        self.save()

    def mark_uploaded(self, full_path: str) -> None:
        """Mark entry as uploaded."""
        for e in self.entries:
            if e.get("full_path") == full_path or e.get("path") == full_path:
                e["uploaded"] = True
        self.save()

    def save(self) -> None:
        """Atomically persist manifest."""
        tmp = self.manifest_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, self.manifest_path)


class QuarantineWriter:
    """Isolates corrupt or schema-violating records in quarantine.jsonl."""

    def __init__(self, output_dir: Path | str, max_error_rate: float = 0.01) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "quarantine.jsonl"
        self.max_error_rate = max_error_rate
        self.error_count = 0
        self.total_docs_seen = 0

    def quarantine(
        self,
        source: str,
        batch_id: int,
        record_id: str,
        code: str,
        message: str,
    ) -> None:
        """Append error entry to quarantine log."""
        self.error_count += 1
        entry = {
            "source": source,
            "batch_id": batch_id,
            "record_id": record_id,
            "code": code,
            "message": message,
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)

    def check_error_rate(self, total_seen: int) -> None:
        """Verify error rate does not exceed budget."""
        self.total_docs_seen = total_seen
        if total_seen > 100:
            rate = self.error_count / float(total_seen)
            if rate > self.max_error_rate:
                raise ScourError(
                    code="E-RATE-001",
                    stage="runtime",
                    message=f"Quarantine error rate ({rate:.2%}) exceeded max_error_rate ({self.max_error_rate:.2%})",
                    hint="Check quarantine.jsonl for recurrent schema or extraction failures.",
                )


def sanitize_source_key(source_key: str) -> str:
    """Sanitize source key into safe filename slug [A-Za-z0-9._-] with stable hash suffix."""
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


class OrderedSourceWriter:
    """Manages reorder buffer, dedup checking, shard rotation, and checkpoint persistence."""

    def __init__(
        self,
        source_key: str,
        cfg: Config,
        ckpt: Checkpoint,
        dedup_index: Any = None,
        manifest: Manifest | None = None,
        uploader_q: Any = None,
        resume_state: dict[str, Any] | None = None,
    ) -> None:
        self.source_key = source_key
        self.cfg = cfg
        self.ckpt = ckpt
        self.dedup = dedup_index
        self.manifest = manifest
        self.uploader_q = uploader_q
        self.src_slug = sanitize_source_key(source_key)
        self.reason_counts: dict[str, int] = {}

        self.out_dir = Path(cfg.output.dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_max_bytes = cfg.output.shard_mb * 1024 * 1024

        res = resume_state or {}
        self.next_batch_id: int = res.get("next_batch_id", 0)
        self.clean_shard_idx: int = res.get("shard_idx", {}).get("clean", 0)
        self.garbage_shard_idx: int = res.get("shard_idx", {}).get("garbage", 0)
        self.records_in: int = res.get("records_in", 0)
        self.records_clean: int = res.get("records_out", {}).get("clean", 0)
        self.records_garbage: int = res.get("records_out", {}).get("garbage", 0)
        self.records_quarantine: int = res.get("records_out", {}).get("quarantine", 0)

        self.reorder_buffer: dict[int, Any] = {}
        self.clean_shard: AtomicShard | None = None
        self.garbage_shard: AtomicShard | None = None
        self.latest_stream_state: dict[str, Any] | None = None

        self._open_shards()

    def _clean_path(self, idx: int) -> Path:
        ext = self.cfg.output.format
        name = self.cfg.output.clean_name
        return self.out_dir / f"{name}-{self.src_slug}-{idx:05d}.{ext}"

    def _garbage_path(self, idx: int) -> Path:
        ext = self.cfg.output.format
        name = self.cfg.output.garbage_name
        return self.out_dir / f"{name}-{self.src_slug}-{idx:05d}.{ext}"

    def _open_shards(self) -> None:
        if self.clean_shard is None or self.clean_shard.is_closed:
            self.clean_shard = AtomicShard(self._clean_path(self.clean_shard_idx), self.cfg.output.format)
        if self.garbage_shard is None or self.garbage_shard.is_closed:
            self.garbage_shard = AtomicShard(self._garbage_path(self.garbage_shard_idx), self.cfg.output.format)

    def _rotate_if_needed(self) -> None:
        clean_full = self.clean_shard and self.clean_shard.bytes_written >= self.shard_max_bytes
        garbage_full = self.garbage_shard and self.garbage_shard.bytes_written >= self.shard_max_bytes

        if clean_full or garbage_full:
            self._commit_and_rotate(is_final=False)

    def _commit_and_rotate(self, is_final: bool = False) -> None:
        """Atomic rotation sequence: shard rename -> dedup save -> checkpoint commit -> upload enqueue."""
        committed_shards: list[tuple[Path, str, int, int]] = []

        if self.clean_shard and not self.clean_shard.is_closed:
            rows = self.clean_shard.records_written
            bytes_sz = self.clean_shard.bytes_written
            p = self.clean_shard.commit()
            if rows > 0:
                committed_shards.append((p, "clean", rows, bytes_sz))
                self.clean_shard_idx += 1

        if self.garbage_shard and not self.garbage_shard.is_closed:
            rows = self.garbage_shard.records_written
            bytes_sz = self.garbage_shard.bytes_written
            p = self.garbage_shard.commit()
            if rows > 0:
                committed_shards.append((p, "garbage", rows, bytes_sz))
                self.garbage_shard_idx += 1

        # 1. Update manifest
        if self.manifest:
            for p, kind, rows, bsz in committed_shards:
                self.manifest.add_shard(p, kind, rows, bsz)

        # 2. Dedup save
        dedup_dir = self.ckpt.state_dir
        dedup_path_str = str(dedup_dir / f"dedup-{self.src_slug}.bin")
        if self.dedup and hasattr(self.dedup, "save"):
            try:
                os.makedirs(str(dedup_dir), exist_ok=True)
                self.dedup.save(dedup_path_str)
            except Exception:
                pass

        # 3. Checkpoint commit
        from textscour.config import critical_config_hash

        cfg_hash = critical_config_hash(self.cfg)
        payload = {
            "source": self.source_key,
            "next_batch_id": self.next_batch_id,
            "reader_states": {"0": self.latest_stream_state or {}},
            "shard_idx": {
                "clean": self.clean_shard_idx,
                "garbage": self.garbage_shard_idx,
            },
            "records_in": self.records_in,
            "records_out": {
                "clean": self.records_clean,
                "garbage": self.records_garbage,
                "quarantine": self.records_quarantine,
            },
            "dedup_file": dedup_path_str,
            "engine_version": "0.1.0",
            "config_hash": cfg_hash,
        }
        self.ckpt.commit(self.source_key, payload)

        # 4. Enqueue to uploader if active
        if self.uploader_q and committed_shards:
            for p, _, _, _ in committed_shards:
                self.uploader_q.put(str(p))

        # Re-open fresh active shards only if not final flush
        if not is_final:
            self._open_shards()
        else:
            self.clean_shard = None
            self.garbage_shard = None

    def push(self, result_batch: Any) -> None:
        """Receive ResultBatch and write records in strict ascending batch_id order."""
        if result_batch.batch_id < self.next_batch_id:
            return
        self.reorder_buffer[result_batch.batch_id] = result_batch

        while self.next_batch_id in self.reorder_buffer:
            batch = self.reorder_buffer.pop(self.next_batch_id)
            self._write_batch(batch)
            self.next_batch_id += 1

    def _write_batch(self, batch: Any) -> None:
        """Process and output records from an ordered batch."""
        self.latest_stream_state = getattr(batch, "stream_state", None)

        for record in batch.records:
            self.records_in += 1
            # Exact and near-duplicate check via DedupIndex
            is_clean = getattr(record, "clean", True)
            reasons = list(getattr(record, "reasons", []))

            if is_clean and self.dedup is not None:
                dup_reason = self.dedup.check_and_insert(record.exact, list(record.bands))
                if dup_reason:
                    is_clean = False
                    reasons.append(dup_reason)

            doc_id = getattr(record, "id", f"doc_{self.records_in}")
            domain = getattr(record, "domain", self.source_key)
            text = getattr(record, "text", "")
            length = getattr(record, "length", len(text))
            score = round(float(getattr(record, "score", 0.0)), 1)

            if is_clean:
                # 5-field clean schema
                clean_dict = {
                    "id": doc_id,
                    "domain": domain,
                    "text": text,
                    "length": length,
                    "score": score,
                }
                self.clean_shard.write_record(clean_dict)  # type: ignore
                self.records_clean += 1
            else:
                # Record reason counts
                for r in reasons:
                    r_str = str(r)
                    self.reason_counts[r_str] = self.reason_counts.get(r_str, 0) + 1

                # Garbage schema
                garbage_dict = {
                    "id": doc_id,
                    "domain": domain,
                    "text": text,
                    "length": length,
                    "score": score,
                    "reasons": reasons,
                    "kind": "doc",
                }
                self.garbage_shard.write_record(garbage_dict)  # type: ignore
                self.records_garbage += 1

                # Emitting fragments if configured
                if self.cfg.output.emit_fragments:
                    frags = getattr(record, "fragments", [])
                    for f_idx, (frag_reason, frag_text) in enumerate(frags):
                        frag_r = f"frag:{frag_reason}"
                        self.reason_counts[frag_r] = self.reason_counts.get(frag_r, 0) + 1
                        frag_dict = {
                            "id": f"{doc_id}#f{f_idx}",
                            "domain": domain,
                            "text": frag_text,
                            "length": len(frag_text),
                            "score": 0.0,
                            "reasons": [frag_reason],
                            "kind": "fragment",
                        }
                        self.garbage_shard.write_record(frag_dict)  # type: ignore

        # Rotate on batch boundaries so checkpoint strictly matches written records
        self._rotate_if_needed()

    def flush(self) -> None:
        """Finalize all remaining records and commit active shards."""
        # Drain any remaining out-of-order batches if possible
        if self.reorder_buffer:
            for b_id in sorted(self.reorder_buffer.keys()):
                self._write_batch(self.reorder_buffer.pop(b_id))
                self.next_batch_id = max(self.next_batch_id, b_id + 1)

        self._commit_and_rotate(is_final=True)

        # Cleanup any empty or leftover .part files in output directory
        for part_file in self.out_dir.glob("*.part"):
            try:
                if part_file.stat().st_size == 0:
                    part_file.unlink()
            except OSError:
                pass


def merge_outputs(cfg: Config) -> dict[str, str]:
    """Concatenate individual shard files into final merged clean and garbage files."""
    out_dir = Path(cfg.output.dir)
    fmt = cfg.output.format
    clean_target = out_dir / f"{cfg.output.clean_name}.{fmt}"
    garbage_target = out_dir / f"{cfg.output.garbage_name}.{fmt}"

    clean_shards = sorted(out_dir.glob(f"{cfg.output.clean_name}-*-*.*"))
    garbage_shards = sorted(out_dir.glob(f"{cfg.output.garbage_name}-*-*.*"))

    # Exclude .part files
    clean_shards = [s for s in clean_shards if not s.name.endswith(".part")]
    garbage_shards = [s for s in garbage_shards if not s.name.endswith(".part")]

    # Check disk space
    total_needed = sum(s.stat().st_size for s in clean_shards + garbage_shards)
    free_bytes = shutil.disk_usage(out_dir).free
    if free_bytes < total_needed:
        raise ScourError(
            code="E-DISK-001",
            stage="storage",
            message=f"Insufficient disk space for final merge: need {total_needed / (1024**3):.2f} GB, free {free_bytes / (1024**3):.2f} GB",
            hint="Increase runtime.disk_budget_gb or disable final_merge.",
        )

    for target, shards in ((clean_target, clean_shards), (garbage_target, garbage_shards)):
        if not shards:
            continue
        tmp_target = target.with_suffix(".merge.tmp")
        with open(tmp_target, "wb") as out_f:
            for s in shards:
                with open(s, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f)
            out_f.flush()
            try:
                os.fsync(out_f.fileno())
            except OSError:
                pass
        os.replace(tmp_target, target)

    return {"clean": str(clean_target), "garbage": str(garbage_target)}
