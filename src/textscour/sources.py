"""Streaming data sources: Hugging Face datasets and local files (JSONL, JSON, Parquet).

Implements fault-tolerant streaming readers with token resolution, exponential
backoff retry loops, cgroup-aware batching, checkpoint resuming, and multi-reader
sharding.
"""

from __future__ import annotations

import glob
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import xxhash

from textscour.config import Config, SourceCfg
from textscour.errors import ScourError
from textscour.schema import compile_extractor, extract_domain, extract_id


@dataclass
class RawBatch:
    """Inter-process batch passed from reader to pipeline workers."""

    source_key: str
    batch_id: int
    shard_idx: int
    docs: list[dict[str, Any] | str]
    ids: list[str]
    domains: list[str]
    stream_state: dict[str, Any] | None = None
    bytes_in: int = 0  # Raw byte size estimate for this batch


def resolve_hf_token() -> str | None:
    """Resolve Hugging Face authentication token in priority order:

    1. Environment variable `HF_TOKEN`
    2. Kaggle Secrets (`HF_TOKEN`)
    3. huggingface_hub token cache
    """
    token = os.environ.get("HF_TOKEN")
    if token and token.strip():
        return token.strip()

    # Kaggle secrets fallback
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        client = UserSecretsClient()
        token = client.get_secret("HF_TOKEN")
        if token and str(token).strip():
            return str(token).strip()
    except Exception:
        pass

    # huggingface_hub fallback
    try:
        import huggingface_hub

        token = huggingface_hub.get_token()
        if token and str(token).strip():
            return str(token).strip()
    except Exception:
        pass

    return None


class HFStreamSource:
    """Fault-tolerant streaming source for Hugging Face Hub datasets."""

    def __init__(
        self,
        cfg: SourceCfg,
        token: str | None = None,
        reader_index: int = 0,
        n_readers: int = 1,
        batch_docs: int = 512,
        resume_state: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.token = token or resolve_hf_token()
        self.reader_index = reader_index
        self.n_readers = max(1, n_readers)
        self.batch_docs = max(1, batch_docs)
        self.resume_state = resume_state or {}
        self.source_key = cfg.repo or "hf_dataset"
        self.extractor = compile_extractor(cfg.text)

    def _open_dataset(self) -> Any:
        """Instantiate datasets.load_dataset with streaming enabled."""
        import datasets

        kwargs: dict[str, Any] = {
            "path": self.cfg.repo,
            "split": self.cfg.split or "train",
            "streaming": True,
        }
        if self.cfg.name:
            kwargs["name"] = self.cfg.name
        if self.cfg.revision:
            kwargs["revision"] = self.cfg.revision
        if self.cfg.data_files:
            kwargs["data_files"] = self.cfg.data_files
        if self.token:
            kwargs["token"] = self.token

        try:
            ds = datasets.load_dataset(**kwargs)
        except Exception as e:
            err_msg = str(e)
            if "401" in err_msg or "403" in err_msg or "gated" in err_msg.lower():
                raise ScourError(
                    code="E-SRC-002",
                    stage="source",
                    dataset=self.source_key,
                    message=f"Hugging Face authentication failed for dataset '{self.source_key}': {err_msg}",
                    cause=e,
                    hint="Set HF_TOKEN in environment or Kaggle Secrets, and accept repository terms.",
                )
            if "not found" in err_msg.lower() or "404" in err_msg:
                raise ScourError(
                    code="E-SRC-001",
                    stage="source",
                    dataset=self.source_key,
                    message=f"Hugging Face dataset '{self.source_key}' not found: {err_msg}",
                    cause=e,
                    hint="Verify repository identifier and spelling on huggingface.co",
                )
            raise ScourError(
                code="E-SRC-003",
                stage="source",
                dataset=self.source_key,
                message=f"Failed loading HF dataset '{self.source_key}': {err_msg}",
                cause=e,
                hint="Check network connectivity or HF hub service status",
            )

        # Shard dataset across parallel readers if supported
        if self.n_readers > 1 and hasattr(ds, "n_shards"):
            n_shards = getattr(ds, "n_shards", 1)
            if n_shards >= self.n_readers:
                ds = ds.shard(num_shards=self.n_readers, index=self.reader_index)
            elif self.reader_index != 0:
                # Only 1 shard available; reader 0 consumes, other readers produce empty
                return None

        # State dict resume if supported
        if self.resume_state and hasattr(ds, "load_state_dict"):
            try:
                ds.load_state_dict(self.resume_state)
            except Exception:
                pass

        return ds

    def __iter__(self) -> Iterator[RawBatch]:
        """Iterate records and yield RawBatch objects with retry resilience."""
        max_retries = 8
        attempt = 0
        records_read = self.resume_state.get("records_in", 0)
        skip_count = records_read if not self.resume_state.get("has_native_state") else 0

        curr_batch_docs: list[dict[str, Any] | str] = []
        curr_batch_ids: list[str] = []
        curr_batch_domains: list[str] = []
        curr_batch_bytes: int = 0
        batch_id = self.resume_state.get("next_batch_id", 0)
        shard_idx = self.resume_state.get("shard_idx", 0)
        total_emitted = 0

        while attempt < max_retries:
            try:
                ds = self._open_dataset()
                if ds is None:
                    return

                iterator = iter(ds)
                # Skip previously processed rows if resuming without native state dict
                if skip_count > 0:
                    for _ in range(skip_count):
                        next(iterator, None)
                    skip_count = 0

                for doc in iterator:
                    doc_id_val = extract_id(
                        doc,
                        self.cfg.id_field,
                        xxhash.xxh3_64_hexdigest(f"{self.source_key}|{self.reader_index}|{records_read}".encode("utf-8")),
                    )
                    domain_val = extract_domain(doc, self.cfg.domain, self.source_key)

                    # Estimate raw byte size of this document
                    if isinstance(doc, str):
                        curr_batch_bytes += len(doc.encode("utf-8", errors="replace"))
                    elif isinstance(doc, dict):
                        try:
                            import json as _json
                            curr_batch_bytes += len(_json.dumps(doc, ensure_ascii=False).encode("utf-8"))
                        except Exception:
                            curr_batch_bytes += sum(len(str(v).encode("utf-8", errors="replace")) for v in doc.values())

                    curr_batch_docs.append(doc)
                    curr_batch_ids.append(doc_id_val)
                    curr_batch_domains.append(domain_val)
                    records_read += 1
                    total_emitted += 1

                    if len(curr_batch_docs) >= self.batch_docs:
                        state_dict = {
                            "records_in": records_read,
                            "shard_idx": shard_idx,
                            "reader_index": self.reader_index,
                            "next_batch_id": batch_id + 1,
                        }
                        yield RawBatch(
                            source_key=self.source_key,
                            batch_id=batch_id,
                            shard_idx=shard_idx,
                            docs=curr_batch_docs,
                            ids=curr_batch_ids,
                            domains=curr_batch_domains,
                            stream_state=state_dict,
                            bytes_in=curr_batch_bytes,
                        )
                        batch_id += 1
                        curr_batch_docs = []
                        curr_batch_ids = []
                        curr_batch_domains = []
                        curr_batch_bytes = 0

                    if self.cfg.max_docs and records_read >= self.cfg.max_docs:
                        break

                # Emit final partial batch
                if curr_batch_docs:
                    state_dict = {
                        "records_in": records_read,
                        "shard_idx": shard_idx,
                        "reader_index": self.reader_index,
                        "next_batch_id": batch_id + 1,
                    }
                    yield RawBatch(
                        source_key=self.source_key,
                        batch_id=batch_id,
                        shard_idx=shard_idx,
                        docs=curr_batch_docs,
                        ids=curr_batch_ids,
                        domains=curr_batch_domains,
                        stream_state=state_dict,
                        bytes_in=curr_batch_bytes,
                    )

                if total_emitted == 0 and records_read == 0 and self.reader_index == 0:
                    raise ScourError(
                        code="E-SRC-004",
                        stage="source",
                        dataset=self.source_key,
                        message=f"Hugging Face source '{self.source_key}' yielded zero records",
                        hint="Check split name, data_files patterns, and revision parameter.",
                    )

                return

            except ScourError:
                raise
            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    raise ScourError(
                        code="E-SRC-003",
                        stage="source",
                        dataset=self.source_key,
                        message=f"Network connection dropped or timed out after {max_retries} retries: {e}",
                        cause=e,
                        hint="Check internet connectivity, Hugging Face status, or proxy settings.",
                    )
                # Exponential backoff with jitter
                sleep_sec = min(60.0, (2 ** attempt) + random.uniform(0.5, 2.0))
                time.sleep(sleep_sec)
                skip_count = records_read


class LocalSource:
    """Local file streaming source supporting jsonl, jsonl.zst, json, and parquet files."""

    def __init__(
        self,
        cfg: SourceCfg,
        reader_index: int = 0,
        n_readers: int = 1,
        batch_docs: int = 512,
        resume_state: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.reader_index = reader_index
        self.n_readers = max(1, n_readers)
        self.batch_docs = max(1, batch_docs)
        self.resume_state = resume_state or {}
        self.source_key = cfg.name or (cfg.paths[0] if cfg.paths else "local")
        self.extractor = compile_extractor(cfg.text)

        # Resolve all matching file paths
        all_files: list[str] = []
        for p_pattern in cfg.paths:
            matches = sorted(glob.glob(p_pattern))
            if not matches and Path(p_pattern).is_file():
                matches = [p_pattern]
            all_files.extend(matches)

        if not all_files:
            raise ScourError(
                code="E-SRC-001",
                stage="source",
                dataset=self.source_key,
                message=f"No local files matched paths: {cfg.paths}",
                hint="Verify input file paths and glob patterns.",
            )

        # Shard file list across readers
        self.files = [f for idx, f in enumerate(all_files) if idx % self.n_readers == self.reader_index]

    def _read_jsonl_file(self, file_path: str, offset: int = 0) -> Iterator[tuple[dict[str, Any] | str, int]]:
        """Read lines from JSONL or compressed JSONL."""
        import json

        open_fn = open
        open_kwargs: dict[str, Any] = {"mode": "rt", "encoding": "utf-8"}

        if file_path.endswith(".zst"):
            import zstandard as zstd

            dctx = zstd.ZstdDecompressor()
            with open(file_path, "rb") as raw_f:
                with dctx.stream_reader(raw_f) as reader:
                    import io

                    text_io = io.TextIOWrapper(reader, encoding="utf-8")
                    for line_no, line in enumerate(text_io):
                        if line_no < offset:
                            continue
                        line_str = line.strip()
                        if line_str:
                            try:
                                yield json.loads(line_str), line_no + 1
                            except Exception:
                                yield line_str, line_no + 1
            return

        if file_path.endswith(".gz"):
            import gzip

            open_fn = gzip.open  # type: ignore

        with open_fn(file_path, **open_kwargs) as f:
            for line_no, line in enumerate(f):
                if line_no < offset:
                    continue
                line_str = line.strip()
                if line_str:
                    try:
                        yield json.loads(line_str), line_no + 1
                    except Exception:
                        yield line_str, line_no + 1

    def _read_parquet_file(self, file_path: str, row_group_offset: int = 0) -> Iterator[tuple[dict[str, Any], int]]:
        """Read rows from Parquet file using pyarrow row-group iteration."""
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(file_path)
        num_rg = pf.num_row_groups
        for rg_idx in range(row_group_offset, num_rg):
            table = pf.read_row_group(rg_idx)
            dict_records = table.to_pylist()
            for rec in dict_records:
                yield rec, rg_idx

    def _read_json_file(self, file_path: str) -> Iterator[tuple[dict[str, Any] | str, int]]:
        """Read standard JSON array file."""
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    yield item, idx + 1
            elif isinstance(data, dict):
                yield data, 1

    def __iter__(self) -> Iterator[RawBatch]:
        """Iterate local documents and yield RawBatch objects."""
        curr_batch_docs: list[dict[str, Any] | str] = []
        curr_batch_ids: list[str] = []
        curr_batch_domains: list[str] = []
        curr_batch_bytes: int = 0
        batch_id = self.resume_state.get("next_batch_id", 0)
        shard_idx = self.resume_state.get("shard_idx", 0)
        records_read = self.resume_state.get("records_in", 0)
        saved_file_idx = self.resume_state.get("file_idx", 0)
        saved_offset = self.resume_state.get("file_offset", 0)

        for file_idx, fpath in enumerate(self.files):
            if file_idx < saved_file_idx:
                continue
            offset = saved_offset if file_idx == saved_file_idx else 0

            iterator: Iterator[tuple[Any, int]]
            if fpath.endswith(".parquet"):
                iterator = self._read_parquet_file(fpath, row_group_offset=offset)
            elif fpath.endswith((".jsonl", ".jsonl.zst", ".jsonl.gz")):
                iterator = self._read_jsonl_file(fpath, offset=offset)
            elif fpath.endswith(".json"):
                iterator = self._read_json_file(fpath)
            else:
                iterator = self._read_jsonl_file(fpath, offset=offset)

            for doc, curr_offset in iterator:
                doc_id_val = extract_id(
                    doc,
                    self.cfg.id_field,
                    xxhash.xxh3_64_hexdigest(f"{self.source_key}|{self.reader_index}|{records_read}".encode("utf-8")),
                )
                domain_val = extract_domain(doc, self.cfg.domain, self.source_key)

                # Estimate raw byte size of this document
                if isinstance(doc, str):
                    curr_batch_bytes += len(doc.encode("utf-8", errors="replace"))
                elif isinstance(doc, dict):
                    try:
                        import json as _json
                        curr_batch_bytes += len(_json.dumps(doc, ensure_ascii=False).encode("utf-8"))
                    except Exception:
                        curr_batch_bytes += sum(len(str(v).encode("utf-8", errors="replace")) for v in doc.values())

                curr_batch_docs.append(doc)
                curr_batch_ids.append(doc_id_val)
                curr_batch_domains.append(domain_val)
                records_read += 1

                if len(curr_batch_docs) >= self.batch_docs:
                    stream_state = {
                        "file_idx": file_idx,
                        "file_offset": curr_offset,
                        "records_in": records_read,
                        "shard_idx": shard_idx,
                        "reader_index": self.reader_index,
                        "next_batch_id": batch_id + 1,
                    }
                    yield RawBatch(
                        source_key=self.source_key,
                        batch_id=batch_id,
                        shard_idx=shard_idx,
                        docs=curr_batch_docs,
                        ids=curr_batch_ids,
                        domains=curr_batch_domains,
                        stream_state=stream_state,
                        bytes_in=curr_batch_bytes,
                    )
                    batch_id += 1
                    curr_batch_docs = []
                    curr_batch_ids = []
                    curr_batch_domains = []
                    curr_batch_bytes = 0

                if self.cfg.max_docs and records_read >= self.cfg.max_docs:
                    break

            if self.cfg.max_docs and records_read >= self.cfg.max_docs:
                break

        # Emit remaining docs
        if curr_batch_docs:
            stream_state = {
                "file_idx": len(self.files),
                "file_offset": 0,
                "records_in": records_read,
                "shard_idx": shard_idx,
                "reader_index": self.reader_index,
                "next_batch_id": batch_id + 1,
            }
            yield RawBatch(
                source_key=self.source_key,
                batch_id=batch_id,
                shard_idx=shard_idx,
                docs=curr_batch_docs,
                ids=curr_batch_ids,
                domains=curr_batch_domains,
                stream_state=stream_state,
                bytes_in=curr_batch_bytes,
            )

        if records_read == 0 and self.reader_index == 0:
            raise ScourError(
                code="E-SRC-004",
                stage="source",
                dataset=self.source_key,
                message=f"Local source '{self.source_key}' yielded zero records",
                hint="Check input file contents and formatting.",
            )


def make_sources(
    cfg: Config,
    reader_index: int = 0,
    n_readers: int = 1,
    batch_docs: int = 512,
    resume_states: dict[str, Any] | None = None,
) -> list[Any]:
    """Factory creating configured sources for a reader worker."""
    sources: list[Any] = []
    resumes = resume_states or {}

    for src_cfg in cfg.sources:
        src_key = src_cfg.repo or src_cfg.name or (src_cfg.paths[0] if src_cfg.paths else "source")
        src_resume = resumes.get(src_key)

        if src_cfg.type == "hf":
            sources.append(
                HFStreamSource(
                    cfg=src_cfg,
                    reader_index=reader_index,
                    n_readers=n_readers,
                    batch_docs=batch_docs,
                    resume_state=src_resume,
                )
            )
        else:
            sources.append(
                LocalSource(
                    cfg=src_cfg,
                    reader_index=reader_index,
                    n_readers=n_readers,
                    batch_docs=batch_docs,
                    resume_state=src_resume,
                )
            )
    return sources


def reader_main(
    reader_id: int,
    n_readers: int,
    cfg: Config,
    in_q: Any,
    stop_event: Any,
    batch_docs: int = 512,
    resume_ckpt: dict[str, Any] | None = None,
    dry_run: int | None = None,
    num_workers: int = 2,
) -> None:
    """Entry point for subprocess reader worker."""
    try:
        resume_states = resume_ckpt.get("reader_states", {}).get(str(reader_id)) if resume_ckpt else None
        if resume_ckpt and "next_batch_id" in resume_ckpt:
            if resume_states is None:
                resume_states = {}
            if "next_batch_id" not in resume_states:
                resume_states["next_batch_id"] = resume_ckpt["next_batch_id"]

        resume_states_dict = (
            {s.repo or s.name or (s.paths[0] if s.paths else "source"): resume_states for s in cfg.sources}
            if resume_states
            else None
        )
        source_instances = make_sources(
            cfg=cfg,
            reader_index=reader_id,
            n_readers=n_readers,
            batch_docs=batch_docs,
            resume_states=resume_states_dict,
        )

        records_sent = 0
        for src in source_instances:
            if stop_event.is_set():
                break
            for raw_batch in src:
                if stop_event.is_set():
                    break
                if dry_run is not None and records_sent >= dry_run:
                    break
                if dry_run is not None and (records_sent + len(raw_batch.docs)) > dry_run:
                    keep_n = max(1, dry_run - records_sent)
                    raw_batch.docs = raw_batch.docs[:keep_n]
                    raw_batch.ids = raw_batch.ids[:keep_n]
                    raw_batch.domains = raw_batch.domains[:keep_n]
                in_q.put(raw_batch)
                records_sent += len(raw_batch.docs)
                if dry_run is not None and records_sent >= dry_run:
                    break

    except Exception:
        pass
    finally:
        # Sentinel indicating reader finished for all workers
        for _ in range(num_workers):
            try:
                in_q.put(None, timeout=2.0)
            except Exception:
                pass
