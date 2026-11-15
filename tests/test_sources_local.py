"""
Unit tests for textscour.sources: Local file streaming (JSONL, JSON, Parquet),
batching, state tracking, resuming, and error conditions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from textscour.config import SourceCfg, TextSpec
from textscour.errors import ScourError
from textscour.sources import LocalSource, RawBatch


def test_local_source_jsonl_reading(corpus_dir: Path):
    """Verify LocalSource streams JSONL files into RawBatch chunks."""
    clean_jsonl = corpus_dir / "sample_clean.jsonl"
    assert clean_jsonl.is_file()

    cfg = SourceCfg(
        type="jsonl",
        paths=[str(clean_jsonl)],
        text=TextSpec(mode="field", field="text"),
    )

    source = LocalSource(cfg, reader_index=0, n_readers=1, batch_docs=20)
    batches = list(source)

    assert len(batches) > 0
    total_docs = sum(len(b.docs) for b in batches)
    assert total_docs == 60

    first_batch = batches[0]
    assert isinstance(first_batch, RawBatch)
    assert len(first_batch.docs) == 20
    assert len(first_batch.ids) == 20
    assert len(first_batch.domains) == 20
    assert first_batch.stream_state is not None


def test_local_source_json_array_reading(corpus_dir: Path):
    """Verify LocalSource reads JSON array files."""
    json_path = corpus_dir / "sample_records.json"
    assert json_path.is_file()

    cfg = SourceCfg(
        type="json",
        paths=[str(json_path)],
        text=TextSpec(mode="field", field="text"),
    )

    source = LocalSource(cfg, reader_index=0, n_readers=1, batch_docs=15)
    batches = list(source)

    assert len(batches) > 0
    total_docs = sum(len(b.docs) for b in batches)
    assert total_docs == 50


def test_local_source_parquet_reading(corpus_dir: Path):
    """Verify LocalSource reads Parquet tables."""
    parquet_path = corpus_dir / "sample.parquet"
    if not parquet_path.is_file():
        pytest.skip("Parquet sample file not generated (pyarrow not present)")

    cfg = SourceCfg(
        type="parquet",
        paths=[str(parquet_path)],
        text=TextSpec(mode="field", field="text"),
    )

    source = LocalSource(cfg, reader_index=0, n_readers=1, batch_docs=25)
    batches = list(source)

    assert len(batches) > 0
    total_docs = sum(len(b.docs) for b in batches)
    assert total_docs == 60


def test_local_source_resume_offset(corpus_dir: Path):
    """Verify resuming skips already processed records based on resume_state."""
    clean_jsonl = corpus_dir / "sample_clean.jsonl"

    cfg = SourceCfg(
        type="jsonl",
        paths=[str(clean_jsonl)],
        text=TextSpec(mode="field", field="text"),
    )

    resume_state = {"file_offset": 35, "records_in": 35}
    source = LocalSource(
        cfg,
        reader_index=0,
        n_readers=1,
        batch_docs=20,
        resume_state=resume_state,
    )
    batches = list(source)

    total_docs = sum(len(b.docs) for b in batches)
    assert total_docs == 25  # 60 total - 35 skipped = 25 remaining


def test_local_source_max_docs_limit(corpus_dir: Path):
    """Verify max_docs stops iteration early."""
    clean_jsonl = corpus_dir / "sample_clean.jsonl"

    cfg = SourceCfg(
        type="jsonl",
        paths=[str(clean_jsonl)],
        text=TextSpec(mode="field", field="text"),
        max_docs=17,
    )

    source = LocalSource(cfg, batch_docs=10)
    batches = list(source)
    total_docs = sum(len(b.docs) for b in batches)
    assert total_docs == 17


def test_local_source_missing_file_raises_error():
    """Verify nonexistent path raises E-SRC-001."""
    cfg = SourceCfg(
        type="jsonl",
        paths=["/nonexistent/path/data_xyz.jsonl"],
    )
    with pytest.raises(ScourError) as exc_info:
        LocalSource(cfg)
    assert exc_info.value.code == "E-SRC-001"


def test_local_source_empty_file_raises_error(tmp_path: Path):
    """Verify empty file raises E-SRC-004."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.touch()

    cfg = SourceCfg(
        type="jsonl",
        paths=[str(empty_file)],
    )
    source = LocalSource(cfg)
    with pytest.raises(ScourError) as exc_info:
        list(source)
    assert exc_info.value.code == "E-SRC-004"
