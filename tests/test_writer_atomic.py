"""
Unit tests for textscour.writer: Atomic shard writing (.part -> rename),
5-field clean schema, garbage/fragment schemas, manifest tracking, quarantine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from textscour.errors import ScourError
from textscour.writer import (
    AtomicShard,
    Manifest,
    QuarantineWriter,
)


def test_atomic_shard_lifecycle(tmp_path: Path):
    """Verify AtomicShard writes to .part and atomically renames on commit."""
    final_file = tmp_path / "shard_00000.jsonl"
    part_file = Path(str(final_file) + ".part")

    shard = AtomicShard(final_path=final_file, fmt="jsonl")
    assert part_file.is_file()
    assert not final_file.is_file()

    rec1 = {"id": "doc_1", "text": "Clean document text.", "domain": "wiki", "length": 20, "score": 8.0}
    rec2 = {"id": "doc_2", "text": "Another document.", "domain": "wiki", "length": 17, "score": 7.5}

    shard.write_record(rec1)
    shard.write_record(rec2)

    assert shard.records_written == 2
    assert shard.bytes_written > 0

    committed_path = shard.commit()
    assert committed_path == final_file
    assert final_file.is_file()
    assert not part_file.is_file()

    with open(final_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2
    assert lines[0]["id"] == "doc_1"


def test_clean_schema_exact_five_fields(tmp_path: Path):
    """Verify clean record format strictly contains the 5 required fields."""
    final_file = tmp_path / "clean_shard.jsonl"
    shard = AtomicShard(final_path=final_file, fmt="jsonl")

    clean_rec = {
        "id": "clean_42",
        "domain": "news",
        "text": "Cleaned article text.",
        "length": 21,
        "score": 9.2,
    }
    shard.write_record(clean_rec)
    shard.commit()

    with open(final_file, "r", encoding="utf-8") as f:
        loaded = json.loads(f.readline())

    assert set(loaded.keys()) == {"id", "domain", "text", "length", "score"}
    assert isinstance(loaded["id"], str)
    assert isinstance(loaded["domain"], str)
    assert isinstance(loaded["text"], str)
    assert isinstance(loaded["length"], int)
    assert isinstance(loaded["score"], (int, float))


def test_garbage_schema_fields(tmp_path: Path):
    """Verify garbage record format includes reasons and kind."""
    final_file = tmp_path / "garbage_shard.jsonl"
    shard = AtomicShard(final_path=final_file, fmt="jsonl")

    garbage_rec = {
        "id": "garbage_101",
        "domain": "web",
        "text": "Spam promotional link text here.",
        "length": 33,
        "score": 2.5,
        "reasons": ["promo_phrase", "url_spam"],
        "kind": "doc",
    }
    shard.write_record(garbage_rec)
    shard.commit()

    with open(final_file, "r", encoding="utf-8") as f:
        loaded = json.loads(f.readline())

    assert loaded["kind"] == "doc"
    assert loaded["reasons"] == ["promo_phrase", "url_spam"]


def test_manifest_tracking(tmp_path: Path):
    """Verify Manifest tracks shard entries, hashes, and uploaded state."""
    manifest = Manifest(output_dir=tmp_path)
    sample_shard = tmp_path / "clean_00001.jsonl"
    with open(sample_shard, "w", encoding="utf-8") as f:
        f.write('{"id":"1"}\n')

    manifest.add_shard(sample_shard, kind="clean", rows=1, bytes_size=sample_shard.stat().st_size)
    assert len(manifest.entries) == 1
    assert manifest.entries[0]["rows"] == 1
    assert manifest.entries[0]["uploaded"] is False
    assert manifest.entries[0]["sha256"] != ""

    # Reload from disk
    manifest2 = Manifest(output_dir=tmp_path)
    assert len(manifest2.entries) == 1

    # Mark uploaded
    manifest2.mark_uploaded(manifest.entries[0]["full_path"])
    assert manifest2.entries[0]["uploaded"] is True


def test_quarantine_writer(tmp_path: Path):
    """Verify QuarantineWriter isolates records and flags error budget violations."""
    qw = QuarantineWriter(output_dir=tmp_path, max_error_rate=0.05)

    qw.quarantine(
        source="src_test",
        batch_id=1,
        record_id="rec_99",
        code="E-SCHEMA-002",
        message="Text field was int instead of string",
    )

    assert qw.error_count == 1
    assert qw.path.is_file()

    with open(qw.path, "r", encoding="utf-8") as f:
        entry = json.loads(f.readline())
    assert entry["record_id"] == "rec_99"
    assert entry["code"] == "E-SCHEMA-002"

    # Rate check: 1 error in 1000 docs is 0.001 <= 0.05 budget (OK)
    qw.check_error_rate(total_seen=1000)

    # Rate check: 10 errors in 150 docs is ~0.066 > 0.05 budget (raises E-RATE-001)
    qw.error_count = 10
    with pytest.raises(ScourError) as exc_info:
        qw.check_error_rate(total_seen=150)
    assert exc_info.value.code == "E-RATE-001"


def test_sanitize_source_key_forbidden_characters():
    """Verify source key sanitization strips forbidden chars and generates strictly safe slugs."""
    import re
    from textscour.writer import sanitize_source_key

    # Keys containing wildcards and forbidden characters
    bad_keys = [
        "tests/data/*.jsonl",
        'source?name:"bad"<tag>|pipe',
        "https://domain.com/repo/name:v1.0",
        "sample*clean?garbage",
    ]

    for k in bad_keys:
        slug = sanitize_source_key(k)
        # Must match [A-Za-z0-9._-]
        assert re.fullmatch(r"[A-Za-z0-9._-]+", slug) is not None
        # Zero forbidden characters
        for forbidden in '*?:"<>|/\\':
            assert forbidden not in slug


def test_part_cleanup_on_flush(tmp_path: Path):
    """Verify clean exit leaves zero .part files in output directory."""
    from textscour.checkpoint import Checkpoint
    from textscour.config import Config, OutputCfg, SourceCfg
    from textscour.writer import OrderedSourceWriter

    cfg = Config(
        sources=[SourceCfg(type="jsonl", paths=["test.jsonl"])],
        output=OutputCfg(dir=str(tmp_path), format="jsonl", shard_mb=10),
    )
    ckpt = Checkpoint(state_dir=tmp_path / "state")

    writer = OrderedSourceWriter(source_key="test/source/*.jsonl", cfg=cfg, ckpt=ckpt)
    writer.flush()

    # Assert zero .part files remain after flush
    part_files = list(tmp_path.glob("*.part"))
    assert len(part_files) == 0

