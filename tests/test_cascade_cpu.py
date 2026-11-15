"""
Unit tests for textscour.cascade: CPU rules-only stage 1 processing,
batch execution, schema quarantine routing, and result batch assembly.
"""

from __future__ import annotations

import queue
from pathlib import Path

import pytest

from textscour.cascade import ResultBatch, get_engine, process_batch
from textscour.config import CascadeCfg, Config, SourceCfg, TextSpec
from textscour.schema import compile_extractor
from textscour.sources import RawBatch


def test_cascade_process_batch_rules_only():
    """Verify process_batch executes Stage 1 on documents in CPU/rules-only mode."""
    cfg = Config(
        sources=[SourceCfg(type="jsonl", paths=["test.jsonl"])],
        cascade=CascadeCfg(
            gpu_mode="off",
            min_doc_chars=100,
            clean_threshold=6.0,
        ),
    )

    clean_text = (
        "Supervised machine learning algorithms build mathematical models based on "
        "training datasets. In physics, energy is a conserved scalar quantity that "
        "cannot be created or destroyed, only transformed between states."
    )
    noisy_text = "Click here to buy now and get 50% discount! Limited time offer!"

    raw_batch = RawBatch(
        source_key="local_source",
        batch_id=1,
        shard_idx=0,
        docs=[
            {"text": clean_text, "domain": "wiki"},
            {"text": noisy_text, "domain": "spam"},
        ],
        ids=["clean_001", "spam_002"],
        domains=["wiki", "spam"],
        stream_state={"offset": 2},
    )

    engine = get_engine(cfg)
    extractor = compile_extractor(TextSpec(mode="field", field="text"))

    result = process_batch(
        worker_id=0,
        raw_batch=raw_batch,
        engine=engine,
        extractor=extractor,
        cfg=cfg,
    )

    assert isinstance(result, ResultBatch)
    assert result.batch_id == 1
    assert result.source_key == "local_source"
    assert len(result.records) == 2

    # Check clean record
    rec_clean = result.records[0]
    assert rec_clean.clean is True
    assert rec_clean.score >= 6.0
    assert rec_clean.length >= 100

    # Check noisy record
    rec_spam = result.records[1]
    assert rec_spam.clean is False
    assert len(rec_spam.reasons) > 0


def test_cascade_quarantine_bad_schema_records():
    """Verify non-string or None text fields are routed to quarantine queue."""
    cfg = Config(
        sources=[SourceCfg(type="jsonl", paths=["test.jsonl"])],
        cascade=CascadeCfg(gpu_mode="off"),
    )

    raw_batch = RawBatch(
        source_key="bad_schema_source",
        batch_id=2,
        shard_idx=0,
        docs=[
            {"text": None},  # Missing / None text -> E-SCHEMA-001
            {"text": 12345},  # Non-string text -> E-SCHEMA-002
            {"text": "Valid document text exceeding minimum character length requirements."},
        ],
        ids=["rec_none", "rec_int", "rec_valid"],
        domains=["dom", "dom", "dom"],
    )

    quarantine_q = queue.Queue()
    engine = get_engine(cfg)
    extractor = compile_extractor(TextSpec(mode="field", field="text"))

    result = process_batch(
        worker_id=0,
        raw_batch=raw_batch,
        engine=engine,
        extractor=extractor,
        cfg=cfg,
        quarantine_q=quarantine_q,
    )

    assert len(result.records) == 1
    assert quarantine_q.qsize() == 2

    q1 = quarantine_q.get()
    assert q1[2] == "rec_none"
    assert q1[3] == "E-SCHEMA-001"

    q2 = quarantine_q.get()
    assert q2[2] == "rec_int"
    assert q2[3] == "E-SCHEMA-002"
