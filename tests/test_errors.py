"""
Unit tests for textscour.errors: ScourError formatting, error catalog, secret scrubbing.
"""

from __future__ import annotations

import pytest

from textscour.errors import ERROR_CATALOG, EXIT_CODE_MAP, ScourError, get_exit_code, redact


REQUIRED_ERROR_CODES = [
    "E-CFG-001",
    "E-CFG-002",
    "E-CFG-003",
    "E-CFG-004",
    "E-CFG-005",
    "E-CFG-006",
    "E-CFG-007",
    "E-CFG-008",
    "E-CFG-009",
    "E-SRC-001",
    "E-SRC-002",
    "E-SRC-003",
    "E-SRC-004",
    "E-SCHEMA-001",
    "E-SCHEMA-002",
    "E-SCHEMA-003",
    "E-RUST-001",
    "E-RUST-002",
    "E-RUST-003",
    "E-GPU-001",
    "E-GPU-002",
    "E-GPU-003",
    "E-DISK-001",
    "E-OUT-001",
    "E-CKPT-001",
    "E-UP-001",
    "E-UP-002",
    "E-WORKER-001",
    "E-RATE-001",
]


def test_error_catalog_completeness():
    """Verify that every error code defined in INTERFACES.md is registered."""
    for code in REQUIRED_ERROR_CODES:
        assert code in ERROR_CATALOG, f"Missing error code {code} in ERROR_CATALOG"
        entry = ERROR_CATALOG[code]
        assert "stage" in entry and entry["stage"], f"{code} missing stage"
        assert "description" in entry and entry["description"], f"{code} missing description"
        assert "hint" in entry and entry["hint"], f"{code} missing hint"


def test_secret_redaction():
    """Verify that sensitive secrets are sanitized from messages."""
    raw = (
        "Failed connecting with hf_1234567890abcdef1234567890 and "
        "Bearer eyJhbGciOiJIUzI1NiJ9.test and "
        "kaggle key 0123456789abcdef0123456789abcdef"
    )
    scrubbed = redact(raw)
    assert "hf_1234567890abcdef1234567890" not in scrubbed
    assert "hf_***REDACTED***" in scrubbed
    assert "Bearer ***REDACTED***" in scrubbed
    assert "0123456789abcdef0123456789abcdef" not in scrubbed


def test_scour_error_formatting():
    """Verify multi-line diagnostic formatting of ScourError."""
    cause = RuntimeError("Authentication failed with token hf_secrettoken123456789012345")
    err = ScourError(
        code="E-SRC-002",
        stage="source",
        message="HF authentication failed (401)",
        dataset="org/private_dataset",
        shard=3,
        record_id="doc_42",
        cause=cause,
    )

    formatted = err.format()
    assert "[E-SRC-002]" in formatted
    assert "stage=source" in formatted
    assert "dataset=org/private_dataset" in formatted
    assert "shard=3" in formatted
    assert "record_id=doc_42" in formatted
    assert "Problem :" in formatted
    assert "Where   :" in formatted
    assert "Cause   :" in formatted
    assert "Fix     :" in formatted
    # Secret must be redacted
    assert "hf_secrettoken123456789012345" not in formatted
    assert "hf_***REDACTED***" in formatted


def test_exit_code_mapping():
    """Verify error category mappings to exit codes."""
    assert get_exit_code("E-CFG-001") == EXIT_CODE_MAP["CFG"]
    assert get_exit_code("E-SRC-002") == EXIT_CODE_MAP["SRC"]
    assert get_exit_code("E-SCHEMA-001") == EXIT_CODE_MAP["SCHEMA"]
    assert get_exit_code("E-RUST-001") == EXIT_CODE_MAP["RUST"]
    assert get_exit_code("E-GPU-001") == EXIT_CODE_MAP["GPU"]
    assert get_exit_code("E-UNKNOWN-999") == 1
