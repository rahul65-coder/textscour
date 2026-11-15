"""
Unit tests for deduplication and resume: exact hash, MinHash-LSH band collisions,
persistence to disk, and index merging across runs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

import scour_core


def _len(idx: Any) -> int:
    return idx.len() if hasattr(idx, "len") else len(idx)


def test_dedup_exact_match():
    """Verify DedupIndex detects exact hash collision."""
    index = scour_core.DedupIndex()

    exact_hash = 1234567890123456789
    bands = [10, 20, 30, 40, 50, 60, 70, 80]

    # First insert is unique
    verdict1 = index.check_and_insert(exact_hash, bands)
    assert verdict1 is None
    assert _len(index) == 1

    # Second insert with identical exact hash is duplicate_exact
    verdict2 = index.check_and_insert(exact_hash, [1, 2, 3, 4, 5, 6, 7, 8])
    assert verdict2 == "duplicate_exact"
    assert _len(index) == 1


def test_dedup_near_match():
    """Verify DedupIndex detects MinHash band collision (near duplicate)."""
    index = scour_core.DedupIndex()

    exact1 = 1111111111
    bands1 = [100, 200, 300, 400, 500, 600, 700, 800]
    assert index.check_and_insert(exact1, bands1) is None

    # Different exact hash, but 1 band matches (near duplicate)
    exact2 = 2222222222
    bands2 = [999, 998, 300, 996, 995, 994, 993, 992]  # Band 2 matches 300
    verdict = index.check_and_insert(exact2, bands2)
    assert verdict == "duplicate_near"


def test_dedup_persistence_and_reload(tmp_path: Path):
    """Verify saving DedupIndex to disk and reloading it preserves collision detection."""
    index_path = str(tmp_path / "dedup_state.bin")
    index1 = scour_core.DedupIndex()

    docs = [
        (1001, [1, 2, 3, 4, 5, 6, 7, 8]),
        (1002, [11, 12, 13, 14, 15, 16, 17, 18]),
        (1003, [21, 22, 23, 24, 25, 26, 27, 28]),
    ]

    for ex, bnds in docs:
        assert index1.check_and_insert(ex, bnds) is None

    assert _len(index1) == 3
    index1.save(index_path)

    # Reload from disk
    index2 = scour_core.DedupIndex.load(index_path)
    assert _len(index2) == 3

    # Documents previously added should be detected as duplicates
    assert index2.check_and_insert(1001, [1, 2, 3, 4, 5, 6, 7, 8]) == "duplicate_exact"
    assert index2.check_and_insert(9999, [11, 99, 99, 99, 99, 99, 99, 99]) == "duplicate_near"

    # Completely new doc should insert cleanly
    assert index2.check_and_insert(8888, [81, 82, 83, 84, 85, 86, 87, 88]) is None
    assert _len(index2) == 4


def test_dedup_merge_from():
    """Verify merge_from combines sets from two indices."""
    idx_a = scour_core.DedupIndex()
    idx_b = scour_core.DedupIndex()

    idx_a.check_and_insert(100, [1, 1, 1, 1, 1, 1, 1, 1])
    idx_b.check_and_insert(200, [2, 2, 2, 2, 2, 2, 2, 2])

    idx_a.merge_from(idx_b)
    assert _len(idx_a) == 2
    assert idx_a.check_and_insert(200, [9, 9, 9, 9, 9, 9, 9, 9]) == "duplicate_exact"
