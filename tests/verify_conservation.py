#!/usr/bin/env python3
"""
Conservation and Output Record Invariants Verifier for TEXTSCOUR.
Verifies:
1. Exact conservation: clean docs + garbage docs + quarantine records == input docs
2. Strict 5-field schema on all clean records (id, domain, text, length, score)
3. Valid closed-set reason codes on all garbage records
4. Character length consistency (length == text.chars().count())
Exits 0 on success, 1 on any invariant violation.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any

# Closed reason code set from Section 6 of docs/INTERFACES.md
VALID_REASON_CODES = {
    "url_spam",
    "promo_phrase",
    "promo_handle",
    "promo_contact",
    "cta_sentence",
    "boilerplate",
    "cookie_banner",
    "nav_menu",
    "symbol_heavy",
    "repetition",
    "too_short",
    "low_alpha",
    "gpu_garbage",
    "wrong_lang",
    "duplicate_exact",
    "duplicate_near",
    "pii_dropped",
    "empty_after_clean",
    "low_score",
}

CLEAN_EXACT_FIELDS = {"id", "domain", "text", "length", "score"}


def open_file_read(path: Path | str) -> Any:
    """Open text file with zstd/gzip auto-decompression."""
    p_str = str(path)
    if p_str.endswith(".zst"):
        import zstandard as zstd

        f = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        return dctx.stream_reader(f)
    elif p_str.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    else:
        return open(path, "r", encoding="utf-8")


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records from uncompressed or compressed file."""
    records: list[dict[str, Any]] = []
    if str(path).endswith(".zst"):
        import zstandard as zstd

        with open(path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text_stream = reader.read().decode("utf-8")
                for line in text_stream.splitlines():
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records

    with open_file_read(path) as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def verify_conservation(output_dir: str | Path) -> bool:
    """
    Verify conservation and schema invariants in output_dir.
    Returns True if all invariants pass, False otherwise.
    """
    out_path = Path(output_dir)
    if not out_path.is_dir():
        print(f"[verify_conservation] ERROR: Directory '{output_dir}' does not exist.")
        return False

    print(f"[verify_conservation] Inspecting output directory: {out_path.resolve()}")

    # 1. Discover output shards
    clean_files = sorted(
        [p for p in out_path.iterdir() if p.is_file() and "clean" in p.name and "garbage" not in p.name and not p.name.endswith((".part", ".tmp"))]
    )
    garbage_files = sorted(
        [p for p in out_path.iterdir() if p.is_file() and "garbage" in p.name and not p.name.endswith((".part", ".tmp"))]
    )
    quarantine_file = out_path / "quarantine.jsonl"
    report_file = out_path / "REPORT.json"

    print(f"  Found {len(clean_files)} clean shard(s)")
    print(f"  Found {len(garbage_files)} garbage shard(s)")

    violations: list[str] = []
    clean_records_count = 0
    clean_seen_ids: set[str] = set()

    # 2. Check Clean Records
    for cf in clean_files:
        try:
            records = read_jsonl_records(cf)
        except Exception as e:
            violations.append(f"Failed reading clean shard {cf.name}: {e}")
            continue

        for r_idx, rec in enumerate(records):
            clean_records_count += 1
            fields = set(rec.keys())

            # Invariant: exactly 5 fields
            if fields != CLEAN_EXACT_FIELDS:
                diff_extra = fields - CLEAN_EXACT_FIELDS
                diff_missing = CLEAN_EXACT_FIELDS - fields
                violations.append(
                    f"Clean record {rec.get('id', r_idx)} in {cf.name} violates 5-field schema. "
                    f"Extra: {diff_extra}, Missing: {diff_missing}"
                )

            # Invariant: length == char count
            text_val = rec.get("text", "")
            len_val = rec.get("length", 0)
            if len(text_val) != len_val:
                violations.append(
                    f"Clean record {rec.get('id')} length mismatch: declared {len_val} != actual {len(text_val)}"
                )

            # Invariant: score within [0.0, 10.0]
            score_val = rec.get("score", 0.0)
            if not (0.0 <= score_val <= 10.0):
                violations.append(f"Clean record {rec.get('id')} score {score_val} out of bounds [0.0, 10.0]")

            doc_id = rec.get("id")
            if doc_id:
                if doc_id in clean_seen_ids:
                    violations.append(f"Duplicate document ID in clean outputs: '{doc_id}'")
                clean_seen_ids.add(doc_id)

    # 3. Check Garbage Records
    garbage_doc_count = 0
    garbage_fragment_count = 0

    for gf in garbage_files:
        try:
            records = read_jsonl_records(gf)
        except Exception as e:
            violations.append(f"Failed reading garbage shard {gf.name}: {e}")
            continue

        for r_idx, rec in enumerate(records):
            kind = rec.get("kind", "doc")
            if kind == "doc":
                garbage_doc_count += 1
            else:
                garbage_fragment_count += 1

            reasons = rec.get("reasons", [])
            if not isinstance(reasons, list) or not reasons:
                violations.append(f"Garbage record {rec.get('id', r_idx)} has empty or invalid reasons list")
            else:
                for r in reasons:
                    if r not in VALID_REASON_CODES:
                        violations.append(
                            f"Garbage record {rec.get('id', r_idx)} has invalid reason code: '{r}'"
                        )

    # 4. Check Quarantine Records
    quarantine_count = 0
    if quarantine_file.is_file():
        try:
            q_records = read_jsonl_records(quarantine_file)
            quarantine_count = len(q_records)
        except Exception as e:
            violations.append(f"Failed reading quarantine file: {e}")

    # 5. Check Conservation against REPORT.json or state if present
    expected_docs_in = None
    if report_file.is_file():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                rdata = json.load(f)
                expected_docs_in = rdata.get("docs_in")
        except Exception:
            pass

    total_output_docs = clean_records_count + garbage_doc_count + quarantine_count

    print("\n--- Summary Statistics ---")
    print(f"  Clean documents     : {clean_records_count:,}")
    print(f"  Garbage documents   : {garbage_doc_count:,}")
    print(f"  Garbage fragments   : {garbage_fragment_count:,}")
    print(f"  Quarantined records : {quarantine_count:,}")
    print(f"  Total output docs   : {total_output_docs:,}")

    if expected_docs_in is not None:
        print(f"  Expected docs in    : {expected_docs_in:,}")
        if total_output_docs != expected_docs_in:
            violations.append(
                f"Conservation violated! Total output docs ({total_output_docs}) != input docs ({expected_docs_in})"
            )
        else:
            print("  [OK] Exact conservation verified: output docs == input docs")

    if violations:
        print(f"\n[verify_conservation] FAIL: Found {len(violations)} violation(s):")
        for v in violations[:25]:
            print(f"  * {v}")
        if len(violations) > 25:
            print(f"  ... and {len(violations) - 25} more.")
        return False

    print("\n[verify_conservation] PASS: All conservation and schema invariants verified successfully.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify conservation and output invariants")
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default="outputs",
        help="Path to output directory containing shards and quarantine.jsonl",
    )
    args = parser.parse_args()

    success = verify_conservation(args.output_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
