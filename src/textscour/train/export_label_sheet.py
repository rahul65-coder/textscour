"""
Ambiguous Chunk Label Sheet Exporter for TEXTSCOUR.
Exports ambiguous chunks stratified by source and language to CSV for human annotation.
CSV Format: chunk_id,source,text,model_p,label
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
from collections import defaultdict
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def export_label_sheet(
    chunks: Sequence[dict[str, Any]],
    out_csv_path: str,
    n: int = 300,
) -> int:
    """
    Exports up to n ambiguous chunks to CSV, stratified by source/language when available.
    Columns: chunk_id, source, text, model_p, label (empty)

    Returns the count of exported records.
    """
    if not chunks:
        logger.warning("No ambiguous chunks provided for label sheet export.")
        records_to_export = []
    else:
        # Group by (source, lang) for stratification
        strata = defaultdict(list)
        for c in chunks:
            src = str(c.get("source", "default"))
            lang = str(c.get("lang", "en"))
            strata[(src, lang)].append(c)

        total_strata = len(strata)
        target_per_stratum = max(1, n // total_strata)

        selected = []
        for key, items in strata.items():
            selected.extend(items[:target_per_stratum])

        # If we need more to reach n, fill from remaining
        if len(selected) < n:
            seen_ids = {c.get("chunk_id") for c in selected}
            for c in chunks:
                if c.get("chunk_id") not in seen_ids:
                    selected.append(c)
                    seen_ids.add(c.get("chunk_id"))
                    if len(selected) >= n:
                        break

        records_to_export = selected[:n]

    os.makedirs(os.path.dirname(os.path.abspath(out_csv_path)), exist_ok=True)
    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "source", "text", "model_p", "label"])
        for r in records_to_export:
            cid = str(r.get("chunk_id", ""))
            src = str(r.get("source", ""))
            # Sanitize newlines in text for single-line or quoted CSV readability
            raw_text = str(r.get("text", ""))
            text = raw_text.replace("\r", " ").replace("\n", " ").strip()
            model_p = r.get("model_p", 0.5)
            if isinstance(model_p, float):
                model_p = f"{model_p:.4f}"
            writer.writerow([cid, src, text, model_p, ""])

    logger.info("Exported %d ambiguous chunks to %s", len(records_to_export), out_csv_path)
    return len(records_to_export)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ambiguous chunks to label sheet CSV")
    parser.add_argument("--dir", type=str, default="outputs", help="Directory containing pipeline outputs or state")
    parser.add_argument("--out", type=str, default="outputs/label_sheet.csv", help="Destination CSV path")
    parser.add_argument("--n", type=int, default=300, help="Number of records to sample")
    args = parser.parse_args()

    # Search for ambiguous chunks in jsonl files in --dir
    chunks: list[dict[str, Any]] = []
    candidates = glob.glob(os.path.join(args.dir, "**", "*ambiguous*.json*"), recursive=True) + \
                 glob.glob(os.path.join(args.dir, "**", "*quarantine*.json*"), recursive=True)

    for cpath in candidates:
        try:
            with open(cpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunks.append(json.loads(line))
        except Exception as ex:
            logger.debug("Failed reading %s: %s", cpath, ex)

    if not chunks:
        # If no real ambiguous files found, generate synthetic template rows for validation
        logger.info("No recorded ambiguous files found in %s; emitting sample structure.", args.dir)
        for i in range(args.n):
            chunks.append({
                "chunk_id": f"chunk_{i:04d}",
                "source": "hf_sample" if i % 2 == 0 else "local_crawl",
                "lang": "en" if i % 3 == 0 else ("hi" if i % 3 == 1 else "hinglish"),
                "text": f"Ambiguous sample chunk content {i} with potential promo and informational text.",
                "model_p": round(0.40 + (i % 50) * 0.01, 3),
            })

    export_label_sheet(chunks, args.out, n=args.n)


if __name__ == "__main__":
    main()
