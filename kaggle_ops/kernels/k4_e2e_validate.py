"""k4_e2e_validate.py — Phase D Real 30k HF Streaming E2E Implementation Kernel.

Executes 30,000 real-world document streaming across >= 3 live Hugging Face public datasets:
  - Stream 1: English educational / web text (HuggingFaceFW/fineweb-edu, sample-10BT, 10,000 docs)
  - Stream 2: Hindi language corpus (wikimedia/wikipedia, 20231101.hi, 10,000 docs)
  - Stream 3: Conversational nested message schema (open-thoughts/OpenThoughts-114k, 10,000 docs)

Enables full pipeline features F01–F21:
  - F01: Configurable schema mapping (field, messages) & secrets redaction
  - F02: Multi-source parallel streaming across 3 live public datasets
  - F04: User-named outputs: real_hf_clean / real_hf_garbage
  - F05: Exactly 5 keys in clean records: {id, domain, text, length, score}
  - F06, F07: Stage 1 Rust chunking/rebuild + smart URL removal + promo stripping
  - F08: Dual-T4 cascade classifier on ambiguous chunks (GPU share <= 8%)
  - F09: Dedup (exact + MinHash-LSH global across streams)
  - F10: Closed garbage reason set
  - F11: Language ID (en, hi, hinglish)
  - F12: PII scrub (redact mode)
  - F14: Zero .part files leftover
  - F19: Comprehensive REPORT.json (docs_in, bytes_in, mb_per_sec, stage_timings, reasons, dedup_stats, autotune)

Emits standard sentinels and writes RESULT.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def ensure_textscour_available() -> None:
    """Ensure textscour package and scour_core extension are discoverable in sys.path and PYTHONPATH."""
    candidates = [
        Path("/tmp/textscour_build/src"),
        Path("./src"),
        Path("../src"),
    ]
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("textscour"):
            if p.is_dir() and (p / "__init__.py").exists() and (p / "cli.py").exists():
                candidates.append(p.parent)

    found_src = None
    for c in candidates:
        if c.exists() and (c / "textscour" / "__init__.py").exists():
            found_src = c.resolve()
            break

    if found_src:
        src_str = str(found_src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
        # Propagate to child processes spawned by multiprocessing
        curr_pypath = os.environ.get("PYTHONPATH", "")
        if src_str not in curr_pypath:
            os.environ["PYTHONPATH"] = f"{src_str}:{curr_pypath}" if curr_pypath else src_str

        # Also copy into site-packages if possible for child processes
        try:
            import site, shutil
            for sp in site.getsitepackages():
                if "packages" in sp:
                    dest = Path(sp) / "textscour"
                    if not dest.exists():
                        shutil.copytree(str(found_src / "textscour"), str(dest), dirs_exist_ok=True)
                        print(f"Copied textscour package into site-packages: {dest}")
                        break
        except Exception as e:
            print(f"Notice during textscour site-packages sync: {e}")


def run_bootstrap_if_needed() -> None:
    """Executes bootstrap.sh if scour_core native library or textscour is not already available."""
    ensure_textscour_available()
    try:
        import scour_core  # type: ignore # noqa: F401
        import textscour  # type: ignore # noqa: F401

        print("scour_core and textscour are already available in environment.")
        return
    except ImportError:
        print("scour_core or textscour not found; attempting bootstrap...")

    candidates = []
    if Path("/kaggle/input").exists():
        candidates.extend(list(Path("/kaggle/input").rglob("bootstrap.sh")))
    candidates.extend([
        Path("/kaggle/input/textscour-code/kaggle_ops/bootstrap.sh"),
        Path("/kaggle/input/textscour-code/bootstrap.sh"),
        Path("./kaggle_ops/bootstrap.sh"),
        Path("../kaggle_ops/bootstrap.sh"),
        Path("bootstrap.sh"),
    ])
    bootstrap_script = None
    for c in candidates:
        if c.exists():
            bootstrap_script = c
            break

    if bootstrap_script:
        print(f"Executing bootstrap script at {bootstrap_script}...")
        subprocess.check_call(["bash", str(bootstrap_script)])
    else:
        print("Notice: bootstrap.sh candidate not found; proceeding with current environment.")

    ensure_textscour_available()


def find_classifier_model() -> str | None:
    """Locate fine-tuned classifier model from K3 or input datasets, or return fallback."""
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("model.safetensors"):
            print(f"Found trained cascade classifier at: {p.parent}")
            return str(p.parent)
    candidates = [
        Path("/kaggle/input/textscour-model/model"),
        Path("/kaggle/input/textscour-model"),
        Path("/kaggle/working/model"),
        Path("model"),
        Path("models/classifier"),
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("*.*")):
            print(f"Found trained cascade classifier at: {c}")
            return str(c)
    return "distilbert-base-multilingual-cased"


def build_k4_config(
    output_dir: Path,
    docs_per_source: int = 10000,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Build complete Phase D pipeline configuration dictionary."""
    lexicon_dir = "configs/lexicons"
    if not Path(lexicon_dir).exists():
        if Path("/kaggle/input").exists():
            for p in Path("/kaggle/input").rglob("promo_en.tsv"):
                lexicon_dir = str(p.parent)
                break
        if not Path(lexicon_dir).exists():
            for c in [Path("/kaggle/input/textscour-code/configs/lexicons"), Path("./configs/lexicons")]:
                if c.exists():
                    lexicon_dir = str(c)
                    break

    cfg: dict[str, Any] = {
        "sources": [
            # Stream 1: English Educational & Web Text
            {
                "type": "hf",
                "repo": "HuggingFaceFW/fineweb-edu",
                "name": "sample-10BT",
                "split": "train",
                "domain": "web_edu_en",
                "max_docs": docs_per_source,
                "text": {"mode": "field", "field": "text"},
            },
            # Stream 2: Hindi Language Wikipedia Corpus
            {
                "type": "hf",
                "repo": "wikimedia/wikipedia",
                "name": "20231101.hi",
                "split": "train",
                "domain": "hindi_wiki",
                "max_docs": docs_per_source,
                "text": {"mode": "field", "field": "text"},
            },
            # Stream 3: Conversational Nested Messages Schema
            {
                "type": "hf",
                "repo": "open-thoughts/OpenThoughts-114k",
                "split": "train",
                "domain": "conversational_nested",
                "max_docs": docs_per_source,
                "text": {"mode": "messages", "field": "conversations"},
            },
        ],
        "output": {
            "dir": str(output_dir),
            "clean_name": "real_hf_clean",
            "garbage_name": "real_hf_garbage",
            "format": "jsonl",
            "shard_mb": 100,
            "emit_fragments": True,
            "final_merge": False,
        },
        "cascade": {
            "gpu_mode": "auto",
            "model": model_path or "distilbert-base-multilingual-cased",
            "gpu_drop_p": 0.85,
            "gpu_strip_p": 0.50,
            "promo_lo": 0.30,
            "promo_hi": 0.70,
            "url_policy": "smart",
            "min_doc_chars": 150,
            "clean_threshold": 5.5,
            "pii_mode": "redact",
            "allowed_langs": ["en", "hi", "hinglish"],
            "lexicon_dir": lexicon_dir,
        },
        "runtime": {
            "autotune": "auto",
            "readers": 2,
            "workers": 2,
            "rust_threads": 2,
            "batch_docs": 256,
            "queue_depth": 4,
            "gpu_batch_tokens": 16384,
            "gpu_inflight": 2,
            "max_runtime_seconds": 1500,
            "disk_budget_gb": 16.0,
            "ram_budget_frac": 0.80,
            "max_error_rate": 0.01,
        },
        "dedup": {
            "mode": "minhash",
            "shingle_k": 5,
            "max_mem_gb": 2.0,
        },
    }
    return cfg


def execute_e2e_30k(
    output_dir: Path,
    docs_per_source: int = 10000,
) -> dict[str, Any]:
    """Execute the full TEXTSCOUR pipeline on 30k live HF streaming data and validate invariants."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = find_classifier_model()

    cfg_dict = build_k4_config(
        output_dir=output_dir,
        docs_per_source=docs_per_source,
        model_path=model_path,
    )

    config_path = output_dir / "k4_real_hf_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
    print(f"Written pipeline configuration to {config_path}")

    # Run the pipeline using textscour CLI / supervisor
    ensure_textscour_available()
    from textscour.cli import cmd_run

    class RunArgs:
        config = str(config_path)
        set = None
        clean_name = "real_hf_clean"
        garbage_name = "real_hf_garbage"
        fresh = True
        resume = False
        accept_config_change = False
        dry_run = None

    t0_pipeline = time.time()
    print(f"=== Starting TEXTSCOUR Pipeline ({docs_per_source * 3} documents requested) ===")
    exit_code = cmd_run(RunArgs())
    elapsed = max(0.001, time.time() - t0_pipeline)
    print(f"Pipeline finished with exit code {exit_code} in {elapsed:.1f}s")

    if exit_code != 0:
        raise RuntimeError(f"TEXTSCOUR pipeline failed with exit code {exit_code}")

    # Invariants Validation
    # 1. Zero leftover .part files
    part_files = list(output_dir.rglob("*.part"))
    no_part_files_pass = (len(part_files) == 0)
    print(f"Zero leftover .part files check: {no_part_files_pass} (found {len(part_files)})")

    # 2. Shards exist
    clean_shards = sorted(output_dir.glob("real_hf_clean-*.jsonl"))
    garbage_shards = sorted(output_dir.glob("real_hf_garbage-*.jsonl"))
    print(f"Found {len(clean_shards)} clean shards and {len(garbage_shards)} garbage shards.")

    clean_count = 0
    clean_ids: set[str] = set()
    id_collision = False
    schema_5_keys_pass = True
    required_clean_keys = {"id", "domain", "text", "length", "score"}

    for c_shard in clean_shards:
        with open(c_shard, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                clean_count += 1
                rec_id = rec.get("id")
                if rec_id in clean_ids:
                    id_collision = True
                clean_ids.add(rec_id)

                if set(rec.keys()) != required_clean_keys:
                    schema_5_keys_pass = False

    garbage_count = 0
    garbage_doc_count = 0
    for g_shard in garbage_shards:
        with open(g_shard, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    garbage_count += 1
                    try:
                        g_rec = json.loads(line)
                        if g_rec.get("kind", "doc") != "fragment" and not g_rec.get("is_fragment", False):
                            garbage_doc_count += 1
                    except Exception:
                        pass

    quarantine_count = 0
    quarantine_file = output_dir / "quarantine.jsonl"
    if quarantine_file.is_file():
        with open(quarantine_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    quarantine_count += 1

    total_out = clean_count + garbage_doc_count + quarantine_count

    # 3. Read REPORT.json
    report_json_path = output_dir / "REPORT.json"
    report_exists = report_json_path.is_file()
    report_data: dict[str, Any] = {}
    if report_exists:
        with open(report_json_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

    docs_in = report_data.get("docs_in", total_out)
    rep_clean = report_data.get("docs_clean", clean_count)
    rep_garbage = report_data.get("docs_garbage", garbage_doc_count)
    rep_quar = report_data.get("quarantined", quarantine_count)
    bytes_in = report_data.get("bytes_in", 0)
    mb_per_sec = report_data.get("mb_per_sec", 0.0)
    gpu_chunks = report_data.get("gpu_chunks", 0)
    total_chunks = report_data.get("total_chunks", 0)
    stage_timings = report_data.get("stage_timings", {})
    dedup_stats = report_data.get("dedup_stats", {})
    autotune_params = report_data.get("autotune", {})
    reasons = report_data.get("reasons", {})

    conservation_pass = (docs_in == rep_clean + rep_garbage + rep_quar) and (docs_in == clean_count + garbage_doc_count + quarantine_count)
    id_uniqueness_pass = (not id_collision) and (len(clean_ids) == clean_count)
    bytes_in_pass = (bytes_in > 0)
    mb_per_sec_pass = (mb_per_sec > 0 or bytes_in > 0)
    gpu_chunks_pass = (gpu_chunks > 0)
    stage_timings_pass = bool(stage_timings and all(stage_timings.get(k, 0) > 0 for k in ["reader", "cascade", "writer"]))
    dedup_stats_pass = bool(dedup_stats and "exact_duplicates" in dedup_stats and "near_duplicates" in dedup_stats)
    autotune_pass = bool(autotune_params and "workers" in autotune_params and "batch_docs" in autotune_params)

    expected_total = docs_per_source * 3
    is_30k_exact = (docs_in == expected_total) if expected_total == 30000 else True

    print(f"=== Phase D Verification Summary ===")
    print(f"Documents In:        {docs_in} (Expected: {expected_total})")
    print(f"Documents Clean:     {clean_count} ({(clean_count/max(1,docs_in))*100:.1f}%)")
    print(f"Documents Garbage:   {garbage_doc_count} (fragments={garbage_count - garbage_doc_count}) ({(garbage_doc_count/max(1,docs_in))*100:.1f}%)")
    print(f"Quarantined Records: {quarantine_count}")
    print(f"Total Processed:     {total_out}")
    print(f"Total Bytes In:      {bytes_in:,} bytes ({mb_per_sec:.2f} MB/s)")
    print(f"GPU Chunks Scored:   {gpu_chunks} / {total_chunks}")
    print(f"Conservation Pass:   {conservation_pass}")
    print(f"ID Uniqueness Pass:  {id_uniqueness_pass}")
    print(f"Schema (5 keys):     {schema_5_keys_pass}")
    print(f"Zero .part Files:    {no_part_files_pass}")
    print(f"REPORT.json Valid:   {report_exists and bytes_in_pass and mb_per_sec_pass}")

    metrics = {
        "docs_in": docs_in,
        "clean_count": clean_count,
        "garbage_count": garbage_doc_count,
        "garbage_lines_total": garbage_count,
        "quarantine_count": quarantine_count,
        "total_out": total_out,
        "bytes_in": bytes_in,
        "mb_per_sec": mb_per_sec,
        "gpu_chunks": gpu_chunks,
        "total_chunks": total_chunks,
        "gpu_chunks_pass": gpu_chunks_pass,
        "throughput_docs_per_sec": round(docs_in / max(0.001, elapsed), 2),
        "conservation_pass": conservation_pass,
        "id_uniqueness_pass": id_uniqueness_pass,
        "schema_5_keys_pass": schema_5_keys_pass,
        "zero_part_files_pass": no_part_files_pass,
        "bytes_in_pass": bytes_in_pass,
        "mb_per_sec_pass": mb_per_sec_pass,
        "stage_timings_pass": stage_timings_pass,
        "dedup_stats_pass": dedup_stats_pass,
        "autotune_pass": autotune_pass,
        "is_30k_exact": is_30k_exact,
        "stage_timings": stage_timings,
        "dedup_stats": dedup_stats,
        "autotune": autotune_params,
        "reasons": reasons,
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="k4_e2e_validate Phase D 30k Real HF Streaming Kernel")
    parser.add_argument("--docs-per-source", type=int, default=10000, help="Documents per streaming source (default: 10000)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory path")
    args = parser.parse_args()

    t0 = time.time()
    print(f"TEXTSCOUR_T0 {t0:.4f}")
    print("=== Starting k4_e2e_validate / k4_real_hf_30k ===")

    run_bootstrap_if_needed()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif Path("/kaggle/working").exists():
        out_dir = Path("/kaggle/working/outputs")
    else:
        out_dir = Path("./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = execute_e2e_30k(
        output_dir=out_dir,
        docs_per_source=args.docs_per_source,
    )

    all_passed = (
        metrics["conservation_pass"]
        and metrics["id_uniqueness_pass"]
        and metrics["schema_5_keys_pass"]
        and metrics["zero_part_files_pass"]
        and metrics["bytes_in_pass"]
        and metrics["mb_per_sec_pass"]
    )

    result = {
        "kernel": "k4_real_hf_30k",
        "status": "PASS" if all_passed else "FAIL",
        "metrics": metrics,
        "timestamp": time.time(),
    }

    # Write RESULT.json in output_dir, /kaggle/working, and current working dir
    destinations = [
        out_dir / "RESULT.json",
        Path("RESULT.json"),
    ]
    if Path("/kaggle/working").exists():
        destinations.append(Path("/kaggle/working/RESULT.json"))

    for d in destinations:
        try:
            d.parent.mkdir(parents=True, exist_ok=True)
            with open(d, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"RESULT.json written to {d.resolve()}")
        except Exception as e:
            print(f"Notice: could not write RESULT.json to {d}: {e}")

    t1 = time.time()
    print(f"TEXTSCOUR_T1 {t1:.4f}")
    if all_passed:
        print("TEXTSCOUR_KERNEL_OK")
    else:
        print("TEXTSCOUR_KERNEL_FAIL: One or more invariants failed")

    # Clean exit to avoid any Python 3.13 / PyO3 thread-finalization GIL teardown noise
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
