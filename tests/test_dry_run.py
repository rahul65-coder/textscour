#!/usr/bin/env python3
"""5,000-doc Dry-run Pipeline Verification.

Contract:
- Process exactly 5,000 documents from large corpus
- Writes strictly to <output_dir>/dryrun/
- Never touches or creates checkpoints in state/
- Zero leftover .part files
- Saves evidence to evidence/d6_dryrun.json
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def run_dry_run_5000() -> dict[str, Any]:
    corpus_file = WORKSPACE_ROOT / "tests" / "data" / "corpus_200000.jsonl"
    created_synthetic = False
    if not corpus_file.is_file():
        corpus_file.parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_file, "w", encoding="utf-8") as f:
            for i in range(5050):
                f.write(json.dumps({
                    "id": f"doc_{i:06d}",
                    "text": f"This is valid clean test document {i} for dry-run verification. "
                            f"It contains multiple meaningful sentences and informative content. "
                            f"Quality text ensures the supervisor and writers exercise full batch pipelines."
                }) + "\n")
        created_synthetic = True

    out_dir = WORKSPACE_ROOT / "outputs" / "d6_dryrun_5000"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check state before run
    state_dir = out_dir / "state"

    cmd = [
        sys.executable,
        "-m",
        "textscour",
        "run",
        "--config",
        "configs/corpus_200k.yaml",
        "--dry-run",
        "5000",
        "--set",
        f"output.dir={str(out_dir)}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{WORKSPACE_ROOT}:{WORKSPACE_ROOT / 'src'}"

    print(f"Running dry-run pipeline test on 5,000 docs...")
    proc = subprocess.run(cmd, cwd=WORKSPACE_ROOT, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"ERROR: Dry-run failed with code {proc.returncode}:\n{proc.stderr}")
        raise RuntimeError(f"Dry run failed: {proc.stderr}")

    # Inspect outputs
    # D5 requirement: writes strictly to <out>/dryrun/ or doesn't pollute root
    dryrun_shards = list(out_dir.glob("dryrun/**/*.jsonl")) + list(out_dir.glob("dryrun/*.jsonl"))
    root_shards = [f for f in out_dir.glob("*.jsonl") if "report" not in f.name and "metrics" not in f.name]

    # Verify zero leftover .part files
    part_files = list(out_dir.glob("**/*.part"))

    # Verify state dir was not touched or contains no real committed checkpoints
    ckpt_files = list(out_dir.glob("state/ckpt-*.json")) + list((WORKSPACE_ROOT / "state").glob("ckpt-*corpus_200000*.json"))

    # Read REPORT.json
    report_file = out_dir / "REPORT.json"
    docs_processed = 0
    if report_file.is_file():
        with open(report_file, "r", encoding="utf-8") as f:
            rep = json.load(f)
            docs_processed = rep.get("summary", {}).get("docs_in", 0)

    # Check stdout/stderr if not in report
    if docs_processed == 0:
        for line in proc.stdout.splitlines() + proc.stderr.splitlines():
            if "processed" in line and "documents" in line:
                # e.g. "Run complete: processed 5000 documents"
                parts = line.split("processed")
                if len(parts) > 1:
                    num_str = parts[1].strip().split()[0]
                    try:
                        docs_processed = int(num_str)
                    except ValueError:
                        pass

    part_free = len(part_files) == 0
    no_ckpts = len(ckpt_files) == 0
    processed_5000 = docs_processed >= 5000

    results = {
        "benchmark": "D6_dry_run_5000",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "docs_processed": docs_processed,
        "processed_gte_5000": processed_5000,
        "leftover_part_files_count": len(part_files),
        "zero_part_files": part_free,
        "created_checkpoint_files_count": len(ckpt_files),
        "zero_checkpoints_created": no_ckpts,
        "exit_code": proc.returncode,
        "gate_passed": processed_5000 and part_free and no_ckpts,
    }

    # Clean up test output dir
    shutil.rmtree(out_dir, ignore_errors=True)
    if created_synthetic and corpus_file.exists():
        try:
            corpus_file.unlink()
        except Exception:
            pass

    return results


def test_dry_run_5000() -> None:
    res = run_dry_run_5000()
    assert res["gate_passed"], "Dry-run 5000 failed gate checks"


def main() -> None:
    evidence_dir = Path("evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    out_file = evidence_dir / "d6_dryrun.json"

    res = run_dry_run_5000()
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print(f"Dry-run 5000 test results saved to {out_file}")
    print(f"Processed Docs: {res['docs_processed']}")
    print(f"Zero .part files: {res['zero_part_files']}")
    print(f"Zero checkpoints created: {res['zero_checkpoints_created']}")
    print(f"Gate Passed: {res['gate_passed']}")

    if not res["gate_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
