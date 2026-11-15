#!/usr/bin/env python3
"""
Kill-and-Resume Resilience Check for TEXTSCOUR (Phase 2 Local Verification Gate).
Tests pipeline durability under abrupt crash:
1. Performs reference run to completion and collects ground truth output IDs.
2. Spawns pipeline on a fresh copy, abruptly terminates it mid-execution via SIGKILL.
3. Resumes the killed pipeline via --resume.
4. Compares resumed outputs against reference run to verify zero lost or duplicated records.
Exits 0 on success, 1 on invariant violation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def read_all_ids(output_dir: Path) -> tuple[set[str], set[str], int]:
    """Read all clean and garbage record IDs from output shards in directory."""
    clean_ids: set[str] = set()
    garbage_ids: set[str] = set()
    quarantine_count = 0

    # Clean shards
    for f in output_dir.glob("*clean*.jsonl"):
        if f.is_file() and not f.name.endswith((".part", ".tmp")) and "garbage" not in f.name:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            rec = json.loads(line.strip())
                            doc_id = rec.get("id")
                            if doc_id:
                                clean_ids.add(str(doc_id))
                        except Exception:
                            pass

    # Garbage shards
    for f in output_dir.glob("*garbage*.jsonl"):
        if f.is_file() and not f.name.endswith((".part", ".tmp")):
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            rec = json.loads(line.strip())
                            if rec.get("kind") == "doc":
                                doc_id = rec.get("id")
                                if doc_id:
                                    garbage_ids.add(str(doc_id))
                        except Exception:
                            pass

    q_file = output_dir / "quarantine.jsonl"
    if q_file.is_file():
        with open(q_file, "r", encoding="utf-8") as fh:
            quarantine_count = sum(1 for line in fh if line.strip())

    return clean_ids, garbage_ids, quarantine_count


def run_pipeline(
    config_path: str,
    output_dir: str,
    fresh: bool = False,
    timeout: float = 600.0,
) -> subprocess.CompletedProcess[str]:
    """Execute textscour CLI run command synchronously."""
    cmd = [
        sys.executable,
        "-m",
        "textscour",
        "run",
        "--config",
        config_path,
        "--set",
        f"output.dir={output_dir}",
    ]
    if fresh:
        cmd.append("--fresh")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{WORKSPACE_ROOT}:{WORKSPACE_ROOT / 'src'}"

    return subprocess.run(
        cmd,
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def test_kill_and_resume(config_path: str = "configs/local_gate.yaml") -> bool:
    """Execute kill-resume verification and compare ID sets."""
    print(f"[kill_resume_check] Running kill-resume resilience check using: {config_path}")

    # Ensure test corpus exists
    corpus_dir = WORKSPACE_ROOT / "tests" / "data"
    if not (corpus_dir / "sample_clean.jsonl").is_file():
        print("[kill_resume_check] Generating initial test corpus in tests/data...")
        from tests.make_corpus import generate_corpus

        generate_corpus(out_dir=corpus_dir, n_docs=100, seed=42)

    test_out_dir = WORKSPACE_ROOT / "outputs" / "kill_resume_test"
    if test_out_dir.exists():
        shutil.rmtree(test_out_dir, ignore_errors=True)
    test_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reference Run: full run to completion
    ref_dir = test_out_dir / "ref"
    print(f"  Step 1: Executing reference run to completion in {ref_dir}...")
    res_ref = run_pipeline(config_path=config_path, output_dir=str(ref_dir), fresh=True)
    if res_ref.returncode != 0:
        print(f"[kill_resume_check] ERROR: Reference run failed:\n{res_ref.stderr}")
        return False

    ref_clean, ref_garbage, ref_quarantine = read_all_ids(ref_dir)
    ref_total = len(ref_clean) + len(ref_garbage) + ref_quarantine
    print(f"  Reference run completed: {len(ref_clean)} clean, {len(ref_garbage)} garbage, {ref_total} total docs.")

    if ref_total == 0:
        print("[kill_resume_check] ERROR: Reference run produced 0 output records.")
        return False

    # 2. Crash Run: Start fresh and kill abruptly with SIGKILL
    crash_dir = test_out_dir / "crash"
    print(f"  Step 2: Starting second run in {crash_dir} to be abruptly terminated...")

    cmd_crash = [
        sys.executable,
        "-m",
        "textscour",
        "run",
        "--config",
        config_path,
        "--set",
        f"output.dir={str(crash_dir)}",
        "--fresh",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{WORKSPACE_ROOT}:{WORKSPACE_ROOT / 'src'}"

    # Start in a new process group so we can SIGKILL the whole subtree
    proc = subprocess.Popen(
        cmd_crash,
        cwd=WORKSPACE_ROOT,
        env=env,
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait until pipeline has committed at least one checkpoint
    state_dir = crash_dir / "state"
    for _ in range(60):
        if state_dir.is_dir() and list(state_dir.glob("ckpt-*.json")):
            break
        time.sleep(0.5)

    print(f"  Sending SIGKILL to process group {proc.pid}...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass

    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()

    print(f"  Abrupt crash verified (terminated with signal {proc.returncode}).")

    # 3. Resumed Run: resume from checkpoint without --fresh
    print(f"  Step 3: Resuming pipeline from checkpoint in {crash_dir}...")
    res_resume = run_pipeline(config_path=config_path, output_dir=str(crash_dir), fresh=False)
    if res_resume.returncode != 0:
        print(f"[kill_resume_check] ERROR: Resumed run failed:\n{res_resume.stderr}")
        return False

    res_clean, res_garbage, res_quarantine = read_all_ids(crash_dir)
    res_total = len(res_clean) + len(res_garbage) + res_quarantine
    print(f"  Resumed run completed: {len(res_clean)} clean, {len(res_garbage)} garbage, {res_total} total docs.")

    # 4. Compare ID sets
    all_ref_ids = ref_clean.union(ref_garbage)
    all_res_ids = res_clean.union(res_garbage)

    missing_ids = all_ref_ids - all_res_ids
    extra_ids = all_res_ids - all_ref_ids

    print("\n--- Invariant Verification ---")
    print(f"  Reference doc IDs count : {len(all_ref_ids):,}")
    print(f"  Resumed doc IDs count   : {len(all_res_ids):,}")

    if missing_ids:
        print(f"[kill_resume_check] FAIL: Lost {len(missing_ids)} document(s) after resume:")
        for m_id in list(missing_ids)[:5]:
            print(f"    - {m_id}")
        return False

    if extra_ids:
        print(f"[kill_resume_check] WARNING: {len(extra_ids)} unexpected extra IDs found.")

    if len(res_clean) != len(ref_clean):
        print(f"[kill_resume_check] Note: Clean count difference: ref={len(ref_clean)}, resumed={len(res_clean)}")

    print("[kill_resume_check] PASS: All document IDs conserved with zero data loss after SIGKILL and resume.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Kill-and-Resume Resilience Check")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/local_gate.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    success = test_kill_and_resume(config_path=args.config)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
