"""kaggle_ops/fetch_output.py — Download and verify Kaggle kernel output and logs.

Executes `kaggle kernels output <slug> -p <out_dir>`, verifies RESULT.json and
sentinel line TEXTSCOUR_KERNEL_OK, extracts timestamps for quota ledger,
and prints the last 80 lines of the kernel log on failure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_ops import accounts

SENTINEL_OK = "TEXTSCOUR_KERNEL_OK"
T0_PATTERN = re.compile(r"TEXTSCOUR_T0\s+([0-9]+(?:\.[0-9]+)?)")
T1_PATTERN = re.compile(r"TEXTSCOUR_T1\s+([0-9]+(?:\.[0-9]+)?)")
ERROR_CODE_PATTERN = re.compile(r"\[(E-[A-Z]+-\d{3})\]")


class FetchVerificationError(RuntimeError):
    """Raised when output download succeeds but kernel result verification fails."""
    pass


def find_log_files(out_dir: Path) -> list[Path]:
    """Finds all potential log files inside output directory."""
    logs: list[Path] = []
    for p in out_dir.rglob("*"):
        if p.is_file():
            # Kaggle outputs often name the log after the kernel slug or .log
            if p.suffix in (".log", ".txt") or p.name.endswith(".log") or "log" in p.name.lower():
                logs.append(p)
    return logs


def inspect_logs(log_paths: list[Path]) -> tuple[bool, float | None, float | None, str, list[str]]:
    """Inspects log files for sentinel, t0, t1, and extracts the last 80 lines."""
    has_sentinel = False
    t0: float | None = None
    t1: float | None = None
    all_lines: list[str] = []

    for lp in log_paths:
        try:
            with open(lp, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                all_lines.extend(lines)
                for line in lines:
                    if SENTINEL_OK in line:
                        has_sentinel = True
                    m0 = T0_PATTERN.search(line)
                    if m0:
                        try:
                            t0 = float(m0.group(1))
                        except ValueError:
                            pass
                    m1 = T1_PATTERN.search(line)
                    if m1:
                        try:
                            t1 = float(m1.group(1))
                        except ValueError:
                            pass
        except Exception:
            continue

    combined_text = "".join(all_lines)
    last_80 = [line.rstrip() for line in all_lines[-80:]] if all_lines else []
    return has_sentinel, t0, t1, combined_text, last_80


def extract_diagnosis(log_text: str) -> dict[str, str]:
    """Extracts diagnosis metadata including error codes and traceback hints."""
    codes = ERROR_CODE_PATTERN.findall(log_text)
    primary_code = codes[-1] if codes else "UNKNOWN_ERROR"

    # Search for traceback or panic
    traceback_lines: list[str] = []
    in_tb = False
    for line in log_text.splitlines():
        if "Traceback (most recent call last):" in line or "panicked at" in line:
            in_tb = True
            traceback_lines = [line]
        elif in_tb:
            traceback_lines.append(line)
            if len(traceback_lines) > 30:
                in_tb = False

    return {
        "error_code": primary_code,
        "traceback": "\n".join(traceback_lines[-25:]),
    }


def fetch(account: str, slug: str, out_dir: str | Path) -> list[Path]:
    """Downloads kernel output via Kaggle CLI and verifies RESULT.json and sentinel."""
    target_dir = Path(out_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if "/" not in slug:
        user = accounts.whoami(account)
        slug = f"{user}/{slug}"

    print(f"[{account}] Fetching outputs for {slug} into {target_dir}...")
    res = accounts.run(account, ["kernels", "output", slug, "-p", str(target_dir)], timeout=300.0)

    if res.returncode != 0:
        err_msg = f"Failed to download outputs for {slug} (code {res.returncode}):\n{res.stderr}"
        print(f"[{account}] ERROR: {err_msg}", file=sys.stderr)
        raise RuntimeError(err_msg)

    # Collect downloaded files
    fetched_files = [p for p in target_dir.rglob("*") if p.is_file()]
    if not fetched_files:
        raise FetchVerificationError(f"Output directory {target_dir} is completely empty after download.")

    print(f"[{account}] Downloaded {len(fetched_files)} file(s).")

    # Check for log files and sentinel
    log_files = find_log_files(target_dir)
    has_sentinel, t0, t1, log_text, last_80 = inspect_logs(log_files)

    # Check for RESULT.json
    result_json_path = target_dir / "RESULT.json"
    result_data: dict[str, Any] | None = None
    if result_json_path.exists():
        try:
            with open(result_json_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            print(f"[{account}] RESULT.json found and loaded successfully.")
        except Exception as e:
            print(f"[{account}] Warning: Failed to parse RESULT.json: {e}", file=sys.stderr)

    # Validate success criteria
    verification_errors: list[str] = []
    if not has_sentinel:
        verification_errors.append(f"Missing sentinel '{SENTINEL_OK}' in kernel log files.")
    if result_data is None:
        verification_errors.append("Missing or unreadable RESULT.json in kernel outputs.")

    if verification_errors:
        print("\n" + "=" * 40 + " KERNEL FAILURE DIAGNOSIS " + "=" * 40, file=sys.stderr)
        for err in verification_errors:
            print(f"VERIFICATION FAILURE: {err}", file=sys.stderr)
        print("\nLast 80 lines of kernel log:", file=sys.stderr)
        print("-" * 80, file=sys.stderr)
        for line in last_80:
            print(line, file=sys.stderr)
        print("-" * 80, file=sys.stderr)

        # Write diagnosis file under evidence/<kernel>/diagnosis.md
        kernel_name = slug.split("/")[-1]
        repo_root = Path(__file__).parent.parent.resolve()
        evidence_dir = repo_root / "evidence" / kernel_name
        evidence_dir.mkdir(parents=True, exist_ok=True)
        diag_path = evidence_dir / "diagnosis.md"

        diagnosis = extract_diagnosis(log_text)
        with open(diag_path, "w", encoding="utf-8") as f:
            f.write(f"# Kernel Failure Diagnosis: {slug}\n\n")
            f.write(f"- **Error Code**: {diagnosis['error_code']}\n")
            f.write(f"- **Verification Failures**: {'; '.join(verification_errors)}\n\n")
            f.write("## Traceback / Failure Summary\n```text\n")
            f.write(diagnosis["traceback"] or "No traceback detected in log.\n")
            f.write("\n```\n\n## Log Tail (Last 80 lines)\n```text\n")
            f.write("\n".join(last_80))
            f.write("\n```\n")

        print(f"[{account}] Diagnostic report written to {diag_path}", file=sys.stderr)
        raise FetchVerificationError(f"Kernel {slug} output verification failed: {'; '.join(verification_errors)}")

    print(f"[{account}] Output verification PASSED for {slug}.")
    return fetched_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify Kaggle kernel output.")
    parser.add_argument("--account", default="acc1", help="Kaggle account name")
    parser.add_argument("--slug", required=True, help="Full kernel slug (e.g. user/slug)")
    parser.add_argument("--out-dir", "--dest", dest="out_dir", required=True, help="Target download directory")

    args = parser.parse_args()

    try:
        files = fetch(account=args.account, slug=args.slug, out_dir=args.out_dir)
        print(f"Successfully fetched {len(files)} files.")
    except Exception as ex:
        print(f"Error fetching kernel outputs: {ex}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
