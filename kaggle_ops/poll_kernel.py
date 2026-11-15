"""kaggle_ops/poll_kernel.py — Poll kernel execution status until completion.

Implements polling loop with status enum {QUEUED, RUNNING, COMPLETE, ERROR, CANCELED, TIMEOUT},
writes timestamped evidence logs, and applies exponential backoff on transient errors.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_ops import accounts


class KernelStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"
    CANCELED = "CANCELED"
    TIMEOUT = "TIMEOUT"

    @property
    def is_terminal(self) -> bool:
        return self in (
            KernelStatus.COMPLETE,
            KernelStatus.ERROR,
            KernelStatus.CANCELED,
            KernelStatus.TIMEOUT,
        )


def parse_status_string(output: str) -> KernelStatus:
    """Parses raw Kaggle CLI status string into KernelStatus enum."""
    text = output.lower()
    if "complete" in text:
        return KernelStatus.COMPLETE
    if "error" in text or "fail" in text:
        return KernelStatus.ERROR
    if "cancel" in text:
        return KernelStatus.CANCELED
    if "running" in text:
        return KernelStatus.RUNNING
    if "queued" in text or "waiting" in text or "pending" in text:
        return KernelStatus.QUEUED

    # If unrecognized but exit was 0 and status has some text
    if "has status" in text:
        # e.g. "shaikmoham/textscour-k1-probe has status 'complete'"
        for st in KernelStatus:
            if st.value.lower() in text:
                return st

    # Default fallback to RUNNING if active
    return KernelStatus.RUNNING


def poll(
    account: str,
    slug: str,
    interval: float = 60.0,
    timeout_s: float = 43200.0,  # 12 hours Kaggle session cap
    log_path: str | Path | None = None,
) -> KernelStatus:
    """Polls kernel status until a terminal state or timeout is reached.

    Appends all status checks to evidence/<kernel>/status.log.
    """
    repo_root = Path(__file__).parent.parent.resolve()

    if log_path is None:
        kernel_name = slug.split("/")[-1]
        log_dir = repo_root / "evidence" / kernel_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "status.log"
    else:
        log_file = Path(log_path).resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    consecutive_errors = 0
    max_backoff = 60.0

    if "/" not in slug:
        user = accounts.whoami(account)
        slug = f"{user}/{slug}"

    print(f"[{account}] Polling status for {slug} (interval: {interval}s, timeout: {timeout_s}s)...")
    print(f"[{account}] Logging status to {log_file}")

    while True:
        elapsed = time.time() - t0
        if elapsed > timeout_s:
            iso_now = datetime.now(timezone.utc).isoformat()
            timeout_msg = f"{iso_now} TIMEOUT after {elapsed:.1f}s\n"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(timeout_msg)
            print(f"[{account}] {slug} reached TIMEOUT ({elapsed:.1f}s)")
            return KernelStatus.TIMEOUT

        try:
            res = accounts.run(account, ["kernels", "status", slug], timeout=30.0)
            raw_out = (res.stdout or "").strip()
            if not raw_out and res.stderr:
                raw_out = res.stderr.strip()

            consecutive_errors = 0
            status = parse_status_string(raw_out)
            iso_now = datetime.now(timezone.utc).isoformat()
            log_line = f"{iso_now} {raw_out}\n"

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_line)

            print(f"[{account}] [{elapsed:.0f}s] {slug} -> {status.value} ({raw_out})")

            if status.is_terminal:
                return status

        except Exception as ex:
            consecutive_errors += 1
            backoff = min(interval * (1.5 ** consecutive_errors), max_backoff)
            iso_now = datetime.now(timezone.utc).isoformat()
            err_line = f"{iso_now} NETWORK_ERROR: {ex} (backing off {backoff:.1f}s)\n"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(err_line)
            print(f"[{account}] Warning during status check: {ex}. Retrying in {backoff:.1f}s...")
            time.sleep(backoff)
            continue

        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll Kaggle kernel status until completion.")
    parser.add_argument("--account", default="acc1", help="Kaggle account name")
    parser.add_argument("--slug", required=True, help="Full kernel slug (e.g. user/slug)")
    parser.add_argument("--interval", type=float, default=60.0, help="Polling interval in seconds")
    parser.add_argument("--timeout", type=float, default=43200.0, help="Maximum timeout in seconds")
    parser.add_argument("--log-path", help="Path to write status.log")

    args = parser.parse_args()

    status = poll(
        account=args.account,
        slug=args.slug,
        interval=args.interval,
        timeout_s=args.timeout,
        log_path=args.log_path,
    )

    print(f"Final status: {status.value}")
    if status != KernelStatus.COMPLETE:
        sys.exit(1)


if __name__ == "__main__":
    main()
