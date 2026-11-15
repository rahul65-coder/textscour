"""kaggle_ops/budget.py — Quota ledger and dispatch budget enforcement for Kaggle GPU runs.

Enforces:
  quota_hours_initial: 30.0
  reserve_hours: 4.0
  max_spend_hours: 20.0

Calculates compute hours from kernel execution timestamps (excluding queue time),
refusing dispatch when budget constraints would be violated.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QUOTA_HOURS_INITIAL: float = 30.0
RESERVE_HOURS: float = 4.0
MAX_SPEND_HOURS: float = 20.0


class Ledger:
    """Manages Kaggle GPU quota consumption across accounts."""

    def __init__(self, path: str | Path = "kaggle_ops/ledger.json") -> None:
        self.path = Path(path).resolve()
        self._lock = threading.Lock()
        self.data: dict[str, Any] = self._load()

    def _default_account_entry(self) -> dict[str, Any]:
        return {
            "quota_hours_initial": QUOTA_HOURS_INITIAL,
            "spent_hours": 0.0,
            "reserve_hours": RESERVE_HOURS,
            "max_spend_hours": MAX_SPEND_HOURS,
            "runs": [],
        }

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "accounts": {
                "acc1": self._default_account_entry(),
                "acc2": self._default_account_entry(),
            }
        }

    def _save_unlocked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(self.path)

    def save(self) -> None:
        """Atomically persists ledger to disk."""
        with self._lock:
            self._save_unlocked()

    def get_account_state(self, account: str) -> dict[str, Any]:
        """Returns the current quota ledger state for an account."""
        with self._lock:
            if "accounts" not in self.data:
                self.data["accounts"] = {}
            if account not in self.data["accounts"]:
                self.data["accounts"][account] = self._default_account_entry()
            return dict(self.data["accounts"][account])

    def can_dispatch(self, account: str, est_hours: float) -> bool:
        """Checks whether dispatching a kernel with est_hours is permitted under quota policy.

        Refuses if:
          spent + est_hours > max_spend_hours
          OR
          quota_hours_initial - spent - est_hours < reserve_hours
        """
        state = self.get_account_state(account)
        spent = float(state.get("spent_hours", 0.0))
        max_spend = float(state.get("max_spend_hours", MAX_SPEND_HOURS))
        reserve = float(state.get("reserve_hours", RESERVE_HOURS))
        initial = float(state.get("quota_hours_initial", QUOTA_HOURS_INITIAL))

        if spent + est_hours > max_spend:
            return False

        remaining_after = initial - spent - est_hours
        if remaining_after < reserve:
            return False

        return True

    def record(
        self,
        account: str,
        slug: str,
        start: float,
        end: float,
        notes: str | None = None,
    ) -> float:
        """Records execution time between start and end epochs and increments spent_hours.

        Returns recorded hours.
        """
        with self._lock:
            if "accounts" not in self.data:
                self.data["accounts"] = {}
            if account not in self.data["accounts"]:
                self.data["accounts"][account] = self._default_account_entry()

            duration_s = max(0.0, end - start)
            hours = duration_s / 3600.0

            run_entry = {
                "slug": slug,
                "start": start,
                "end": end,
                "duration_seconds": duration_s,
                "hours": round(hours, 4),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "notes": notes or "",
            }

            self.data["accounts"][account]["runs"].append(run_entry)
            current_spent = float(self.data["accounts"][account].get("spent_hours", 0.0))
            self.data["accounts"][account]["spent_hours"] = round(current_spent + hours, 4)

            self._save_unlocked()
            return hours


def print_summary(ledger_path: str | Path = "kaggle_ops/ledger.json") -> None:
    ledger = Ledger(ledger_path)
    print("=" * 60)
    print("           TEXTSCOUR KAGGLE QUOTA LEDGER")
    print("=" * 60)
    print(f"{'Account':<10} {'Initial (h)':<12} {'Spent (h)':<12} {'Remaining':<12} {'Can Spend (h)':<14}")
    print("-" * 60)

    for acc in ["acc1", "acc2"]:
        st = ledger.get_account_state(acc)
        initial = st["quota_hours_initial"]
        spent = st["spent_hours"]
        rem = initial - spent
        max_sp = st["max_spend_hours"]
        can_sp = max(0.0, max_sp - spent)
        print(f"{acc:<10} {initial:<12.2f} {spent:<12.2f} {rem:<12.2f} {can_sp:<14.2f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View or manage Kaggle quota ledger.")
    parser.add_argument("--path", default="kaggle_ops/ledger.json", help="Path to ledger.json")
    args = parser.parse_args()
    print_summary(args.path)
