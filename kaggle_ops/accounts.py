"""kaggle_ops/accounts.py — Multi-account handling and credential management for Kaggle CLI.

Provides account discovery, identity verification, environment configuration,
and redacted subprocess execution.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

# Default account paths as specified in MASTER_PLAN
DEFAULT_ACCOUNTS: dict[str, Path] = {
    "acc1": Path("~/.kaggle_acc1").expanduser(),
    "acc2": Path("~/.kaggle_acc2").expanduser(),
}

# Regex patterns for sensitive credential redaction
SECRET_PATTERNS = [
    re.compile(r'(["\']?key["\']?\s*[:=]\s*["\'])([a-f0-9]{32,})(["\'])', re.IGNORECASE),
    re.compile(r'([a-f0-9]{32})', re.IGNORECASE),
    re.compile(r'(hf_[A-Za-z0-9]{20,})'),
]


def redact_text(text: str) -> str:
    """Masks secret keys and tokens from logs or error outputs."""
    if not text:
        return text
    result = text
    # Redact HF tokens first
    result = re.sub(r'hf_[A-Za-z0-9]{20,}', 'hf_[REDACTED]', result)
    # Redact JSON key values
    result = re.sub(
        r'(["\']?(?:key|token|secret)["\']?\s*[:=]\s*["\'])([^"\']{8,})(["\'])',
        r'\1[REDACTED]\3',
        result,
        flags=re.IGNORECASE,
    )
    return result


def discover_accounts() -> dict[str, Path]:
    """Discovers available Kaggle account directories or sets them up from ~/.kaggle files."""
    accounts: dict[str, Path] = {}

    acc1_dir = Path("~/.kaggle_acc1").expanduser()
    acc2_dir = Path("~/.kaggle_acc2").expanduser()

    # Check existing ~/.kaggle_acc1 and ~/.kaggle_acc2
    if (acc1_dir / "kaggle.json").exists():
        accounts["acc1"] = acc1_dir
    if (acc2_dir / "kaggle.json").exists():
        accounts["acc2"] = acc2_dir

    # If missing, check ~/.kaggle directory for alternate account configs
    kaggle_home = Path("~/.kaggle").expanduser()
    if kaggle_home.exists():
        # Check for named configs like kaggle_*.json
        named_configs = sorted(list(kaggle_home.glob("kaggle_*.json")))
        if "acc1" not in accounts and len(named_configs) >= 1:
            acc1_dir.mkdir(parents=True, exist_ok=True)
            target = acc1_dir / "kaggle.json"
            if not target.exists():
                shutil.copy2(named_configs[0], target)
                target.chmod(0o600)
            accounts["acc1"] = acc1_dir

        if "acc2" not in accounts and len(named_configs) >= 2:
            acc2_dir.mkdir(parents=True, exist_ok=True)
            target = acc2_dir / "kaggle.json"
            if not target.exists():
                shutil.copy2(named_configs[1], target)
                target.chmod(0o600)
            accounts["acc2"] = acc2_dir

        # Fallback to single ~/.kaggle/kaggle.json if still no acc1
        if "acc1" not in accounts and (kaggle_home / "kaggle.json").exists():
            accounts["acc1"] = kaggle_home

    return accounts


# Global registry of active accounts
ACCOUNTS: dict[str, Path] = discover_accounts()


def kaggle_env(account: str) -> dict[str, str]:
    """Returns a copy of the environment with KAGGLE_CONFIG_DIR set to the account path."""
    if account not in ACCOUNTS:
        # Refresh discovery
        discovered = discover_accounts()
        ACCOUNTS.update(discovered)
        if account not in ACCOUNTS:
            raise ValueError(f"Unknown Kaggle account: {account}. Available: {list(ACCOUNTS.keys())}")

    env = os.environ.copy()
    config_dir = str(ACCOUNTS[account].resolve())
    env["KAGGLE_CONFIG_DIR"] = config_dir
    return env


def whoami(account: str) -> str:
    """Returns verified Kaggle username for the given account without leaking secret keys."""
    if account not in ACCOUNTS:
        raise ValueError(f"Account {account} not found in {list(ACCOUNTS.keys())}")

    account_dir = ACCOUNTS[account]
    kaggle_json = account_dir / "kaggle.json"

    # Attempt reading username from kaggle.json safely
    if kaggle_json.exists():
        try:
            with open(kaggle_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                username = data.get("username")
                if username and isinstance(username, str):
                    _save_account_username(account, username)
                    return username
        except Exception:
            pass

    # Fallback to CLI verification: `kaggle config view`
    res = run(account, ["config", "view"], timeout=30.0)
    for line in res.stdout.splitlines():
        if "username:" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                username = parts[1].strip()
                if username:
                    _save_account_username(account, username)
                    return username

    raise RuntimeError(f"Unable to determine Kaggle username for account {account}")


def _save_account_username(account: str, username: str) -> None:
    """Records discovered username into ~/.kaggle/ if accessible."""
    cache_path = Path("~/.kaggle/.accounts.json").expanduser()
    cache: dict[str, str] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    cache[account] = username
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def run(
    account: str,
    args: list[str],
    timeout: float | None = 300.0,
    check: bool = False,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Executes a Kaggle CLI command with credentials for the specified account and redacts secrets."""
    env = kaggle_env(account)
    cmd = ["kaggle"] + args

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            timeout=timeout,
            check=check,
            capture_output=capture_output,
            text=True,
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=redact_text(proc.stdout) if proc.stdout else "",
            stderr=redact_text(proc.stderr) if proc.stderr else "",
        )
    except subprocess.TimeoutExpired as e:
        redacted_stdout = redact_text(e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""))
        redacted_stderr = redact_text(e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or ""))
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=timeout or 0.0,
            output=redacted_stdout,
            stderr=redacted_stderr,
        ) from e
    except subprocess.CalledProcessError as e:
        redacted_stdout = redact_text(e.stdout if isinstance(e.stdout, str) else "")
        redacted_stderr = redact_text(e.stderr if isinstance(e.stderr, str) else "")
        raise subprocess.CalledProcessError(
            returncode=e.returncode,
            cmd=cmd,
            output=redacted_stdout,
            stderr=redacted_stderr,
        ) from e


if __name__ == "__main__":
    print("=== Discovered Kaggle Accounts ===")
    for acc, path in ACCOUNTS.items():
        try:
            user = whoami(acc)
            print(f"  {acc}: {path} (user: {user})")
        except Exception as ex:
            print(f"  {acc}: {path} (error resolving user: {ex})")
