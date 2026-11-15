"""kaggle_ops/package_model.py — Upload trained classifier model weights to Kaggle dataset.

Packages models/classifier/ into Kaggle dataset textscour-model and polls until ready.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_ops import accounts, metadata


def upload_model_dataset(
    account: str,
    model_dir: str | Path = "models/classifier",
    slug: str = "textscour-model",
    title: str = "textscour-model",
    timeout_s: float = 600.0,
) -> str:
    m_dir = Path(model_dir).resolve()
    if not m_dir.exists() or not (m_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"Trained model.safetensors not found at {m_dir}")

    user = accounts.whoami(account)
    full_slug = f"{user}/{slug}"

    meta = metadata.dataset_metadata(user=user, slug=slug, title=title, is_private=True)
    metadata.write(m_dir, meta)
    print(f"[{account}] Written metadata for {full_slug} in {m_dir}")

    # Check if dataset already exists
    status_proc = accounts.run(account, ["datasets", "status", full_slug], timeout=60.0)
    stdout_lower = (status_proc.stdout or "").lower()

    dataset_exists = (
        status_proc.returncode == 0
        and ("ready" in stdout_lower or "pending" in stdout_lower or "running" in stdout_lower)
    )

    git_rev = str(int(time.time()))

    if not dataset_exists:
        print(f"[{account}] Dataset {full_slug} does not exist. Creating new dataset...")
        create_res = accounts.run(
            account,
            ["datasets", "create", "-p", str(m_dir), "--dir-mode", "zip"],
            timeout=600.0,
        )
        if create_res.returncode != 0:
            if "already exists" in (create_res.stderr or "").lower():
                print(f"[{account}] Already exists reported. Uploading new version...")
                accounts.run(
                    account,
                    ["datasets", "version", "-p", str(m_dir), "-m", git_rev, "--dir-mode", "zip"],
                    timeout=600.0,
                    check=True,
                )
            else:
                raise RuntimeError(f"Failed to create dataset {full_slug}: {create_res.stderr}")
    else:
        print(f"[{account}] Dataset {full_slug} exists. Uploading new version (msg: {git_rev})...")
        accounts.run(
            account,
            ["datasets", "version", "-p", str(m_dir), "-m", git_rev, "--dir-mode", "zip"],
            timeout=600.0,
            check=True,
        )

    print(f"[{account}] Polling dataset status for {full_slug}...")
    start_time = time.time()
    while time.time() - start_time < timeout_s:
        chk = accounts.run(account, ["datasets", "status", full_slug], timeout=30.0)
        status_text = (chk.stdout or "").strip()
        print(f"[{account}] Status: {status_text}")
        lower_status = status_text.lower()
        if "ready" in lower_status:
            print(f"[{account}] Dataset {full_slug} is ready.")
            return full_slug
        if "error" in lower_status or "failed" in lower_status:
            raise RuntimeError(f"Dataset {full_slug} entered error state: {status_text}")
        time.sleep(15.0)

    raise TimeoutError(f"Dataset {full_slug} did not reach ready state within {timeout_s}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload trained classifier weights to Kaggle.")
    parser.add_argument("--account", default="all", help="Kaggle account name (acc1, acc2, or all)")
    parser.add_argument("--model-dir", default="models/classifier", help="Model directory")
    args = parser.parse_args()

    target_accounts = [args.account] if args.account != "all" else list(accounts.ACCOUNTS.keys())
    for acc in target_accounts:
        upload_model_dataset(account=acc, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
