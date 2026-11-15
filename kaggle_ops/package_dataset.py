"""kaggle_ops/package_dataset.py — Package and upload code/model dataset to Kaggle.

Builds build/textscour-code/ containing:
  src/
  rust/ (excluding target/)
  configs/
  Cargo.toml
  pyproject.toml
  scripts/build_core.sh
  kaggle_ops/bootstrap.sh

Uploads or versions the dataset on Kaggle and polls status until 'ready'.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_ops import accounts, metadata


def copy_tree_filtered(src: Path, dst: Path, ignore_names: set[str]) -> None:
    """Recursively copies src to dst ignoring specified filenames or folder names."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        target_root = dst / rel_root

        # Filter out directories
        dirs[:] = [d for d in dirs if d not in ignore_names and not d.startswith(".")]

        target_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            if f in ignore_names or f.endswith(".pyc") or f.endswith(".tmp"):
                continue
            src_file = Path(root) / f
            dst_file = target_root / f
            shutil.copy2(src_file, dst_file)


def package_code(repo_root: Path, build_dir: Path) -> Path:
    """Collects required repository source files into build_dir."""
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    ignore_patterns = {
        "target",
        ".git",
        "__pycache__",
        "outputs",
        "evidence",
        "tests",
        "build",
        "dist",
        ".egg-info",
    }

    # 1. Copy src/
    copy_tree_filtered(repo_root / "src", build_dir / "src", ignore_patterns)

    # 2. Copy rust/
    copy_tree_filtered(repo_root / "rust", build_dir / "rust", ignore_patterns)

    # 3. Copy configs/
    copy_tree_filtered(repo_root / "configs", build_dir / "configs", ignore_patterns)

    # 4. Copy root manifest files
    for manifest_name in ["Cargo.toml", "pyproject.toml", "Cargo.lock"]:
        src_manifest = repo_root / manifest_name
        if src_manifest.exists():
            shutil.copy2(src_manifest, build_dir / manifest_name)

    # 5. Copy scripts/build_core.sh
    dest_scripts = build_dir / "scripts"
    dest_scripts.mkdir(parents=True, exist_ok=True)
    src_build_script = repo_root / "scripts" / "build_core.sh"
    if src_build_script.exists():
        dst_build_script = dest_scripts / "build_core.sh"
        shutil.copy2(src_build_script, dst_build_script)
        dst_build_script.chmod(0o755)

    # 6. Copy kaggle_ops/bootstrap.sh
    dest_ops = build_dir / "kaggle_ops"
    dest_ops.mkdir(parents=True, exist_ok=True)
    src_bootstrap = repo_root / "kaggle_ops" / "bootstrap.sh"
    if src_bootstrap.exists():
        dst_bootstrap = dest_ops / "bootstrap.sh"
        shutil.copy2(src_bootstrap, dst_bootstrap)
        dst_bootstrap.chmod(0o755)

    return build_dir


def get_git_revision(repo_root: Path) -> str:
    """Returns short git revision or current unix timestamp."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        sha = res.stdout.strip()
        if sha:
            return sha
    except Exception:
        pass
    return str(int(time.time()))


def package_and_upload(
    account: str,
    slug: str = "textscour-code",
    title: str = "textscour-code",
    build_dir: str | Path = "build/textscour-code",
    wait_ready: bool = True,
    timeout_s: float = 300.0,
) -> str:
    """Packages code, writes metadata, uploads/versions dataset, and polls until ready."""
    repo_root = Path(__file__).parent.parent.resolve()
    target_build = (repo_root / build_dir).resolve() if not Path(build_dir).is_absolute() else Path(build_dir)

    print(f"[{account}] Packaging code into {target_build}...")
    package_code(repo_root, target_build)

    user = accounts.whoami(account)
    full_slug = f"{user}/{slug}"

    meta = metadata.dataset_metadata(user=user, slug=slug, title=title, is_private=True)
    metadata.write(target_build, meta)
    print(f"[{account}] Written metadata for {full_slug}")

    # Check if dataset already exists
    status_proc = accounts.run(account, ["datasets", "status", full_slug], timeout=60.0)
    stdout_lower = (status_proc.stdout or "").lower()
    stderr_lower = (status_proc.stderr or "").lower()

    dataset_exists = (
        status_proc.returncode == 0
        and ("ready" in stdout_lower or "pending" in stdout_lower or "running" in stdout_lower)
    )

    git_rev = get_git_revision(repo_root)

    if not dataset_exists:
        print(f"[{account}] Dataset {full_slug} does not exist. Creating new dataset...")
        create_res = accounts.run(
            account,
            ["datasets", "create", "-p", str(target_build), "--dir-mode", "zip"],
            timeout=300.0,
        )
        if create_res.returncode != 0:
            # If failed because it actually existed or race condition, try versioning
            if "already exists" in (create_res.stderr or "").lower():
                print(f"[{account}] Already exists reported. Uploading new version...")
                accounts.run(
                    account,
                    ["datasets", "version", "-p", str(target_build), "-m", git_rev, "--dir-mode", "zip"],
                    timeout=300.0,
                    check=True,
                )
            else:
                raise RuntimeError(f"Failed to create dataset {full_slug}: {create_res.stderr}")
    else:
        print(f"[{account}] Dataset {full_slug} exists. Uploading new version (msg: {git_rev})...")
        accounts.run(
            account,
            ["datasets", "version", "-p", str(target_build), "-m", git_rev, "--dir-mode", "zip"],
            timeout=300.0,
            check=True,
        )

    if wait_ready:
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
            time.sleep(10.0)

        raise TimeoutError(f"Dataset {full_slug} did not reach ready state within {timeout_s}s")

    return full_slug


def main() -> None:
    parser = argparse.ArgumentParser(description="Package and upload textscour code dataset to Kaggle.")
    parser.add_argument("--account", default="acc1", help="Kaggle account name (e.g. acc1, acc2, all)")
    parser.add_argument("--slug", default="textscour-code", help="Dataset slug")
    parser.add_argument("--title", default="textscour-code", help="Dataset title")
    parser.add_argument("--build-dir", default="build/textscour-code", help="Directory to assemble package")
    parser.add_argument("--timeout", type=float, default=300.0, help="Status polling timeout in seconds")
    parser.add_argument("--no-wait", action="store_true", help="Do not wait for ready status")

    args = parser.parse_args()

    target_accounts = [args.account] if args.account != "all" else list(accounts.ACCOUNTS.keys())
    for acc in target_accounts:
        package_and_upload(
            account=acc,
            slug=args.slug,
            title=args.title,
            build_dir=args.build_dir,
            wait_ready=not args.no_wait,
            timeout_s=args.timeout,
        )


if __name__ == "__main__":
    main()
