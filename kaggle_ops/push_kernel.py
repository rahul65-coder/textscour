"""kaggle_ops/push_kernel.py — Push kernel to Kaggle with metadata and error handling.

Prepares kernel metadata, copies script, executes `kaggle kernels push -p <dir>`,
handles accelerator flags, and parses output URL and errors.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_ops import accounts, metadata

# Custom exception classes for structured error handling
class KernelPushError(RuntimeError):
    """Base error for kernel push failures."""
    pass


class QuotaExceededError(KernelPushError):
    """Raised when Kaggle GPU quota or session limits are exceeded."""
    pass


class DatasetNotFoundError(KernelPushError):
    """Raised when a referenced dataset source does not exist or is not ready."""
    pass


def supports_accelerator_flag(account: str = "acc1") -> bool:
    """Checks whether the installed Kaggle CLI supports the --accelerator option."""
    try:
        res = accounts.run(account, ["kernels", "push", "--help"], timeout=15.0)
        output = (res.stdout or "") + (res.stderr or "")
        return "--accelerator" in output
    except Exception:
        return False


def push(
    account: str,
    kernel_dir: str | Path,
    kernel_script: str | Path | None = None,
    slug: str | None = None,
    title: str | None = None,
    dataset_sources: list[str] | None = None,
    enable_gpu: bool = True,
    enable_internet: bool = True,
    accelerator: str | None = "NvidiaTeslaT4",
    timeout: float = 120.0,
) -> str:
    """Prepares and pushes a kernel directory to Kaggle.

    Returns the full kernel slug (<user>/<slug>).
    """
    k_dir = Path(kernel_dir).resolve()
    k_dir.mkdir(parents=True, exist_ok=True)

    user = accounts.whoami(account)

    # If a script is provided, copy it to kernel_dir
    code_filename = "main.py"
    if kernel_script is not None:
        src_script = Path(kernel_script).resolve()
        if not src_script.exists():
            raise FileNotFoundError(f"Kernel script not found: {src_script}")
        code_filename = src_script.name
        dst_script = k_dir / code_filename
        if src_script != dst_script:
            shutil.copy2(src_script, dst_script)
    else:
        # Search for existing python script in k_dir
        py_files = list(k_dir.glob("*.py"))
        if py_files:
            code_filename = py_files[0].name

    if slug is None:
        slug = f"textscour-{k_dir.name}"
    if title is None:
        title = slug

    # Default datasets if not provided
    if dataset_sources is None:
        dataset_sources = [f"{user}/textscour-code"]

    full_slug = f"{user}/{slug}" if "/" not in slug else slug

    # Write kernel-metadata.json
    meta = metadata.kernel_metadata(
        user=user,
        slug=slug,
        title=title,
        code_file=code_filename,
        dataset_sources=dataset_sources,
        enable_gpu=enable_gpu,
        enable_internet=enable_internet,
        is_private=True,
    )
    metadata.write(k_dir, meta)

    # Build CLI push command
    cmd_args = ["kernels", "push", "-p", str(k_dir)]

    if enable_gpu and accelerator:
        if supports_accelerator_flag(account):
            cmd_args.extend(["--accelerator", accelerator])

    print(f"[{account}] Pushing kernel {full_slug} from {k_dir}...")
    res = accounts.run(account, cmd_args, timeout=timeout)

    output = (res.stdout or "") + "\n" + (res.stderr or "")
    output_lower = output.lower()

    # Check for quota / session limits
    if "quota" in output_lower or "limit exceeded" in output_lower or "maximum number of concurrent" in output_lower:
        raise QuotaExceededError(f"Kaggle quota or concurrency limit reached for {account}: {output.strip()}")

    # Check for dataset source not found
    if "dataset source not found" in output_lower or "dataset not found" in output_lower:
        raise DatasetNotFoundError(f"Dataset source not found for {full_slug}: {output.strip()}")

    # Check for failure returncode
    if res.returncode != 0:
        raise KernelPushError(f"Failed to push kernel {full_slug} (exit {res.returncode}):\n{output.strip()}")

    # Parse URL or confirmation
    # Standard output: "Kernel version N successfully pushed. URL: https://www.kaggle.com/code/user/slug"
    match = re.search(r"https://www\.kaggle\.com/code/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)", output)
    if match:
        pushed_slug = match.group(1)
        print(f"[{account}] Kernel pushed successfully: {pushed_slug}")
        return pushed_slug

    print(f"[{account}] Kernel push accepted for {full_slug}")
    return full_slug


def main() -> None:
    parser = argparse.ArgumentParser(description="Push a kernel to Kaggle.")
    parser.add_argument("--account", default="acc1", help="Kaggle account name")
    parser.add_argument("--dir", "--kernel-dir", dest="dir", required=True, help="Kernel directory")
    parser.add_argument("--script", "--kernel-script", dest="script", help="Path to Python kernel script")
    parser.add_argument("--slug", help="Kernel slug")
    parser.add_argument("--title", help="Kernel title")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4", help="Accelerator type")
    parser.add_argument("--dataset-sources", nargs="*", default=None, help="Dataset sources to attach")

    args = parser.parse_args()

    slug = push(
        account=args.account,
        kernel_dir=args.dir,
        kernel_script=args.script,
        slug=args.slug,
        title=args.title,
        dataset_sources=args.dataset_sources,
        enable_gpu=not args.no_gpu,
        accelerator=args.accelerator if not args.no_gpu else None,
    )
    print(f"Pushed: {slug}")


if __name__ == "__main__":
    main()
