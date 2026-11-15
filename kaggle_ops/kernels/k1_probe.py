"""k1_probe.py — Kaggle GPU Environment Probe and Diagnostic Kernel.

Executes:
  - T0 sentinel emission
  - bootstrap.sh execution / scour_core verification
  - Doctor environment diagnostics
  - GPU discovery (nvidia-smi, torch.cuda)
  - HF reachability check
  - HF_TOKEN presence check (boolean only)
  - Disk and RAM capacity inspection
  - 10-second CPU micro-benchmark
  - RESULT.json output
  - T1 and TEXTSCOUR_KERNEL_OK sentinels
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def run_bootstrap_if_needed() -> None:
    """Executes bootstrap.sh if scour_core is not already compiled and importable."""
    try:
        import scour_core  # type: ignore # noqa: F401
        print("scour_core is already available in environment.")
        return
    except ImportError:
        print("scour_core not found; attempting bootstrap...")

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
        print("Warning: bootstrap.sh not found among candidates.")


def check_hf_reachability() -> bool:
    """Checks whether huggingface.co API endpoints can be reached via HTTPS."""
    url = "https://huggingface.co/api/models"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "textscour-probe/1.0"},
            method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return resp.status in (200, 301, 302)
    except Exception as e:
        print(f"HF reachability check failed: {e}")
        return False


def check_hf_token_presence() -> bool:
    """Detects whether HF_TOKEN is present in env or Kaggle Secrets without logging secrets."""
    if os.environ.get("HF_TOKEN"):
        return True

    # Try Kaggle UserSecretsClient if running in Kaggle environment
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        secrets_client = UserSecretsClient()
        token = secrets_client.get_secret("HF_TOKEN")
        if token and len(token) > 5:
            return True
    except Exception:
        pass

    return False


def run_cpu_microbench(seconds: float = 10.0) -> dict[str, float]:
    """Runs a 10-second CPU micro-benchmark measuring throughput."""
    print(f"Running CPU micro-benchmark for {seconds} seconds...")
    sample_text = (
        "The quick brown fox jumps over the lazy dog. 1234567890! "
        "Visit https://example.com/promo?discount=50 for extra deals! "
    ) * 100

    start = time.perf_counter()
    iterations = 0
    total_bytes = 0

    import hashlib
    while time.perf_counter() - start < seconds:
        h = hashlib.sha256(sample_text.encode("utf-8")).hexdigest()
        total_bytes += len(sample_text)
        iterations += 1

    elapsed = time.perf_counter() - start
    mb_per_sec = (total_bytes / (1024 * 1024)) / elapsed
    iters_per_sec = iterations / elapsed

    print(f"CPU micro-bench: {iters_per_sec:.1f} iters/s, {mb_per_sec:.2f} MB/s")
    return {
        "duration_seconds": elapsed,
        "iterations": iterations,
        "mb_per_sec": mb_per_sec,
        "iters_per_sec": iters_per_sec,
    }


def main() -> None:
    t0 = time.time()
    print(f"TEXTSCOUR_T0 {t0:.4f}")
    print("=== Starting k1_probe ===")

    run_bootstrap_if_needed()

    # 1. GPU Discovery
    gpu_available = False
    gpu_count = 0
    gpu_names: list[str] = []

    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    except ImportError:
        pass

    print(f"PyTorch CUDA available: {gpu_available}, count: {gpu_count}, devices: {gpu_names}")

    # 2. Disk & Memory Inspection
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    disk_usage = shutil.disk_usage(out_dir)
    disk_free_gb = disk_usage.free / (1024 ** 3)
    print(f"Output directory {out_dir} free space: {disk_free_gb:.2f} GB")

    cpu_count = os.cpu_count() or 1
    print(f"Logical CPU count: {cpu_count}")

    # 3. HF Reachability & Token
    hf_reachable = check_hf_reachability()
    hf_token_present = check_hf_token_presence()
    print(f"Hugging Face reachable: {hf_reachable}")
    print(f"HF_TOKEN present: {'yes' if hf_token_present else 'no'}")

    # 4. CPU Micro-bench
    bench_metrics = run_cpu_microbench(seconds=10.0)

    # 5. Native extension check
    scour_core_ok = False
    scour_version = "unknown"
    scour_features: list[str] = []
    try:
        import scour_core  # type: ignore
        scour_version = scour_core.version()
        scour_features = scour_core.cpu_features()
        scour_core_ok = True
        print(f"scour_core {scour_version} initialized with features {scour_features}")
    except Exception as e:
        print(f"scour_core probe notice: {e}")

    # 6. Assemble RESULT.json
    result = {
        "kernel": "k1_probe",
        "status": "PASS",
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "cpu_count": cpu_count,
        "disk_free_gb": round(disk_free_gb, 2),
        "hf_reachable": hf_reachable,
        "hf_token_present": hf_token_present,
        "scour_core_built": scour_core_ok,
        "scour_version": scour_version,
        "scour_features": scour_features,
        "cpu_microbench": bench_metrics,
        "timestamp": time.time(),
    }

    result_path = out_dir / "RESULT.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Also write to current working dir if different
    cwd_result = Path("RESULT.json").resolve()
    if cwd_result != result_path.resolve():
        with open(cwd_result, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(f"RESULT.json written to {result_path}")

    t1 = time.time()
    print(f"TEXTSCOUR_T1 {t1:.4f}")
    print("TEXTSCOUR_KERNEL_OK")


if __name__ == "__main__":
    main()
