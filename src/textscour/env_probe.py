"""Environment discovery, cgroup-aware resource detection, and doctor diagnostics.

Detects CPU cores (cgroup v1/v2 quotas vs host count), RAM limits, GPU memory
and topology, disk budgets, compiler toolchains, Hugging Face reachability,
and generates hardware environment fingerprints for autotuning cache keys.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_total_mb: float
    memory_used_mb: float


@dataclass
class EnvInfo:
    cpu_cores_effective: int
    cpu_cores_host: int
    ram_total_gb: float
    ram_avail_gb: float
    disk_free_gb: float
    workdir: str
    gpus: list[GpuInfo] = field(default_factory=list)
    cuda: bool = False
    torch_version: str | None = None
    python: str = ""
    rustc: str | None = None
    cpu_flags: list[str] = field(default_factory=list)
    is_kaggle: bool = False
    hf_reachable: bool = False
    fingerprint: str = ""

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)


def _detect_cgroup_cores() -> float | None:
    """Detect CPU limit from cgroup v2 (cpu.max) or cgroup v1 (cpu.cfs_quota_us)."""
    # cgroup v2: /sys/fs/cgroup/cpu.max -> "quota period"
    cg2_path = Path("/sys/fs/cgroup/cpu.max")
    if cg2_path.is_file():
        try:
            content = cg2_path.read_text().strip().split()
            if len(content) >= 2 and content[0] != "max":
                quota = float(content[0])
                period = float(content[1])
                if period > 0:
                    return quota / period
        except Exception:
            pass

    # cgroup v1: /sys/fs/cgroup/cpu/cpu.cfs_quota_us and cpu.cfs_period_us
    cg1_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    cg1_period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if cg1_quota.is_file() and cg1_period.is_file():
        try:
            q = float(cg1_quota.read_text().strip())
            p = float(cg1_period.read_text().strip())
            if q > 0 and p > 0:
                return q / p
        except Exception:
            pass

    return None


def _detect_cgroup_ram() -> float | None:
    """Detect RAM limit in GB from cgroup v2 (memory.max) or cgroup v1 (memory.limit_in_bytes)."""
    # cgroup v2
    cg2_mem = Path("/sys/fs/cgroup/memory.max")
    if cg2_mem.is_file():
        try:
            content = cg2_mem.read_text().strip()
            if content != "max":
                b = float(content)
                if 0 < b < 10**16:
                    return b / (1024**3)
        except Exception:
            pass

    # cgroup v1
    cg1_mem = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if cg1_mem.is_file():
        try:
            content = cg1_mem.read_text().strip()
            b = float(content)
            if 0 < b < 10**16:
                return b / (1024**3)
        except Exception:
            pass

    return None


def _detect_proc_meminfo() -> tuple[float, float]:
    """Parse /proc/meminfo for total and available RAM in GB."""
    mem_total_gb = 0.0
    mem_avail_gb = 0.0
    proc_mem = Path("/proc/meminfo")
    if proc_mem.is_file():
        try:
            for line in proc_mem.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    kb = float(line.split()[1])
                    mem_total_gb = kb / (1024 * 1024)
                elif line.startswith("MemAvailable:"):
                    kb = float(line.split()[1])
                    mem_avail_gb = kb / (1024 * 1024)
        except Exception:
            pass
    return mem_total_gb, mem_avail_gb


def _detect_cpu_flags() -> list[str]:
    """Extract notable CPU acceleration flags from /proc/cpuinfo."""
    notable = {"avx", "avx2", "avx512f", "fma", "sse4_1", "sse4_2", "bmi1", "bmi2", "neon", "asimd"}
    found: set[str] = set()
    proc_cpu = Path("/proc/cpuinfo")
    if proc_cpu.is_file():
        try:
            for line in proc_cpu.read_text().splitlines():
                if line.startswith("flags") or line.startswith("Features"):
                    parts = line.split(":", 1)[1].strip().split()
                    for p in parts:
                        p_lower = p.lower()
                        if p_lower in notable:
                            found.add(p_lower)
                    break
        except Exception:
            pass
    return sorted(found)


def _detect_gpus() -> list[GpuInfo]:
    """Detect available GPUs via pynvml or nvidia-smi CLI."""
    gpus: list[GpuInfo] = []

    # Attempt 1: pynvml
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                GpuInfo(
                    index=idx,
                    name=name,
                    memory_total_mb=round(mem_info.total / (1024 * 1024), 1),
                    memory_used_mb=round(mem_info.used / (1024 * 1024), 1),
                )
            )
        return gpus
    except Exception:
        pass

    # Attempt 2: nvidia-smi
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            for line in res.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append(
                        GpuInfo(
                            index=int(parts[0]),
                            name=parts[1],
                            memory_total_mb=float(parts[2]),
                            memory_used_mb=float(parts[3]),
                        )
                    )
            return gpus
    except Exception:
        pass

    return gpus


def _detect_rustc() -> str | None:
    """Retrieve installed rustc compiler version."""
    try:
        res = subprocess.run(["rustc", "-V"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def _check_hf_reachable(timeout: float = 5.0) -> bool:
    """Check network connectivity to Hugging Face API / Hub."""
    try:
        req = urllib.request.Request(
            "https://huggingface.co",
            headers={"User-Agent": "textscour-envprobe/0.1.0"},
            method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status in (200, 301, 302)
    except Exception:
        return False


def _compute_fingerprint(
    cores: int,
    ram_gb: float,
    gpus: list[GpuInfo],
    cpu_flags: list[str],
) -> str:
    """Generate hardware fingerprint for autotuning configuration cache."""
    # Try importing scour_core for extension version
    scour_ver = "none"
    try:
        import scour_core  # type: ignore

        if hasattr(scour_core, "version"):
            scour_ver = scour_core.version()
    except Exception:
        pass

    gpu_str = ",".join(f"{g.name}:{int(g.memory_total_mb)}" for g in gpus)
    flags_str = ",".join(sorted(cpu_flags))
    payload = (
        f"cores={cores}|"
        f"ram_bucket={int(ram_gb // 2 * 2)}|"
        f"gpus={len(gpus)}[{gpu_str}]|"
        f"py={sys.version_info[0]}.{sys.version_info[1]}|"
        f"scour={scour_ver}|"
        f"flags={flags_str}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def probe(workdir: str = ".") -> EnvInfo:
    """Probe system environment, resources, accelerators, and network connectivity."""
    # 1. CPU cores
    host_cores = os.cpu_count() or 1
    cg_cores = _detect_cgroup_cores()
    if cg_cores is not None:
        eff_cores = max(1, int(round(cg_cores)))
    else:
        try:
            eff_cores = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            eff_cores = host_cores

    # 2. RAM
    proc_total, proc_avail = _detect_proc_meminfo()
    cg_ram = _detect_cgroup_ram()
    total_ram_gb = min(proc_total, cg_ram) if cg_ram is not None and proc_total > 0 else (cg_ram or proc_total or 8.0)
    avail_ram_gb = proc_avail if proc_avail > 0 else total_ram_gb * 0.7

    # 3. Disk free
    try:
        free_bytes = shutil.disk_usage(workdir).free
        disk_free_gb = round(free_bytes / (1024**3), 2)
    except Exception:
        disk_free_gb = 10.0

    # 4. GPU & CUDA
    gpus = _detect_gpus()
    cuda_avail = False
    torch_ver: str | None = None
    try:
        import torch  # type: ignore

        torch_ver = torch.__version__
        cuda_avail = bool(gpus and torch.cuda.is_available())
    except Exception:
        pass

    # 5. Toolchains & Environment
    rustc_ver = _detect_rustc()
    cpu_flags = _detect_cpu_flags()
    is_kaggle = Path("/kaggle").is_dir() or ("KAGGLE_KERNEL_RUN_TYPE" in os.environ)
    hf_ok = _check_hf_reachable(timeout=5.0)

    # 6. Fingerprint
    fingerprint = _compute_fingerprint(eff_cores, total_ram_gb, gpus, cpu_flags)

    return EnvInfo(
        cpu_cores_effective=eff_cores,
        cpu_cores_host=host_cores,
        ram_total_gb=round(total_ram_gb, 2),
        ram_avail_gb=round(avail_ram_gb, 2),
        disk_free_gb=disk_free_gb,
        workdir=str(Path(workdir).resolve()),
        gpus=gpus,
        cuda=cuda_avail,
        torch_version=torch_ver,
        python=sys.version.split()[0],
        rustc=rustc_ver,
        cpu_flags=cpu_flags,
        is_kaggle=is_kaggle,
        hf_reachable=hf_ok,
        fingerprint=fingerprint,
    )


def print_doctor(env: EnvInfo) -> str:
    """Format doctor diagnostics report for user inspection."""
    lines: list[str] = [
        "============================================================",
        "               TEXTSCOUR ENVIRONMENT DOCTOR                 ",
        "============================================================",
        f"Workdir            : {env.workdir}",
        f"Environment Target : {'Kaggle Kernel' if env.is_kaggle else 'Local Linux / Server'}",
        f"Hardware Fingerprint: {env.fingerprint}",
        "",
        "--- CPU & Memory ---",
        f"Effective CPU Cores: {env.cpu_cores_effective} (Host visible: {env.cpu_cores_host})",
        f"CPU Flags          : {', '.join(env.cpu_flags) if env.cpu_flags else 'None detected'}",
        f"Total RAM          : {env.ram_total_gb:.2f} GB (Available: {env.ram_avail_gb:.2f} GB)",
        f"Free Disk Space    : {env.disk_free_gb:.2f} GB",
        "",
        "--- Accelerators & CUDA ---",
        f"CUDA Available     : {'YES' if env.cuda else 'NO'}",
    ]

    if env.gpus:
        for g in env.gpus:
            lines.append(f"  GPU #{g.index}: {g.name} | VRAM: {g.memory_used_mb:.0f}/{g.memory_total_mb:.0f} MB")
    else:
        lines.append("  No NVIDIA GPUs detected (CPU rules_only or fallback mode active)")

    lines.extend(
        [
            "",
            "--- Software & Toolchains ---",
            f"Python Version     : {env.python}",
            f"PyTorch Version    : {env.torch_version or 'Not installed'}",
            f"Rust Toolchain     : {env.rustc or 'Not installed / not in PATH'}",
            "",
            "--- Network Connectivity ---",
            f"HuggingFace Reach  : {'[OK] Connected' if env.hf_reachable else '[WARNING] Unreachable / Offline'}",
            "============================================================",
        ]
    )

    return "\n".join(lines)
