"""kaggle_ops/metadata.py — JSON metadata generators for Kaggle datasets and kernels.

Generates dataset-metadata.json and kernel-metadata.json matching Section D.2
specifications exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dataset_metadata(
    user: str,
    slug: str,
    title: str,
    licenses: list[dict[str, str]] | None = None,
    is_private: bool = True,
) -> dict[str, Any]:
    """Generates a dataset metadata dictionary conforming to Kaggle CLI specs.

    Example:
    {
      "title": "textscour-code",
      "id": "ACC1_USER/textscour-code",
      "licenses": [{"name": "CC0-1.0"}],
      "isPrivate": true
    }
    """
    dataset_id = f"{user}/{slug}" if "/" not in slug else slug
    return {
        "title": title,
        "id": dataset_id,
        "licenses": licenses if licenses is not None else [{"name": "CC0-1.0"}],
        "isPrivate": is_private,
    }


def kernel_metadata(
    user: str,
    slug: str,
    title: str,
    code_file: str,
    dataset_sources: list[str] | None = None,
    kernel_sources: list[str] | None = None,
    enable_gpu: bool = True,
    enable_internet: bool = True,
    is_private: bool = True,
    language: str = "python",
    kernel_type: str = "script",
    competition_sources: list[str] | None = None,
    model_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Generates a kernel metadata dictionary conforming to Kaggle CLI specs.

    Example:
    {
      "id": "ACC_USER/textscour-k2-autotune",
      "title": "textscour-k2-autotune",
      "code_file": "k2_autotune_bench.py",
      "language": "python",
      "kernel_type": "script",
      "is_private": "true",
      "enable_gpu": "true",
      "enable_tpu": "false",
      "enable_internet": "true",
      "dataset_sources": ["ACC_USER/textscour-code"],
      "competition_sources": [],
      "kernel_sources": [],
      "model_sources": []
    }
    """
    kernel_id = f"{user}/{slug}" if "/" not in slug else slug
    resolved_datasets: list[str] = []
    if dataset_sources:
        for ds in dataset_sources:
            if "/" not in ds:
                resolved_datasets.append(f"{user}/{ds}")
            else:
                resolved_datasets.append(ds)

    resolved_kernels: list[str] = []
    if kernel_sources:
        for ks in kernel_sources:
            if "/" not in ks:
                resolved_kernels.append(f"{user}/{ks}")
            else:
                resolved_kernels.append(ks)

    return {
        "id": kernel_id,
        "title": title,
        "code_file": code_file,
        "language": language,
        "kernel_type": kernel_type,
        "is_private": "true" if is_private else "false",
        "enable_gpu": "true" if enable_gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true" if enable_internet else "false",
        "dataset_sources": resolved_datasets,
        "competition_sources": competition_sources if competition_sources is not None else [],
        "kernel_sources": resolved_kernels,
        "model_sources": model_sources if model_sources is not None else [],
    }


def write(target_dir: str | Path, meta: dict[str, Any], filename: str | None = None) -> Path:
    """Writes metadata dictionary to target_dir as JSON and returns the written path."""
    dest_dir = Path(target_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        if "code_file" in meta:
            filename = "kernel-metadata.json"
        else:
            filename = "dataset-metadata.json"

    out_path = dest_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_path
