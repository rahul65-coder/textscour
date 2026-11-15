"""Configuration loading, cross-validation, CLI dotted-key overrides, and engine mapping.

Defines dataclasses for all configuration sections, strict validation with
actionable error codes, and mapping to the native Rust engine dictionary.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field as dc_field, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from textscour.errors import ScourError


@dataclass
class TextSpec:
    mode: Literal["field", "path", "concat", "template", "messages", "auto"] = "auto"
    field: str | None = None
    fields: list[str] = dc_field(default_factory=list)
    sep: str = "\n\n"
    template: str | None = None
    role_filter: list[str] | None = None


@dataclass
class SourceCfg:
    type: Literal["hf", "jsonl", "json", "parquet"]
    repo: str | None = None
    name: str | None = None
    split: str = "train"
    revision: str | None = None
    data_files: str | list[str] | None = None
    paths: list[str] = dc_field(default_factory=list)
    text: TextSpec = dc_field(default_factory=TextSpec)
    id_field: str | None = None
    domain: str | None = None
    weight: float = 1.0
    max_docs: int | None = None


@dataclass
class OutputCfg:
    dir: str = "outputs"
    clean_name: str = "clean"
    garbage_name: str = "garbage"
    format: Literal["jsonl", "jsonl.zst", "parquet"] = "jsonl"
    shard_mb: int = 500
    emit_fragments: bool = False
    extended_columns: bool = False
    final_merge: bool = False


@dataclass
class UploadCfg:
    enabled: bool = False
    repo: str | None = None
    private: bool = True
    commit_every_shards: int = 4
    commit_every_seconds: int = 300
    delete_after_upload: bool = False
    state_backend: Literal["hf", "local"] = "local"


@dataclass
class GopherCfg:
    min_alpha_ratio: float = 0.50
    max_symbol_ratio: float = 0.10
    max_dup_line_frac: float = 0.30
    max_top_2gram_cov: float = 0.20
    max_top_3gram_cov: float = 0.18
    max_top_4gram_cov: float = 0.16
    max_short_line_frac: float = 0.70
    min_avg_line_len: float = 40.0
    max_boilerplate_hits: int = 2
    min_stopword_ratio: float = 0.06


@dataclass
class CascadeCfg:
    gpu_mode: Literal["auto", "off"] = "auto"
    model: str | None = "distilbert-base-multilingual-cased"
    gpu_drop_p: float = 0.85
    gpu_strip_p: float = 0.50
    promo_lo: float = 0.30
    promo_hi: float = 0.70
    url_policy: Literal["smart", "remove_all", "allowlist_only", "keep_all", "blocklist"] = "smart"
    allow_hosts: list[str] = dc_field(default_factory=list)
    block_hosts: list[str] = dc_field(default_factory=list)
    min_doc_chars: int = 200
    clean_threshold: float = 6.0
    pii_mode: Literal["redact", "drop", "off"] = "redact"
    pii_drop_min_hits: int = 1
    allowed_langs: list[str] = dc_field(default_factory=lambda: ["en", "hi", "hinglish"])
    lexicon_dir: str = "configs/lexicons"
    domain_mode: Literal["field", "url_host", "dataset", "topic"] = "dataset"
    keep_urls_in_code: bool = True
    chunk_target_chars: int = 800
    chunk_min_chars: int = 100
    chunk_max_chars: int = 2000
    max_doc_chars: int = 2000000
    normalize_unicode: bool = True
    gopher: GopherCfg = dc_field(default_factory=GopherCfg)


@dataclass
class RuntimeCfg:
    autotune: Literal["auto", "cached", "off"] = "auto"
    readers: int | Literal["auto"] = "auto"
    workers: int | Literal["auto"] = "auto"
    rust_threads: int | Literal["auto"] = "auto"
    batch_docs: int | Literal["auto"] = "auto"
    queue_depth: int | Literal["auto"] = "auto"
    gpu_batch_tokens: int | Literal["auto"] = "auto"
    gpu_inflight: int = 2
    gpu_job_chunks: int = 128
    max_runtime_seconds: int | None = None
    disk_budget_gb: float = 16.0
    ram_budget_frac: float = 0.80
    max_error_rate: float = 0.01


@dataclass
class DedupCfg:
    mode: Literal["exact", "minhash", "off"] = "minhash"
    shingle_k: int = 5
    max_mem_gb: float = 2.0


@dataclass
class Config:
    sources: list[SourceCfg]
    output: OutputCfg = dc_field(default_factory=OutputCfg)
    upload: UploadCfg = dc_field(default_factory=UploadCfg)
    cascade: CascadeCfg = dc_field(default_factory=CascadeCfg)
    runtime: RuntimeCfg = dc_field(default_factory=RuntimeCfg)
    dedup: DedupCfg = dc_field(default_factory=DedupCfg)


def _parse_scalar(val: str) -> Any:
    """Parse string scalar into bool, int, float, or string."""
    lower = val.lower().strip()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("none", "null"):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _apply_dotted_override(d: dict[str, Any], dot_key: str, val: Any) -> None:
    """Apply dotted key override like cascade.promo_lo=0.35 to nested dict."""
    keys = dot_key.strip().split(".")
    curr = d
    for k in keys[:-1]:
        if k not in curr or not isinstance(curr[k], dict):
            curr[k] = {}
        curr = curr[k]
    curr[keys[-1]] = val


TYPE_MAP: dict[str, type] = {
    "output": OutputCfg,
    "upload": UploadCfg,
    "cascade": CascadeCfg,
    "runtime": RuntimeCfg,
    "dedup": DedupCfg,
    "gopher": GopherCfg,
    "text": TextSpec,
}


def _get_known_keys(cls: type) -> set[str]:
    """Retrieve all known field names for a dataclass recursively."""
    if not is_dataclass(cls):
        return set()
    result = set()
    for fname in cls.__dataclass_fields__:
        result.add(fname)
        if fname in TYPE_MAP:
            sub_cls = TYPE_MAP[fname]
            sub_keys = _get_known_keys(sub_cls)
            for sk in sub_keys:
                result.add(f"{fname}.{sk}")
    return result


def _check_unknown_keys(raw: dict[str, Any], cls: type, prefix: str = "") -> None:
    """Recursively verify that raw dict keys belong to the dataclass."""
    valid_fields = cls.__dataclass_fields__
    for key, val in raw.items():
        full_key = f"{prefix}{key}" if prefix else key
        if key not in valid_fields:
            all_known = _get_known_keys(Config)
            matches = difflib.get_close_matches(full_key, list(all_known), n=1, cutoff=0.5)
            suggestion = f"; did you mean '{matches[0]}'?" if matches else ""
            raise ScourError(
                code="E-CFG-002",
                stage="config",
                message=f"Unknown configuration key '{full_key}'{suggestion}",
                hint="Check docs/INTERFACES.md or configs/default.yaml for valid keys.",
            )
        if key in TYPE_MAP and isinstance(val, dict):
            _check_unknown_keys(val, TYPE_MAP[key], prefix=f"{full_key}.")


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a dataclass from a dict, handling nested dataclasses."""
    init_kwargs = {}
    for f_name in cls.__dataclass_fields__:
        if f_name in data:
            val = data[f_name]
            if f_name in TYPE_MAP and isinstance(val, dict):
                init_kwargs[f_name] = _build_dataclass(TYPE_MAP[f_name], val)
            elif f_name == "sources" and isinstance(val, list):
                built_sources = []
                for s in val:
                    if isinstance(s, dict):
                        text_val = s.get("text", {})
                        s_copy = dict(s)
                        if isinstance(text_val, dict):
                            s_copy["text"] = _build_dataclass(TextSpec, text_val)
                        built_sources.append(_build_dataclass(SourceCfg, s_copy))
                    else:
                        built_sources.append(s)
                init_kwargs[f_name] = built_sources
            elif f_name == "gopher" and isinstance(val, dict):
                init_kwargs[f_name] = _build_dataclass(GopherCfg, val)
            else:
                init_kwargs[f_name] = val
    return cls(**init_kwargs)


def validate(cfg: Config) -> None:
    """Perform rigorous cross-validation of loaded configuration."""
    # Source validation
    if not cfg.sources:
        raise ScourError(
            code="E-CFG-001",
            stage="config",
            message="No data sources configured in sources list",
            hint="Add at least one SourceCfg entry under sources:",
        )

    for idx, src in enumerate(cfg.sources):
        if src.type not in ("hf", "jsonl", "json", "parquet"):
            raise ScourError(
                code="E-CFG-003",
                stage="config",
                message=f"Invalid source type '{src.type}' in source #{idx}",
                hint="Source type must be one of: 'hf', 'jsonl', 'json', 'parquet'",
            )
        if src.type == "hf" and not src.repo:
            raise ScourError(
                code="E-CFG-001",
                stage="config",
                message=f"Hugging Face source #{idx} requires 'repo' identifier",
                hint="Specify repo: 'org/dataset-name'",
            )
        if src.type in ("jsonl", "json", "parquet") and not src.paths:
            raise ScourError(
                code="E-CFG-001",
                stage="config",
                message=f"Local source #{idx} ({src.type}) requires 'paths' list",
                hint="Specify paths: ['data/train-*.parquet']",
            )
        if src.text.mode not in ("field", "path", "concat", "template", "messages", "auto"):
            raise ScourError(
                code="E-CFG-003",
                stage="config",
                message=f"Invalid text extractor mode '{src.text.mode}' in source #{idx}",
                hint="Mode must be one of: 'field', 'path', 'concat', 'template', 'messages', 'auto'",
            )

    # Output validation
    if cfg.output.clean_name == cfg.output.garbage_name:
        raise ScourError(
            code="E-CFG-007",
            stage="config",
            message=f"clean_name and garbage_name cannot be identical ('{cfg.output.clean_name}')",
            hint="Set different names like clean_name: 'clean' and garbage_name: 'garbage'",
        )

    if cfg.output.format not in ("jsonl", "jsonl.zst", "parquet"):
        raise ScourError(
            code="E-CFG-003",
            stage="config",
            message=f"Invalid output format '{cfg.output.format}'",
            hint="Output format must be one of: 'jsonl', 'jsonl.zst', 'parquet'",
        )

    # Upload validation
    if cfg.upload.enabled and not cfg.upload.repo:
        raise ScourError(
            code="E-CFG-008",
            stage="config",
            message="upload.enabled is True but upload.repo is not specified",
            hint="Set upload.repo to destination Hugging Face dataset identifier (e.g. 'org/dataset')",
        )

    if cfg.upload.state_backend not in ("hf", "local"):
        raise ScourError(
            code="E-CFG-003",
            stage="config",
            message=f"Invalid upload.state_backend '{cfg.upload.state_backend}'",
            hint="Must be 'hf' or 'local'",
        )

    # Cascade validation
    cas = cfg.cascade
    if cas.gpu_mode not in ("auto", "off"):
        raise ScourError(
            code="E-CFG-003",
            stage="config",
            message=f"Invalid cascade.gpu_mode '{cas.gpu_mode}'",
            hint="Must be 'auto' or 'off'",
        )

    if cas.url_policy not in ("smart", "remove_all", "allowlist_only", "keep_all", "blocklist"):
        raise ScourError(
            code="E-CFG-003",
            stage="config",
            message=f"Invalid cascade.url_policy '{cas.url_policy}'",
            hint="Must be one of: 'smart', 'remove_all', 'allowlist_only', 'keep_all', 'blocklist'",
        )

    if cas.pii_mode not in ("redact", "drop", "off"):
        raise ScourError(
            code="E-CFG-003",
            stage="config",
            message=f"Invalid cascade.pii_mode '{cas.pii_mode}'",
            hint="Must be one of: 'redact', 'drop', 'off'",
        )

    if cas.domain_mode not in ("field", "url_host", "dataset", "topic"):
        raise ScourError(
            code="E-CFG-003",
            stage="config",
            message=f"Invalid cascade.domain_mode '{cas.domain_mode}'",
            hint="Must be one of: 'field', 'url_host', 'dataset', 'topic'",
        )

    if not (0.0 <= cas.promo_lo < cas.promo_hi <= 1.0):
        raise ScourError(
            code="E-CFG-004",
            stage="config",
            message=f"Invalid promo thresholds: promo_lo ({cas.promo_lo}) must be < promo_hi ({cas.promo_hi}) in [0, 1]",
            hint="Ensure 0.0 <= promo_lo < promo_hi <= 1.0 (defaults: 0.30 and 0.70)",
        )

    if not (0.0 <= cas.gpu_strip_p <= cas.gpu_drop_p <= 1.0):
        raise ScourError(
            code="E-CFG-004",
            stage="config",
            message=f"Invalid GPU thresholds: gpu_strip_p ({cas.gpu_strip_p}) must be <= gpu_drop_p ({cas.gpu_drop_p})",
            hint="Ensure 0.0 <= gpu_strip_p <= gpu_drop_p <= 1.0 (defaults: 0.50 and 0.85)",
        )

    if not (cas.chunk_min_chars <= cas.chunk_target_chars <= cas.chunk_max_chars):
        raise ScourError(
            code="E-CFG-005",
            stage="config",
            message=f"Invalid chunk sizes: min ({cas.chunk_min_chars}) <= target ({cas.chunk_target_chars}) <= max ({cas.chunk_max_chars}) violated",
            hint="Adjust chunk_min_chars, chunk_target_chars, chunk_max_chars in cascade configuration",
        )

    # Runtime validation
    rt = cfg.runtime
    if rt.autotune not in ("auto", "cached", "off"):
        raise ScourError(
            code="E-CFG-003",
            stage="config",
            message=f"Invalid runtime.autotune '{rt.autotune}'",
            hint="Must be one of: 'auto', 'cached', 'off'",
        )

    if rt.disk_budget_gb <= 0:
        raise ScourError(
            code="E-CFG-009",
            stage="config",
            message=f"disk_budget_gb must be > 0 (got {rt.disk_budget_gb})",
            hint="Set runtime.disk_budget_gb to a positive number in gigabytes",
        )

    if not (0.0 < rt.ram_budget_frac <= 1.0):
        raise ScourError(
            code="E-CFG-009",
            stage="config",
            message=f"ram_budget_frac must be in (0.0, 1.0] (got {rt.ram_budget_frac})",
            hint="Set runtime.ram_budget_frac to a fraction like 0.80",
        )

    for field_name in ("readers", "workers", "rust_threads", "batch_docs", "queue_depth", "gpu_batch_tokens"):
        val = getattr(rt, field_name)
        if isinstance(val, int) and val <= 0:
            raise ScourError(
                code="E-CFG-009",
                stage="config",
                message=f"runtime.{field_name} must be > 0 or 'auto' (got {val})",
                hint=f"Set runtime.{field_name} to a positive integer or 'auto'",
            )

    # Dedup validation
    if cfg.dedup.mode not in ("exact", "minhash", "off"):
        raise ScourError(
            code="E-CFG-003",
            stage="config",
            message=f"Invalid dedup.mode '{cfg.dedup.mode}'",
            hint="Must be one of: 'exact', 'minhash', 'off'",
        )


def load_config(
    path_or_dict: str | Path | dict[str, Any],
    overrides: list[str] | dict[str, Any] | None = None,
) -> Config:
    """Load configuration from YAML file or dictionary, apply dotted overrides, and validate."""
    raw_data: dict[str, Any] = {}
    if isinstance(path_or_dict, (str, Path)):
        p = Path(path_or_dict)
        if "\n" not in str(path_or_dict) and p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if loaded is not None:
                        if not isinstance(loaded, dict):
                            raise ValueError("YAML root must be a dictionary mapping")
                        raw_data = loaded
            except Exception as e:
                raise ScourError(
                    code="E-CFG-001",
                    stage="config",
                    message=f"Failed to parse config YAML: {e}",
                    cause=e,
                    hint="Check YAML syntax, indentation, and characters",
                )
        else:
            # Try parsing directly as YAML string
            try:
                loaded = yaml.safe_load(str(path_or_dict))
                if isinstance(loaded, dict):
                    raw_data = loaded
                else:
                    raise ScourError(
                        code="E-CFG-001",
                        stage="config",
                        message=f"Config file not found or invalid YAML: {path_or_dict}",
                        hint="Verify that the path to configuration YAML is correct",
                    )
            except ScourError:
                raise
            except Exception as e:
                raise ScourError(
                    code="E-CFG-001",
                    stage="config",
                    message=f"Failed to parse config YAML string: {e}",
                    cause=e,
                    hint="Check YAML syntax, indentation, and characters",
                )
    elif isinstance(path_or_dict, dict):
        raw_data = dict(path_or_dict)
    else:
        raise ScourError(
            code="E-CFG-001",
            stage="config",
            message=f"Invalid config input type: {type(path_or_dict)}",
            hint="Pass a valid file path or dictionary",
        )

    # Apply dotted-key CLI overrides e.g. ["cascade.promo_lo=0.35", "runtime.workers=4"]
    if overrides:
        if isinstance(overrides, dict):
            for k, v in overrides.items():
                _apply_dotted_override(raw_data, k, v)
        elif isinstance(overrides, list):
            for item in overrides:
                if "=" in item:
                    k, v = item.split("=", 1)
                    parsed_val = _parse_scalar(v)
                    _apply_dotted_override(raw_data, k.strip(), parsed_val)

    # Check unknown keys against Config schema
    _check_unknown_keys(raw_data, Config)

    # Build Config dataclass
    cfg = _build_dataclass(Config, raw_data)
    validate(cfg)
    return cfg


def to_engine_dict(cfg: Config) -> dict[str, Any]:
    """Convert Config instance to dictionary contract expected by scour_core.Engine."""
    cas = cfg.cascade
    lex_dir = cas.lexicon_dir
    if not os.path.exists(lex_dir):
        # Resolve against known repo and package locations
        candidates = [
            Path("/kaggle/input/textscour-code/configs/lexicons"),
            Path("/kaggle/input/datasets/shaikmoham/textscour-code/configs/lexicons"),
            Path("/kaggle/input/datasets/sksamsuddin52/textscour-code/configs/lexicons"),
            Path("/tmp/textscour_build/configs/lexicons"),
            Path(__file__).parent.parent.parent / "configs" / "lexicons",
            Path(__file__).parent.parent / "configs" / "lexicons",
            Path("./configs/lexicons"),
        ]
        for cand in candidates:
            if cand.exists():
                lex_dir = str(cand.resolve())
                break

    return {
        "chunk_target_chars": cas.chunk_target_chars,
        "chunk_min_chars": cas.chunk_min_chars,
        "chunk_max_chars": cas.chunk_max_chars,
        "url_policy": cas.url_policy,
        "allow_hosts": list(cas.allow_hosts),
        "block_hosts": list(cas.block_hosts),
        "promo_lo": float(cas.promo_lo),
        "promo_hi": float(cas.promo_hi),
        "gpu_drop_p": float(cas.gpu_drop_p),
        "gpu_strip_p": float(cas.gpu_strip_p),
        "min_doc_chars": cas.min_doc_chars,
        "clean_threshold": float(cas.clean_threshold),
        "pii_mode": cas.pii_mode,
        "pii_drop_min_hits": cas.pii_drop_min_hits,
        "allowed_langs": list(cas.allowed_langs),
        "rust_threads": cfg.runtime.rust_threads if isinstance(cfg.runtime.rust_threads, int) else 0,
        "lexicon_dir": lex_dir,
        "domain_mode": cas.domain_mode,
        "emit_fragments": cfg.output.emit_fragments,
        "normalize_unicode": cas.normalize_unicode,
        "dedup_shingle_k": cfg.dedup.shingle_k,
        "keep_urls_in_code": cas.keep_urls_in_code,
        "max_doc_chars": cas.max_doc_chars,
        "gopher": asdict(cas.gopher),
    }


def critical_config_hash(cfg: Config) -> str:
    """Compute deterministic SHA-256 hash of critical cascade configuration fields.

    Used by checkpoint verification to ensure that resumed runs share identical
    filtering heuristics and thresholds.
    """
    cas = cfg.cascade
    critical_data = {
        "url_policy": cas.url_policy,
        "promo_lo": cas.promo_lo,
        "promo_hi": cas.promo_hi,
        "min_doc_chars": cas.min_doc_chars,
        "clean_threshold": cas.clean_threshold,
        "allowed_langs": sorted(cas.allowed_langs),
        "pii_mode": cas.pii_mode,
    }
    encoded = json.dumps(critical_data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
