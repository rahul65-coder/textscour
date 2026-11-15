"""TEXTSCOUR actionable diagnostic error catalog and exception system.

Every pipeline failure surfaces as a structured ScourError with an error code,
stage, problem description, where it happened, root cause with secret redaction,
and prescriptive remedy hint.
"""

from __future__ import annotations

import inspect
import re
import sys
from typing import Any, Mapping

# Regex patterns for credential and secret scrubbing
RE_HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]{15,}")
RE_BEARER = re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{8,}", re.IGNORECASE)
RE_KAGGLE_KEY = re.compile(r"\b[a-f0-9]{32}\b", re.IGNORECASE)
RE_GENERIC_SECRET = re.compile(
    r"""(?i)(?:api_key|token|secret|password|access_token)\s*[:=]\s*['"]?[A-Za-z0-9_\-]{8,}['"]?"""
)


def redact(msg: Any) -> str:
    """Scrub sensitive credentials, Hugging Face tokens, and API keys from messages."""
    if msg is None:
        return ""
    text = str(msg)
    text = RE_HF_TOKEN.sub("hf_***REDACTED***", text)
    text = RE_BEARER.sub("Bearer ***REDACTED***", text)
    text = RE_KAGGLE_KEY.sub("***REDACTED_KEY***", text)
    text = RE_GENERIC_SECRET.sub("secret: ***REDACTED***", text)
    return text


ERROR_CATALOG: dict[str, dict[str, str]] = {
    "E-CFG-001": {
        "stage": "config",
        "description": "Config file missing or invalid YAML syntax",
        "hint": "Verify config file path and check YAML syntax with a validator",
    },
    "E-CFG-002": {
        "stage": "config",
        "description": "Unknown configuration key encountered",
        "hint": "Remove or fix unknown key; check configs/default.yaml for valid schema",
    },
    "E-CFG-003": {
        "stage": "config",
        "description": "Invalid enum value provided",
        "hint": "Check allowed values in docs/INTERFACES.md (e.g. url_policy, pii_mode)",
    },
    "E-CFG-004": {
        "stage": "config",
        "description": "Contradictory threshold configuration",
        "hint": "Ensure promo_lo < promo_hi and gpu_strip_p <= gpu_drop_p",
    },
    "E-CFG-005": {
        "stage": "config",
        "description": "Invalid chunk sizing parameters",
        "hint": "Ensure chunk_min_chars <= chunk_target_chars <= chunk_max_chars",
    },
    "E-CFG-006": {
        "stage": "config",
        "description": "Lexicon directory or required lexicon file not found",
        "hint": "Check lexicon_dir path and verify all required TSV/TXT files exist",
    },
    "E-CFG-007": {
        "stage": "config",
        "description": "Identical clean and garbage output names",
        "hint": "Specify distinct names for clean_name and garbage_name",
    },
    "E-CFG-008": {
        "stage": "config",
        "description": "Upload enabled without destination repository",
        "hint": "Set upload.repo to a valid Hugging Face dataset identifier",
    },
    "E-CFG-009": {
        "stage": "config",
        "description": "Invalid resource budget (workers, threads, or memory)",
        "hint": "Set positive values for workers and ensure budgets are within host limits",
    },
    "E-SRC-001": {
        "stage": "source",
        "description": "Dataset or local source file path not found",
        "hint": "Verify Hugging Face dataset ID or local path existence",
    },
    "E-SRC-002": {
        "stage": "source",
        "description": "HF authentication failed (HTTP 401/403)",
        "hint": "Set HF_TOKEN in env / Kaggle Secrets or accept dataset license terms",
    },
    "E-SRC-003": {
        "stage": "source",
        "description": "Network timeout or connection drop after retries",
        "hint": "Check internet connectivity or Hugging Face service status; increase retries",
    },
    "E-SRC-004": {
        "stage": "source",
        "description": "Source yielded zero records",
        "hint": "Verify dataset split, revision, and data_files filtering parameters",
    },
    "E-SCHEMA-001": {
        "stage": "schema",
        "description": "Document text field missing or extracted as None",
        "hint": "Check text.field or text.mode configuration against document schema",
    },
    "E-SCHEMA-002": {
        "stage": "schema",
        "description": "Document text field is not a string",
        "hint": "Verify extractor expression or inspect source schema types",
    },
    "E-SCHEMA-003": {
        "stage": "schema",
        "description": "Duplicate document ID collision within same source",
        "hint": "Ensure id_field is unique or use auto-generated hash IDs",
    },
    "E-RUST-001": {
        "stage": "engine",
        "description": "Native extension scour_core import failure",
        "hint": "Run bash scripts/build_core.sh to compile PyO3 module",
    },
    "E-RUST-002": {
        "stage": "engine",
        "description": "Engine rejected configuration dictionary",
        "hint": "Verify dictionary parameters passed to scour_core.Engine",
    },
    "E-RUST-003": {
        "stage": "engine",
        "description": "Unhandled panic caught during document processing",
        "hint": "Inspect document characters; report panic traceback in quarantine",
    },
    "E-GPU-001": {
        "stage": "gpu",
        "description": "CUDA requested but no compatible GPU detected",
        "hint": "Set cascade.gpu_mode: 'off' or run on a GPU-enabled machine",
    },
    "E-GPU-002": {
        "stage": "gpu",
        "description": "GPU Out-Of-Memory (OOM) at minimum batch size",
        "hint": "Reduce gpu_batch_tokens or free competing GPU processes",
    },
    "E-GPU-003": {
        "stage": "gpu",
        "description": "Classifier model or tokenizer loading failure",
        "hint": "Verify model path or HF model identifier and check disk cache",
    },
    "E-DISK-001": {
        "stage": "storage",
        "description": "Available disk space below safety budget",
        "hint": "Increase runtime.disk_budget_gb, enable upload-and-delete, or prune disk",
    },
    "E-OUT-001": {
        "stage": "writer",
        "description": "Failed writing shard or compression stream error",
        "hint": "Check disk write permissions, zstandard library, or disk quotas",
    },
    "E-CKPT-001": {
        "stage": "resume",
        "description": "Checkpoint file corrupt or config hash mismatch",
        "hint": "Use --fresh to restart or verify config compatibility",
    },
    "E-UP-001": {
        "stage": "upload",
        "description": "Hugging Face upload failed after max retries",
        "hint": "Check write permissions for token on destination repo and network",
    },
    "E-UP-002": {
        "stage": "upload",
        "description": "Hugging Face commit rate limit exceeded (HTTP 429)",
        "hint": "Increase upload.commit_every_shards and commit_every_seconds",
    },
    "E-WORKER-001": {
        "stage": "runtime",
        "description": "Subprocess worker died or heartbeat timed out",
        "hint": "Check worker logs for segfaults or OOM killer interventions",
    },
    "E-RATE-001": {
        "stage": "runtime",
        "description": "Record error rate exceeded max_error_rate",
        "hint": "Inspect quarantine.jsonl; check source schema formatting consistency",
    },
}

EXIT_CODE_MAP: dict[str, int] = {
    "CFG": 2,
    "SRC": 3,
    "SCHEMA": 4,
    "RUST": 5,
    "GPU": 6,
    "DISK": 7,
    "OUT": 8,
    "CKPT": 9,
    "UP": 10,
    "WORKER": 11,
    "RATE": 12,
}


def get_exit_code(code: str) -> int:
    """Map error code category to process exit code."""
    parts = code.split("-")
    if len(parts) >= 2:
        cat = parts[1]
        return EXIT_CODE_MAP.get(cat, 1)
    return 1


class ScourError(Exception):
    """Actionable, structured error with diagnostic formatting and secret redaction."""

    def __init__(
        self,
        code: str,
        stage: str | None = None,
        message: str | None = None,
        *,
        dataset: str | None = None,
        shard: int | None = None,
        record_id: str | None = None,
        cause: BaseException | None = None,
        hint: str | None = None,
        where: str | None = None,
    ) -> None:
        self.code = code
        meta = ERROR_CATALOG.get(code, {})
        self.stage = stage or meta.get("stage", "unknown")
        catalog_desc = meta.get("description", "")
        self.message = message or catalog_desc or code
        self.dataset = dataset
        self.shard = shard
        self.record_id = record_id
        self.cause = cause
        self.hint = hint or meta.get("hint", "Consult documentation for resolution")

        if where:
            self.where = where
        else:
            # Auto-detect caller module and function if possible
            try:
                frame = inspect.currentframe()
                caller = frame.f_back if frame else None
                if caller:
                    mod = caller.f_globals.get("__name__", "unknown")
                    func = caller.f_code.co_name
                    self.where = f"{mod}:{func}"
                else:
                    self.where = "unknown"
            except Exception:
                self.where = "unknown"

        super().__init__(self.format())

    @property
    def exit_code(self) -> int:
        return get_exit_code(self.code)

    def format(self) -> str:
        """Format into multi-line actionable diagnostic report."""
        header_parts = [f"[{self.code}]", f"stage={self.stage}"]
        if self.dataset is not None:
            header_parts.append(f"dataset={self.dataset}")
        if self.shard is not None:
            header_parts.append(f"shard={self.shard}")
        if self.record_id is not None:
            header_parts.append(f"record_id={self.record_id}")
        header = " ".join(header_parts)

        cause_str = ""
        if self.cause is not None:
            c_name = self.cause.__class__.__name__
            c_msg = redact(str(self.cause))
            cause_str = f"{c_name}: {c_msg}"
        else:
            cause_str = redact(self.message)

        problem_str = redact(self.message)
        fix_str = redact(self.hint)

        lines = [
            header,
            f"  Problem : {problem_str}",
            f"  Where   : {self.where}",
            f"  Cause   : {cause_str}",
            f"  Fix     : {fix_str}",
        ]
        return "\n".join(lines)
