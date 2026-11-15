"""
Pytest configuration and interface-level fakes for TEXTSCOUR testing.
Provides fakes for scour_core (Engine, Stage1Batch, DocResult, DedupIndex)
to enable isolated Python runtime tests before or alongside native Rust extension.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure workspace root and src/ are in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
SRC_DIR = WORKSPACE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest

from tests.make_corpus import generate_corpus


@dataclass
class FakeDocResult:
    """Python fake matching scour_core.DocResult."""
    clean: bool
    text: str
    length: int
    score: float
    reasons: list[str]
    fragments: list[tuple[str, str]]
    exact: int
    bands: list[int]
    orig_len: int
    lang: str = "en"
    domain: str = ""


class FakeStage1Batch:
    """Python fake matching scour_core.Stage1Batch."""

    def __init__(self, texts: list[str], cfg: dict[str, Any]) -> None:
        self.texts = texts
        self.cfg = cfg
        self._ambiguous_indices: list[int] = []
        self._ambiguous_texts: list[str] = []

        # Find ambiguous candidates
        for idx, text in enumerate(texts):
            lower = text.lower()
            if "ambiguous" in lower or "maybe_promo" in lower:
                self._ambiguous_indices.append(idx)
                self._ambiguous_texts.append(text[:200])

    def n_docs(self) -> int:
        return len(self.texts)

    def ambiguous(self) -> tuple[list[int], list[int], list[str]]:
        doc_indices = list(self._ambiguous_indices)
        chunk_indices = [0] * len(doc_indices)
        return (doc_indices, chunk_indices, list(self._ambiguous_texts))

    def finalize(self, probs: list[float], mode: str) -> list[FakeDocResult]:
        results: list[FakeDocResult] = []
        min_doc_chars = int(self.cfg.get("min_doc_chars", 200))
        clean_threshold = float(self.cfg.get("clean_threshold", 6.0))
        gpu_drop_p = float(self.cfg.get("gpu_drop_p", 0.85))

        prob_map = dict(zip(self._ambiguous_indices, probs))

        for idx, text in enumerate(self.texts):
            orig_len = len(text)
            reasons: list[str] = []
            fragments: list[tuple[str, str]] = []
            score = 8.5
            cleaned_text = text

            # Simulate promo detection
            lower = text.lower()
            if "click here to buy" in lower or "promo code" in lower or "discount" in lower:
                reasons.append("promo_phrase")
                score -= 3.5

            if "http://spam-tracker" in lower or "https://discount-zone" in lower:
                reasons.append("url_spam")
                score -= 4.0

            if "all rights reserved" in lower or "cookie policy" in lower:
                reasons.append("boilerplate")
                score -= 2.0

            if "developer.test@example.org" in lower:
                reasons.append("pii_dropped")
                score -= 5.0

            # GPU resolution if ambiguous
            if idx in prob_map:
                p = prob_map[idx]
                if p >= gpu_drop_p:
                    reasons.append("gpu_garbage")
                    score -= 4.0

            if len(cleaned_text) < min_doc_chars:
                reasons.append("too_short")
                score -= 3.0

            score = max(0.0, min(10.0, round(score, 1)))
            clean = len(reasons) == 0 and score >= clean_threshold and len(cleaned_text) >= min_doc_chars

            # Deterministic hash calculation for exact and bands
            h = hashlib.sha256(cleaned_text.encode("utf-8")).digest()
            exact = int.from_bytes(h[:8], "big")
            bands = [
                int.from_bytes(hashlib.sha256(f"{exact}_{b}".encode("utf-8")).digest()[:8], "big")
                for b in range(8)
            ]

            results.append(
                FakeDocResult(
                    clean=clean,
                    text=cleaned_text,
                    length=len(cleaned_text),
                    score=score,
                    reasons=reasons,
                    fragments=fragments,
                    exact=exact,
                    bands=bands,
                    orig_len=orig_len,
                    lang="en" if any(c.isascii() for c in cleaned_text) else "hi",
                    domain=self.cfg.get("domain", ""),
                )
            )
        return results


class FakeEngine:
    """Python fake matching scour_core.Engine."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = dict(cfg)
        # Validation checks matching Section B.2
        if self.cfg.get("promo_lo", 0.3) >= self.cfg.get("promo_hi", 0.7):
            raise ValueError("promo_lo must be strictly less than promo_hi")

    def stage1(self, texts: list[str]) -> FakeStage1Batch:
        return FakeStage1Batch(texts, self.cfg)


class FakeDedupIndex:
    """Python fake matching scour_core.DedupIndex."""

    def __init__(self) -> None:
        self.exact_set: set[int] = set()
        self.band_sets: list[set[int]] = [set() for _ in range(8)]
        self._count = 0

    def check_and_insert(self, exact: int, bands: list[int]) -> str | None:
        if exact in self.exact_set:
            return "duplicate_exact"

        for b_idx, band_val in enumerate(bands):
            if band_val in self.band_sets[b_idx]:
                return "duplicate_near"

        # Unique doc: insert
        self.exact_set.add(exact)
        for b_idx, band_val in enumerate(bands):
            self.band_sets[b_idx].add(band_val)
        self._count += 1
        return None

    def __len__(self) -> int:
        return self._count

    def len(self) -> int:
        return self._count

    def mem_bytes(self) -> int:
        return (len(self.exact_set) + sum(len(b) for b in self.band_sets)) * 16

    def save(self, path: str) -> None:
        tmp_path = path + ".tmp"
        data = {
            "exact": list(self.exact_set),
            "bands": [list(b) for b in self.band_sets],
            "count": self._count,
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    @staticmethod
    def load(path: str) -> FakeDedupIndex:
        idx = FakeDedupIndex()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        idx.exact_set = set(data.get("exact", []))
        idx.band_sets = [set(b) for b in data.get("bands", [[]] * 8)]
        idx._count = data.get("count", len(idx.exact_set))
        return idx

    def merge_from(self, other: FakeDedupIndex) -> None:
        self.exact_set.update(other.exact_set)
        for i in range(8):
            self.band_sets[i].update(other.band_sets[i])
        self._count = len(self.exact_set)


def fake_version() -> str:
    return "0.1.0-fake"


def fake_cpu_features() -> list[str]:
    return ["avx2", "fma", "bmi2"]


def fake_bench_stage1(texts: list[str], cfg: dict[str, Any], seconds: float) -> float:
    return float(len(texts)) / max(seconds, 0.001)


# Register fake scour_core in sys.modules if real one is not available
if "scour_core" not in sys.modules:
    try:
        import scour_core  # type: ignore # noqa: F401
    except ImportError:
        fake_mod = types.ModuleType("scour_core")
        fake_mod.Engine = FakeEngine  # type: ignore
        fake_mod.Stage1Batch = FakeStage1Batch  # type: ignore
        fake_mod.DocResult = FakeDocResult  # type: ignore
        fake_mod.DedupIndex = FakeDedupIndex  # type: ignore
        fake_mod.version = fake_version  # type: ignore
        fake_mod.cpu_features = fake_cpu_features  # type: ignore
        fake_mod.bench_stage1 = fake_bench_stage1  # type: ignore
        sys.modules["scour_core"] = fake_mod


@pytest.fixture
def default_cfg_dict() -> dict[str, Any]:
    """Default engine configuration dictionary matching Section 2.2."""
    return {
        "chunk_target_chars": 800,
        "chunk_min_chars": 100,
        "chunk_max_chars": 2000,
        "url_policy": "smart",
        "allow_hosts": ["wikipedia.org", "github.com"],
        "block_hosts": ["spam.com"],
        "promo_lo": 0.30,
        "promo_hi": 0.70,
        "gpu_drop_p": 0.85,
        "gpu_strip_p": 0.50,
        "min_doc_chars": 200,
        "clean_threshold": 6.0,
        "pii_mode": "redact",
        "pii_drop_min_hits": 1,
        "allowed_langs": ["en", "hi", "hinglish"],
        "rust_threads": 2,
        "lexicon_dir": "configs/lexicons",
        "domain_mode": "dataset",
        "emit_fragments": False,
        "normalize_unicode": True,
        "dedup_shingle_k": 5,
        "keep_urls_in_code": True,
        "max_doc_chars": 2000000,
    }


@pytest.fixture
def fake_engine(default_cfg_dict: dict[str, Any]) -> FakeEngine:
    return FakeEngine(default_cfg_dict)


@pytest.fixture
def fake_dedup() -> FakeDedupIndex:
    return FakeDedupIndex()


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("test_corpus")
    generate_corpus(out_dir=out, n_docs=60, seed=123)
    return out
