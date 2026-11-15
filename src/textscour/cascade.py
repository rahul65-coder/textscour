"""Pipeline worker state machine, Stage 1 execution, and GPU classifier cascade routing.

Runs as spawned worker processes consuming RawBatch messages, running native
Rust Stage 1 cleaning automata, routing ambiguous candidate chunks to the GPU
inference server (or rules-only fallback), and emitting ResultBatch messages.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Any

from textscour.config import Config, to_engine_dict
from textscour.errors import ScourError
from textscour.schema import compile_extractor


@dataclass
class ProcessedDoc:
    """Wrapper holding finalized document attributes with id and domain."""

    id: str
    domain: str
    clean: bool
    text: str
    length: int
    score: float
    reasons: list[str] = field(default_factory=list)
    fragments: list[tuple[str, str]] = field(default_factory=list)
    exact: int = 0
    bands: list[int] = field(default_factory=lambda: [0] * 8)
    orig_len: int = 0
    lang: str = "en"


@dataclass
class DocResultFallback:
    """Interface-compatible fallback container when scour_core is absent."""

    clean: bool
    text: str
    length: int
    score: float
    reasons: list[str] = field(default_factory=list)
    fragments: list[tuple[str, str]] = field(default_factory=list)
    exact: int = 0
    bands: list[int] = field(default_factory=lambda: [0] * 8)
    orig_len: int = 0
    lang: str = "en"
    domain: str = ""
    id: str = ""


from textscour.gpu_scorer import GpuJob, GpuResult


@dataclass
class ResultBatch:
    """Final processed batch emitted from pipeline worker to OrderedSourceWriter."""

    batch_id: int
    source_key: str
    records: list[Any]
    stats: dict[str, Any]
    stream_state: dict[str, Any] | None = None
    bytes_in: int = 0  # Raw bytes from source batch, forwarded for throughput accounting


class FallbackStage1Batch:
    """Fallback Stage1Batch implementation when scour_core native library is compiling."""

    def __init__(self, texts: list[str], cfg: dict[str, Any]) -> None:
        self.texts = texts
        self.cfg = cfg
        self.min_doc_chars = cfg.get("min_doc_chars", 200)
        self.clean_threshold = cfg.get("clean_threshold", 6.0)

    def n_docs(self) -> int:
        return len(self.texts)

    def ambiguous(self) -> tuple[list[int], list[int], list[str]]:
        # Heuristic: return empty ambiguous queue for fallback
        return ([], [], [])

    def finalize(self, probs: list[float], mode: str) -> list[DocResultFallback]:
        results = []
        for idx, text in enumerate(self.texts):
            clean = len(text) >= self.min_doc_chars
            reasons = [] if clean else ["too_short"]
            score = 8.0 if clean else 3.0
            # XXH3-like hash fallback
            import xxhash

            exact_h = xxhash.xxh3_64_intdigest(text.encode("utf-8")) if text else 0
            results.append(
                DocResultFallback(
                    clean=clean,
                    text=text,
                    length=len(text),
                    score=score,
                    reasons=reasons,
                    fragments=[],
                    exact=exact_h,
                    bands=[exact_h ^ i for i in range(8)],
                    orig_len=len(text),
                    lang="en",
                    domain="",
                )
            )
        return results


class FallbackEngine:
    """Fallback Engine implementation when scour_core is absent."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def stage1(self, texts: list[str]) -> FallbackStage1Batch:
        return FallbackStage1Batch(texts, self.cfg)


def get_engine(cfg: Config) -> Any:
    """Instantiate scour_core.Engine or FallbackEngine if native core not yet built."""
    engine_dict = to_engine_dict(cfg)
    try:
        import scour_core

        return scour_core.Engine(engine_dict)
    except ImportError:
        return FallbackEngine(engine_dict)


def process_batch(
    worker_id: int,
    raw_batch: Any,
    engine: Any,
    extractor: Any,
    cfg: Config,
    gpu_q: Any | None = None,
    gpu_res_q: Any | None = None,
    quarantine_q: Any | None = None,
) -> ResultBatch:
    """Process a single RawBatch through Stage 1 cleaning and optional GPU cascade."""
    start_t = time.time()
    batch_id = raw_batch.batch_id
    source_key = raw_batch.source_key
    docs = raw_batch.docs
    ids = raw_batch.ids
    domains = raw_batch.domains

    texts: list[str] = []
    valid_indices: list[int] = []

    # 1. Schema extraction with quarantine handling
    for idx, doc in enumerate(docs):
        rec_id = ids[idx] if idx < len(ids) else f"doc_{idx}"
        try:
            extracted = extractor(doc)
            if extracted is None:
                if quarantine_q:
                    quarantine_q.put((source_key, batch_id, rec_id, "E-SCHEMA-001", "Text extracted as None"))
                continue
            if not isinstance(extracted, str):
                if quarantine_q:
                    quarantine_q.put((source_key, batch_id, rec_id, "E-SCHEMA-002", f"Text is {type(extracted)}"))
                continue
            if not extracted.strip():
                if quarantine_q:
                    quarantine_q.put((source_key, batch_id, rec_id, "E-SCHEMA-001", "Text extracted as empty"))
                continue
            texts.append(extracted.strip())
            valid_indices.append(idx)
        except Exception as e:
            if quarantine_q:
                err_code = getattr(e, "code", "E-SCHEMA-001")
                quarantine_q.put((source_key, batch_id, rec_id, err_code, str(e)))

    if not texts:
        return ResultBatch(
            batch_id=batch_id,
            source_key=source_key,
            records=[],
            stats={"duration": time.time() - start_t, "ambiguous_count": 0, "docs_in": len(docs)},
            stream_state=raw_batch.stream_state,
            bytes_in=getattr(raw_batch, "bytes_in", 0),
        )

    # 2. Stage 1 processing
    stage1_batch = engine.stage1(texts)
    doc_indices, chunk_indices, ambiguous_chunks = stage1_batch.ambiguous()

    # 3. GPU routing or rules-only resolution
    gpu_enabled = (
        cfg.cascade.gpu_mode != "off"
        and cfg.cascade.model is not None
        and gpu_q is not None
        and gpu_res_q is not None
        and len(ambiguous_chunks) > 0
    )

    if gpu_enabled:
        max_chunk_slice = cfg.runtime.gpu_job_chunks
        n_chunks = len(ambiguous_chunks)
        total_slices = (n_chunks + max_chunk_slice - 1) // max_chunk_slice

        # Dispatch jobs
        for slice_idx in range(total_slices):
            start_idx = slice_idx * max_chunk_slice
            end_idx = min(n_chunks, start_idx + max_chunk_slice)
            sub_texts = ambiguous_chunks[start_idx:end_idx]
            gpu_q.put(GpuJob(worker_id=worker_id, batch_id=batch_id, job_idx=slice_idx, texts=sub_texts))

        # Collect results directly from dedicated worker response queue
        collected_probs: dict[int, list[float]] = {}
        while len(collected_probs) < total_slices:
            res = gpu_res_q.get()
            r_bid = getattr(res, "batch_id", res[1] if isinstance(res, (tuple, list)) else None)
            if r_bid == batch_id:
                r_jidx = getattr(res, "job_idx", res[2] if isinstance(res, (tuple, list)) else 0)
                r_probs = getattr(res, "probs", res[3] if isinstance(res, (tuple, list)) else [])
                collected_probs[r_jidx] = r_probs

        # Assemble probabilities in order
        all_probs: list[float] = []
        for s_idx in range(total_slices):
            all_probs.extend(collected_probs[s_idx])

        doc_results = stage1_batch.finalize(all_probs, "gpu")
    else:
        # Rules-only fallback mode
        doc_results = stage1_batch.finalize([], "rules_only")

    # 4. Bind IDs, domains, and build ResultBatch with pickleable ProcessedDoc
    final_records = []
    for out_idx, doc_res in enumerate(doc_results):
        orig_idx = valid_indices[out_idx]
        cur_id = ids[orig_idx]
        cur_dom = domains[orig_idx]
        wrapped = ProcessedDoc(
            id=cur_id,
            domain=cur_dom,
            clean=doc_res.clean,
            text=doc_res.text,
            length=doc_res.length,
            score=doc_res.score,
            reasons=list(doc_res.reasons),
            fragments=list(doc_res.fragments),
            exact=doc_res.exact,
            bands=list(doc_res.bands),
            orig_len=doc_res.orig_len,
            lang=doc_res.lang,
        )
        final_records.append(wrapped)

    duration = time.time() - start_t
    stats = {
        "duration": duration,
        "docs_in": len(docs),
        "docs_processed": len(final_records),
        "ambiguous_count": len(ambiguous_chunks),
        "gpu_chunks": len(ambiguous_chunks) if gpu_enabled else 0,
        "total_chunks": max(len(ambiguous_chunks), len(docs)),
        "gpu_routed": gpu_enabled,
    }

    return ResultBatch(
        batch_id=batch_id,
        source_key=source_key,
        records=final_records,
        stats=stats,
        stream_state=raw_batch.stream_state,
        bytes_in=getattr(raw_batch, "bytes_in", 0),
    )


def worker_main(
    worker_id: int,
    cfg: Config,
    in_q: Any,
    out_q: Any,
    gpu_q: Any | None,
    gpu_res_q: Any | None,
    quarantine_q: Any | None,
    stop_event: Any,
    heartbeat_dict: Any | None = None,
) -> None:
    """Subprocess pipeline worker executing Stage 1 and GPU routing loop."""
    # Only consumer queue can cancel join thread if worker exits early
    if in_q is not None and hasattr(in_q, "cancel_join_thread"):
        try:
            in_q.cancel_join_thread()
        except Exception:
            pass

    try:
        engine = get_engine(cfg)
        # Precompile extractors per source for heterogeneous schemas
        extractors: dict[str, Any] = {}
        for s in cfg.sources:
            ext = compile_extractor(s.text)
            if s.repo:
                extractors[s.repo] = ext
            if s.name:
                extractors[s.name] = ext
            if s.paths:
                extractors[s.paths[0]] = ext
        default_extractor = compile_extractor(cfg.sources[0].text if cfg.sources else None)  # type: ignore

        while not stop_event.is_set():
            if heartbeat_dict is not None:
                heartbeat_dict[worker_id] = time.time()

            try:
                batch = in_q.get(timeout=0.25)
                if batch is None:
                    break
            except queue.Empty:
                continue

            extractor = extractors.get(batch.source_key, default_extractor)
            result_batch = process_batch(
                worker_id=worker_id,
                raw_batch=batch,
                engine=engine,
                extractor=extractor,
                cfg=cfg,
                gpu_q=gpu_q,
                gpu_res_q=gpu_res_q,
                quarantine_q=quarantine_q,
            )
            out_q.put(result_batch)

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Fatal worker crash handled by supervisor
        raise
    finally:
        try:
            out_q.put(None, timeout=1.0)
        except Exception:
            pass
