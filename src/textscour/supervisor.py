"""Multiprocessing supervisor, process lifecycle management, and AIMD dynamic control loop.

Orchestrates spawned child processes (readers, stage 1 workers, GPU server,
and writer), provides shared memory telemetry and heartbeat monitoring, and executes
an Additive-Increase / Multiplicative-Decrease (AIMD) feedback loop adjusting queue
depths, reader allocations, batch hints, and memory ceilings.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from textscour.cascade import worker_main
from textscour.checkpoint import Checkpoint
from textscour.config import Config
from textscour.env_probe import EnvInfo, probe
from textscour.errors import ScourError
from textscour.metrics import Dashboard, write_report
from textscour.sources import reader_main
from textscour.uploader import HFUploader
from textscour.writer import Manifest, OrderedSourceWriter, QuarantineWriter, merge_outputs


def writer_proc_main(
    cfg: Config,
    out_q: Any,
    quarantine_q: Any,
    stop_event: Any,
    shared_counters: Any,
    resume_ckpt: dict[str, Any] | None = None,
    num_workers: int = 2,
) -> None:
    """Subprocess writer managing OrderedSourceWriter, dedup, and quarantine."""
    for q_obj in (out_q, quarantine_q):
        if q_obj is not None and hasattr(q_obj, "cancel_join_thread"):
            try:
                q_obj.cancel_join_thread()
            except Exception:
                pass

    try:
        state_dir = Path(cfg.output.dir) / "state" if cfg.output.dir != "outputs" else Path("state")
        ckpt = Checkpoint(state_dir=state_dir)
        manifest = Manifest(cfg.output.dir)
        quarantine = QuarantineWriter(cfg.output.dir, max_error_rate=cfg.runtime.max_error_rate)

        # Load DedupIndex if available
        dedup_index = None
        if cfg.dedup.mode != "off":
            try:
                import scour_core

                dedup_index = scour_core.DedupIndex()
                dedup_file = resume_ckpt.get("dedup_file") if resume_ckpt else None
                if dedup_file and Path(dedup_file).is_file():
                    dedup_index = scour_core.DedupIndex.load(dedup_file)
            except Exception:
                pass

        # Per-source writers
        writers: dict[str, OrderedSourceWriter] = {}
        for src in cfg.sources:
            s_key = src.repo or src.name or (src.paths[0] if src.paths else "default")
            writers[s_key] = OrderedSourceWriter(
                source_key=s_key,
                cfg=cfg,
                ckpt=ckpt,
                dedup_index=dedup_index,
                manifest=manifest,
                uploader_q=None,
                resume_state=resume_ckpt,
            )

        # bytes_in accumulator (from ResultBatch.bytes_in carried from RawBatch)
        total_bytes_in: int = 0
        total_gpu_chunks: int = 0
        total_all_chunks: int = 0
        active_workers = num_workers

        def _sync_telemetry() -> None:
            c_clean = sum(w.records_clean for w in writers.values())
            c_garbage = sum(w.records_garbage for w in writers.values())
            c_quar = quarantine.error_count
            shared_counters["docs_clean"] = c_clean
            shared_counters["docs_garbage"] = c_garbage
            shared_counters["quarantined"] = c_quar
            shared_counters["docs_in"] = c_clean + c_garbage + c_quar
            shared_counters["gpu_chunks"] = total_gpu_chunks
            shared_counters["total_chunks"] = max(1, total_all_chunks)

            merged_reasons: dict[str, int] = {}
            for w in writers.values():
                for r, cnt in w.reason_counts.items():
                    merged_reasons[r] = merged_reasons.get(r, 0) + cnt
            shared_counters["reasons"] = merged_reasons

            shared_counters["per_source"] = {
                w.source_key: {
                    "in": w.records_clean + w.records_garbage,
                    "clean": w.records_clean,
                    "garbage": w.records_garbage,
                }
                for w in writers.values()
            }

            # Accumulate dedup stats
            exact_dups = sum(w.reason_counts.get("duplicate_exact", 0) for w in writers.values())
            near_dups = sum(w.reason_counts.get("duplicate_near", 0) for w in writers.values())
            idx_entries = 0
            if dedup_index is not None:
                try:
                    idx_entries = dedup_index.len() if hasattr(dedup_index, "len") else 0
                except Exception:
                    idx_entries = 0
            shared_counters["dedup_stats"] = {
                "exact_duplicates": exact_dups,
                "near_duplicates": near_dups,
                "unique_seen": c_clean,
                "index_entries": idx_entries,
            }

            # Sync accumulated bytes_in
            shared_counters["bytes_in"] = total_bytes_in

        while active_workers > 0:
            # Drain quarantine entries
            try:
                while True:
                    q_entry = quarantine_q.get_nowait()
                    quarantine.quarantine(*q_entry)
            except queue.Empty:
                pass

            # Read result batch
            try:
                result_batch = out_q.get(timeout=0.5)
                if result_batch is None:
                    active_workers -= 1
                    continue

                s_key = result_batch.source_key
                if s_key in writers:
                    writers[s_key].push(result_batch)

                # Accumulate bytes and chunk telemetry from this batch
                batch_bytes = getattr(result_batch, "bytes_in", 0)
                if not batch_bytes and getattr(result_batch, "records", None):
                    batch_bytes = sum(getattr(r, "length", 0) or len(getattr(r, "text", "")) for r in result_batch.records)
                total_bytes_in += batch_bytes

                batch_stats = getattr(result_batch, "stats", {})
                total_gpu_chunks += batch_stats.get("gpu_chunks", 0)
                total_all_chunks += batch_stats.get("total_chunks", 0)

                _sync_telemetry()

            except queue.Empty:
                _sync_telemetry()
                continue

        # Final flush
        for w in writers.values():
            w.flush()

        _sync_telemetry()

    except Exception:
        raise


class ControlLoop:
    """Runs every 10 seconds adjusting batch hints and resource limits via AIMD rules."""

    def __init__(
        self,
        supervisor: Supervisor,
        interval_sec: float = 10.0,
    ) -> None:
        self.supervisor = supervisor
        self.interval_sec = interval_sec
        self.last_run = time.time()
        self.low_rss_streak = 0
        self.ewma_dps = 0.0
        self.dps_drop_streak = 0

    def step(self) -> None:
        """Evaluate telemetry metrics and apply AIMD tuning decisions."""
        now = time.time()
        dt = max(1.0, now - self.last_run)
        self.last_run = now

        sup = self.supervisor
        in_depth = sup.in_q.qsize()
        out_depth = sup.out_q.qsize()
        gpu_depth = sup.gpu_q.qsize() if sup.gpu_q else 0

        # Update queue telemetry in counters
        sup.counters["in_q_depth"] = in_depth
        sup.counters["out_q_depth"] = out_depth
        sup.counters["gpu_q_depth"] = gpu_depth

        # 1. RSS check
        rss_gb = sup.dashboard._get_rss_gb()
        ram_budget = sup.env.ram_total_gb * sup.cfg.runtime.ram_budget_frac

        if rss_gb > 0.85 * ram_budget:
            # RSS > 85% budget -> multiplicative decrease
            old_hint = sup.batch_docs_hint.value
            new_hint = max(128, old_hint // 2)
            sup.batch_docs_hint.value = new_hint
            sup.dashboard.log_control_event("RSS_OVERCOMMIT_HALVE_BATCH", f"RSS={rss_gb:.1f}GB, new_batch={new_hint}")
            self.low_rss_streak = 0
        elif rss_gb < 0.60 * ram_budget:
            self.low_rss_streak += 1
            if self.low_rss_streak >= 5:
                # Additive increase
                old_hint = sup.batch_docs_hint.value
                new_hint = int(old_hint * 1.25)
                sup.batch_docs_hint.value = new_hint
                sup.dashboard.log_control_event("LOW_RSS_INCREASE_BATCH", f"streak=5, new_batch={new_hint}")
                self.low_rss_streak = 0
        else:
            self.low_rss_streak = 0

        # 2. Disk space check
        env_now = probe(sup.cfg.output.dir)
        disk_budget = sup.cfg.runtime.disk_budget_gb
        if env_now.disk_free_gb < 0.10 * disk_budget:
            sup.paused_flag.value = 1
            sup.dashboard.log_control_event("DISK_LOW_PAUSE_READERS", f"disk_free={env_now.disk_free_gb:.1f}GB")
        elif env_now.disk_free_gb > 0.25 * disk_budget and sup.paused_flag.value == 1:
            sup.paused_flag.value = 0
            sup.dashboard.log_control_event("DISK_RESUMED", f"disk_free={env_now.disk_free_gb:.1f}GB")

        # 3. GPU backlog check
        if sup.gpu_q and gpu_depth > (2 * sup.cfg.runtime.gpu_inflight):
            old_tokens = sup.gpu_batch_tokens_hint.value
            new_tokens = min(65536, int(old_tokens * 1.5))
            sup.gpu_batch_tokens_hint.value = new_tokens
            sup.dashboard.log_control_event("GPU_BACKLOG_INCREASE_TOKENS", f"backlog={gpu_depth}, tokens={new_tokens}")


class Supervisor:
    """Central pipeline coordinator orchestrating worker processes, telemetry, and health."""

    def __init__(
        self,
        cfg: Config,
        env: EnvInfo,
        tuned: dict[str, Any] | None = None,
        dry_run: int | None = None,
        resume_checkpoint: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.env = env
        self.tuned = dict(tuned) if tuned else {}
        self.dry_run = dry_run
        self.resume_ckpt = resume_checkpoint

        # If autotune params not provided, load from autotune_cache.json if available
        if not self.tuned:
            from textscour.autotune import CACHE_FILE_DEFAULT, load_cache

            cached = load_cache(CACHE_FILE_DEFAULT, env.fingerprint)
            if cached and isinstance(cached, dict):
                self.tuned = dict(cached)
            else:
                try:
                    p = Path(CACHE_FILE_DEFAULT)
                    if p.is_file():
                        with open(p, "r", encoding="utf-8") as f:
                            c_data = json.load(f)
                            if isinstance(c_data, dict):
                                for v in c_data.values():
                                    if isinstance(v, dict) and "params" in v:
                                        self.tuned = dict(v["params"])
                                        break
                except Exception:
                    pass

        # Ensure all 7 effective runtime tuning parameters are populated
        eff_cores = env.cpu_cores_effective
        default_params = {
            "workers": cfg.runtime.workers if isinstance(cfg.runtime.workers, int) else max(1, eff_cores // 2),
            "rust_threads": getattr(cfg.runtime, "rust_threads", 0) or 0,
            "readers": cfg.runtime.readers if isinstance(cfg.runtime.readers, int) else 1,
            "batch_docs": cfg.runtime.batch_docs if isinstance(cfg.runtime.batch_docs, int) else 512,
            "queue_depth": cfg.runtime.queue_depth if isinstance(cfg.runtime.queue_depth, int) else 4,
            "gpu_batch_tokens": cfg.runtime.gpu_batch_tokens if isinstance(cfg.runtime.gpu_batch_tokens, int) else 16384,
            "chunk_target_chars": getattr(cfg.stage1, "chunk_target_chars", 800) if hasattr(cfg, "stage1") else 800,
        }
        for k, v in default_params.items():
            if k not in self.tuned or self.tuned[k] is None:
                self.tuned[k] = v

        self.ctx = mp.get_context("spawn")
        self.stop_event = self.ctx.Event()
        self.paused_flag = self.ctx.Value("i", 0)

        # Tune runtime parameters
        num_workers = self.tuned["workers"]
        num_readers = self.tuned["readers"]
        batch_docs = self.tuned["batch_docs"]
        gpu_tokens = self.tuned["gpu_batch_tokens"]

        self.num_workers = max(1, num_workers)
        self.num_readers = max(1, num_readers)
        self.batch_docs_hint = self.ctx.Value("i", batch_docs)
        self.gpu_batch_tokens_hint = self.ctx.Value("i", gpu_tokens)

        # Queues
        q_depth = cfg.runtime.queue_depth if isinstance(cfg.runtime.queue_depth, int) else 4
        self.in_q = self.ctx.Queue(maxsize=max(32, q_depth * self.num_workers * 2))
        self.out_q = self.ctx.Queue(maxsize=max(32, q_depth * self.num_workers * 2))
        self.quarantine_q = self.ctx.Queue()

        self.gpu_q: Any | None = None
        self.gpu_res_qs: list[Any] | None = None
        can_gpu = (cfg.cascade.gpu_mode != "off") and env.cuda and (cfg.cascade.model is not None)
        if can_gpu:
            self.gpu_q = self.ctx.Queue()
            self.gpu_res_qs = [self.ctx.Queue() for _ in range(self.num_workers)]

        # Shared telemetry counters
        self.manager = self.ctx.Manager()
        self.counters = self.manager.dict(
            {
                "docs_in": 0,
                "docs_clean": 0,
                "docs_garbage": 0,
                "quarantined": 0,
                "bytes_in": 0,
                "gpu_chunks": 0,
                "total_chunks": 1,
                "in_q_depth": 0,
                "out_q_depth": 0,
                "gpu_q_depth": 0,
                "reasons": self.manager.dict(),
            }
        )
        self.heartbeats = self.manager.dict()

        self.dashboard = Dashboard(cfg.output.dir, self.counters)
        self.control_loop = ControlLoop(self)

        self.uploader: HFUploader | None = None
        if cfg.upload.enabled and cfg.upload.repo:
            self.uploader = HFUploader(
                repo=cfg.upload.repo,
                private=cfg.upload.private,
                commit_every_shards=cfg.upload.commit_every_shards,
                commit_every_seconds=cfg.upload.commit_every_seconds,
                delete_after_upload=cfg.upload.delete_after_upload,
                state_backend=cfg.upload.state_backend,
            )

        self.processes: list[mp.Process] = []

    def _setup_signals(self) -> None:
        """Register graceful termination on SIGINT and SIGTERM."""
        def _handle_signal(signum: int, frame: Any) -> None:
            self.stop_event.set()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    def run(self) -> dict[str, Any]:
        """Spawn workers, monitor execution, run control loop, and finalize."""
        self._setup_signals()
        start_time = time.time()
        self.dashboard.start()

        if self.uploader:
            self.uploader.start()

        # 1. Spawn Writer
        writer_proc = self.ctx.Process(
            target=writer_proc_main,
            args=(self.cfg, self.out_q, self.quarantine_q, self.stop_event, self.counters, self.resume_ckpt, self.num_workers),
            name="textscour-writer",
        )
        writer_proc.start()
        self.processes.append(writer_proc)

        # 1.5 Spawn Dedicated GPU Server Processes (Dual-T4 saturation)
        gpu_procs: list[mp.Process] = []
        if self.gpu_q is not None and self.gpu_res_qs is not None:
            from textscour.gpu_scorer import gpu_main

            env_gpus = getattr(self.env, "gpus", [])
            gpu_cnt = getattr(self.env, "gpu_count", len(env_gpus))
            num_gpus = gpu_cnt if (self.env.cuda and gpu_cnt > 0) else 1
            for dev_idx in range(num_gpus):
                gp = self.ctx.Process(
                    target=gpu_main,
                    args=(dev_idx, self.cfg, self.counters, self.gpu_q, self.gpu_res_qs),
                    name=f"textscour-gpu-{dev_idx}",
                )
                gp.start()
                gpu_procs.append(gp)
                self.processes.append(gp)

        # 2. Spawn Pipeline Workers
        for w_idx in range(self.num_workers):
            w_res_q = self.gpu_res_qs[w_idx] if self.gpu_res_qs is not None else None
            p = self.ctx.Process(
                target=worker_main,
                args=(
                    w_idx,
                    self.cfg,
                    self.in_q,
                    self.out_q,
                    self.gpu_q,
                    w_res_q,
                    self.quarantine_q,
                    self.stop_event,
                    self.heartbeats,
                ),
                name=f"textscour-worker-{w_idx}",
            )
            p.start()
            self.processes.append(p)

        # 3. Spawn Readers
        reader_procs = []
        for r_idx in range(self.num_readers):
            p = self.ctx.Process(
                target=reader_main,
                args=(
                    r_idx,
                    self.num_readers,
                    self.cfg,
                    self.in_q,
                    self.stop_event,
                    self.batch_docs_hint.value,
                    self.resume_ckpt,
                    self.dry_run,
                    self.num_workers,
                ),
                name=f"textscour-reader-{r_idx}",
            )
            p.start()
            reader_procs.append(p)
            self.processes.append(p)

        # 4. Supervision Loop
        last_ctrl_t = time.time()
        max_sec = self.cfg.runtime.max_runtime_seconds
        workers_signaled = False
        consecutive_drained = 0

        try:
            while not self.stop_event.is_set():
                now = time.time()
                elapsed = now - start_time

                # Max runtime threshold check (graceful shutdown before timeout)
                if max_sec and (elapsed >= max_sec):
                    self.stop_event.set()
                    break

                # Dry-run threshold check
                if self.dry_run and self.counters.get("docs_in", 0) >= self.dry_run:
                    self.stop_event.set()
                    break

                # Control Loop trigger every 10s
                if now - last_ctrl_t >= self.control_loop.interval_sec:
                    self.control_loop.step()
                    last_ctrl_t = now

                # Check reader completion
                all_readers_done = all(not p.is_alive() for p in reader_procs)
                if all_readers_done and not workers_signaled:
                    # All readers done: push EOF sentinel (None) per worker to in_q (FIFO preserves unread docs)
                    for _ in range(self.num_workers):
                        try:
                            self.in_q.put(None, timeout=1.0)
                        except Exception:
                            pass
                    workers_signaled = True

                # Check if all workers have finished
                all_workers_done = all(not p.is_alive() for p in self.processes if "worker" in p.name)

                # Safeguard: do NOT exit solely when queues appear empty if workers are still mid-batch (gpu_res_qs in flight)
                if all_readers_done and all_workers_done:
                    gpu_busy = (self.gpu_q is not None and not self.gpu_q.empty())
                    res_busy = any(not rq.empty() for rq in self.gpu_res_qs if rq is not None) if self.gpu_res_qs else False
                    out_busy = not self.out_q.empty()

                    if not gpu_busy and not res_busy and not out_busy:
                        consecutive_drained += 1
                        if consecutive_drained >= 2:
                            if writer_proc.is_alive():
                                writer_proc.join(timeout=15.0)
                            break
                    else:
                        consecutive_drained = 0
                else:
                    consecutive_drained = 0

                time.sleep(0.05)

        finally:
            self.stop_event.set()

            # Wait for readers
            t_readers_start = start_time
            for p in reader_procs:
                p.join(timeout=2.0)
                if p.is_alive():
                    p.terminate()
            t_readers_done = time.time()

            # Signal workers to drain
            for _ in range(self.num_workers):
                try:
                    self.in_q.put_nowait(None)
                except Exception:
                    pass

            # Wait for workers
            for p in self.processes:
                if "worker" in p.name:
                    p.join(timeout=2.0)
                    if p.is_alive():
                        p.terminate()
            t_workers_done = time.time()

            # Signal GPU workers to drain & terminate cleanly
            for _ in gpu_procs:
                if self.gpu_q is not None:
                    try:
                        self.gpu_q.put_nowait(None)
                    except Exception:
                        pass
            for gp in gpu_procs:
                if gp.is_alive():
                    gp.join(timeout=5.0)
                    if gp.is_alive():
                        gp.terminate()

            # Signal writer
            if writer_proc.is_alive():
                try:
                    self.out_q.put_nowait(None)
                except Exception:
                    pass
                writer_proc.join(timeout=5.0)
                if writer_proc.is_alive():
                    writer_proc.terminate()
            t_writer_done = time.time()

            if self.uploader:
                self.uploader.stop()

            self.dashboard.stop()

            # Final merge if configured
            if self.cfg.output.final_merge:
                merge_outputs(self.cfg)

            # Close and cancel join thread on all queues to prevent Python atexit deadlock
            all_qs = [self.in_q, self.out_q, self.gpu_q, self.quarantine_q]
            if self.gpu_res_qs:
                all_qs.extend(self.gpu_res_qs)
            for q in all_qs:
                if q is not None:
                    try:
                        if hasattr(q, "cancel_join_thread"):
                            q.cancel_join_thread()
                        if hasattr(q, "close"):
                            q.close()
                    except Exception:
                        pass

            elapsed_sec = round(time.time() - start_time, 1)
            bytes_in = int(self.counters.get("bytes_in", 0))
            mb_per_sec = round((bytes_in / (1024 * 1024)) / max(0.001, elapsed_sec), 3)

            stage_timings = {
                "reader": max(0.001, round(t_readers_done - t_readers_start, 3)),
                "cascade": max(0.001, round(t_workers_done - t_readers_start, 3)),
                "writer": max(0.001, round(t_writer_done - t_readers_start, 3)),
            }

            dedup_stats = dict(self.counters.get("dedup_stats", {}))
            if not dedup_stats:
                dedup_stats = {
                    "exact_duplicates": 0,
                    "near_duplicates": 0,
                    "unique_seen": self.counters.get("docs_clean", 0),
                    "index_entries": 0,
                }

            # Write final reports
            summary_data = {
                "docs_in": self.counters.get("docs_in", 0),
                "docs_clean": self.counters.get("docs_clean", 0),
                "docs_garbage": self.counters.get("docs_garbage", 0),
                "quarantined": self.counters.get("quarantined", 0),
                "bytes_in": bytes_in,
                "mb_per_sec": mb_per_sec,
                "gpu_chunks": int(self.counters.get("gpu_chunks", 0)),
                "total_chunks": int(self.counters.get("total_chunks", 1)),
                "stage_timings": stage_timings,
                "dedup_stats": dedup_stats,
                "elapsed_sec": elapsed_sec,
                "autotune": dict(self.tuned),
                "reasons": dict(self.counters.get("reasons", {})),
                "per_source": dict(self.counters.get("per_source", {})),
            }
            write_report(self.cfg.output.dir, summary_data)

        return summary_data
