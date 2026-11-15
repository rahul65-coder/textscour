"""
GPU-Accelerated & CPU-Fallback Classifier Scorer for TEXTSCOUR.
Provides ModelBundle for calibrated inference and gpu_main worker for dynamic batching.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from textscour.errors import ScourError

logger = logging.getLogger(__name__)

try:
    import pynvml  # type: ignore
except ImportError:
    pynvml = None


@dataclass
class GpuJob:
    """Queue message sent from pipeline workers to GPU scorer."""
    worker_id: int
    batch_id: int
    job_idx: int
    texts: list[str]

    def __iter__(self):
        yield self.worker_id
        yield self.batch_id
        yield self.job_idx
        yield self.texts


@dataclass
class GpuResult:
    """Queue message sent from GPU scorer to pipeline workers."""
    worker_id: int
    batch_id: int
    job_idx: int
    probs: list[float]

    def __iter__(self):
        yield self.worker_id
        yield self.batch_id
        yield self.job_idx
        yield self.probs


class ModelBundle:
    """
    Encapsulates sequence classification model, tokenizer, and calibration parameters.
    Applies temperature scaling: p = sigmoid(logit / T).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device,
        temperature: float = 1.0,
        gpu_drop_p: float = 0.85,
        gpu_strip_p: float = 0.50,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = max(float(temperature), 1e-4)
        self.gpu_drop_p = float(gpu_drop_p)
        self.gpu_strip_p = float(gpu_strip_p)

    @classmethod
    def load(
        cls,
        path_or_repo: str | None,
        device: str = "auto",
        dtype: torch.dtype | None = None,
    ) -> ModelBundle:
        """
        Loads tokenizer, model, and optional calibration.json metadata.
        Falls back to CPU when CUDA is unavailable or requested.
        """
        if not path_or_repo:
            raise ScourError(
                code="E-GPU-003",
                stage="gpu",
                message="Model path or repository identifier cannot be None or empty.",
                hint="Specify a valid model path in cascade.model configuration.",
            )

        # Resolve device
        if device == "auto":
            target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ScourError(
                    code="E-GPU-001",
                    stage="gpu",
                    message="CUDA was explicitly requested but no compatible CUDA device is available.",
                    hint="Set cascade.gpu_mode: 'off' or run on a GPU-enabled instance.",
                )
            target_device = torch.device(device)
        else:
            target_device = torch.device("cpu")

        # Resolve dtype
        if dtype is None:
            if target_device.type == "cuda":
                target_dtype = torch.float16
            else:
                target_dtype = torch.float32
        else:
            target_dtype = dtype

        # Attempt to load calibration.json
        temperature = 1.0
        gpu_drop_p = 0.85
        gpu_strip_p = 0.50

        calib_candidates = [
            os.path.join(path_or_repo, "calibration.json"),
            "calibration.json",
            "configs/calibration.json",
        ]
        calib_loaded = False
        for cpath in calib_candidates:
            if os.path.exists(cpath):
                try:
                    with open(cpath, "r", encoding="utf-8") as f:
                        cdata = json.load(f)
                        temperature = float(cdata.get("temperature", 1.0))
                        gpu_drop_p = float(cdata.get("gpu_drop_p", 0.85))
                        gpu_strip_p = float(cdata.get("gpu_strip_p", 0.50))
                        logger.info("Loaded calibration from %s: T=%.3f, drop_p=%.3f, strip_p=%.3f",
                                    cpath, temperature, gpu_drop_p, gpu_strip_p)
                        calib_loaded = True
                        break
                except Exception as ex:
                    logger.warning("Failed reading calibration file %s: %s", cpath, ex)

        if not calib_loaded and not os.path.isdir(path_or_repo):
            try:
                from huggingface_hub import hf_hub_download
                calib_file = hf_hub_download(repo_id=path_or_repo, filename="calibration.json")
                with open(calib_file, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    temperature = float(cdata.get("temperature", 1.0))
                    gpu_drop_p = float(cdata.get("gpu_drop_p", 0.85))
                    gpu_strip_p = float(cdata.get("gpu_strip_p", 0.50))
                    logger.info("Downloaded calibration from Hugging Face Hub (%s): T=%.3f, drop_p=%.3f, strip_p=%.3f",
                                path_or_repo, temperature, gpu_drop_p, gpu_strip_p)
            except Exception as hf_ex:
                logger.debug("No remote calibration.json on HF Hub for '%s': %s", path_or_repo, hf_ex)

        # Load Hugging Face components
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(path_or_repo)
            model = AutoModelForSequenceClassification.from_pretrained(path_or_repo)
            model.to(device=target_device, dtype=target_dtype)
            model.eval()

            # Warm-up pass
            with torch.inference_mode():
                warm_tokens = tokenizer(
                    ["TEXTSCOUR warmup probe"],
                    padding=True,
                    truncation=True,
                    max_length=16,
                    return_tensors="pt",
                ).to(target_device)
                _ = model(**warm_tokens)

            return cls(
                model=model,
                tokenizer=tokenizer,
                device=target_device,
                temperature=temperature,
                gpu_drop_p=gpu_drop_p,
                gpu_strip_p=gpu_strip_p,
            )

        except Exception as e:
            raise ScourError(
                code="E-GPU-003",
                stage="gpu",
                message=f"Classifier model or tokenizer failed to load from '{path_or_repo}': {e}",
                cause=e,
                hint="Verify model path or Hugging Face repo access. For offline mode, ensure model weights are cached.",
            ) from e

    def predict_proba(self, texts: Sequence[str], max_length: int = 256) -> np.ndarray:
        """
        Runs calibrated inference on a list of texts and returns class probabilities.
        Returns float32 probabilities in [0.0, 1.0] for the noise/garbage class.
        """
        if not texts:
            return np.empty((0,), dtype=np.float32)

        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if hasattr(enc, "to"):
            enc = enc.to(self.device)
        elif isinstance(enc, dict):
            enc = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in enc.items()}

        with torch.inference_mode():
            if isinstance(enc, dict) or hasattr(enc, "items"):
                outputs = self.model(**enc)
            elif hasattr(enc, "to_dict"):
                outputs = self.model(**enc.to_dict())
            else:
                outputs = self.model(enc)
            logits = outputs.logits

            # For binary classification with 2 output logits: index 1 represents positive (noise/promo)
            # Calibrated softmax: sigmoid((logits[:, 1] - logits[:, 0]) / T) == softmax(logits / T, dim=-1)[:, 1]
            if logits.ndim == 2 and logits.shape[1] >= 2:
                probs = torch.softmax(logits / self.temperature, dim=-1)[:, 1].cpu().numpy().astype(np.float32)
            else:
                scaled_logits = logits.squeeze(-1) / self.temperature
                probs = torch.sigmoid(scaled_logits).cpu().numpy().astype(np.float32)

        return probs


def _get_shared_val(shared: Any, attr_name: str, default: Any) -> Any:
    """Helper to safely read a shared variable or multiprocessing Value."""
    if shared is None or not hasattr(shared, attr_name):
        return default
    val = getattr(shared, attr_name)
    if hasattr(val, "value"):
        return val.value
    return val


def _set_shared_val(shared: Any, attr_name: str, value: Any) -> None:
    """Helper to safely update a shared variable or multiprocessing Value."""
    if shared is None or not hasattr(shared, attr_name):
        return
    val = getattr(shared, attr_name)
    if hasattr(val, "value"):
        val.value = value
    else:
        try:
            setattr(shared, attr_name, value)
        except Exception:
            pass


def _route_result(gpu_res_qs: Any, res: GpuResult) -> None:
    """Routes inference result directly to worker-specific response queue if a list is provided."""
    if isinstance(gpu_res_qs, (list, tuple)):
        if 0 <= res.worker_id < len(gpu_res_qs):
            gpu_res_qs[res.worker_id].put(res)
        else:
            gpu_res_qs[0].put(res)
    else:
        gpu_res_qs.put(res)


def gpu_main(
    device_index: int,
    cfg: Any,
    shared: Any,
    gpu_q: Any,
    gpu_res_q: Any,
) -> None:
    """
    Dedicated GPU server subprocess.
    Performs dynamic batching over incoming GpuJobs from pipeline workers.
    Supports graceful CPU fallback when no GPU is present.
    """
    cascade_cfg = getattr(cfg, "cascade", None)
    gpu_mode = getattr(cascade_cfg, "gpu_mode", "auto") if cascade_cfg else "auto"

    if gpu_mode == "off":
        logger.info("GPU stage disabled: cascade.gpu_mode is 'off'.")
        # Drain any residual poison pills and exit
        _drain_until_stop(gpu_q, gpu_res_q)
        return

    # Check CUDA availability and configure device
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        try:
            torch.cuda.set_device(device_index)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
            device_str = f"cuda:{device_index}"
        except Exception as ex:
            logger.warning("Failed setting CUDA device %d: %s. Falling back to cuda:0", device_index, ex)
            device_str = "cuda:0"
    else:
        device_str = "cpu"
        threads = getattr(getattr(cfg, "runtime", None), "rust_threads", 2)
        if isinstance(threads, int) and threads > 0:
            torch.set_num_threads(threads)
        logger.info("CUDA not detected. GPU stage initializing on CPU fallback.")

    # Initialize NVML for GPU utilization metrics if available
    nvml_handle = None
    if pynvml is not None and has_cuda:
        try:
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as ex:
            logger.debug("pynvml initialization failed: %s", ex)
            nvml_handle = None

    # Load Model Bundle
    model_name = getattr(cascade_cfg, "model", None)
    bundle: ModelBundle | None = None
    try:
        bundle = ModelBundle.load(model_name, device=device_str)
        logger.info("ModelBundle loaded successfully on %s for worker %d", device_str, device_index)
    except ScourError as se:
        if gpu_mode == "auto":
            logger.info("GPU stage disabled: model '%s' unavailable (%s). Falling back to rules_only.",
                        model_name, se.message)
            _drain_until_stop(gpu_q, gpu_res_q)
            return
        raise se
    except Exception as ex:
        if gpu_mode == "auto":
            logger.info("GPU stage disabled: unexpected loading error (%s). Falling back to rules_only.", ex)
            _drain_until_stop(gpu_q, gpu_res_q)
            return
        raise ScourError(
            code="E-GPU-003",
            stage="gpu",
            message=f"Failed loading model '{model_name}': {ex}",
            cause=ex,
        ) from ex

    # Dynamic token budget hint
    base_budget = _get_shared_val(shared, "gpu_batch_tokens_hint", 16384)
    token_budget = int(base_budget) if isinstance(base_budget, (int, float)) and base_budget > 0 else 16384

    logger.info("GPU worker %d started on %s (initial token budget: %d)", device_index, device_str, token_budget)

    # Server loop
    while True:
        try:
            # Poll with timeout to allow heartbeat and queue check
            job = gpu_q.get(timeout=0.1)
        except Exception:
            # Timeout: queue idle; update health heartbeat and loop
            _update_telemetry(shared, nvml_handle)
            continue

        if job is None:
            # Sentinel / Poison pill received -> exit gracefully
            logger.info("GPU worker %d received stop sentinel. Shutting down.", device_index)
            break

        # Unpack job (handles GpuJob dataclass, duck-typing, or tuple)
        w_id = getattr(job, "worker_id", None)
        b_id = getattr(job, "batch_id", None)
        j_idx = getattr(job, "job_idx", 0)
        texts = getattr(job, "texts", None)

        if w_id is None and isinstance(job, (tuple, list)) and len(job) >= 4:
            w_id, b_id, j_idx, texts = job[0], job[1], job[2], job[3]

        if w_id is None or texts is None:
            logger.warning("Unrecognized job format received by GPU worker: %s", type(job))
            if w_id is not None and b_id is not None:
                _route_result(gpu_res_q, GpuResult(worker_id=w_id, batch_id=b_id, job_idx=j_idx, probs=[]))
            continue

        if not texts:
            _route_result(gpu_res_q, GpuResult(worker_id=w_id, batch_id=b_id, job_idx=j_idx, probs=[]))
            continue

        # Inference with dynamic batching & OOM recovery
        probs_list: list[float] = []
        try:
            probs_arr = _run_inference_dynamic(bundle, texts, token_budget)
            probs_list = probs_arr.tolist()
        except torch.cuda.OutOfMemoryError as oom:
            logger.warning("GPU OOM encountered during batch inference. Halving token budget.")
            torch.cuda.empty_cache()
            token_budget = token_budget // 2
            if token_budget < 512:
                raise ScourError(
                    code="E-GPU-002",
                    stage="gpu",
                    message="GPU Out-Of-Memory (OOM) at minimum batch size (< 512 tokens).",
                    cause=oom,
                    hint="Reduce gpu_batch_tokens or free competing GPU processes.",
                ) from oom

            # Retry with reduced budget
            probs_arr = _run_inference_dynamic(bundle, texts, token_budget)
            probs_list = probs_arr.tolist()

        # Send result back
        _route_result(gpu_res_q, GpuResult(worker_id=w_id, batch_id=b_id, job_idx=j_idx, probs=probs_list))
        _update_telemetry(shared, nvml_handle)


def _run_inference_dynamic(bundle: ModelBundle, texts: list[str], token_budget: int) -> np.ndarray:
    """
    Chunks texts into mini-batches fitting within token_budget and computes probabilities.
    """
    results: list[np.ndarray] = []
    # Heuristic: average 20 tokens per chunk or max_length=256
    batch_size = max(1, token_budget // 256)

    for i in range(0, len(texts), batch_size):
        slice_texts = texts[i : i + batch_size]
        sub_probs = bundle.predict_proba(slice_texts, max_length=256)
        results.append(sub_probs)

    if not results:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(results)


def _update_telemetry(shared: Any, nvml_handle: Any) -> None:
    """Publishes GPU utilization metrics and heartbeats to shared memory block."""
    if shared is None:
        return
    _set_shared_val(shared, "gpu_heartbeat", time.time())

    if nvml_handle is not None and pynvml is not None:
        try:
            rates = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
            _set_shared_val(shared, "gpu_util", float(rates.gpu))
        except Exception:
            pass


def _drain_until_stop(gpu_q: Any, gpu_res_q: Any) -> None:
    """Drains pending items and responds with empty results until stop sentinel arrives."""
    while True:
        try:
            job = gpu_q.get(timeout=0.2)
            if job is None:
                break
            w_id = getattr(job, "worker_id", None)
            b_id = getattr(job, "batch_id", None)
            j_idx = getattr(job, "job_idx", 0)
            if w_id is None and isinstance(job, (tuple, list)) and len(job) >= 3:
                w_id, b_id, j_idx = job[0], job[1], job[2]
            if w_id is not None and b_id is not None:
                _route_result(gpu_res_q, GpuResult(worker_id=w_id, batch_id=b_id, job_idx=j_idx, probs=[]))
        except Exception:
            break
