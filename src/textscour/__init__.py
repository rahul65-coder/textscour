"""TEXTSCOUR: High-Throughput Web Corpus Cleaning Engine.

Combines Rust automata, streaming readers, heuristics, deduplication,
and dynamic GPU classifier cascade.
"""

from __future__ import annotations

from textscour.checkpoint import Checkpoint
from textscour.config import Config, load_config, to_engine_dict
from textscour.env_probe import EnvInfo, probe
from textscour.errors import ScourError
from textscour.metrics import Dashboard, write_report
from textscour.schema import compile_extractor, extract_domain, extract_id
from textscour.sources import HFStreamSource, LocalSource, RawBatch
from textscour.uploader import HFUploader
from textscour.writer import AtomicShard, Manifest, OrderedSourceWriter, QuarantineWriter, merge_outputs

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ScourError",
    "Config",
    "load_config",
    "to_engine_dict",
    "Checkpoint",
    "EnvInfo",
    "probe",
    "compile_extractor",
    "extract_id",
    "extract_domain",
    "RawBatch",
    "HFStreamSource",
    "LocalSource",
    "AtomicShard",
    "Manifest",
    "OrderedSourceWriter",
    "QuarantineWriter",
    "merge_outputs",
    "Dashboard",
    "write_report",
    "HFUploader",
]
