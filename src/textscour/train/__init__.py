"""
TEXTSCOUR Training, Synthetic Generation, Calibration, and Evaluation Subpackage.
"""

from textscour.train.bench_accuracy import run_benchmark
from textscour.train.calibrate import calibrate_and_save, optimize_temperature
from textscour.train.eval_labels import evaluate_label_sheet
from textscour.train.export_label_sheet import export_label_sheet
from textscour.train.synth_noise import (
    BenchDoc,
    Template,
    inject,
    load_templates,
    make_benchmark,
)
from textscour.train.train_classifier import train_classifier

__all__ = [
    "Template",
    "BenchDoc",
    "load_templates",
    "inject",
    "make_benchmark",
    "calibrate_and_save",
    "optimize_temperature",
    "train_classifier",
    "export_label_sheet",
    "evaluate_label_sheet",
    "run_benchmark",
]
