"""
Human Label Sheet Evaluator for TEXTSCOUR.
Reads an annotated label sheet (chunk_id, source, text, model_p, label)
and computes classification metrics (Precision, Recall, F1, Accuracy)
evaluating the model's decision vs ground-truth human annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


def parse_label(raw_label: str) -> int:
    """
    Normalizes human label to integer:
    1 = drop / garbage
    0 = keep / clean
    Raises ValueError on unrecognized labels.
    """
    clean = str(raw_label).strip().lower()
    if clean in ("drop", "garbage", "bad", "spam", "promo", "1", "true", "yes"):
        return 1
    if clean in ("keep", "clean", "good", "valid", "0", "false", "no"):
        return 0
    raise ValueError(f"Unrecognized annotation label: '{raw_label}'")


def evaluate_label_sheet(
    sheet_csv_path: str,
    drop_threshold: float | None = None,
    calibration_path: str = "calibration.json",
) -> dict[str, Any]:
    """
    Evaluates annotations in sheet_csv_path against model decisions.

    Returns dict of evaluation metrics:
    {
      "total_records": int,
      "evaluated_records": int,
      "tp": int, "fp": int, "tn": int, "fn": int,
      "precision": float,
      "recall": float,
      "f1": float,
      "accuracy": float,
      "drop_threshold": float
    }
    """
    if not os.path.exists(sheet_csv_path):
        raise FileNotFoundError(f"Label sheet CSV not found: {sheet_csv_path}")

    # If drop_threshold not explicitly given, try reading from calibration_path, else default to 0.85
    if drop_threshold is None:
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    drop_threshold = float(cdata.get("gpu_drop_p", 0.85))
                    logger.info("Using calibrated gpu_drop_p threshold: %.3f", drop_threshold)
            except Exception as ex:
                logger.debug("Could not read calibration file: %s", ex)
                drop_threshold = 0.85
        else:
            drop_threshold = 0.85

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    skipped = 0
    total = 0

    with open(sheet_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            raw_label = row.get("label", "").strip()
            if not raw_label:
                skipped += 1
                continue

            try:
                y_true = parse_label(raw_label)
            except ValueError:
                skipped += 1
                continue

            model_p_str = row.get("model_p", "0.0")
            try:
                model_p = float(model_p_str)
            except ValueError:
                model_p = 0.0

            y_pred = 1 if model_p >= drop_threshold else 0

            if y_true == 1 and y_pred == 1:
                tp += 1
            elif y_true == 0 and y_pred == 1:
                fp += 1
            elif y_true == 0 and y_pred == 0:
                tn += 1
            elif y_true == 1 and y_pred == 0:
                fn += 1

    evaluated = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / evaluated if evaluated > 0 else 0.0

    metrics = {
        "total_records": total,
        "evaluated_records": evaluated,
        "skipped_records": skipped,
        "drop_threshold": drop_threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }

    print("\n" + "=" * 60)
    print(f"TEXTSCOUR LABEL EVALUATION REPORT ({os.path.basename(sheet_csv_path)})")
    print("=" * 60)
    print(f"Evaluated Records : {evaluated} / {total} (Skipped empty: {skipped})")
    print(f"Decision Threshold: gpu_drop_p >= {drop_threshold:.3f} -> DROP")
    print("-" * 60)
    print(f"Confusion Matrix  : TP={tp} | FP={fp} | TN={tn} | FN={fn}")
    print(f"Precision (Drop)  : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall (Drop)     : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1 Score          : {f1:.4f}")
    print(f"Overall Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 60 + "\n")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate annotated human label sheet")
    parser.add_argument("--sheet", type=str, required=True, help="Path to filled label sheet CSV")
    parser.add_argument("--drop-p", type=float, default=0.85, help="Decision drop threshold")
    parser.add_argument("--calib", type=str, default="calibration.json", help="Path to calibration.json")
    args = parser.parse_args()

    evaluate_label_sheet(args.sheet, drop_threshold=args.drop_p, calibration_path=args.calib)


if __name__ == "__main__":
    main()
