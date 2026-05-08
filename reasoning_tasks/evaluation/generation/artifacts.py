"""
evaluation/generation/artifacts.py

Artifact builders for Task 4 generation post-validation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from evaluation.generation.metrics import (
    bleu2_score,
    build_generation_metrics,
    compute_bertscore_values,
    rouge_l_f1,
)
from evaluation.record_utils import unpack_sample_indices


def build_generation_prediction_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build unified readable prediction records for generation tasks.

    Args:
        records: Raw generation records from inference.

    Returns:
        JSONL-ready generation prediction records.
    """
    preds = [record.get("prediction", "") for record in records]
    refs = [record.get("reference", "") for record in records]
    bertscore_values = compute_bertscore_values(preds, refs)

    prediction_records: List[Dict[str, Any]] = []
    for fallback_index, record in enumerate(records):
        dialog_index, turn_index = unpack_sample_indices(record)
        prediction = record.get("prediction", "")
        reference = record.get("reference", "")

        prediction_records.append(
            {
                "record_index": record.get("record_index", fallback_index),
                "dialog_index": dialog_index,
                "turn_index": turn_index,
                "prediction": prediction,
                "reference": reference,
                "confidence": None,
                "margin": None,
                "is_correct": None,
                "metrics": {
                    "bleu_2": round(float(bleu2_score(prediction, reference)), 6),
                    "rouge_l": round(float(rouge_l_f1(prediction, reference)), 6),
                    "bertscore_f1": (
                        round(float(bertscore_values[fallback_index]), 6)
                        if bertscore_values[fallback_index] is not None
                        else None
                    ),
                    "nll": (
                        round(float(record["nll"]), 6)
                        if record.get("nll") is not None
                        else None
                    ),
                    "ppl": (
                        round(float(record["ppl"]), 6)
                        if record.get("ppl") is not None
                        else None
                    ),
                },
            }
        )

    return prediction_records


def build_generation_artifacts(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build all Task 4 post-validation artifacts in memory.

    Args:
        records: Raw generation records.

    Returns:
        Dictionary containing metrics and unified predictions.
    """
    prediction_records = build_generation_prediction_records(records)
    return {
        "metrics": build_generation_metrics(prediction_records),
        "predictions": prediction_records,
    }
