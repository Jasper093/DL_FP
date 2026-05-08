"""
evaluation/classification/artifacts.py

Artifact builders for classification post-validation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from evaluation.classification.metrics import (
    build_classification_report_dict,
    build_confusion_matrix_dict,
    build_final_metrics,
    build_label_distribution,
    build_prediction_distribution,
)
from evaluation.record_utils import unpack_sample_indices


def build_classification_prediction_records(full_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build unified readable prediction records for classification tasks.

    Args:
        full_records: Full prediction records from inference.

    Returns:
        JSONL-ready prediction records.
    """
    prediction_records: List[Dict[str, Any]] = []

    for fallback_index, record in enumerate(full_records):
        dialog_index, turn_index = unpack_sample_indices(record)
        pred = record.get("pred", record.get("prediction"))
        label = record.get("label", record.get("reference"))

        prediction_records.append(
            {
                "record_index": record.get("record_index", fallback_index),
                "dialog_index": dialog_index,
                "turn_index": turn_index,
                "prediction": pred,
                "reference": label,
                "pred": pred,
                "label": label,
                "is_correct": pred == label,
                "confidence": record.get("confidence"),
                "margin": record.get("margin"),
                "metrics": None,
            }
        )

    return prediction_records


def build_classification_artifacts(
    full_records: List[Dict[str, Any]],
    true_ids: List[int],
    pred_ids: List[int],
    label_names: List[str],
) -> Dict[str, Any]:
    """
    Build all classification post-validation artifacts in memory.

    Args:
        full_records: Full prediction records.
        true_ids: Ground-truth label ids.
        pred_ids: Predicted label ids.
        label_names: Ordered label names.

    Returns:
        Dictionary containing metrics, diagnostic reports, and unified predictions.
    """
    prediction_records = build_classification_prediction_records(full_records)
    final_metrics = build_final_metrics(true_ids=true_ids, pred_ids=pred_ids)

    metrics_payload: Dict[str, Any] = dict(final_metrics)
    metrics_payload["num_val_samples"] = len(true_ids)
    metrics_payload["num_misclassifications"] = sum(
        1 for record in prediction_records if record.get("is_correct") is False
    )
    metrics_payload["label_distribution"] = build_label_distribution(true_ids, label_names)
    metrics_payload["prediction_distribution"] = build_prediction_distribution(pred_ids, label_names)

    return {
        "metrics": metrics_payload,
        "predictions": prediction_records,
        "classification_report": build_classification_report_dict(
            true_ids=true_ids,
            pred_ids=pred_ids,
            label_names=label_names,
        ),
        "confusion_matrix": build_confusion_matrix_dict(
            true_ids=true_ids,
            pred_ids=pred_ids,
            label_names=label_names,
        ),
    }
