"""
evaluation/classification/metrics.py

Metric builders for classification-style post-validation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def build_classification_report_dict(
    true_ids: List[int],
    pred_ids: List[int],
    label_names: List[str],
) -> Dict[str, Any]:
    """
    Build a JSON-friendly classification report.

    Args:
        true_ids: Ground-truth label ids.
        pred_ids: Predicted label ids.
        label_names: Ordered label names.

    Returns:
        Classification report dictionary.
    """
    report = classification_report(
        true_ids,
        pred_ids,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    normalized_report: Dict[str, Any] = {}
    for key, value in report.items():
        if isinstance(value, dict):
            normalized_report[key] = {
                sub_key: round(float(sub_value), 6)
                for sub_key, sub_value in value.items()
            }
        else:
            normalized_report[key] = round(float(value), 6)

    return normalized_report


def build_confusion_matrix_dict(
    true_ids: List[int],
    pred_ids: List[int],
    label_names: List[str],
) -> Dict[str, Any]:
    """
    Build a labeled confusion matrix payload.

    Args:
        true_ids: Ground-truth label ids.
        pred_ids: Predicted label ids.
        label_names: Ordered label names.

    Returns:
        Confusion matrix payload.
    """
    matrix = confusion_matrix(
        true_ids,
        pred_ids,
        labels=list(range(len(label_names))),
    )

    return {
        "labels": label_names,
        "matrix": matrix.tolist(),
    }


def build_final_metrics(true_ids: List[int], pred_ids: List[int]) -> Dict[str, float]:
    """
    Build compact summary classification metrics.

    Args:
        true_ids: Ground-truth label ids.
        pred_ids: Predicted label ids.

    Returns:
        Dictionary with accuracy, weighted F1, and macro F1.
    """
    accuracy = accuracy_score(true_ids, pred_ids)
    weighted_f1 = f1_score(true_ids, pred_ids, average="weighted", zero_division=0)
    macro_f1 = f1_score(true_ids, pred_ids, average="macro", zero_division=0)

    return {
        "accuracy": round(float(accuracy), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "macro_f1": round(float(macro_f1), 6),
    }


def build_prediction_distribution(pred_ids: List[int], label_names: List[str]) -> Dict[str, int]:
    """
    Count predicted class distribution.

    Args:
        pred_ids: Predicted label ids.
        label_names: Ordered label names.

    Returns:
        Mapping from label name to prediction count.
    """
    counts = {label: 0 for label in label_names}
    for pred_id in pred_ids:
        counts[label_names[pred_id]] += 1
    return counts


def build_label_distribution(true_ids: List[int], label_names: List[str]) -> Dict[str, int]:
    """
    Count ground-truth class distribution.

    Args:
        true_ids: Ground-truth label ids.
        label_names: Ordered label names.

    Returns:
        Mapping from label name to label count.
    """
    counts = {label: 0 for label in label_names}
    for true_id in true_ids:
        counts[label_names[true_id]] += 1
    return counts
