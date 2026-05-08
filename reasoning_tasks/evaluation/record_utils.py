"""
evaluation/record_utils.py

Shared record-normalization helpers for post-validation artifacts.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


def unpack_sample_indices(record: Dict[str, Any]) -> Tuple[int | None, int | None]:
    """
    Extract dialog_index and turn_index from a prediction record.

    Args:
        record: Prediction record that may contain either top-level indices or
            a nested sample_id dictionary.

    Returns:
        Tuple of dialog_index and turn_index. Missing values return None.
    """
    sample_id = record.get("sample_id", {})
    if not isinstance(sample_id, dict):
        sample_id = {}

    dialog_index = record.get("dialog_index", sample_id.get("dialog_index"))
    turn_index = record.get("turn_index", sample_id.get("turn_index"))

    return dialog_index, turn_index


def truncate_text(text: Any, max_chars: int = 160) -> str:
    """
    Convert text to a compact single-line preview.

    Args:
        text: Input text-like object.
        max_chars: Maximum output length.

    Returns:
        Preview string with whitespace normalized.
    """
    clean_text = " ".join(str(text if text is not None else "").split())
    if len(clean_text) <= max_chars:
        return clean_text
    return clean_text[: max_chars - 3] + "..."
