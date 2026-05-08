"""
Utility helpers for the SMES preprocessing pipeline.

This module intentionally keeps only low-level helpers:
- path / directory helpers
- JSON / JSONL loading and writing
- text normalization
- fragment joining
- simple preview dumping
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.

    Args:
        path: Directory path.

    Returns:
        Path object pointing to the ensured directory.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_json(path: str | Path) -> Any:
    """
    Load a JSON file.

    Args:
        path: JSON file path.

    Returns:
        Parsed Python object.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """
    Write records to a strict JSONL file.

    Args:
        path: Destination file path.
        records: Iterable of JSON-serializable dictionaries.
    """
    path_obj = Path(path)
    ensure_dir(path_obj.parent)

    with path_obj.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_pretty_json(path: str | Path, obj: Any) -> None:
    """
    Write a pretty JSON file for debugging / preview.

    Args:
        path: Destination file path.
        obj: JSON-serializable object.
    """
    path_obj = Path(path)
    ensure_dir(path_obj.parent)

    with path_obj.open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace without changing semantics.

    Args:
        text: Raw text.

    Returns:
        Text with collapsed whitespace and stripped edges.
    """
    return re.sub(r"\s+", " ", text).strip()


def normalize_fragment_list(items: list[str] | None) -> list[str]:
    """
    Normalize each text fragment inside a list.

    Args:
        items: List of raw text fragments.

    Returns:
        Normalized list with empty fragments removed.
    """
    if not items:
        return []

    normalized_items: list[str] = []
    for item in items:
        if item is None:
            continue
        normalized_item = normalize_whitespace(str(item))
        if normalized_item:
            normalized_items.append(normalized_item)
    return normalized_items


def join_fragments(items: list[str] | None) -> str:
    """
    Join text fragments into one string.

    Args:
        items: List of text fragments.

    Returns:
        Joined normalized text.
    """
    return normalize_whitespace(" ".join(normalize_fragment_list(items)))


def load_text(path: str | Path) -> str:
    """
    Load a plain-text file.

    Args:
        path: Text file path.

    Returns:
        File contents as a string.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        return file.read().strip()


def dump_preview_samples(
    path: str | Path,
    task_to_records: dict[str, list[dict[str, Any]]],
    preview_count: int,
) -> None:
    """
    Save a small pretty JSON preview of records for manual inspection.

    Args:
        path: Output JSON path.
        task_to_records: Mapping from task name to list of rendered records.
        preview_count: Number of samples per task to keep.
    """
    preview_payload = {
        task_name: records[:preview_count]
        for task_name, records in task_to_records.items()
    }
    write_pretty_json(path, preview_payload)
