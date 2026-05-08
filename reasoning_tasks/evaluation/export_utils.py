"""
evaluation/export_utils.py

Shared filesystem helpers for post-validation outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def ensure_dir(dir_path: Path) -> None:
    """
    Create a directory recursively if it does not already exist.

    Args:
        dir_path: Target directory path.
    """
    dir_path.mkdir(parents=True, exist_ok=True)


def load_json(file_path: Path) -> Any:
    """
    Load a JSON file.

    Args:
        file_path: Input JSON path.

    Returns:
        Parsed JSON payload.
    """
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(obj: Any, file_path: Path) -> None:
    """
    Save a Python object as pretty JSON.

    Args:
        obj: Object to serialize.
        file_path: Output JSON path.
    """
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def save_jsonl(records: List[Dict[str, Any]], file_path: Path) -> None:
    """
    Save a list of dictionaries as JSONL.

    Args:
        records: Records to save.
        file_path: Output JSONL path.
    """
    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
