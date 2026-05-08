"""
Entry-point script for generating structured processed splits.

Default behavior:
- load config.yaml
- read raw train / val / test split files
- build structured records only
- optionally dump a preview file for manual inspection

This script intentionally does NOT render final prompt text.
That is deferred to the training dataset stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .core import process_split
from .utils import dump_preview_samples, ensure_dir, load_json, write_jsonl


TASK_TO_OUTPUT_DIR_KEY = {
    "user_emotion": "user_emotion_dir",
    "therapist_strategy": "therapist_strategy_dir",
    "therapist_emotion": "therapist_emotion_dir",
    "therapist_response": "therapist_response_dir",
}


def load_config(config_path: str | Path) -> dict:
    """
    Load YAML config.

    Args:
        config_path: Config file path.

    Returns:
        Parsed config dictionary.
    """
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def export_one_split(
    split_name: str,
    raw_file_name: str,
    config: dict,
) -> None:
    """
    Export one split into per-task structured JSONL files.

    Args:
        split_name: One of train / val / test.
        raw_file_name: Raw JSON file name from config.
        config: Parsed config dictionary.
    """
    raw_data_dir = Path(config["paths"]["raw_data_dir"])
    processed_data_dir = Path(config["paths"]["processed_data_dir"])
    output_cfg = config["output"]
    processing_cfg = config["processing"]
    tasks_cfg = config["tasks"]

    raw_path = raw_data_dir / raw_file_name
    raw_split = load_json(raw_path)

    task_to_records = process_split(raw_split, tasks_cfg)

    file_name = output_cfg[f"{split_name}_file"]

    for task_name, records in task_to_records.items():
        output_subdir = output_cfg[TASK_TO_OUTPUT_DIR_KEY[task_name]]
        output_path = processed_data_dir / output_subdir / file_name
        write_jsonl(output_path, records)

    if processing_cfg.get("save_preview", True):
        preview_dir = ensure_dir(processed_data_dir / "_preview")
        preview_path = preview_dir / f"{split_name}_preview.json"
        dump_preview_samples(
            preview_path,
            task_to_records,
            preview_count=processing_cfg.get("preview_count_per_task", 3),
        )

    print(f"[DONE] Exported split: {split_name}")


def main() -> None:
    """
    Run the preprocessing export pipeline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str("/home/jsp/DL_final_project/reasoning_tasks/configs/preprocess.yaml"),
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    files_cfg = config["files"]

    export_one_split("train", files_cfg["train"], config)
    export_one_split("val", files_cfg["val"], config)
    export_one_split("test", files_cfg["test"], config)


if __name__ == "__main__":
    main()
