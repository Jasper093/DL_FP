#!/usr/bin/env python3
"""
evaluation/run_eval.py

Unified post-validation and inference entry point.

Classification tasks:
    python -m evaluation.run_eval --config configs/task1.yaml --mdl_config configs/model.yaml

Generation task:
    python -m evaluation.run_eval --config configs/task4.yaml --mdl_config configs/model.yaml

Final output layout:
    runs/<task_name>/run_xxx/post_val/ (or inference)
        metrics.json
        predictions.jsonl
        classification_rankings.xlsx or generation_rankings.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import json

from common.config_utils import PCSPrinter, load_config
from evaluation.export_utils import ensure_dir, save_json, save_jsonl
from evaluation.excel_rankings import (
    save_classification_rankings_excel,
    save_generation_rankings_excel,
)
from evaluation.loaders import (
    build_classification_val_dataloader,
    load_classification_model_and_tokenizer,
    load_generation_model_tokenizer_dataset,
    resolve_label_names,
)
from evaluation.classification.artifacts import build_classification_artifacts
from evaluation.classification.inference import run_classification_inference
from evaluation.generation.artifacts import build_generation_artifacts
from evaluation.generation.inference import run_generation_inference


def resolve_task_type(config: Dict[str, Any]) -> str:
    """
    Resolve evaluation task type.

    Args:
        config: Full merged config dictionary.

    Returns:
        "classification" or "generation".
    """
    task_type = str(config.get("task_type", "")).strip().lower()

    if task_type:
        if task_type not in {"classification", "generation"}:
            raise ValueError(
                "task_type must be either 'classification' or 'generation'."
            )
        return task_type

    task_name = str(config.get("task_name", "")).strip().lower()
    if task_name in {"task4", "therapist_response", "response_generation", "generation"}:
        return "generation"

    return "classification"


def run_classification_eval(config: Dict[str, Any], printer: PCSPrinter) -> None:
    """
    Run evaluation for Tasks 1-3 classification.

    Args:
        config: Full merged config dictionary.
        printer: Structured logger.
    """
    eval_mode = str(config.get("eval_mode", "post_val")).strip().lower()

    if eval_mode == "inference":
        runtime_config = config["inference"]
        output_dir = Path(runtime_config["run_dir"]) / "inference"
    elif eval_mode == "post_val":
        runtime_config = config["post_val"]
        output_dir = Path(runtime_config["run_dir"]) / "post_val"
    else:
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")

    ensure_dir(output_dir)

    label_names = resolve_label_names(config)
    device = torch.device(
        runtime_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    percentile = float(runtime_config.get("percentile", 0.2))

    printer.separator()
    printer.stage("Classification evaluation start")
    printer.info(f"output_dir : {output_dir}")
    printer.info(f"task_name    : {config.get('task_name', 'unresolved')}")
    printer.info(f"val_jsonl    : {config['val_path']}")
    printer.info(f"labels       : {label_names}")
    printer.info(f"device       : {device}")
    printer.separator()

    model, tokenizer = load_classification_model_and_tokenizer(config, printer)
    model.to(device)
    printer.info("Model moved to target device")

    val_dataloader = build_classification_val_dataloader(config, tokenizer, printer)

    printer.stage("Run classification inference")
    full_records, true_ids, pred_ids = run_classification_inference(
        model=model,
        dataloader=val_dataloader,
        label_names=label_names,
        device=device,
        tokenizer=tokenizer,
    )
    printer.info(f"Inference finished | num_samples={len(true_ids)}")

    printer.stage("Build classification artifacts")
    artifacts = build_classification_artifacts(
        full_records=full_records,
        true_ids=true_ids,
        pred_ids=pred_ids,
        label_names=label_names,
    )

    printer.stage("Save classification outputs")
    save_json(artifacts["metrics"], output_dir / "metrics.json")
    save_jsonl(artifacts["predictions"], output_dir / "predictions.jsonl")
    save_classification_rankings_excel(
        records=artifacts["predictions"],
        output_path=output_dir / "classification_rankings.xlsx",
        yellow_ratio= percentile
    )

    if bool(runtime_config.get("save_diagnostics", False)):
        save_json(artifacts["classification_report"], output_dir / "classification_report.json")
        save_json(artifacts["confusion_matrix"], output_dir / "confusion_matrix.json")

    printer.info(f"Saved metrics.json")
    printer.info(f"Saved predictions.jsonl")
    printer.info(f"Saved classification_rankings.xlsx")
    printer.info(f"Saved classification evaluation artifacts to: {output_dir}")


def run_generation_eval(config: Dict[str, Any], printer: PCSPrinter) -> None:
    """
    Run evaluation for Task 4 generation.

    Args:
        config: Full merged config dictionary.
        printer: Structured logger.
    """
    eval_mode = str(config.get("eval_mode", "post_val")).strip().lower()

    if eval_mode == "inference":
        runtime_config = config["inference"]
        output_dir = Path(runtime_config["run_dir"]) / "inference"
    elif eval_mode == "post_val":
        runtime_config = config["post_val"]
        output_dir = Path(runtime_config["run_dir"]) / "post_val"
    else:
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")

    percentile = float(runtime_config.get("percentile", 0.2))

    ensure_dir(output_dir)

    printer.separator()
    printer.stage("Generation evaluation start")
    printer.info(f"output_dir : {output_dir}")
    printer.info(f"task_name    : {config.get('task_name', 'generation')}")
    printer.info(f"val_jsonl    : {config['val_path']}")
    printer.separator()

    model, tokenizer, dataset, device = load_generation_model_tokenizer_dataset(config, printer)

    printer.stage("Run generation inference")
    records = run_generation_inference(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_new_tokens=int(runtime_config.get("max_new_tokens", config.get("max_new_tokens", 192))),
        device=device,
    )
    printer.info(f"Generation finished | num_samples={len(records)}")

    printer.stage("Build generation artifacts")
    artifacts = build_generation_artifacts(records=records)

    printer.stage("Save generation outputs")
    save_json(artifacts["metrics"], output_dir / "metrics.json")
    save_jsonl(artifacts["predictions"], output_dir / "predictions.jsonl")
    save_generation_rankings_excel(
        records=artifacts["predictions"],
        output_path=output_dir / "generation_rankings.xlsx",
        yellow_ratio=percentile
    )
    if config['task_type'] == "generation":
        with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    printer.info(f"Saved metrics.json")
    printer.info(f"Saved predictions.jsonl")
    printer.info(f"Saved generation_rankings.xlsx")
    printer.info(f"Saved generation evaluation artifacts to: {output_dir}")


def main(config_path: str, mdl_config_path: str | None = None) -> None:
    """
    Run the unified evaluation pipeline.

    Args:
        config_path: Task config path.
        mdl_config_path: Model config path.
    """
    printer = PCSPrinter(debug=True)

    printer.stage("Load merged config")
    config = load_config(config_path, mdl_config_path)

    task_type = resolve_task_type(config)
    printer.info(f"Resolved evaluation task_type: {task_type}")

    if task_type == "classification":
        run_classification_eval(config, printer)
    elif task_type == "generation":
        run_generation_eval(config, printer)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    printer.stage("evaluation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation launcher.")
    parser.add_argument("--config", type=str, default="configs/task1.yaml")
    parser.add_argument("--mdl_config", type=str, default="configs/model.yaml")
    args = parser.parse_args()

    main(args.config, args.mdl_config)
