"""
evaluation/loaders.py

Shared loaders for post-validation.

This module owns:
- label-name resolution
- tokenizer loading
- classification validation dataloader creation
- classification/generation model + adapter loading
- Task 4 generation dataset creation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from torch.utils.data import DataLoader
from peft.peft_model import PeftModel

from common.config_utils import (
    PCSPrinter,
    apply_special_token_ids_to_model,
    debug_dataset_sample,
    load_text,
    setup_qwen_special_tokens,
)
from evaluation.export_utils import load_json


DEFAULT_LABELS: List[str] = [
    "anger",
    "sadness",
    "disgust",
    "depression",
    "fear",
    "neutral",
    "joy",
]

def get_eval_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select runtime config section based on eval_mode.

    eval_mode:
        - "post_val": use config["post_val"]
        - "inference": use config["inference"]
    """
    eval_mode = str(config.get("eval_mode", "post_val")).strip().lower()

    if eval_mode not in {"post_val", "inference"}:
        raise ValueError(
            f"Unsupported eval_mode: {eval_mode}. "
            "Expected 'post_val' or 'inference'."
        )

    if eval_mode not in config:
        raise KeyError(f"Missing config section: {eval_mode}")

    return config[eval_mode]

def resolve_inference_model_path(
    config: Dict[str, Any],
    runtime_config: Dict[str, Any],
) -> tuple[str, str | None]:
    """
    Resolve base/full model path and optional LoRA adapter path.

    Returns:
        model_path:
            Base model path for pretrained/lora mode,
            or full model path for full mode.

        adapter_path:
            LoRA adapter path for lora mode,
            otherwise None.
    """
    model_config = config["model"]
    mode = str(runtime_config.get("mode", "lora")).strip().lower()

    if mode == "pretrained":
        return model_config["path"], None

    if mode == "lora":
        adapter_path = runtime_config.get("adapter_path", "")
        if not adapter_path:
            raise ValueError("inference.mode='lora' requires inference.adapter_path.")
        return model_config["path"], adapter_path

    if mode == "full":
        full_path = runtime_config.get("full_path", "")
        if not full_path:
            raise ValueError("inference.mode='full' requires inference.full_path.")
        return full_path, None

    raise ValueError(
        f"Unsupported inference model mode: {mode}. "
        "Expected pretrained | lora | full."
    )

def resolve_model_dtype_config(model_config: dict) -> dict:
    """
    Resolve model dtype settings for Qwen-style model loading.

    Args:
        model_config: Model config dictionary.

    Returns:
        Dictionary containing:
            - torch_dtype: PyTorch dtype object.
            - fp16: Whether to force Qwen fp16 mode.
            - bf16: Whether to force Qwen bf16 mode.
    """
    dtype_str = model_config.get("dtype", "float16")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unsupported model dtype: {dtype_str}. "
            f"Expected one of {sorted(dtype_map)}."
        )

    return {
        "torch_dtype": dtype_map[dtype_str],
        "fp16": dtype_str == "float16",
        "bf16": dtype_str == "bfloat16",
    }

def resolve_label_names(config: Dict[str, Any]) -> List[str]:
    """
    Resolve label names from external JSON, root config labels, or defaults.

    Priority:
        1. config["post_val"]["label_names_json"]
        2. config["labels"]
        3. DEFAULT_LABELS

    Args:
        config: Full merged config dictionary.

    Returns:
        List of label names.
    """
    post_val_config = config.get("post_val", {})
    label_names_json = post_val_config.get("label_names_json", "")

    if label_names_json:
        label_file = Path(label_names_json)
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")

        payload = load_json(label_file)

        if isinstance(payload, dict) and "label_names" in payload:
            return list(payload["label_names"])

        if isinstance(payload, list):
            return list(payload)

        raise ValueError(
            "label_names_json must be either a list or a dict with key 'label_names'."
        )

    if "labels" in config and config["labels"]:
        return list(config["labels"])

    return DEFAULT_LABELS


def build_classification_val_dataloader(
    config: Dict[str, Any],
    tokenizer: Any,
    printer: PCSPrinter,
) -> DataLoader:
    """
    Build the validation dataloader for classification tasks.

    Args:
        config: Full merged config dictionary.
        tokenizer: Tokenizer used by the model.
        printer: Structured logger.

    Returns:
        Validation DataLoader.
    """
    try:
        from common.dataset import EmotionDataset
    except ImportError as exc:
        raise ImportError(
            "Edit build_classification_val_dataloader() to match your actual dataset stack."
        ) from exc

    runtime_config = get_eval_section(config)
    eval_mode = str(config.get("eval_mode", "post_val")).strip().lower()

    if eval_mode == "inference":
        input_path = runtime_config["input_path"]
    else:
        input_path = config["val_path"]

    printer.stage("Build classification validation dataloader")
    prompt_text = load_text(prompt_dir=config["prompt_dir"], task_name=config["task_name"], sys_reaction=config["formatting"]["sys_reaction"])

    dataset = EmotionDataset(
        jsonl_path=input_path,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_length=runtime_config.get("max_length", config["max_length"]),
        return_idx=True,
        sph_arng_mod=config["formatting"]["sph_arng_mod"],
        sys_reaction=config["formatting"]["sys_reaction"],
        ablation=runtime_config.get("ablation", {}),
    )

    printer.info(f"Validation samples: {len(dataset)}")

    debug_dataset_sample(
        dataset=dataset,
        printer=printer,
        sample_idx=0,
        stage_name="Validation Dataset Sample Check",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=runtime_config.get("batch_size", config["batch_size"]),
        shuffle=False,
    )

    printer.info(
        f"Validation dataloader ready | batch_size="
        f"{runtime_config.get('batch_size', config['batch_size'])}"
    )

    return dataloader

def load_classification_model_and_tokenizer(
    config: Dict[str, Any],
    printer: PCSPrinter,
) -> Tuple[torch.nn.Module, Any]:
    """
    Load causal LM and tokenizer for classification evaluation/inference.

    Supports:
        eval_mode: post_val | inference

    In inference mode:
        inference.mode:
            - pretrained: base model only
            - lora: base model + adapter
            - full: full fine-tuned model path
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for the default loader.") from exc

    eval_mode = str(config.get("eval_mode", "post_val")).strip().lower()
    runtime_config = get_eval_section(config)
    model_config = config["model"]
    label_names = resolve_label_names(config)

    if eval_mode == "inference":
        model_path, adapter_path = resolve_inference_model_path(config, runtime_config)
    elif eval_mode == "post_val":
        model_path = model_config["path"]
        adapter_path = str(Path(runtime_config["run_dir"]) / "best_adapter")
    else:
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")

    printer.stage("Load tokenizer")
    printer.info(f"eval_mode  : {eval_mode}")
    printer.info(f"model_path : {model_path}")
    printer.info(f"adapter    : {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=model_config.get("trust_remote_code", True),
        local_files_only=model_config.get("local_files_only", False),
    )

    token_info = setup_qwen_special_tokens(tokenizer)

    printer.stage("Load classification model")
    dtype_kwargs = resolve_model_dtype_config(model_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        num_labels=len(label_names),
        trust_remote_code=model_config.get("trust_remote_code", True),
        local_files_only=model_config.get("local_files_only", True),
        use_flash_attn=False,
        **dtype_kwargs,
    )

    apply_special_token_ids_to_model(model, token_info)

    if adapter_path is not None:
        adapter_path = Path(adapter_path)

        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

        printer.stage("Load LoRA adapter")
        printer.info(f"adapter_path: {adapter_path}")

        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )

        model = model.merge_and_unload()
    else:
        printer.info("No LoRA adapter loaded")

    printer.info("Classification model and tokenizer loaded successfully")
    return model, tokenizer


def load_generation_model_tokenizer_dataset(
    config: Dict[str, Any],
    printer: PCSPrinter,
) -> Tuple[torch.nn.Module, Any, Any, torch.device]:
    """
    Load model, tokenizer, dataset, and device for Task 4 generation.

    Supports:
        eval_mode: post_val | inference

    In inference mode:
        inference.mode:
            - pretrained: base model only
            - lora: base model + adapter
            - full: full fine-tuned model path
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common.dataset import ResponseGenerationDataset

    eval_mode = str(config.get("eval_mode", "post_val")).strip().lower()
    runtime_config = get_eval_section(config)
    model_config = config["model"]
    run_dir = Path(runtime_config["run_dir"])

    if eval_mode == "inference":
        input_path = runtime_config["input_path"]
        model_path, adapter_path = resolve_inference_model_path(config, runtime_config)
        output_dir = run_dir / "inference"
    else:
        input_path = config["val_path"]
        model_path = model_config["path"]
        adapter_path = str(run_dir / "best_adapter")
        output_dir = run_dir / "post_val"

    printer.stage("Load tokenizer")
    printer.info(f"eval_mode  : {eval_mode}")
    printer.info(f"model_path : {model_path}")
    printer.info(f"input_path : {input_path}")
    printer.info(f"output_dir : {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=model_config.get("trust_remote_code", True),
        local_files_only=model_config.get("local_files_only", True),
    )

    token_info = setup_qwen_special_tokens(tokenizer)

    prompt_text = load_text(
        prompt_dir=config["prompt_dir"],
        task_name=config["task_name"],
        sys_reaction=config["formatting"]["sys_reaction"],
    )

    printer.stage("Load generation model")
    dtype_kwargs = resolve_model_dtype_config(model_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=model_config.get("trust_remote_code", True),
        local_files_only=model_config.get("local_files_only", True),
        device_map=model_config.get("device_map", "auto"),
        use_flash_attn=False,
        **dtype_kwargs,
    )

    apply_special_token_ids_to_model(model, token_info)

    if adapter_path is not None:
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

        printer.stage("Load LoRA adapter")
        printer.info(f"adapter_path: {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )
    else:
        printer.info("No LoRA adapter loaded")

    device = next(model.parameters()).device

    dataset = ResponseGenerationDataset(
        input_path,
        tokenizer,
        prompt_text,
        max_length=int(runtime_config.get("max_length", config["max_length"])),
        return_idx=True,
        sph_arng_mod=config["formatting"]["sph_arng_mod"],
        sys_reaction=config["formatting"]["sys_reaction"],
        ablation=runtime_config.get("ablation", {}),
    )

    debug_dataset_sample(
        dataset=dataset,
        printer=printer,
        sample_idx=0,
        stage_name="Generation Dataset Sample Check",
    )

    printer.info("Generation model, tokenizer, and dataset loaded successfully")
    return model, tokenizer, dataset, device
