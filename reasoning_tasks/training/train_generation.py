import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

from peft import LoraConfig, TaskType, get_peft_model, PeftModel # type: ignore

from common.dataset import ResponseGenerationDataset
from training.trainer_utils import train_one_epoch, validate_loss
from common.config_utils import (
    debug_dataset_sample,
    setup_qwen_special_tokens,
    apply_special_token_ids_to_model,
    load_text,
    load_config,
    PCSPrinter,
)


def main(config_path, mdl_config_path=None):
    printer = PCSPrinter(debug=True)
    printer.stage("Starting Task 4 generation training")
    config = load_config(config_path, mdl_config_path)

    model_path = config["model"]["path"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=config["model"].get("trust_remote_code", True),
        local_files_only=config["model"].get("local_files_only", True),
    )

    special_token_info = setup_qwen_special_tokens(tokenizer)
    prompt_text = load_text(prompt_dir=config["prompt_dir"], task_name=config["task_name"], sys_reaction=config["formatting"]["sys_reaction"])

    printer.stage("Building Task 4 dataset")
    train_dataset = ResponseGenerationDataset(
        config["train_path"],
        tokenizer,
        prompt_text,
        max_length=int(config["max_length"]),
        sph_arng_mod=config["formatting"]["sph_arng_mod"],
    )
    val_dataset = ResponseGenerationDataset(
        config["val_path"],
        tokenizer,
        prompt_text,
        max_length=int(config["max_length"]),
        sph_arng_mod=config["formatting"]["sph_arng_mod"],
    )

    printer.info(f"Train size: {len(train_dataset)}")
    printer.info(f"Val size: {len(val_dataset)}")
    debug_dataset_sample(train_dataset, printer, sample_idx=0, stage_name="Task 4 Dataset Sample Check")

    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False)

    dtype_str = config["model"].get("dtype", "bfloat16")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype_str]

    printer.stage("Loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=config["model"].get("trust_remote_code", True),
        local_files_only=config["model"].get("local_files_only", True),
        device_map=config["model"].get("device_map", "auto"),
        torch_dtype=torch_dtype,
        use_flash_attn=False,
    )
    apply_special_token_ids_to_model(model, special_token_info)

    printer.stage("Preparing LoRA adapter")
    init_adapter_path = config.get("init_adapter_path", "").strip()
    if init_adapter_path:
        printer.info(f"Loading previous adapter from: {init_adapter_path}")
        model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
    else:
        printer.info("No init_adapter_path provided. Creating fresh LoRA adapter.")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=int(config["lora"]["r"]),
            lora_alpha=int(config["lora"]["alpha"]),
            lora_dropout=float(config["lora"]["dropout"]),
            bias=str(config["lora"].get("bias", "none")),
            target_modules=list(config["lora"]["target_modules"]),
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config["lr"]))

    os.makedirs(config["output_dir"], exist_ok=True)
    log_path = os.path.join(config["output_dir"], "train.log")
    metrics_path = os.path.join(config["output_dir"], "metrics.json")

    best_val_loss = float("inf")
    metrics = {
        "max_epoch": int(config["epochs"]),
        "train_loss": [],
        "val_loss": [],
    }

    printer.stage("Starting Task 4 training loop")
    for epoch in range(int(config["epochs"])):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate_loss(model, val_loader)

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)

        prfx = printer._prefix(f"Epoch {epoch}")
        log_line = (
            f"{prfx} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )
        print(log_line)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            adapter_dir = os.path.join(config["output_dir"], "best_adapter")
            model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/task4.yaml")
    parser.add_argument("--mdl_config", type=str, default="configs/model.yaml")
    args = parser.parse_args()

    main(args.config, args.mdl_config)
