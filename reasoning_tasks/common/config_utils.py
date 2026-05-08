import yaml
import time
import os
from pathlib import Path

PROMPT_FILE_MAP = {
    "user_emotion": {
        "default": "user_emotion_prompt.txt",
    },
    "therapist_strategy": {
        "default": "therapist_strategy_prompt.txt",
    },
    "therapist_emotion": {
        "default": "therapist_emotion_prompt.txt",
    },
    "therapist_response": {
        "verbal": "therapist_response_v_prompt.txt",
        "verbal_and_physical": "therapist_response_vp_prompt.txt",
    },
}

def select_prompt_path(
    prompt_dir: str,
    task_name: str,
    sys_reaction: str = "verbal",
) -> Path:
    """
    Select the prompt file path according to task name and formatting mode.

    Args:
        prompt_dir:
            Directory containing prompt .txt files.
        task_name:
            Task name, e.g. user_emotion, therapist_strategy,
            therapist_emotion, therapist_response.
        sys_reaction:
            Response-generation output mode.
            Used only for therapist_response.

    Returns:
        Path to selected prompt file.
    """
    if task_name not in PROMPT_FILE_MAP:
        raise ValueError(f"Unsupported task_name={task_name}")

    task_prompt_map = PROMPT_FILE_MAP[task_name]

    if task_name == "therapist_response":
        if sys_reaction not in task_prompt_map:
            raise ValueError(
                f"Unsupported sys_reaction={sys_reaction}. "
                f"Expected one of {sorted(task_prompt_map.keys())}."
            )
        filename = task_prompt_map[sys_reaction]
    else:
        filename = task_prompt_map["default"]

    prompt_path = Path(prompt_dir) / filename

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_text(
    prompt_dir: str,
    task_name: str,
    sys_reaction: str = "verbal",
) -> str:
    prompt_path = select_prompt_path(
        prompt_dir=prompt_dir,
        task_name=task_name,
        sys_reaction=sys_reaction,
    )

    return prompt_path.read_text(encoding="utf-8").strip()

def load_config(task_config_path, model_config_path):
    task_cfg = load_yaml(task_config_path)
    model_cfg = load_yaml(model_config_path)

    # merge
    config = {**task_cfg, **model_cfg}

    return config


def debug_dataset_sample(dataset, printer, sample_idx=0, stage_name="Sample Debug"):
    """
    Inspect one dataset sample and print basic label sanity information.

    Args:
        dataset:
            Dataset object that returns a dict containing at least:
            - "input_ids"
            - "labels"

        printer:
            Logger / printer object with methods:
            - separator()
            - stage(msg)
            - info(msg)
            - warn(msg)

        sample_idx:
            Index of the sample to inspect.

        stage_name:
            Title shown in the stage printout.

    Returns:
        dict:
            A summary dictionary containing:
            - sample_idx
            - input_length
            - valid_label_count
            - all_labels_masked
    """
    sample = dataset[sample_idx]
    labels = sample["labels"]

    valid_label_count = (labels != -100).sum().item()
    input_length = len(sample["input_ids"])
    all_labels_masked = (valid_label_count == 0)

    printer.separator()
    printer.stage(stage_name)
    printer.info(f"Sample index: {sample_idx}")
    printer.info(f"Input length: {input_length}")
    printer.info(f"Valid labels: {valid_label_count}")

    if all_labels_masked:
        printer.warn("ALL LABELS ARE -100 -> NaN LOSS RISK")

    printer.separator()

    return {
        "sample_idx": sample_idx,
        "input_length": input_length,
        "valid_label_count": valid_label_count,
        "all_labels_masked": all_labels_masked,
    }

def setup_qwen_special_tokens(tokenizer):
    eos_id = None

    if hasattr(tokenizer, "eod_id") and tokenizer.eod_id is not None:
        eos_id = int(tokenizer.eod_id)
    elif hasattr(tokenizer, "im_end_id") and tokenizer.im_end_id is not None:
        eos_id = int(tokenizer.im_end_id)
    elif tokenizer.eos_token_id is not None:
        eos_id = int(tokenizer.eos_token_id)

    if eos_id is None:
        raise ValueError("Could not find a valid EOS token id.")

    eos_token = tokenizer.convert_ids_to_tokens(eos_id)

    tokenizer.eos_token = eos_token
    tokenizer.pad_token = eos_token

    if tokenizer.pad_token_id is None:
        raise ValueError("Failed to assign tokenizer.pad_token_id.")

    return {
        "eos_id": eos_id,
        "pad_id": int(tokenizer.pad_token_id),
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
    }

def apply_special_token_ids_to_model(model, token_info):
    """
    Apply tokenizer-derived special token ids to model.config.

    Args:
        model:
            Hugging Face model instance.

        token_info:
            Dict returned by setup_qwen_special_tokens().
    """
    model.config.eos_token_id = int(token_info["eos_id"])
    model.config.pad_token_id = int(token_info["pad_id"])

class PCSPrinter:
    """
    Process Control System Printer
    Lightweight structured logger for training/debugging

    Features:
    - Stage tracking
    - Timestamp
    - PID tagging
    - Optional debug verbosity
    """

    def __init__(self, debug=True):
        self.debug_mode = debug
        self.start_time = time.time()
        self.last_time = self.start_time
        self.pid = os.getpid()

    def _format_time(self, seconds: float) -> str:
        if seconds >= 86400:
            d = int(seconds // 86400)
            h = int((seconds % 86400) // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{d}d {h}h {m}m {s:.2f}s"
        elif seconds >= 3600:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{h}h {m}m {s:.2f}s"
        elif seconds >= 60:
            m = int(seconds // 60)
            s = seconds % 60
            return f"{m}m {s:.2f}s"
        else:
            return f"{seconds:.2f}s"

    def _prefix(self, level):
        now = time.time()
        delta = now - self.last_time
        total = now - self.start_time
        self.last_time = now

        delta_str = self._format_time(delta)
        total_str = self._format_time(total)

        return f"[{level}] [PID:{self.pid}] [Δ:{delta_str} | T:{total_str}]"

    def stage(self, msg):
        print(f"\n{self._prefix('STAGE')}: {msg}")

    def info(self, msg):
        print(f"{self._prefix('INFO')}: {msg}")

    def debug(self, msg):
        if self.debug_mode:
            print(f"{self._prefix('DEBUG')}: {msg}")

    def warn(self, msg):
        print(f"{self._prefix('WARN')}: {msg}")

    def separator(self):
        print("=" * 60)