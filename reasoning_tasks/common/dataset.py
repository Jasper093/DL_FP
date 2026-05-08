import json
import torch
from torch.utils.data import Dataset

VALID_SPH_ARNG_MODES = {"paired", "field"}
VALID_SYS_REACTION_MODES = {"verbal", "verbal_and_physical"}

def _clean_text(value: str | None) -> str:
    """
    Clean a text value safely.

    Args:
        value: Raw string or None.

    Returns:
        Stripped string. Empty string if input is None.
    """
    if value is None:
        return ""
    return str(value).strip()


def _clean_list(values: list[str] | None) -> list[str]:
    """
    Clean a list of strings safely.

    Args:
        values: Raw list of strings or None.

    Returns:
        List of non-empty stripped strings.
    """
    if not values:
        return []

    return [
        _clean_text(value)
        for value in values
        if _clean_text(value)
    ]

def render_turn_modalities(
    turn_like: dict,
    mode: str = "paired",
    utterance_prefix: str = "Utterance",
    video_prefix: str = "Video Cue",
    include_video: bool = True,
) -> str:
    utterance_list = _clean_list(turn_like.get("utterance_list", []))
    video_list = _clean_list(turn_like.get("video_list", [])) if include_video else []

    utterance_text = _clean_text(turn_like.get("utterance_text", ""))
    video_text = _clean_text(turn_like.get("video_text", "")) if include_video else ""

    final_utterance_text = " ".join(utterance_list) if utterance_list else utterance_text
    final_video_text = " ".join(video_list) if video_list else video_text

    if mode == "field":
        if include_video:
            return (
                "Utterance:\n"
                f"{final_utterance_text if final_utterance_text else '(empty)'}\n\n"
                "Video Cues:\n"
                f"{final_video_text if final_video_text else '(empty)'}"
            )

        return (
            "Utterance:\n"
            f"{final_utterance_text if final_utterance_text else '(empty)'}"
        )

    if mode == "paired":
        if include_video and utterance_list and video_list and len(utterance_list) == len(video_list):
            return "\n\n".join(
                f"[{utterance_prefix}{i + 1}] {utterance}\n"
                f"[{video_prefix}{i + 1}] {video}"
                for i, (utterance, video) in enumerate(zip(utterance_list, video_list))
            )

        if not include_video and utterance_list:
            return "\n".join(
                f"[{utterance_prefix}{i + 1}] {utterance}"
                for i, utterance in enumerate(utterance_list)
            )

    if include_video:
        return (
            "Utterance:\n"
            f"{final_utterance_text if final_utterance_text else '(empty)'}\n\n"
            "Video Cues:\n"
            f"{final_video_text if final_video_text else '(empty)'}"
        )

    return (
        "Utterance:\n"
        f"{final_utterance_text if final_utterance_text else '(empty)'}"
    )

def render_current_user_turn(
    current_user_turn: dict,
    sph_arng_mod: str = "paired",
    include_video: bool = True,
) -> str:
    return render_turn_modalities(
        turn_like=current_user_turn,
        mode=sph_arng_mod,
        utterance_prefix="Utterance",
        video_prefix="Video Cue",
        include_video=include_video,
    )

def format_history_turn(
    turn: dict,
    sph_arng_mod: str = "paired",
    sys_reaction: str = "verbal",
    include_video: bool = True,
) -> str:
    speaker_raw = turn["speaker"]
    speaker = speaker_raw.capitalize()

    is_therapist = speaker_raw.lower() in {"sys", "therapist", "system"}

    include_video = include_video and not (
        is_therapist and sys_reaction == "verbal"
    )

    turn_like = dict(turn)

    rendered = render_turn_modalities(
        turn_like=turn_like,
        mode=sph_arng_mod,
        utterance_prefix="Utterance",
        video_prefix="Video Cue",
        include_video=include_video,
    )

    return f"{speaker}:\n{rendered}"

def format_history(
    history: list[dict],
    sph_arng_mod: str = "paired",
    sys_reaction: str = "verbal",
    include_history: bool = True,
    include_video: bool = True,
) -> str:
    if not include_history:
        return "(dropped)"

    if not history:
        return "(empty)"

    rendered_turns = [
        format_history_turn(
            turn,
            sph_arng_mod=sph_arng_mod,
            sys_reaction=sys_reaction,
            include_video=include_video,
        )
        for turn in history
    ]

    rendered_turns = [x for x in rendered_turns if x.strip()]

    return "\n\n".join(rendered_turns) if rendered_turns else "(dropped)"


def render_common_prefix(
    sample: dict,
    sph_arng_mod: str = "paired",
    sys_reaction: str = "verbal",
    ablation: dict | None = None,
) -> str:
    ablation = ablation or {}

    include_history = ablation.get("history", True)
    include_video = ablation.get("video", True)

    history_text = format_history(
        sample["history"],
        sph_arng_mod=sph_arng_mod,
        sys_reaction=sys_reaction,
        include_history=include_history,
        include_video=include_video,
    )

    current_turn_text = render_current_user_turn(
        sample["current_user_turn"],
        sph_arng_mod=sph_arng_mod,
        include_video=include_video,
    )

    return (
        f"Problem Type: {sample['problem_type']}\n"
        f"Situation: {sample['situation']}\n\n"
        "Dialogue History:\n"
        f"{history_text}\n\n"
        "Current User Turn:\n"
        f"{current_turn_text}"
    )

def render_input_by_task(
    sample: dict,
    sph_arng_mod: str = "paired",
    sys_reaction: str = "verbal",
    ablation: dict | None = None,
) -> str:
    ablation = ablation or {}

    task_name = sample.get("task", "")

    base_text = render_common_prefix(
        sample,
        sph_arng_mod=sph_arng_mod,
        sys_reaction=sys_reaction,
        ablation=ablation,
    )

    fields = sample.get("fields", {})

    include_emotion = ablation.get("emotion", True)
    include_strategy = ablation.get("strategy", True)

    extra_lines = []

    if task_name == "user_emotion":
        return base_text

    if task_name == "therapist_strategy":
        extra_lines.append(f"User Emotion: {fields['user_emotion']}")

    elif task_name == "therapist_emotion":
        extra_lines.append(f"User Emotion: {fields['user_emotion']}")
        extra_lines.append(f"Therapist Strategy: {fields['therapist_strategy']}")

    elif task_name == "therapist_response":
        if include_emotion:
            extra_lines.append(f"User Emotion: {fields['user_emotion']}")
            extra_lines.append(f"Therapist Emotion: {fields['therapist_emotion']}")
        if include_strategy:
            extra_lines.append(f"Therapist Strategy: {fields['therapist_strategy']}")

    else:
        raise ValueError(f"Unsupported task name: {task_name}")

    if not extra_lines:
        return base_text

    return f"{base_text}\n\n" + "\n".join(extra_lines)


def serialize_response_target(
    target: dict,
    sys_reaction: str = "verbal",
) -> str:
    """
    Convert Task 4 structured response target into the exact prompt format.

    Args:
        target:
            Structured response target containing optional video_list
            and utterance_list.
        sys_reaction:
            - "verbal": output utterance_list only
            - "verbal_and_physical": output both video_list and utterance_list

    Returns:
        Serialized target text for supervised generation.
    """
    if sys_reaction not in VALID_SYS_REACTION_MODES:
        raise ValueError(
            f"Unsupported sys_reaction={sys_reaction}. "
            f"Expected one of {sorted(VALID_SYS_REACTION_MODES)}."
        )

    video_list = _clean_list(target.get("video_list", []))
    utterance_list = _clean_list(target.get("utterance_list", []))

    utterance_block = "\n".join(f"- {item}" for item in utterance_list)

    if sys_reaction == "verbal":
        return (
            "utterance_list:\n"
            f"{utterance_block}"
        ).strip()

    video_block = "\n".join(f"- {item}" for item in video_list)

    return (
        "video_list:\n"
        f"{video_block}\n\n"
        "utterance_list:\n"
        f"{utterance_block}"
    ).strip()


class BaseTaskDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        tokenizer,
        prompt_text,
        max_length=512,
        return_idx=False,
        sph_arng_mod="paired",
        sys_reaction="verbal",
        ablation=None
    ):
        if sph_arng_mod not in VALID_SPH_ARNG_MODES:
            raise ValueError(
                f"Unsupported sph_arng_mod={sph_arng_mod}. "
                f"Expected one of {sorted(VALID_SPH_ARNG_MODES)}."
            )

        if sys_reaction not in VALID_SYS_REACTION_MODES:
            raise ValueError(
                f"Unsupported sys_reaction={sys_reaction}. "
                f"Expected one of {sorted(VALID_SYS_REACTION_MODES)}."
            )

        self.samples = []
        self.tokenizer = tokenizer
        self.prompt_text = prompt_text.strip()
        self.max_length = max_length
        self.return_idx = return_idx
        self.sph_arng_mod = sph_arng_mod
        self.sys_reaction = sys_reaction
        self.ablation = ablation or {}

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def build_inference_text(self, sample: dict) -> str:
        input_text = render_input_by_task(
            sample,
            sph_arng_mod=self.sph_arng_mod,
            sys_reaction=self.sys_reaction,
            ablation=self.ablation,
        )
        return f"{self.prompt_text}\n\n{input_text}\n\nAnswer:"
    
    def __len__(self):
        return len(self.samples)

    def build_target_text(self, sample: dict) -> str:
        raise NotImplementedError

    def additional_info(self, idx):
        sample = self.samples[idx]

        return {
            "sample_id": {
                "dialog_index": sample["dialog_index"],
                "turn_index": sample["turn_index"],
            },
            "raw_input": {
                "problem_type": sample.get("problem_type", ""),
                "situation": sample.get("situation", ""),
                "history": sample.get("history", []),
                "utterance_list": sample["current_user_turn"].get("utterance_list", []),
                "video_list": sample["current_user_turn"].get("video_list", []),
                "fields": sample.get("fields", {}),
            },
        }

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt_only = self.build_inference_text(sample)
        target_text = self.build_target_text(sample)

        prompt_ids = self.tokenizer(
            prompt_only,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        target_ids = self.tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        if prompt_ids is None or target_ids is None:
            raise ValueError(
                f"Tokenizer returned None at idx={idx}. "
                f"prompt_ids is None: {prompt_ids is None}, "
                f"target_ids is None: {target_ids is None}"
            )

        if any(x is None for x in prompt_ids):
            raise ValueError(f"Found None in prompt_ids at idx={idx}")

        if any(x is None for x in target_ids):
            raise ValueError(f"Found None in target_ids at idx={idx}")

        if len(target_ids) >= self.max_length:
            target_ids = target_ids[: self.max_length - 1]

        max_prompt_len = self.max_length - len(target_ids)

        if max_prompt_len <= 0:
            raise ValueError(
                f"No room left for prompt at idx={idx}. "
                f"target_len={len(target_ids)}, max_length={self.max_length}"
            )

        prompt_ids = prompt_ids[:max_prompt_len]

        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_ids)) + target_ids

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError(
                "tokenizer.pad_token_id is None. "
                "Call setup_qwen_special_tokens(tokenizer) before building the dataset."
            )
        pad_id = int(pad_id)

        pad_len = self.max_length - len(input_ids)
        if pad_len < 0:
            raise ValueError(
                f"input_ids exceeds max_length at idx={idx}. "
                f"len(input_ids)={len(input_ids)}, max_length={self.max_length}"
            )

        input_ids = input_ids + ([pad_id] * pad_len)
        attention_mask = attention_mask + ([0] * pad_len)
        labels = labels + ([-100] * pad_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        ################# debug ################### 
        if (idx < 3):
            print("\n" + "="*50)
            print(f"[DEBUG SAMPLE {idx}]")

            print("\n[PROMPT TEXT]")
            print(prompt_only)

            print("\n[TARGET TEXT]")
            print(target_text)
            
            supervised_ids = input_ids[labels != -100]
            decoded_supervised = self.tokenizer.decode(
                supervised_ids.tolist(),
                skip_special_tokens=False,
            )
            
            if decoded_supervised.strip() != target_text.strip():
                raise ValueError(
                    f"[MASK ERROR]\nExpected: {repr(target_text)}\nGot: {repr(decoded_supervised)}"
                )

            n_valid = (labels != -100).sum().item()
            if n_valid == 0:
                raise ValueError(
                    f"All labels are masked at idx={idx}. "
                    f"prompt_len={len(prompt_ids)}, target_len={len(target_ids)}, max_length={self.max_length}"
                )
        ################# debug ################### 

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if self.return_idx:
            output["idx"] = torch.tensor(idx, dtype=torch.long)

        return output


class EmotionDataset(BaseTaskDataset):
    def build_target_text(self, sample: dict) -> str:
        return " " + str(sample["target"])


class ResponseGenerationDataset(BaseTaskDataset):
    def build_target_text(self, sample: dict) -> str:
        return "\n" + serialize_response_target(
            sample["target"],
            sys_reaction=self.sys_reaction,
        )