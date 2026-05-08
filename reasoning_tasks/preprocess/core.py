"""
Core preprocessing logic for anchor-based sample generation.

This file intentionally focuses on:
- turn normalization
- anchor validation
- history slicing
- structured sample generation

It does NOT do:
- final prompt rendering
- tokenization
- model training
"""

from __future__ import annotations

from typing import Any

from .utils import normalize_fragment_list


USER_EMOTION_TASK = "user_emotion"
THERAPIST_STRATEGY_TASK = "therapist_strategy"
THERAPIST_EMOTION_TASK = "therapist_emotion"
THERAPIST_RESPONSE_TASK = "therapist_response"


def normalize_turn(raw_turn: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a single turn.

    Args:
        raw_turn: Raw turn dictionary from the turn-wise dataset.

    Returns:
        Normalized turn dictionary containing both raw lists and joined text.
    """
    utterance_list = normalize_fragment_list(raw_turn.get("utterance_list", []))
    video_list = normalize_fragment_list(raw_turn.get("video_list", []))

    return {
        "speaker": raw_turn.get("speaker"),
        "utterance_list": utterance_list,
        "video_list": video_list,
        # "utterance_text": join_fragments(utterance_list),
        # "video_text": join_fragments(video_list),
        "emotion": raw_turn.get("emotion"),
        "strategy": raw_turn.get("strategy"),
    }


def normalize_dialog(raw_dialog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize all turns in one dialogue.

    Args:
        raw_dialog: Raw dialogue turn list.

    Returns:
        List of normalized turns.
    """
    return [normalize_turn(turn) for turn in raw_dialog]


def build_history(dialog: list[dict[str, Any]], anchor_index: int) -> list[dict[str, Any]]:
    """
    Build history from all turns before the anchor turn.

    Args:
        dialog: Normalized dialogue.
        anchor_index: Anchor turn index.

    Returns:
        Structured history list.
    """
    history: list[dict[str, Any]] = []

    for turn in dialog[:anchor_index]:
        history.append(
            {
                "speaker": turn["speaker"],
                # "utterance_text": turn["utterance_text"],
                # "video_text": turn["video_text"],
                "utterance_list": turn["utterance_list"],
                "video_list": turn["video_list"],
            }
        )

    return history


def has_valid_next_sys_turn(dialog: list[dict[str, Any]], turn_index: int) -> bool:
    """
    Check whether the immediate next turn exists and is a therapist/system turn.

    Args:
        dialog: Normalized dialogue.
        turn_index: Current anchor turn index.

    Returns:
        True if the next turn exists and has speaker == 'sys'.
    """
    if turn_index + 1 >= len(dialog):
        return False
    return dialog[turn_index + 1]["speaker"] == "sys"


def build_task1_sample(
    dialog: list[dict[str, Any]],
    dialog_index: int,
    turn_index: int,
    problem_type: str,
    situation: str,
) -> dict[str, Any] | None:
    """
    Build a structured Task 1 sample.

    Args:
        dialog: Normalized dialogue.
        dialog_index: Dialogue index.
        turn_index: Anchor turn index.
        problem_type: Dialogue problem type.
        situation: Dialogue situation.

    Returns:
        Structured sample or None if invalid.
    """
    current_turn = dialog[turn_index]

    if current_turn["speaker"] != "user":
        return None
    if not current_turn.get("emotion"):
        return None

    return {
        "dialog_index": dialog_index,
        "turn_index": turn_index,
        "task": USER_EMOTION_TASK,
        "problem_type": problem_type,
        "situation": situation,
        "history": build_history(dialog, turn_index),
        "current_user_turn": {
            # "utterance_text": current_turn["utterance_text"],
            # "video_text": current_turn["video_text"],
            "utterance_list": current_turn["utterance_list"],
            "video_list": current_turn["video_list"],
        },
        "fields": {},
        "target": current_turn["emotion"],
    }


def build_task2_sample(
    dialog: list[dict[str, Any]],
    dialog_index: int,
    turn_index: int,
    problem_type: str,
    situation: str,
) -> dict[str, Any] | None:
    """
    Build a structured Task 2 sample.

    Args:
        dialog: Normalized dialogue.
        dialog_index: Dialogue index.
        turn_index: Anchor turn index.
        problem_type: Dialogue problem type.
        situation: Dialogue situation.

    Returns:
        Structured sample or None if invalid.
    """
    current_turn = dialog[turn_index]

    if current_turn["speaker"] != "user":
        return None
    if not current_turn.get("emotion"):
        return None
    if not has_valid_next_sys_turn(dialog, turn_index):
        return None

    next_turn = dialog[turn_index + 1]
    if not next_turn.get("strategy"):
        return None

    return {
        "dialog_index": dialog_index,
        "turn_index": turn_index,
        "task": THERAPIST_STRATEGY_TASK,
        "problem_type": problem_type,
        "situation": situation,
        "history": build_history(dialog, turn_index),
        "current_user_turn": {
            # "utterance_text": current_turn["utterance_text"],
            # "video_text": current_turn["video_text"],
            "utterance_list": current_turn["utterance_list"],
            "video_list": current_turn["video_list"],
        },
        "fields": {
            "user_emotion": current_turn["emotion"],
        },
        "target": next_turn["strategy"],
    }


def build_task3_sample(
    dialog: list[dict[str, Any]],
    dialog_index: int,
    turn_index: int,
    problem_type: str,
    situation: str,
) -> dict[str, Any] | None:
    """
    Build a structured Task 3 sample.

    Args:
        dialog: Normalized dialogue.
        dialog_index: Dialogue index.
        turn_index: Anchor turn index.
        problem_type: Dialogue problem type.
        situation: Dialogue situation.

    Returns:
        Structured sample or None if invalid.
    """
    current_turn = dialog[turn_index]

    if current_turn["speaker"] != "user":
        return None
    if not current_turn.get("emotion"):
        return None
    if not has_valid_next_sys_turn(dialog, turn_index):
        return None

    next_turn = dialog[turn_index + 1]
    if not next_turn.get("strategy"):
        return None
    if not next_turn.get("emotion"):
        return None

    return {
        "dialog_index": dialog_index,
        "turn_index": turn_index,
        "task": THERAPIST_EMOTION_TASK,
        "problem_type": problem_type,
        "situation": situation,
        "history": build_history(dialog, turn_index),
        "current_user_turn": {
            # "utterance_text": current_turn["utterance_text"],
            # "video_text": current_turn["video_text"],
            "utterance_list": current_turn["utterance_list"],
            "video_list": current_turn["video_list"],
        },
        "fields": {
            "user_emotion": current_turn["emotion"],
            "therapist_strategy": next_turn["strategy"],
        },
        "target": next_turn["emotion"],
    }


def build_task4_sample(
    dialog: list[dict[str, Any]],
    dialog_index: int,
    turn_index: int,
    problem_type: str,
    situation: str,
) -> dict[str, Any] | None:
    """
    Build a structured Task 4 sample.

    Args:
        dialog: Normalized dialogue.
        dialog_index: Dialogue index.
        turn_index: Anchor turn index.
        problem_type: Dialogue problem type.
        situation: Dialogue situation.

    Returns:
        Structured sample or None if invalid.
    """
    current_turn = dialog[turn_index]

    if current_turn["speaker"] != "user":
        return None
    if not current_turn.get("emotion"):
        return None
    if not has_valid_next_sys_turn(dialog, turn_index):
        return None

    next_turn = dialog[turn_index + 1]
    if not next_turn.get("strategy"):
        return None
    if not next_turn.get("emotion"):
        return None
    if not next_turn.get("video_list"):
        return None
    if not next_turn.get("utterance_list"):
        return None

    return {
        "dialog_index": dialog_index,
        "turn_index": turn_index,
        "task": THERAPIST_RESPONSE_TASK,
        "problem_type": problem_type,
        "situation": situation,
        "history": build_history(dialog, turn_index),
        "current_user_turn": {
            # "utterance_text": current_turn["utterance_text"],
            # "video_text": current_turn["video_text"],
            "utterance_list": current_turn["utterance_list"],
            "video_list": current_turn["video_list"],
        },
        "fields": {
            "user_emotion": current_turn["emotion"],
            "therapist_strategy": next_turn["strategy"],
            "therapist_emotion": next_turn["emotion"],
        },
        "target": {
            "video_list": next_turn["video_list"],
            "utterance_list": next_turn["utterance_list"],
        },
    }


def process_dialog(raw_dialog_record: dict[str, Any], dialog_index: int, tasks_cfg: dict[str, bool]) -> dict[str, list[dict[str, Any]]]:
    """
    Process a single dialogue into task-wise structured samples.

    Args:
        raw_dialog_record: One dialogue object from the raw split file.
        dialog_index: Dialogue index in the split.
        tasks_cfg: Task enable / disable config.

    Returns:
        Dictionary mapping task name to structured samples.
    """
    problem_type = raw_dialog_record["problem_type"]
    situation = raw_dialog_record["situation"]
    dialog = normalize_dialog(raw_dialog_record["dialog"])

    task_to_records: dict[str, list[dict[str, Any]]] = {
        USER_EMOTION_TASK: [],
        THERAPIST_STRATEGY_TASK: [],
        THERAPIST_EMOTION_TASK: [],
        THERAPIST_RESPONSE_TASK: [],
    }

    for turn_index, _ in enumerate(dialog):
        if tasks_cfg.get("enable_user_emotion", True):
            sample = build_task1_sample(dialog, dialog_index, turn_index, problem_type, situation)
            if sample is not None:
                task_to_records[USER_EMOTION_TASK].append(sample)

        if tasks_cfg.get("enable_therapist_strategy", True):
            sample = build_task2_sample(dialog, dialog_index, turn_index, problem_type, situation)
            if sample is not None:
                task_to_records[THERAPIST_STRATEGY_TASK].append(sample)

        if tasks_cfg.get("enable_therapist_emotion", True):
            sample = build_task3_sample(dialog, dialog_index, turn_index, problem_type, situation)
            if sample is not None:
                task_to_records[THERAPIST_EMOTION_TASK].append(sample)

        if tasks_cfg.get("enable_therapist_response", True):
            sample = build_task4_sample(dialog, dialog_index, turn_index, problem_type, situation)
            if sample is not None:
                task_to_records[THERAPIST_RESPONSE_TASK].append(sample)

    return task_to_records


def process_split(raw_split: list[dict[str, Any]], tasks_cfg: dict[str, bool]) -> dict[str, list[dict[str, Any]]]:
    """
    Process one split into task-wise structured samples.

    Args:
        raw_split: Full raw split list.
        tasks_cfg: Task enable / disable config.

    Returns:
        Dictionary mapping task name to structured sample list.
    """
    merged_task_to_records: dict[str, list[dict[str, Any]]] = {
        USER_EMOTION_TASK: [],
        THERAPIST_STRATEGY_TASK: [],
        THERAPIST_EMOTION_TASK: [],
        THERAPIST_RESPONSE_TASK: [],
    }

    for dialog_index, raw_dialog_record in enumerate(raw_split):
        dialog_task_to_records = process_dialog(raw_dialog_record, dialog_index, tasks_cfg)
        for task_name, records in dialog_task_to_records.items():
            merged_task_to_records[task_name].extend(records)

    return merged_task_to_records
