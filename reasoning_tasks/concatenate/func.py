import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


CUE_MARKERS = [
    "The speaker",
    "The emotion state of the speaker",
    "The speaker's emotional expression",
    "It is difficult to determine",
]


def normalize_text(raw_text: str) -> str:
    """
    Normalize whitespace in a dialog item without changing semantics.

    Args:
        raw_text: Original text from the dataset.

    Returns:
        Cleaned text with stripped edges and collapsed internal whitespace.
    """
    return " ".join(raw_text.strip().split())


def normalize_speaker(raw_speaker: str) -> str:
    """
    Normalize speaker labels to a controlled set.

    Args:
        raw_speaker: Original speaker field from the dataset.

    Returns:
        'user' or 'sys'.

    Raises:
        ValueError: If the speaker label is unknown.
    """
    speaker = raw_speaker.strip().lower()

    if speaker in {"user", "client"}:
        return "user"
    if speaker in {"sys", "system", "therapist"}:
        return "sys"

    raise ValueError(f"Unknown speaker label: {raw_speaker}")


def majority_vote(label_sequence: List[str]) -> Optional[str]:
    """
    Compute the majority-vote label from a sequence.

    Policy:
    1. Majority vote
    2. If tied, choose the last occurring tied label

    Args:
        label_sequence: Ordered list of labels.

    Returns:
        A single label, or None if the sequence is empty.
    """
    if not label_sequence:
        return None

    counter = Counter(label_sequence)
    max_count = max(counter.values())
    candidates = [label for label, count in counter.items() if count == max_count]

    if len(candidates) == 1:
        return candidates[0]

    for label in reversed(label_sequence):
        if label in candidates:
            return label

    return None


def find_earliest_marker(text: str, markers: List[str]) -> Optional[int]:
    """
    Find the earliest occurrence of any cue marker in the text.

    Args:
        text: Normalized text.
        markers: Candidate cue-start markers.

    Returns:
        The earliest marker index, or None if no marker is found.
    """
    earliest_pos: Optional[int] = None

    for marker in markers:
        # Use regex word boundary for safer matching when possible
        if marker == "The speaker":
            match = re.search(r"\bThe speaker\b", text)
            pos = match.start() if match else -1
        else:
            pos = text.find(marker)

        if pos != -1:
            if earliest_pos is None or pos < earliest_pos:
                earliest_pos = pos

    return earliest_pos


def split_utterance_and_video(raw_text: str) -> Dict[str, str]:
    """
    Split one mixed fragment into utterance_part and video_part.

    Strategy:
    - Split at the earliest cue marker.
    - If no marker exists, keep the full text as utterance_part.
    - This is a heuristic split, not ground truth.

    Args:
        raw_text: Original mixed text field from the dataset.

    Returns:
        Dictionary containing:
        - raw_text
        - utterance_part
        - video_part
        - split_confidence
        - split_method
    """
    text = normalize_text(raw_text)
    split_pos = find_earliest_marker(text, CUE_MARKERS)

    if split_pos is None:
        return {
            "raw_text": text,
            "utterance_part": text,
            "video_part": "",
            "split_confidence": "low",
            "split_method": "cue_marker_v1",
        }

    utterance_part = text[:split_pos].strip()
    video_part = text[split_pos:].strip()

    if not utterance_part:
        return {
            "raw_text": text,
            "utterance_part": text,
            "video_part": "",
            "split_confidence": "low",
            "split_method": "cue_marker_v1",
        }

    return {
        "raw_text": text,
        "utterance_part": utterance_part,
        "video_part": video_part,
        "split_confidence": "high",
        "split_method": "cue_marker_v1",
    }


def build_fragment_record(dialog_item: Dict[str, Any], fragment_id: int) -> Dict[str, Any]:
    """
    Convert one raw dialog item into one normalized fragment record,
    and split its mixed text into utterance and video parts.

    Args:
        dialog_item: One item from sample["dialog"].
        fragment_id: Index of the dialog item within the conversation.

    Returns:
        A normalized fragment dictionary.
    """
    split_result = split_utterance_and_video(dialog_item["text"])

    return {
        "raw_text": split_result["raw_text"],
        "utterance_part": split_result["utterance_part"],
        "video_part": split_result["video_part"],
        "split_confidence": split_result["split_confidence"],
        "split_method": split_result["split_method"],
        "speaker": normalize_speaker(dialog_item["speaker"]),
        "emotion": dialog_item.get("emotion"),
        "strategy": dialog_item.get("strategy"),
        "fragment_id": fragment_id,
    }


def build_merged_turn(turn_fragments):
    """
    Merge consecutive same-speaker fragments into one turn.

    The turn keeps:
    - utterance_list
    - video_list

    Args:
        turn_fragments: Consecutive same-speaker fragment records.

    Returns:
        One merged turn dictionary.
    """
    utterance_list = []
    video_list = []
    emotion_sequence = []
    strategy_sequence = []
    fragment_ids = []

    for frag in turn_fragments:
        utterance_list.append(frag["utterance_part"] or "")
        video_list.append(frag["video_part"] or "")
        fragment_ids.append(frag["fragment_id"])

        if frag.get("emotion") is not None:
            emotion_sequence.append(frag["emotion"])

        if frag.get("strategy") is not None:
            strategy_sequence.append(frag["strategy"])

    return {
        "utterance_list": utterance_list,
        "video_list": video_list,
        "speaker": turn_fragments[0]["speaker"],
        "emotion": majority_vote(emotion_sequence),
        "strategy": majority_vote(strategy_sequence),
        "emotion_sequence": emotion_sequence,
        "strategy_sequence": strategy_sequence,
        "fragment_ids": fragment_ids,
    }

def merge_dialog_into_turns(dialog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive same-speaker dialog items into turns.

    Args:
        dialog: Original sample["dialog"] list.

    Returns:
        A list of merged turn dictionaries.
    """
    if not dialog:
        return []

    fragments = [
        build_fragment_record(dialog_item=dialog_item, fragment_id=fragment_id)
        for fragment_id, dialog_item in enumerate(dialog)
    ]

    merged_turns: List[Dict[str, Any]] = []
    current_turn_fragments: List[Dict[str, Any]] = [fragments[0]]

    for fragment in fragments[1:]:
        if fragment["speaker"] == current_turn_fragments[-1]["speaker"]:
            current_turn_fragments.append(fragment)
        else:
            merged_turns.append(build_merged_turn(current_turn_fragments))
            current_turn_fragments = [fragment]

    merged_turns.append(build_merged_turn(current_turn_fragments))
    return merged_turns


def build_processed_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one raw sample into the final processed sample structure.

    Final structure:
    {
        "problem_type": ...,
        "situation": ...,
        "dialog": [merged turns]
    }

    Args:
        sample: One raw sample from the dataset.

    Returns:
        One processed sample.
    """
    dialog = sample.get("dialog", [])
    if not isinstance(dialog, list):
        raise TypeError("sample['dialog'] must be a list.")

    return {
        "problem_type": sample.get("problem_type"),
        "situation": sample.get("situation"),
        "dialog": merge_dialog_into_turns(dialog),
    }


def process_dataset(input_path: str, output_path: str) -> List[Dict[str, Any]]:
    """
    Process the raw dataset and save the merged-turn version.

    Args:
        input_path: Path to the original JSON file.
        output_path: Path to save the processed JSON file.

    Returns:
        The processed dataset as a list of samples.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError("The dataset root must be a list of samples.")

    processed_data = [build_processed_sample(sample) for sample in data]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    return processed_data


if __name__ == "__main__":
    input_path = "/home/jsp/DL_final_project/text/val.json"
    output_path = "/home/jsp/MESC/reasoning_tasks/data/inf_audio_raw/val_ccnt.json"

    processed_data = process_dataset(input_path=input_path, output_path=output_path)
    print(f"Saved {len(processed_data)} processed samples to {output_path}")