"""
evaluation/classification/inference.py

Closed-set validation inference for classification-style tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move all tensor values in a batch dictionary to the target device.

    Args:
        batch: Batch dictionary.
        device: Target torch device.

    Returns:
        Batch with tensor values moved to device.
    """
    moved_batch: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value
    return moved_batch


def run_classification_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    label_names: List[str],
    device: torch.device,
    tokenizer: Any,
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    """
    Run closed-set label scoring for classification tasks.

    Strategy:
        1. Recover the prompt-only prefix from labels == -100 positions.
        2. Score every candidate label as a continuation of the prompt.
        3. Normalize candidate scores with softmax to get class confidence.
        4. Choose argmax candidate as prediction.

    Args:
        model: Causal language model.
        dataloader: Validation dataloader.
        label_names: Ordered label names.
        device: Target torch device.
        tokenizer: Tokenizer used by the model.

    Returns:
        full_records, true label ids, predicted label ids.
    """
    model.eval()

    full_records: List[Dict[str, Any]] = []
    all_true_ids: List[int] = []
    all_pred_ids: List[int] = []

    label_to_id = {label: idx for idx, label in enumerate(label_names)}

    candidate_token_ids: Dict[str, List[int]] = {}
    for label in label_names:
        token_ids = tokenizer(
            " " + label,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if len(token_ids) == 0:
            raise ValueError(f"Empty tokenization for label: {label}")
        candidate_token_ids[label] = token_ids

    def score_candidate_label(prompt_ids: List[int], candidate_ids: List[int]) -> float:
        """
        Compute average log-probability of candidate tokens given the prompt.

        Args:
            prompt_ids: Prompt token ids.
            candidate_ids: Candidate label token ids.

        Returns:
            Average candidate-token log probability.
        """
        full_ids = prompt_ids + candidate_ids
        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
        attention_tensor = torch.ones_like(input_tensor, device=device)

        outputs = model(input_ids=input_tensor, attention_mask=attention_tensor)
        logits = outputs.logits[:, :-1, :]
        target_tokens = input_tensor[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)

        candidate_start = len(prompt_ids) - 1
        candidate_end = candidate_start + len(candidate_ids)

        candidate_log_probs = []
        for pos in range(candidate_start, candidate_end):
            token_id = int(target_tokens[0, pos].item())
            token_log_prob = log_probs[0, pos, token_id]
            candidate_log_probs.append(token_log_prob)

        return float(torch.stack(candidate_log_probs).mean().item())

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)

            input_ids = batch["input_ids"]
            labels = batch["labels"]
            indices = batch["idx"]

            batch_size = input_ids.size(0)

            for item_index in range(batch_size):
                sample_idx = int(indices[item_index].item())
                meta = dataloader.dataset.additional_info(sample_idx)  # pyright: ignore[reportAttributeAccessIssue]

                label_mask = labels[item_index] != -100
                valid_positions = label_mask.nonzero(as_tuple=False).squeeze(-1)
                if valid_positions.numel() == 0:
                    continue

                true_token_ids = labels[item_index, valid_positions].detach().cpu().tolist()
                true_label = tokenizer.decode(
                    true_token_ids,
                    skip_special_tokens=True,
                ).strip()

                if true_label not in label_to_id:
                    continue

                prompt_mask = (labels[item_index] == -100) & (
                    input_ids[item_index] != tokenizer.pad_token_id
                )
                prompt_ids = input_ids[item_index, prompt_mask].detach().cpu().tolist()

                if len(prompt_ids) == 0:
                    continue

                candidate_scores = [
                    score_candidate_label(
                        prompt_ids=prompt_ids,
                        candidate_ids=candidate_token_ids[label],
                    )
                    for label in label_names
                ]

                scores_tensor = torch.tensor(candidate_scores, dtype=torch.float32)
                probs_tensor = torch.softmax(scores_tensor, dim=0)
                sorted_probs, _ = torch.sort(probs_tensor, descending=True)

                pred_id = int(torch.argmax(probs_tensor).item())
                pred_label = label_names[pred_id]
                confidence = float(probs_tensor[pred_id].item())
                margin = float(sorted_probs[0].item() - sorted_probs[1].item()) if len(sorted_probs) > 1 else confidence
                true_id = label_to_id[true_label]

                record = {
                    "record_index": len(full_records),
                    "pred": pred_label,
                    "label": true_label,
                    "prediction": pred_label,
                    "reference": true_label,
                    "is_correct": pred_label == true_label,
                    "confidence": round(confidence, 6),
                    "margin": round(margin, 6),
                    "sample_id": meta["sample_id"],
                    "input": meta["raw_input"],
                    "candidate_scores": {
                        label_names[i]: round(float(scores_tensor[i].item()), 6)
                        for i in range(len(label_names))
                    },
                    "candidate_probs": {
                        label_names[i]: round(float(probs_tensor[i].item()), 6)
                        for i in range(len(label_names))
                    },
                }

                full_records.append(record)
                all_true_ids.append(true_id)
                all_pred_ids.append(pred_id)

    return full_records, all_true_ids, all_pred_ids
