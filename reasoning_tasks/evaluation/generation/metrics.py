"""
evaluation/generation/metrics.py

Metric helpers for Task 4 generation post-validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import math

def lcs_length(a: List[str], b: List[str]) -> int:
    """
    Compute longest common subsequence length.

    Args:
        a: First token sequence.
        b: Second token sequence.

    Returns:
        LCS length.
    """
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1(pred: str, ref: str) -> float:
    """
    Compute ROUGE-L F1 using whitespace tokenization.

    Args:
        pred: Predicted text.
        ref: Reference text.

    Returns:
        ROUGE-L F1 score.
    """
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    prec = lcs / len(pred_tokens)
    rec = lcs / len(ref_tokens)
    if prec + rec == 0:
        return 0.0

    return 2 * prec * rec / (prec + rec)


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Build n-grams from a token sequence.

    Args:
        tokens: Token sequence.
        n: N-gram order.

    Returns:
        List of n-gram tuples.
    """
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu2_score(pred: str, ref: str) -> float:
    """
    Compute a simple BLEU-2 score using whitespace tokenization.

    Args:
        pred: Predicted text.
        ref: Reference text.

    Returns:
        BLEU-2 score.
    """
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    ref_1grams = set(ngrams(ref_tokens, 1))
    ref_2grams = set(ngrams(ref_tokens, 2))
    pred_1grams = ngrams(pred_tokens, 1)
    pred_2grams = ngrams(pred_tokens, 2)

    p1_matches = sum(1 for ng in pred_1grams if ng in ref_1grams)
    p2_matches = sum(1 for ng in pred_2grams if ng in ref_2grams)

    p1_total = max(len(pred_1grams), 1)
    p2_total = max(len(pred_2grams), 1)

    p1 = p1_matches / p1_total
    p2 = p2_matches / p2_total

    if p1 == 0 or p2 == 0:
        return 0.0

    bp = 1.0
    if len(pred_tokens) < len(ref_tokens):
        bp = torch.exp(
            torch.tensor(
                1 - len(ref_tokens) / max(len(pred_tokens), 1),
                dtype=torch.float32,
            )
        ).item()

    return bp * ((p1 * p2) ** 0.5)


def compute_bertscore_values(preds: List[str], refs: List[str]) -> List[float | None]:
    """
    Compute per-sample BERTScore F1 values if bert_score is installed.

    Args:
        preds: Predicted texts.
        refs: Reference texts.

    Returns:
        Per-sample BERTScore F1 list. If unavailable, returns None for each sample.
    """
    try:
        from bert_score import score as bertscore_score
    except ImportError:
        return [None for _ in preds]

    _, _, f1 = bertscore_score(preds, refs, lang="en", verbose=False)
    return [float(value) for value in f1.detach().cpu().tolist()]

def mean_float(values: List[float | None]) -> float | None:
    """
    Compute mean for values while ignoring None.

    Args:
        values: Numeric or None values.

    Returns:
        Mean value or None.
    """
    valid_values = [float(value) for value in values if value is not None]
    if not valid_values:
        return None
    return float(sum(valid_values) / len(valid_values))


def build_generation_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build aggregate generation metrics from prediction records.

    Args:
        records: Full generation records with per-sample metrics.

    Returns:
        Dictionary with BLEU-2, ROUGE-L, optional BERTScore, and sample count.
    """
    bleu2_values = []
    rouge_values = []
    bertscore_values = []
    nll_values = []
    ppl_values = []

    for record in records:
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        bleu2_values.append(float(metrics.get("bleu_2", 0.0)))
        rouge_values.append(float(metrics.get("rouge_l", 0.0)))
        bertscore_values.append(metrics.get("bertscore_f1"))
        nll_values.append(metrics.get("nll"))
        ppl_values.append(metrics.get("ppl"))

    return {
        "bleu_2": float(sum(bleu2_values) / max(len(bleu2_values), 1)),
        "rouge_l": float(sum(rouge_values) / max(len(rouge_values), 1)),
        "bertscore_f1": mean_float(bertscore_values),
        "nll": mean_float(nll_values),
        "ppl": mean_float(ppl_values),
        "num_val_samples": len(records),
    }