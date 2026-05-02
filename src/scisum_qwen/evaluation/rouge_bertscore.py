from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import re
from typing import Any

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1))


def _safe_f1(overlap: int, prediction_total: int, reference_total: int) -> tuple[float, float, float]:
    precision = overlap / prediction_total if prediction_total else 0.0
    recall = overlap / reference_total if reference_total else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def rouge_n(prediction: str, reference: str, n: int) -> dict[str, float]:
    prediction_tokens = _tokenize(prediction)
    reference_tokens = _tokenize(reference)
    prediction_counts = _ngram_counts(prediction_tokens, n)
    reference_counts = _ngram_counts(reference_tokens, n)
    overlap = sum(min(prediction_counts[gram], reference_counts[gram]) for gram in prediction_counts.keys() & reference_counts.keys())
    precision, recall, f1 = _safe_f1(overlap, sum(prediction_counts.values()), sum(reference_counts.values()))
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for left_idx, left_token in enumerate(left, start=1):
        for right_idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                dp[left_idx][right_idx] = dp[left_idx - 1][right_idx - 1] + 1
            else:
                dp[left_idx][right_idx] = max(dp[left_idx - 1][right_idx], dp[left_idx][right_idx - 1])
    return dp[-1][-1]


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    prediction_tokens = _tokenize(prediction)
    reference_tokens = _tokenize(reference)
    lcs = _lcs_length(prediction_tokens, reference_tokens)
    precision, recall, f1 = _safe_f1(lcs, len(prediction_tokens), len(reference_tokens))
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def aggregate_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    if not predictions:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
        }
    rouge1_scores = [rouge_n(prediction, reference, 1)["f1"] for prediction, reference in zip(predictions, references, strict=True)]
    rouge2_scores = [rouge_n(prediction, reference, 2)["f1"] for prediction, reference in zip(predictions, references, strict=True)]
    rougeL_scores = [rouge_l(prediction, reference)["f1"] for prediction, reference in zip(predictions, references, strict=True)]
    return {
        "rouge1": round(sum(rouge1_scores) / len(rouge1_scores), 4),
        "rouge2": round(sum(rouge2_scores) / len(rouge2_scores), 4),
        "rougeL": round(sum(rougeL_scores) / len(rougeL_scores), 4),
    }


def try_compute_bertscore(
    predictions: list[str],
    references: list[str],
    *,
    cache_dir: str | Path = "data/raw/evaluate_cache",
) -> dict[str, Any]:
    try:
        import evaluate
    except ImportError:
        return {"bertscore_f1": None, "status": "evaluate_not_installed"}

    try:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        scorer = evaluate.load("bertscore", cache_dir=str(cache_path))
        result = scorer.compute(predictions=predictions, references=references, lang="en")
    except Exception as exc:  # pragma: no cover - network/model availability depends on environment
        return {"bertscore_f1": None, "status": f"unavailable: {exc}"}

    mean_f1 = sum(result["f1"]) / len(result["f1"]) if result["f1"] else 0.0
    return {"bertscore_f1": round(mean_f1, 4), "status": "ok"}
