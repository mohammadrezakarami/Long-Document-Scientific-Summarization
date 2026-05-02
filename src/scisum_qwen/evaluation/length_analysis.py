from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def count_words(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(text))


def compute_length_stats(predictions: list[str], references: list[str]) -> dict[str, float]:
    if not predictions:
        return {
            "avg_prediction_words": 0.0,
            "avg_reference_words": 0.0,
            "avg_compression_ratio": 0.0,
        }
    prediction_lengths = [count_words(text) for text in predictions]
    reference_lengths = [count_words(text) for text in references]
    compression = [
        prediction / reference
        for prediction, reference in zip(prediction_lengths, reference_lengths, strict=True)
        if reference > 0
    ]
    return {
        "avg_prediction_words": round(sum(prediction_lengths) / len(prediction_lengths), 4),
        "avg_reference_words": round(sum(reference_lengths) / len(reference_lengths), 4),
        "avg_compression_ratio": round(sum(compression) / len(compression), 4) if compression else 0.0,
    }

