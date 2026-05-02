from __future__ import annotations

from dataclasses import dataclass
import re

REQUIRED_STRUCTURED_FIELDS = (
    "TL;DR",
    "Problem",
    "Method",
    "Key Contributions",
    "Results",
    "Limitations",
)


@dataclass(frozen=True)
class FormatEvaluation:
    format_valid: int
    field_completion_rate: float
    present_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]


def evaluate_structured_format(text: str, required_fields: tuple[str, ...] = REQUIRED_STRUCTURED_FIELDS) -> FormatEvaluation:
    present_fields = []
    for field in required_fields:
        pattern = re.compile(rf"^\s*{re.escape(field)}\s*:", flags=re.IGNORECASE | re.MULTILINE)
        if pattern.search(text):
            present_fields.append(field)
    missing_fields = [field for field in required_fields if field not in present_fields]
    completion_rate = len(present_fields) / len(required_fields) if required_fields else 1.0
    return FormatEvaluation(
        format_valid=int(len(missing_fields) == 0),
        field_completion_rate=round(completion_rate, 4),
        present_fields=tuple(present_fields),
        missing_fields=tuple(missing_fields),
    )


def aggregate_format_evaluations(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {"format_validity": 0.0, "field_completion_rate": 0.0}
    evaluations = [evaluate_structured_format(text) for text in texts]
    return {
        "format_validity": round(sum(item.format_valid for item in evaluations) / len(evaluations), 4),
        "field_completion_rate": round(sum(item.field_completion_rate for item in evaluations) / len(evaluations), 4),
    }

