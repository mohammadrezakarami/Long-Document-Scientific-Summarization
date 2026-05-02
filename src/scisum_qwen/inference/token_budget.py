from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenBudgetConfig:
    max_input_tokens: int = 4096
    chunk_token_limit: int = 1200
    safety_margin: float = 0.9
    fallback_chars_per_token: float = 4.0


def estimate_text_tokens(text: str, *, fallback_chars_per_token: float = 4.0) -> int:
    if not text:
        return 0
    return max(1, int(round(len(text) / fallback_chars_per_token)))


def should_use_hierarchical(
    total_tokens: int,
    *,
    max_input_tokens: int = 4096,
    safety_margin: float = 0.9,
) -> bool:
    return total_tokens > int(max_input_tokens * safety_margin)


def allocate_section_word_budgets(
    section_token_counts: dict[str, int],
    *,
    total_word_budget: int,
    minimum_section_words: int = 24,
) -> dict[str, int]:
    if not section_token_counts:
        return {}
    if total_word_budget <= 0:
        total_word_budget = 180

    total_tokens = sum(section_token_counts.values()) or 1
    budgets: dict[str, int] = {}
    for section_name, token_count in section_token_counts.items():
        proportional = round((token_count / total_tokens) * total_word_budget)
        budgets[section_name] = max(minimum_section_words, proportional)
    return budgets


def infer_target_word_budget(reference_summary: str | None = None, *, default_budget: int = 180) -> int:
    if not reference_summary:
        return default_budget
    word_count = len(reference_summary.split())
    if word_count <= 0:
        return default_budget
    return max(40, min(320, int(round(word_count * 1.15))))
