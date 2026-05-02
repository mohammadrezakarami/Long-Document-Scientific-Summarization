from __future__ import annotations

from collections import Counter
import math
import re

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_PATTERN.split(text.strip()) if sentence.strip()]
    return sentences


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def sentence_similarity(left: str, right: str) -> float:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0

    left_counts = Counter(left_tokens)
    right_counts = Counter(right_tokens)
    overlap = sum(min(left_counts[token], right_counts[token]) for token in set(left_counts) & set(right_counts))
    denominator = math.sqrt(sum(value * value for value in left_counts.values())) * math.sqrt(
        sum(value * value for value in right_counts.values())
    )
    if denominator == 0:
        return 0.0
    return overlap / denominator


def build_similarity_matrix(sentences: list[str]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for sentence in sentences:
        row: list[float] = []
        for other in sentences:
            row.append(sentence_similarity(sentence, other))
        matrix.append(row)
    return matrix


def pagerank_scores(matrix: list[list[float]], *, damping: float = 0.85, max_iter: int = 50, tol: float = 1e-5) -> list[float]:
    size = len(matrix)
    if size == 0:
        return []
    scores = [1.0 / size] * size
    for _ in range(max_iter):
        updated = [(1.0 - damping) / size] * size
        for column in range(size):
            row_sum = sum(matrix[column]) - matrix[column][column]
            if row_sum <= 0:
                continue
            for row in range(size):
                if row == column:
                    continue
                updated[row] += damping * scores[column] * (matrix[column][row] / row_sum)
        delta = sum(abs(updated[idx] - scores[idx]) for idx in range(size))
        scores = updated
        if delta < tol:
            break
    return scores


def infer_sentence_budget(text: str, *, target_word_budget: int | None = None, max_sentences: int = 6) -> int:
    sentences = split_sentences(text)
    if not sentences:
        return 0
    if target_word_budget is None or target_word_budget <= 0:
        return min(max_sentences, max(2, math.ceil(len(sentences) * 0.2)))

    avg_sentence_words = sum(len(tokenize(sentence)) for sentence in sentences) / len(sentences)
    if avg_sentence_words <= 0:
        return min(max_sentences, 3)
    estimated = max(1, round(target_word_budget / avg_sentence_words))
    return min(max_sentences, estimated)


def textrank_summarize(text: str, *, target_word_budget: int | None = None, max_sentences: int = 6) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= 2:
        return " ".join(sentences)

    sentence_budget = infer_sentence_budget(text, target_word_budget=target_word_budget, max_sentences=max_sentences)
    matrix = build_similarity_matrix(sentences)
    scores = pagerank_scores(matrix)
    ranked_indices = sorted(range(len(sentences)), key=lambda idx: scores[idx], reverse=True)[:sentence_budget]
    selected = sorted(ranked_indices)
    return " ".join(sentences[idx] for idx in selected)

