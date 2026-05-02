from __future__ import annotations

from dataclasses import dataclass

from scisum_qwen.inference.extractive import split_sentences


STRUCTURED_PREFIXES = (
    "tl;dr:",
    "problem:",
    "method:",
    "results:",
    "limitations:",
    "introduction:",
    "conclusion:",
    "discussion:",
    "abstract:",
)


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    source_sentence_index: int


def normalize_claim_text(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in STRUCTURED_PREFIXES:
        if lowered.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return stripped


def extract_claims(summary_text: str, *, claim_prefix: str = "claim") -> list[Claim]:
    claims: list[Claim] = []
    for index, sentence in enumerate(split_sentences(summary_text), start=1):
        normalized = normalize_claim_text(sentence)
        if not normalized:
            continue
        claims.append(
            Claim(
                claim_id=f"{claim_prefix}_{index:02d}",
                text=normalized,
                source_sentence_index=index - 1,
            )
        )
    return claims
