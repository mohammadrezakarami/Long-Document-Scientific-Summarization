from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from scisum_qwen.evidence.chunker import DEFAULT_SECTION_ORDER, PaperChunk, chunk_section_aware_text
from scisum_qwen.inference.extractive import split_sentences, textrank_summarize
from scisum_qwen.inference.token_budget import (
    TokenBudgetConfig,
    allocate_section_word_budgets,
    estimate_text_tokens,
    infer_target_word_budget,
    should_use_hierarchical,
)


@dataclass(frozen=True)
class SectionSummary:
    section: str
    summary: str
    chunk_count: int
    source_token_count: int


@dataclass(frozen=True)
class HierarchicalSummaryResult:
    summary: str
    section_summaries: tuple[SectionSummary, ...]
    used_hierarchical: bool
    chunk_count: int
    section_count: int
    title: str | None


def build_section_summary_prompt(section_name: str, section_text: str) -> str:
    return (
        "Summarize the following section of a scientific paper.\n"
        "Focus only on information present in this section.\n"
        f"Section name:\n{section_name}\n\n"
        f"Section text:\n{section_text}"
    )


def build_final_synthesis_prompt(section_summaries: list[SectionSummary], *, title: str | None = None) -> str:
    heading = f"Title: {title}\n\n" if title else ""
    sections_text = "\n\n".join(
        f"{section.section.title()} Summary:\n{section.summary}" for section in section_summaries
    )
    return (
        "You are given section-level summaries of a scientific paper.\n"
        "Generate a faithful structured summary with these fields:\n"
        "TL;DR\nProblem\nMethod\nKey Contributions\nResults\nLimitations\n\n"
        f"{heading}Section summaries:\n{sections_text}"
    )


def _extract_contribution_candidates(text: str, *, limit: int = 3) -> list[str]:
    candidates = []
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in ("propose", "introduce", "present", "novel", "improve")):
            candidates.append(sentence.strip())
        if len(candidates) >= limit:
            break
    return candidates


def _find_limitations_text(section_texts: dict[str, str]) -> str:
    for section_name in ("discussion", "conclusion"):
        section_text = section_texts.get(section_name, "")
        for sentence in split_sentences(section_text):
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in ("limitation", "future work", "challenge", "constraint")):
                return sentence.strip()
    return "Not specified"


def build_structured_summary(section_texts: dict[str, str], section_summaries: list[SectionSummary]) -> str:
    summary_lookup = {item.section: item.summary for item in section_summaries}
    combined_summaries = "\n".join(item.summary for item in section_summaries if item.summary)
    tldr = textrank_summarize(combined_summaries, target_word_budget=28, max_sentences=2) or "Not specified"
    problem = summary_lookup.get("introduction") or summary_lookup.get("abstract") or "Not specified"
    method = summary_lookup.get("method") or summary_lookup.get("experiments") or "Not specified"
    results = summary_lookup.get("results") or summary_lookup.get("discussion") or summary_lookup.get("conclusion") or "Not specified"
    contribution_source = "\n".join(
        filter(
            None,
            [
                summary_lookup.get("introduction", ""),
                summary_lookup.get("method", ""),
                summary_lookup.get("results", ""),
            ],
        )
    )
    contributions = _extract_contribution_candidates(contribution_source)
    if not contributions:
        contributions = ["Not specified"]
    limitations = _find_limitations_text(section_texts)

    lines = [
        f"TL;DR: {tldr}",
        f"Problem: {problem}",
        f"Method: {method}",
        "Key Contributions:",
    ]
    for index, contribution in enumerate(contributions, start=1):
        lines.append(f"{index}. {contribution}")
    lines.extend(
        [
            f"Results: {results}",
            f"Limitations: {limitations}",
        ]
    )
    return "\n".join(lines)


class HierarchicalSummarizer:
    def __init__(
        self,
        *,
        budget: TokenBudgetConfig | None = None,
        section_summary_fn: Callable[[str, int], str] | None = None,
        final_summary_fn: Callable[[str, int], str] | None = None,
    ) -> None:
        self.budget = budget or TokenBudgetConfig()
        self.section_summary_fn = section_summary_fn or (
            lambda text, word_budget: textrank_summarize(text, target_word_budget=word_budget, max_sentences=4)
        )
        self.final_summary_fn = final_summary_fn or (
            lambda text, word_budget: textrank_summarize(text, target_word_budget=word_budget, max_sentences=4)
        )

    def summarize_record(
        self,
        record: dict[str, str],
        *,
        summary_type: str = "abstract",
        target_word_budget: int | None = None,
    ) -> HierarchicalSummaryResult:
        paper_id = record["paper_id"]
        source_text = record["source_text"]
        total_tokens = estimate_text_tokens(
            source_text,
            fallback_chars_per_token=self.budget.fallback_chars_per_token,
        )
        target_word_budget = target_word_budget or infer_target_word_budget(record.get("target_summary"))
        use_hierarchical = should_use_hierarchical(
            total_tokens,
            max_input_tokens=self.budget.max_input_tokens,
            safety_margin=self.budget.safety_margin,
        )
        title, chunks, section_texts = chunk_section_aware_text(
            paper_id=paper_id,
            source_text=source_text,
            max_tokens=self.budget.chunk_token_limit,
            fallback_chars_per_token=self.budget.fallback_chars_per_token,
            allowed_sections=DEFAULT_SECTION_ORDER,
        )

        if not chunks:
            fallback_summary = self.final_summary_fn(source_text, target_word_budget)
            return HierarchicalSummaryResult(
                summary=fallback_summary,
                section_summaries=tuple(),
                used_hierarchical=False,
                chunk_count=0,
                section_count=0,
                title=title,
            )

        chunks_by_section: dict[str, list[PaperChunk]] = defaultdict(list)
        for chunk in chunks:
            chunks_by_section[chunk.section].append(chunk)

        section_budgets = allocate_section_word_budgets(
            {section: sum(chunk.token_count for chunk in section_chunks) for section, section_chunks in chunks_by_section.items()},
            total_word_budget=target_word_budget,
            minimum_section_words=28,
        )

        section_summaries: list[SectionSummary] = []
        for section_name in DEFAULT_SECTION_ORDER:
            section_chunks = chunks_by_section.get(section_name, [])
            if not section_chunks:
                continue
            chunk_summaries = []
            for chunk in section_chunks:
                chunk_word_budget = max(22, round(section_budgets.get(section_name, 60) / max(len(section_chunks), 1)))
                chunk_summaries.append(self.section_summary_fn(chunk.text, chunk_word_budget))
            if len(chunk_summaries) == 1:
                section_summary_text = chunk_summaries[0]
            else:
                section_summary_text = self.final_summary_fn(
                    "\n\n".join(chunk_summaries),
                    section_budgets.get(section_name, 60),
                )
            section_summaries.append(
                SectionSummary(
                    section=section_name,
                    summary=section_summary_text,
                    chunk_count=len(section_chunks),
                    source_token_count=sum(chunk.token_count for chunk in section_chunks),
                )
            )

        if summary_type == "structured":
            final_summary = build_structured_summary(section_texts, section_summaries)
        else:
            combined_section_text = "\n\n".join(
                f"{section.section.title()}: {section.summary}" for section in section_summaries
            )
            final_summary = self.final_summary_fn(combined_section_text, target_word_budget)

        return HierarchicalSummaryResult(
            summary=final_summary,
            section_summaries=tuple(section_summaries),
            used_hierarchical=use_hierarchical or len(chunks) > len(section_summaries),
            chunk_count=len(chunks),
            section_count=len(section_summaries),
            title=title,
        )
