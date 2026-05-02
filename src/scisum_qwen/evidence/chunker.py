from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from scisum_qwen.inference.token_budget import estimate_text_tokens

DEFAULT_SECTION_ORDER = (
    "abstract",
    "introduction",
    "related work",
    "method",
    "experiments",
    "results",
    "discussion",
    "conclusion",
)


@dataclass(frozen=True)
class PaperChunk:
    paper_id: str
    chunk_id: str
    section: str
    text: str
    token_count: int


def parse_section_aware_text(source_text: str) -> tuple[str | None, dict[str, str]]:
    title: str | None = None
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for raw_line in source_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if current_section:
                sections.setdefault(current_section, []).append("")
            continue
        if line.startswith("Title: "):
            title = line.replace("Title: ", "", 1).strip()
            current_section = None
            continue
        if line.endswith(":") and len(line) < 60:
            current_section = line[:-1].strip().lower()
            sections.setdefault(current_section, [])
            continue
        if current_section:
            sections.setdefault(current_section, []).append(line)

    normalized = {
        section_name: "\n".join(lines).strip()
        for section_name, lines in sections.items()
        if "\n".join(lines).strip()
    }
    return title, normalized


def split_into_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]


def chunk_section_text(
    *,
    paper_id: str,
    section_name: str,
    section_text: str,
    max_tokens: int,
    fallback_chars_per_token: float = 4.0,
) -> list[PaperChunk]:
    paragraphs = split_into_paragraphs(section_text)
    if not paragraphs:
        return []

    chunks: list[PaperChunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    chunk_index = 1

    for paragraph in paragraphs:
        paragraph_tokens = estimate_text_tokens(paragraph, fallback_chars_per_token=fallback_chars_per_token)
        if current_parts and current_tokens + paragraph_tokens > max_tokens:
            chunk_text = "\n\n".join(current_parts).strip()
            chunks.append(
                PaperChunk(
                    paper_id=paper_id,
                    chunk_id=f"{paper_id}_{section_name.replace(' ', '_')}_{chunk_index:02d}",
                    section=section_name,
                    text=chunk_text,
                    token_count=current_tokens,
                )
            )
            chunk_index += 1
            current_parts = []
            current_tokens = 0

        current_parts.append(paragraph)
        current_tokens += paragraph_tokens

    if current_parts:
        chunk_text = "\n\n".join(current_parts).strip()
        chunks.append(
            PaperChunk(
                paper_id=paper_id,
                chunk_id=f"{paper_id}_{section_name.replace(' ', '_')}_{chunk_index:02d}",
                section=section_name,
                text=chunk_text,
                token_count=current_tokens,
            )
        )
    return chunks


def chunk_section_aware_text(
    *,
    paper_id: str,
    source_text: str,
    max_tokens: int,
    fallback_chars_per_token: float = 4.0,
    allowed_sections: Iterable[str] = DEFAULT_SECTION_ORDER,
) -> tuple[str | None, list[PaperChunk], dict[str, str]]:
    title, sections = parse_section_aware_text(source_text)
    allowed_lookup = {section.lower() for section in allowed_sections}
    chunks: list[PaperChunk] = []
    filtered_sections: dict[str, str] = {}
    for section_name in allowed_sections:
        normalized = section_name.lower()
        if normalized not in allowed_lookup:
            continue
        if normalized not in sections:
            continue
        filtered_sections[normalized] = sections[normalized]
        chunks.extend(
            chunk_section_text(
                paper_id=paper_id,
                section_name=normalized,
                section_text=sections[normalized],
                max_tokens=max_tokens,
                fallback_chars_per_token=fallback_chars_per_token,
            )
        )
    return title, chunks, filtered_sections

