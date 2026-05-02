from __future__ import annotations

from dataclasses import dataclass
import re

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "abstract": ("abstract",),
    "introduction": ("introduction", "1 introduction"),
    "related_work": ("related work", "background", "literature review"),
    "method": ("method", "methods", "methodology", "approach", "model"),
    "experiments": ("experiments", "experimental setup", "evaluation setup"),
    "results": ("results", "findings", "evaluation results"),
    "discussion": ("discussion", "analysis"),
    "conclusion": ("conclusion", "conclusions", "concluding remarks"),
    "appendix": ("appendix", "appendices"),
    "references": ("references", "bibliography"),
}

DEFAULT_SECTION_ORDER = (
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiments",
    "results",
    "discussion",
    "conclusion",
)

_HEADING_CLEAN_PATTERN = re.compile(r"[^a-z0-9\s]+")


@dataclass(frozen=True)
class SectionSpan:
    name: str
    start: int
    end: int


def canonicalize_heading(line: str) -> str | None:
    normalized = _HEADING_CLEAN_PATTERN.sub(" ", line.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return None
    if len(normalized.split()) > 5:
        return None
    for canonical, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return canonical
    return None


def parse_sections(text: str) -> dict[str, str]:
    lines = text.splitlines(keepends=True)
    headings: list[tuple[str, int, int]] = []
    offset = 0

    for line in lines:
        canonical = canonicalize_heading(line.strip())
        if canonical:
            headings.append((canonical, offset, offset + len(line)))
        offset += len(line)

    if not headings:
        return {}

    sections: dict[str, str] = {}
    for index, (name, _start, body_start) in enumerate(headings):
        body_end = headings[index + 1][1] if index + 1 < len(headings) else len(text)
        body = text[body_start:body_end].strip()
        if not body:
            continue
        if name in sections:
            sections[name] = f"{sections[name]}\n\n{body}".strip()
        else:
            sections[name] = body
    return sections


def select_sections(
    sections: dict[str, str],
    keep_sections: tuple[str, ...] | list[str] = DEFAULT_SECTION_ORDER,
) -> dict[str, str]:
    ordered: dict[str, str] = {}
    for section_name in keep_sections:
        if section_name in sections and sections[section_name].strip():
            ordered[section_name] = sections[section_name].strip()
    return ordered


def build_section_aware_text(
    *,
    title: str | None,
    paper_text: str,
    include_title: bool = True,
    include_section_headers: bool = True,
    keep_sections: tuple[str, ...] | list[str] = DEFAULT_SECTION_ORDER,
) -> str:
    sections = select_sections(parse_sections(paper_text), keep_sections=keep_sections)
    if not sections:
        if title and include_title:
            return f"Title: {title}\n\nPaper:\n{paper_text.strip()}".strip()
        return paper_text.strip()

    blocks: list[str] = []
    if title and include_title:
        blocks.append(f"Title: {title}")

    for section_name, content in sections.items():
        if include_section_headers:
            label = section_name.replace("_", " ").title()
            blocks.append(f"{label}:\n{content}")
        else:
            blocks.append(content)
    return "\n\n".join(blocks).strip()

