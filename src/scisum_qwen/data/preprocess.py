from __future__ import annotations

from dataclasses import dataclass
import re

_CITATION_PATTERN = re.compile(r"\[(?:\d+(?:\s*,\s*\d+)*)\]")
_WHITESPACE_PATTERN = re.compile(r"[ \t]+")
_BLANK_LINE_PATTERN = re.compile(r"\n{3,}")
_REFERENCE_HEADING_PATTERN = re.compile(
    r"\n\s*(references|bibliography)\s*\n",
    flags=re.IGNORECASE,
)
_LATEX_FRAGMENT_PATTERN = re.compile(r"(\\cite\{.*?\}|\\ref\{.*?\}|\\label\{.*?\})")
_CAPTION_LINE_PATTERN = re.compile(r"^\s*(figure|fig\.|table)\s+\d+[:. -]", flags=re.IGNORECASE)


@dataclass(frozen=True)
class PreprocessConfig:
    remove_citation_markers: bool = True
    remove_reference_section: bool = True
    remove_figure_and_table_captions: bool = False
    min_article_chars: int = 2000
    min_abstract_chars: int = 120
    max_article_chars: int | None = 80000


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def collapse_whitespace(text: str) -> str:
    lines = []
    for line in normalize_newlines(text).split("\n"):
        lines.append(_WHITESPACE_PATTERN.sub(" ", line).strip())
    collapsed = "\n".join(lines)
    return _BLANK_LINE_PATTERN.sub("\n\n", collapsed).strip()


def strip_citation_markers(text: str) -> str:
    return _CITATION_PATTERN.sub("", text)


def strip_latex_fragments(text: str) -> str:
    return _LATEX_FRAGMENT_PATTERN.sub("", text)


def strip_reference_section(text: str) -> str:
    match = _REFERENCE_HEADING_PATTERN.search(text)
    if not match:
        return text
    return text[: match.start()].rstrip()


def strip_caption_lines(text: str) -> str:
    kept_lines = [line for line in normalize_newlines(text).split("\n") if not _CAPTION_LINE_PATTERN.match(line)]
    return "\n".join(kept_lines)


def clean_paper_text(text: str, config: PreprocessConfig | None = None) -> str:
    config = config or PreprocessConfig()
    cleaned = normalize_newlines(text)
    cleaned = strip_latex_fragments(cleaned)
    if config.remove_citation_markers:
        cleaned = strip_citation_markers(cleaned)
    if config.remove_reference_section:
        cleaned = strip_reference_section(cleaned)
    if config.remove_figure_and_table_captions:
        cleaned = strip_caption_lines(cleaned)
    return collapse_whitespace(cleaned)


def clean_abstract_text(text: str) -> str:
    return collapse_whitespace(strip_latex_fragments(text))


def is_usable_sample(
    article: str,
    abstract: str,
    *,
    min_article_chars: int,
    min_abstract_chars: int,
    max_article_chars: int | None = None,
) -> bool:
    if len(article) < min_article_chars:
        return False
    if len(abstract) < min_abstract_chars:
        return False
    if max_article_chars is not None and len(article) > max_article_chars:
        return False
    return True

