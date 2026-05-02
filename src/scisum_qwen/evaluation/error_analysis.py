from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any

from scisum_qwen.evaluation.compare_models import infer_run_tags
from scisum_qwen.evaluation.length_analysis import count_words
from scisum_qwen.utils.io import read_jsonl

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
_SECTION_HEADER_PATTERN = re.compile(r"^\s*(abstract|introduction|method|results|discussion|conclusion)\s*:", re.IGNORECASE)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it", "of", "on",
    "or", "our", "that", "the", "their", "this", "to", "we", "with", "were", "was", "using", "use", "used",
}
RESULT_KEYWORDS = {"result", "results", "improve", "improved", "rouge", "f1", "accuracy", "score", "scores", "performance"}
CONTRIBUTION_KEYWORDS = {"propose", "proposed", "introduce", "introduced", "present", "presented", "novel", "contribution"}
GENERIC_METHOD_WORDS = {"method", "approach", "model", "system"}


def content_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_PATTERN.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    ]


def lexical_coverage(prediction: str, source: str) -> float:
    pred_tokens = content_tokens(prediction)
    source_set = set(content_tokens(source))
    if not pred_tokens:
        return 1.0
    covered = sum(1 for token in pred_tokens if token in source_set)
    return covered / len(pred_tokens)


def extract_numbers(text: str) -> set[str]:
    return set(_NUMBER_PATTERN.findall(text))


def detect_error_flags(record: dict[str, Any]) -> list[str]:
    prediction = record["generated_summary"]
    source = record.get("source_text", "")
    reference = record.get("reference_summary", "")
    prompt_style = record.get("prompt_style") or ""

    flags: list[str] = []
    pred_tokens = set(content_tokens(prediction))
    source_tokens = set(content_tokens(source))
    reference_tokens = set(content_tokens(reference))
    pred_numbers = extract_numbers(prediction)
    source_numbers = extract_numbers(source)
    length_ratio = count_words(prediction) / max(count_words(reference), 1)
    coverage = lexical_coverage(prediction, source)

    if pred_numbers and not pred_numbers.issubset(source_numbers):
        flags.append("incorrect_numerical_result")
    if length_ratio < 0.65:
        flags.append("too_short")
    if length_ratio > 1.45:
        flags.append("too_verbose")
    if coverage < 0.65 and count_words(prediction) >= 10:
        flags.append("unsupported_claim")
    if not pred_tokens.intersection(CONTRIBUTION_KEYWORDS) and reference_tokens.intersection(CONTRIBUTION_KEYWORDS):
        flags.append("missing_key_contribution")
    if source_tokens.intersection(RESULT_KEYWORDS) and not pred_tokens.intersection(RESULT_KEYWORDS):
        flags.append("weak_result_extraction")
    if prompt_style != "structured" and _SECTION_HEADER_PATTERN.match(prediction):
        flags.append("section_confusion")
    technical_tokens = {token for token in source_tokens.union(reference_tokens) if len(token) >= 8 and token not in GENERIC_METHOD_WORDS}
    if pred_tokens.intersection(GENERIC_METHOD_WORDS) and technical_tokens and not pred_tokens.intersection(technical_tokens):
        flags.append("over_generalized_method")

    return flags or ["no_major_issue_detected"]


def primary_error(flags: list[str]) -> str:
    priority = [
        "incorrect_numerical_result",
        "unsupported_claim",
        "missing_key_contribution",
        "weak_result_extraction",
        "section_confusion",
        "over_generalized_method",
        "too_short",
        "too_verbose",
        "no_major_issue_detected",
    ]
    for candidate in priority:
        if candidate in flags:
            return candidate
    return flags[0]


def analyze_prediction_file(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = read_jsonl(path)
    tags = infer_run_tags(path, records)
    analyzed_rows: list[dict[str, Any]] = []
    for record in records:
        flags = detect_error_flags(record)
        analyzed_rows.append(
            {
                "paper_id": record["paper_id"],
                "title": record.get("title"),
                "run_name": tags["run_name"],
                "method": record.get("method", "unknown"),
                "prompt_style": record.get("prompt_style") or "",
                "primary_error": primary_error(flags),
                "error_flags": flags,
                "generated_summary": record["generated_summary"],
                "reference_summary": record.get("reference_summary", ""),
            }
        )
    counts = Counter(row["primary_error"] for row in analyzed_rows)
    return analyzed_rows, {"run_name": tags["run_name"], "counts": counts, "total": len(analyzed_rows)}


def build_error_report(rows_by_run: dict[str, list[dict[str, Any]]], summaries: list[dict[str, Any]]) -> str:
    lines = ["# Error Analysis", ""]
    if not summaries:
        lines.append("No prediction files were found.")
        return "\n".join(lines) + "\n"

    for summary in summaries:
        lines.extend(
            [
                f"## {summary['run_name']}",
                "",
                "| Error Category | Count |",
                "| --- | ---: |",
            ]
        )
        for category, count in summary["counts"].most_common():
            lines.append(f"| {category} | {count} |")
        lines.append("")

        grouped_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows_by_run[summary["run_name"]]:
            grouped_examples[row["primary_error"]].append(row)

        for category, examples in grouped_examples.items():
            if category == "no_major_issue_detected":
                continue
            example = examples[0]
            lines.extend(
                [
                    f"### {category}",
                    "",
                    f"- Paper: `{example['paper_id']}`",
                    f"- Title: {example.get('title') or 'Untitled'}",
                    f"- Generated: {example['generated_summary']}",
                    f"- Reference: {example['reference_summary']}",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heuristic error analysis over summarization outputs.")
    parser.add_argument("--input-dir", type=str, default="reports")
    parser.add_argument("--pattern", type=str, default="*_outputs.jsonl")
    parser.add_argument("--output-md", type=str, default="reports/error_analysis.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_by_run: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []

    for file_path in sorted(Path(args.input_dir).glob(args.pattern)):
        analyzed_rows, summary = analyze_prediction_file(file_path)
        rows_by_run[summary["run_name"]] = analyzed_rows
        summaries.append(summary)

    Path(args.output_md).write_text(build_error_report(rows_by_run, summaries), encoding="utf-8")


if __name__ == "__main__":
    main()

