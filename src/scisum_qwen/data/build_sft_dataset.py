from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from scisum_qwen.data.load_arxiv import iter_split_records, limit_split, load_arxiv_split, load_data_config
from scisum_qwen.data.preprocess import PreprocessConfig, clean_abstract_text, clean_paper_text, is_usable_sample
from scisum_qwen.data.section_parser import build_section_aware_text


def build_user_prompt(instruction: str, paper_text: str) -> str:
    return f"{instruction}\n\nPaper:\n{paper_text}".strip()


def build_messages(system_prompt: str, user_prompt: str, target_summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": target_summary},
    ]


def generate_paper_id(title: str | None, article: str, abstract: str) -> str:
    digest_source = "\n".join(filter(None, [title, article[:1000], abstract[:500]]))
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()
    return digest[:16]


def estimate_token_count(text: str, *, fallback_chars_per_token: float = 4.0) -> int:
    if not text:
        return 0
    return max(1, int(round(len(text) / fallback_chars_per_token)))


def preprocess_record(
    raw_record: dict[str, Any],
    *,
    preprocess_config: PreprocessConfig,
    formatting_config: Any,
    fallback_chars_per_token: float,
) -> dict[str, Any] | None:
    cleaned_article = clean_paper_text(raw_record["article"], config=preprocess_config)
    cleaned_abstract = clean_abstract_text(raw_record["abstract"])

    if not is_usable_sample(
        cleaned_article,
        cleaned_abstract,
        min_article_chars=preprocess_config.min_article_chars,
        min_abstract_chars=preprocess_config.min_abstract_chars,
        max_article_chars=preprocess_config.max_article_chars,
    ):
        return None

    title = raw_record.get("title")
    paper_id = raw_record.get("paper_id") or generate_paper_id(title, cleaned_article, cleaned_abstract)
    section_aware_text = build_section_aware_text(
        title=title,
        paper_text=cleaned_article,
        include_title=formatting_config.include_title,
        include_section_headers=formatting_config.include_section_headers,
        keep_sections=formatting_config.keep_sections,
    )
    user_prompt = build_user_prompt(formatting_config.user_instruction, section_aware_text)
    messages = build_messages(formatting_config.system_prompt, user_prompt, cleaned_abstract)

    return {
        "paper_id": paper_id,
        "title": title,
        "source_article": cleaned_article,
        "source_text": section_aware_text,
        "target_summary": cleaned_abstract,
        "article_char_length": len(cleaned_article),
        "summary_char_length": len(cleaned_abstract),
        "article_token_estimate": estimate_token_count(
            section_aware_text,
            fallback_chars_per_token=fallback_chars_per_token,
        ),
        "summary_token_estimate": estimate_token_count(
            cleaned_abstract,
            fallback_chars_per_token=fallback_chars_per_token,
        ),
        "messages": messages,
    }


def summarize_split(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        return {
            "count": 0,
            "avg_article_tokens": 0,
            "avg_summary_tokens": 0,
            "avg_compression_ratio": 0,
        }
    article_tokens = [record["article_token_estimate"] for record in records]
    summary_tokens = [record["summary_token_estimate"] for record in records]
    compression_ratios = [
        summary / article for article, summary in zip(article_tokens, summary_tokens, strict=True) if article
    ]
    return {
        "count": len(records),
        "avg_article_tokens": round(mean(article_tokens), 2),
        "avg_summary_tokens": round(mean(summary_tokens), 2),
        "avg_compression_ratio": round(mean(compression_ratios), 4) if compression_ratios else 0,
    }


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_dataset_report(
    path: str | Path,
    *,
    split_stats: dict[str, dict[str, float | int]],
    filtered_counts: dict[str, int],
) -> None:
    lines = [
        "# Dataset Report",
        "",
        "This report summarizes the processed arXiv scientific summarization dataset.",
        "",
        "| Split | Count | Avg Article Tokens | Avg Summary Tokens | Avg Compression Ratio | Filtered Samples |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in ("train", "validation", "test"):
        stats = split_stats.get(split_name, {})
        lines.append(
            "| {split} | {count} | {article} | {summary} | {ratio} | {filtered} |".format(
                split=split_name,
                count=stats.get("count", 0),
                article=stats.get("avg_article_tokens", 0),
                summary=stats.get("avg_summary_tokens", 0),
                ratio=stats.get("avg_compression_ratio", 0),
                filtered=filtered_counts.get(split_name, 0),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Preprocessing preserves scientific numbers and metric-heavy sentences whenever possible.",
            "- Reference sections are removed conservatively using heading detection.",
            "- Token counts are approximate until a model tokenizer is wired into the preprocessing stage.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_processed_splits(config_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    data_config = load_data_config(config_path)
    preprocess_config = PreprocessConfig(**data_config.preprocessing)

    processed_splits: dict[str, list[dict[str, Any]]] = {}
    filtered_counts: dict[str, int] = defaultdict(int)

    for split_name in ("train", "validation", "test"):
        split_data = load_arxiv_split(data_config.dataset, split_name)
        split_limit = data_config.dataset.subset_limits.get(split_name)
        split_data = limit_split(split_data, split_limit)

        processed_records: list[dict[str, Any]] = []
        for index, raw_record in enumerate(iter_split_records(split_data)):
            standardized = {
                "paper_id": raw_record.get(data_config.dataset.paper_id_field)
                if data_config.dataset.paper_id_field
                else f"{split_name}_{index:07d}",
                "title": raw_record.get(data_config.dataset.title_field) if data_config.dataset.title_field else None,
                "article": raw_record[data_config.dataset.article_field],
                "abstract": raw_record[data_config.dataset.abstract_field],
            }
            record = preprocess_record(
                standardized,
                preprocess_config=preprocess_config,
                formatting_config=data_config.formatting,
                fallback_chars_per_token=data_config.tokenization.fallback_chars_per_token,
            )
            if record is None:
                filtered_counts[split_name] += 1
                continue
            processed_records.append(record)
        processed_splits[split_name] = processed_records

    output_dir = Path(data_config.outputs.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in processed_splits.items():
        filename = data_config.outputs.split_filenames[split_name]
        write_jsonl(output_dir / filename, records)

    split_stats = {split_name: summarize_split(records) for split_name, records in processed_splits.items()}
    write_dataset_report(
        data_config.outputs.report_path,
        split_stats=split_stats,
        filtered_counts=dict(filtered_counts),
    )
    return processed_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned SFT-format data from the arXiv summarization dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to the data YAML configuration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_processed_splits(args.config)


if __name__ == "__main__":
    main()
