from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scisum_qwen.inference.decoding import DecodingConfig
from scisum_qwen.inference.extractive import textrank_summarize
from scisum_qwen.inference.generator import HuggingFaceGenerator
from scisum_qwen.inference.prompts import build_prompt_bundle
from scisum_qwen.utils.io import read_jsonl, write_jsonl


def build_output_record(
    source_record: dict[str, Any],
    *,
    generated_summary: str,
    method: str,
    prompt_style: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "paper_id": source_record["paper_id"],
        "title": source_record.get("title"),
        "method": method,
        "prompt_style": prompt_style,
        "generated_summary": generated_summary,
        "reference_summary": source_record["target_summary"],
        "source_text": source_record["source_text"],
        "metadata": metadata or {},
    }


def run_textrank_baseline(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs = []
    for record in records:
        target_word_budget = len(record["target_summary"].split())
        summary = textrank_summarize(record["source_text"], target_word_budget=target_word_budget)
        outputs.append(build_output_record(record, generated_summary=summary, method="textrank"))
    return outputs


def run_llm_baseline(
    records: list[dict[str, Any]],
    *,
    model_name: str,
    prompt_style: str,
    decoding: DecodingConfig,
    limit: int | None = None,
    max_source_words: int | None = None,
) -> list[dict[str, Any]]:
    generator = HuggingFaceGenerator(model_name=model_name)
    outputs = []
    active_records = records[:limit] if limit else records
    for record in active_records:
        source_text = record["source_text"]
        if max_source_words and max_source_words > 0:
            source_text = " ".join(source_text.split()[:max_source_words])
        bundle = build_prompt_bundle(
            title=record.get("title"),
            paper_text=source_text,
            style=prompt_style,  # type: ignore[arg-type]
        )
        result = generator.generate_from_bundle(bundle, decoding)
        outputs.append(
            build_output_record(
                record,
                generated_summary=result.text,
                method="qwen",
                prompt_style=prompt_style,
                metadata={
                    **result.metadata,
                    "max_source_words": max_source_words,
                },
            )
        )
    return outputs


def write_baseline_markdown(path: str | Path, records: list[dict[str, Any]]) -> None:
    lines = ["# Baseline Outputs", ""]
    for record in records:
        lines.extend(
            [
                f"## {record['paper_id']} - {record['method']}",
                "",
                f"**Title:** {record.get('title') or 'Untitled'}",
                "",
                f"**Prompt Style:** {record.get('prompt_style') or 'n/a'}",
                "",
                "**Generated Summary**",
                "",
                record["generated_summary"],
                "",
                "**Reference Summary**",
                "",
                record["reference_summary"],
                "",
            ]
        )
    Path(path).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run summarization baselines on processed scientific papers.")
    parser.add_argument("--input", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--include-structured", action="store_true")
    parser.add_argument("--max-source-words", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    textrank_outputs = run_textrank_baseline(records[: args.limit or None])
    write_jsonl(output_dir / "baseline_textrank_outputs.jsonl", textrank_outputs)
    qualitative_records = textrank_outputs[:5]

    if args.skip_llm:
        write_baseline_markdown(output_dir / "baseline_outputs.md", qualitative_records)
        return

    decoding = DecodingConfig(max_new_tokens=args.max_new_tokens)
    zero_shot_outputs = run_llm_baseline(
        records,
        model_name=args.model_name,
        prompt_style="zero_shot",
        decoding=decoding,
        limit=args.limit or None,
        max_source_words=args.max_source_words or None,
    )
    prompted_outputs = run_llm_baseline(
        records,
        model_name=args.model_name,
        prompt_style="faithful_abstract",
        decoding=decoding,
        limit=args.limit or None,
        max_source_words=args.max_source_words or None,
    )
    qualitative_records.extend(zero_shot_outputs[:3])
    qualitative_records.extend(prompted_outputs[:3])
    write_jsonl(output_dir / "baseline_zero_shot_outputs.jsonl", zero_shot_outputs)
    write_jsonl(output_dir / "baseline_prompted_outputs.jsonl", prompted_outputs)
    if args.include_structured:
        structured_outputs = run_llm_baseline(
            records,
            model_name=args.model_name,
            prompt_style="structured",
            decoding=decoding,
            limit=args.limit or None,
            max_source_words=args.max_source_words or None,
        )
        qualitative_records.extend(structured_outputs[:3])
        write_jsonl(output_dir / "baseline_structured_outputs.jsonl", structured_outputs)
    write_baseline_markdown(output_dir / "baseline_outputs.md", qualitative_records)


if __name__ == "__main__":
    main()
