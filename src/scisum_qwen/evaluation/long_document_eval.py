from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scisum_qwen.evaluation.run_eval import evaluate_prediction_records
from scisum_qwen.inference.extractive import textrank_summarize
from scisum_qwen.inference.hierarchical import HierarchicalSummarizer
from scisum_qwen.utils.io import read_jsonl, write_csv, write_jsonl


def build_output_record(
    source_record: dict[str, Any],
    *,
    generated_summary: str,
    inference_mode: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "paper_id": source_record["paper_id"],
        "title": source_record.get("title"),
        "method": "textrank",
        "prompt_style": "faithful_abstract",
        "generated_summary": generated_summary,
        "reference_summary": source_record["target_summary"],
        "source_text": source_record["source_text"],
        "metadata": {"run_name": inference_mode, "inference_mode": inference_mode, **(metadata or {})},
    }


def run_single_pass(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs = []
    for record in records:
        target_word_budget = len(record["target_summary"].split())
        summary = textrank_summarize(record["source_text"], target_word_budget=target_word_budget)
        outputs.append(
            build_output_record(
                record,
                generated_summary=summary,
                inference_mode="single_pass",
                metadata={"chunk_count": 1, "section_count": 0},
            )
        )
    return outputs


def run_hierarchical(records: list[dict[str, Any]], *, summary_type: str = "abstract") -> list[dict[str, Any]]:
    summarizer = HierarchicalSummarizer()
    outputs = []
    for record in records:
        result = summarizer.summarize_record(record, summary_type=summary_type)
        outputs.append(
            build_output_record(
                record,
                generated_summary=result.summary,
                inference_mode="hierarchical",
                metadata={
                    "chunk_count": result.chunk_count,
                    "section_count": result.section_count,
                    "used_hierarchical": result.used_hierarchical,
                    "summary_type": summary_type,
                },
            )
        )
    return outputs


def build_long_document_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Long-Document Results",
        "",
        "| Inference Mode | Count | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg Pred Words | Avg Ref Words | Avg Compression | Format Validity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        mode = row.get("run_name") or row.get("inference_mode") or row.get("method")
        lines.append(
            f"| {mode} | {row['count']} | {row['rouge1']} | {row['rouge2']} | {row['rougeL']} | {row['avg_prediction_words']} | {row['avg_reference_words']} | {row['avg_compression_ratio']} | {row['format_validity']} |"
        )
    lines.extend(["", "## Notes", ""])
    if len(rows) >= 2:
        by_mode = {row.get("run_name", row.get("inference_mode", "")): row for row in rows}
        single_pass = by_mode.get("single_pass")
        hierarchical = by_mode.get("hierarchical")
        if single_pass and hierarchical:
            rouge_delta = round(hierarchical["rougeL"] - single_pass["rougeL"], 4)
            lines.append(f"- ROUGE-L delta (`hierarchical - single_pass`): `{rouge_delta}`.")
            lines.append("- Use these outputs as a lightweight proxy before running the heavier LLM-backed hierarchical path.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight single-pass vs hierarchical comparison.")
    parser.add_argument("--input", type=str, default="data/samples/sample_processed.jsonl")
    parser.add_argument("--output-dir", type=str, default="reports/longdoc")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summary-type", type=str, default="abstract", choices=["abstract", "structured"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    if args.limit > 0:
        records = records[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    single_pass_outputs = run_single_pass(records)
    hierarchical_outputs = run_hierarchical(records, summary_type=args.summary_type)

    write_jsonl(output_dir / "single_pass_outputs.jsonl", single_pass_outputs)
    write_jsonl(output_dir / "hierarchical_outputs.jsonl", hierarchical_outputs)

    rows = []
    for run_name, outputs in (("single_pass", single_pass_outputs), ("hierarchical", hierarchical_outputs)):
        row = evaluate_prediction_records(outputs)
        row["run_name"] = run_name
        rows.append(row)

    write_csv(
        output_dir / "long_document_comparison.csv",
        rows,
        [
            "run_name",
            "method",
            "prompt_style",
            "count",
            "rouge1",
            "rouge2",
            "rougeL",
            "bertscore_f1",
            "bertscore_status",
            "format_validity",
            "field_completion_rate",
            "avg_prediction_words",
            "avg_reference_words",
            "avg_compression_ratio",
        ],
    )
    (output_dir / "long_document_results.md").write_text(build_long_document_report(rows), encoding="utf-8")


if __name__ == "__main__":
    main()

