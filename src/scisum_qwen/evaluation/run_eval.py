from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scisum_qwen.evaluation.format_eval import aggregate_format_evaluations
from scisum_qwen.evaluation.length_analysis import compute_length_stats
from scisum_qwen.evaluation.rouge_bertscore import aggregate_rouge, try_compute_bertscore
from scisum_qwen.utils.io import read_jsonl, write_csv


def evaluate_prediction_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [record["generated_summary"] for record in records]
    references = [record["reference_summary"] for record in records]
    methods = [record.get("method", "unknown") for record in records]
    prompt_style = records[0].get("prompt_style") if records else None

    rouge_scores = aggregate_rouge(predictions, references)
    bertscore = try_compute_bertscore(predictions, references)
    format_scores = aggregate_format_evaluations(predictions)
    length_scores = compute_length_stats(predictions, references)

    return {
        "method": methods[0] if methods else "unknown",
        "prompt_style": prompt_style or "",
        **rouge_scores,
        **length_scores,
        **format_scores,
        "bertscore_f1": bertscore["bertscore_f1"],
        "bertscore_status": bertscore["status"],
        "count": len(records),
    }


def write_markdown_report(path: str | Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Baseline Evaluation",
        "",
        "| Method | Prompt Style | Count | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | Format Validity | Field Completion | Avg Pred Words | Avg Ref Words | Avg Compression |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {prompt_style} | {count} | {rouge1} | {rouge2} | {rougeL} | {bertscore_f1} | {format_validity} | {field_completion_rate} | {avg_prediction_words} | {avg_reference_words} | {avg_compression_ratio} |".format(
                **row
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manual_review_template(path: str | Path, records: list[dict[str, Any]], sample_size: int = 30) -> None:
    sampled = records[:sample_size]
    rows = []
    for record in sampled:
        rows.append(
            {
                "paper_id": record["paper_id"],
                "method": record.get("method", "unknown"),
                "prompt_style": record.get("prompt_style", ""),
                "faithfulness": "",
                "coverage": "",
                "technical_accuracy": "",
                "conciseness": "",
                "structure_quality": "",
                "notes": "",
            }
        )
    write_csv(
        path,
        rows,
        [
            "paper_id",
            "method",
            "prompt_style",
            "faithfulness",
            "coverage",
            "technical_accuracy",
            "conciseness",
            "structure_quality",
            "notes",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate summarization baseline outputs.")
    parser.add_argument("--input-dir", type=str, default="reports")
    parser.add_argument("--output-csv", type=str, default="reports/baseline_metrics.csv")
    parser.add_argument("--output-md", type=str, default="reports/baseline_eval.md")
    parser.add_argument("--manual-review-csv", type=str, default="reports/manual_review_template.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    prediction_files = sorted(input_dir.glob("baseline_*_outputs.jsonl"))
    rows: list[dict[str, Any]] = []
    manual_review_source: list[dict[str, Any]] = []

    for prediction_file in prediction_files:
        records = read_jsonl(prediction_file)
        if not records:
            continue
        rows.append(evaluate_prediction_records(records))
        if not manual_review_source:
            manual_review_source = records

    write_csv(
        args.output_csv,
        rows,
        [
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
    write_markdown_report(args.output_md, rows)
    write_manual_review_template(args.manual_review_csv, manual_review_source)


if __name__ == "__main__":
    main()

