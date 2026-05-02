from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scisum_qwen.evaluation.run_eval import evaluate_prediction_records
from scisum_qwen.utils.io import read_jsonl, write_csv


def infer_run_tags(file_path: str | Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    stem = Path(file_path).stem.replace("_outputs", "")
    first = records[0] if records else {}
    metadata = first.get("metadata", {}) or {}
    run_name = metadata.get("run_name", stem)
    context_window = metadata.get("context_window", "")
    if not context_window:
        if "8k" in stem:
            context_window = "8k"
        elif "4k" in stem:
            context_window = "4k"

    inference_mode = metadata.get("inference_mode", "")
    if not inference_mode:
        if "hierarchical" in stem:
            inference_mode = "hierarchical"
        elif "single_pass" in stem or "singlepass" in stem:
            inference_mode = "single_pass"

    evidence_prompt = metadata.get("evidence_aware_prompt", "")
    if evidence_prompt == "":
        if "evidence" in stem:
            evidence_prompt = "on"
        elif "no_evidence" in stem:
            evidence_prompt = "off"

    return {
        "run_name": run_name,
        "context_window": context_window,
        "inference_mode": inference_mode,
        "evidence_aware_prompt": evidence_prompt,
    }


def load_comparison_rows(input_dir: str | Path, pattern: str = "*_outputs.jsonl") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction_file in sorted(Path(input_dir).glob(pattern)):
        records = read_jsonl(prediction_file)
        if not records:
            continue
        row = evaluate_prediction_records(records)
        row.update(infer_run_tags(prediction_file, records))
        rows.append(row)
    return rows


def build_experiment_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Experiment Report",
        "",
        "## Model Comparison",
        "",
        "| Run | Method | Prompt Style | Inference Mode | Context | Evidence Prompt | Count | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | Format Validity | Field Completion |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {method} | {prompt_style} | {inference_mode} | {context_window} | {evidence_aware_prompt} | {count} | {rouge1} | {rouge2} | {rougeL} | {bertscore_f1} | {format_validity} | {field_completion_rate} |".format(
                **row
            )
        )

    lines.extend(["", "## Highlights", ""])
    if not rows:
        lines.append("- No prediction runs were found.")
        return "\n".join(lines) + "\n"

    best_rouge = max(rows, key=lambda row: row["rougeL"])
    best_format = max(rows, key=lambda row: row["field_completion_rate"])
    lines.append(
        f"- Best ROUGE-L so far: `{best_rouge['run_name']}` with `{best_rouge['rougeL']}`."
    )
    lines.append(
        f"- Best structured format completion so far: `{best_format['run_name']}` with `{best_format['field_completion_rate']}`."
    )
    if len(rows) == 1:
        lines.append("- Only one run is available so far; add zero-shot, prompted, and QLoRA outputs for a fuller comparison.")
    else:
        sorted_rows = sorted(rows, key=lambda row: row["rougeL"], reverse=True)
        ordering = ", ".join(f"{row['run_name']} ({row['rougeL']})" for row in sorted_rows[:5])
        lines.append(f"- Current ordering by ROUGE-L: {ordering}.")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare summarization runs and build experiment reports.")
    parser.add_argument("--input-dir", type=str, default="reports")
    parser.add_argument("--pattern", type=str, default="*_outputs.jsonl")
    parser.add_argument("--output-csv", type=str, default="reports/model_comparison.csv")
    parser.add_argument("--output-md", type=str, default="reports/experiment_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_comparison_rows(args.input_dir, pattern=args.pattern)
    write_csv(
        args.output_csv,
        rows,
        [
            "run_name",
            "method",
            "prompt_style",
            "inference_mode",
            "context_window",
            "evidence_aware_prompt",
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
    Path(args.output_md).write_text(build_experiment_report(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
