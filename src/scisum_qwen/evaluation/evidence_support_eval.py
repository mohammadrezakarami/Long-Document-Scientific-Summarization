from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scisum_qwen.evidence.support_scorer import score_summary_support
from scisum_qwen.utils.io import read_jsonl, write_jsonl


def build_support_record(source_record: dict[str, Any], support_result) -> dict[str, Any]:
    claims = []
    for claim in support_result.claims:
        claims.append(
            {
                "claim_id": claim.claim_id,
                "claim": claim.claim,
                "label": claim.label,
                "support_score": claim.support_score,
                "top_evidence": {
                    "chunk_id": claim.top_evidence.chunk.chunk_id,
                    "section": claim.top_evidence.chunk.section,
                    "text": claim.top_evidence.chunk.text,
                    "score": round(claim.top_evidence.score, 4),
                },
                "evidence_candidates": [
                    {
                        "chunk_id": item.chunk.chunk_id,
                        "section": item.chunk.section,
                        "score": round(item.score, 4),
                    }
                    for item in claim.evidence_candidates
                ],
            }
        )
    return {
        "paper_id": source_record["paper_id"],
        "summary": source_record["generated_summary"],
        "overall_support_score": support_result.overall_support_score,
        "claims": claims,
    }


def build_markdown_report(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Evidence Examples",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['paper_id']}",
                "",
                f"- Overall support score: `{record['overall_support_score']}`",
                "",
            ]
        )
        for claim in record["claims"]:
            lines.extend(
                [
                    f"### {claim['claim_id']} - {claim['label']}",
                    "",
                    f"- Claim: {claim['claim']}",
                    f"- Support score: `{claim['support_score']}`",
                    f"- Top evidence section: `{claim['top_evidence']['section']}`",
                    f"- Top evidence text: {claim['top_evidence']['text']}",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run claim-level evidence support scoring over generated summaries.")
    parser.add_argument("--predictions", type=str, default="reports/longdoc/hierarchical_outputs.jsonl")
    parser.add_argument("--output-jsonl", type=str, default="reports/evidence_support.jsonl")
    parser.add_argument("--output-md", type=str, default="reports/evidence_examples.md")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_records = read_jsonl(args.predictions)
    if args.limit > 0:
        prediction_records = prediction_records[: args.limit]

    support_records = []
    for record in prediction_records:
        support_result = score_summary_support(
            paper_id=record["paper_id"],
            summary_text=record["generated_summary"],
            source_text=record["source_text"],
        )
        support_records.append(build_support_record(record, support_result))

    write_jsonl(args.output_jsonl, support_records)
    Path(args.output_md).write_text(build_markdown_report(support_records), encoding="utf-8")


if __name__ == "__main__":
    main()

