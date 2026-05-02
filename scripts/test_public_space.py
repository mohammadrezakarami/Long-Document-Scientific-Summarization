from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an end-to-end test against the public Hugging Face Space.")
    parser.add_argument("--space-id", default="mokarami/scisum-qwen")
    parser.add_argument("--title", default="Faithful Long-Document Scientific Summarization with Section-Aware Hierarchical Inference")
    parser.add_argument("--paper-path", default="data/samples/featured_demo_paper.txt")
    parser.add_argument("--mode", default="hierarchical")
    parser.add_argument("--summary-type", default="structured")
    args = parser.parse_args()

    try:
        from gradio_client import Client
    except ImportError as exc:
        raise SystemExit("gradio_client is required to test the public Space.") from exc

    paper_text = Path(args.paper_path).read_text(encoding="utf-8")
    client = Client(args.space_id, verbose=False)
    summary, evidence_table, overall_score, payload = client.predict(
        title=args.title,
        paper_text=paper_text,
        mode=args.mode,
        summary_type=args.summary_type,
        enable_evidence_support=True,
        api_name="/run_demo",
    )

    print("summary_preview:", summary[:600])
    print("overall_score:", overall_score)
    if isinstance(evidence_table, dict):
        print("evidence_rows:", len(evidence_table.get("data", [])))
    if isinstance(payload, dict):
        print("paper_id:", payload.get("paper_id"))


if __name__ == "__main__":
    main()
