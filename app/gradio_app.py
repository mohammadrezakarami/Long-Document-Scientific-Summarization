from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import pandas as pd

from scisum_qwen.api.schemas import SummarizationRequest
from scisum_qwen.api.service import service

APP_ROOT = Path(__file__).resolve().parents[1]
FEATURED_DEMO_PATH = APP_ROOT / "data" / "samples" / "featured_demo_paper.txt"
FEATURED_DEMO_TITLE = "Faithful Long-Document Scientific Summarization with Section-Aware Hierarchical Inference"

CUSTOM_CSS = """
body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(72, 145, 255, 0.15), transparent 28%),
    radial-gradient(circle at top right, rgba(27, 196, 125, 0.12), transparent 24%),
    linear-gradient(180deg, #f4f7fb 0%, #eef3f9 100%);
  color: #0f172a;
  font-family: "Avenir Next", "Segoe UI", sans-serif;
}
#hero {
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.92));
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 28px;
  padding: 28px 30px;
  box-shadow: 0 24px 60px rgba(15, 23, 42, 0.12);
  color: #f8fafc;
}
#hero h1 {
  margin: 0 0 10px 0;
  font-size: 2.6rem;
  letter-spacing: -0.04em;
}
#hero p {
  margin: 0;
  color: rgba(226, 232, 240, 0.92);
  line-height: 1.6;
  font-size: 1rem;
}
.status-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-top: 22px;
}
.status-card {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 20px;
  padding: 16px 18px;
}
.status-card .label {
  display: block;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: rgba(191, 219, 254, 0.86);
  margin-bottom: 8px;
}
.status-card .value {
  display: block;
  font-size: 1.15rem;
  line-height: 1.4;
  color: #ffffff;
  font-weight: 700;
}
.panel {
  background: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 24px;
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
}
.panel-tight {
  padding: 14px;
}
.section-title {
  font-size: 0.86rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #475569;
  margin-bottom: 8px;
}
.run-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.07);
  border: 1px solid rgba(148, 163, 184, 0.2);
  font-size: 0.92rem;
  color: #0f172a;
  margin-right: 10px;
  margin-bottom: 10px;
}
.summary-markdown {
  background: rgba(255, 255, 255, 0.76);
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 22px;
  padding: 18px 20px;
}
"""


def _load_featured_demo_text() -> str:
    if FEATURED_DEMO_PATH.exists():
        return FEATURED_DEMO_PATH.read_text(encoding="utf-8").strip()
    return (
        "Introduction:\nThis paper studies scientific summarization.\n\n"
        "Method:\nThe method uses section-aware summarization and hierarchical inference.\n\n"
        "Results:\nIt improves ROUGE-L by 2.4 points over the baseline.\n\n"
        "Conclusion:\nThe approach is practical for long documents."
    )


FEATURED_DEMO_TEXT = _load_featured_demo_text()


def _format_summary_markdown(payload: dict) -> str:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    title = payload.get("title") or payload.get("paper_id", "Generated Summary")

    contributions = summary.get("contributions") or []
    contribution_lines = "\n".join(f"- {item}" for item in contributions) if contributions else "- Not specified"

    evidence = payload.get("evidence_support") or {}
    overall_score = evidence.get("overall_score")
    score_line = f"**Evidence Support Score:** `{overall_score:.2f}`" if isinstance(overall_score, (int, float)) else ""

    parts = [
        f"## {title}",
        score_line,
        f"**Backend:** `{metadata.get('backend', 'unknown')}`",
        "",
        f"### TL;DR\n{summary.get('tldr') or 'Not specified'}",
        "",
        f"### Problem\n{summary.get('problem') or 'Not specified'}",
        "",
        f"### Method\n{summary.get('method') or summary.get('abstract') or 'Not specified'}",
        "",
        "### Key Contributions",
        contribution_lines,
        "",
        f"### Results\n{summary.get('results') or 'Not specified'}",
        "",
        f"### Limitations\n{summary.get('limitations') or 'Not specified'}",
    ]
    return "\n".join(line for line in parts if line is not None).strip()


def run_demo(
    title: str,
    paper_text: str,
    mode: str,
    summary_type: str,
    enable_evidence_support: bool,
):
    if not paper_text.strip():
        empty = pd.DataFrame(columns=["claim", "label", "support_score", "section"])
        return "Paste paper text to summarize it.", empty, 0.0, {}

    request = SummarizationRequest(
        title=title or None,
        paper_text=paper_text,
        mode=mode,
        summary_type=summary_type,
        enable_evidence_support=enable_evidence_support,
    )
    response = service.summarize(request)
    payload = response.model_dump(mode="json")

    evidence_rows = []
    overall_score = 0.0
    if response.evidence_support:
        overall_score = response.evidence_support.overall_score
        for claim in response.evidence_support.claims:
            evidence_rows.append(
                {
                    "claim": claim.claim,
                    "label": claim.label,
                    "support_score": claim.support_score,
                    "section": claim.top_evidence.section,
                }
            )
    evidence_df = pd.DataFrame(evidence_rows, columns=["claim", "label", "support_score", "section"])
    return _format_summary_markdown(payload), evidence_df, overall_score, payload


def _load_example() -> tuple[str, str]:
    return FEATURED_DEMO_TITLE, FEATURED_DEMO_TEXT


MODEL_INFO = service.model_info()
BADGES_HTML = f"""
<div class="status-grid">
  <div class="status-card">
    <span class="label">Inference Backend</span>
    <span class="value">{MODEL_INFO.summary_backend}</span>
  </div>
  <div class="status-card">
    <span class="label">Base Model</span>
    <span class="value">{MODEL_INFO.base_model}</span>
  </div>
  <div class="status-card">
    <span class="label">Evidence Layer</span>
    <span class="value">{MODEL_INFO.evidence_backend}</span>
  </div>
</div>
"""


with gr.Blocks(title="SciSum-Qwen") as demo:
    with gr.Column(elem_id="hero"):
        gr.Markdown(
            """
            # SciSum-Qwen

            Faithful long-document scientific paper summarization powered by a
            QLoRA-adapted LLM, hierarchical section-aware inference, and claim-level
            evidence support scoring.
            """
        )
        gr.HTML(BADGES_HTML)

    with gr.Row():
        with gr.Column(scale=5, elem_classes=["panel", "panel-tight"]):
            gr.Markdown("### Input Workspace")
            gr.Markdown(
                "Paste a full research paper, or load the featured long-document example to inspect structured summarization and evidence grounding."
            )
            title = gr.Textbox(label="Paper Title", placeholder="Enter the paper title")
            paper_text = gr.Textbox(
                label="Paper Text",
                lines=22,
                placeholder="Paste a scientific paper with section headers such as Abstract, Introduction, Method, Results, and Conclusion.",
            )
            with gr.Row():
                mode = gr.Dropdown(
                    label="Summarization Mode",
                    choices=["single_pass", "hierarchical"],
                    value="hierarchical",
                )
                summary_type = gr.Dropdown(
                    label="Output Style",
                    choices=["abstract", "structured"],
                    value="structured",
                )
                enable_evidence = gr.Checkbox(label="Enable Evidence Support", value=True)

            with gr.Row():
                load_example_btn = gr.Button("Load Featured Example", variant="secondary")
                clear_btn = gr.Button("Clear", variant="secondary")
                submit = gr.Button("Run Summarization", variant="primary")

            gr.Examples(
                examples=[[FEATURED_DEMO_TITLE, FEATURED_DEMO_TEXT, "hierarchical", "structured", True]],
                inputs=[title, paper_text, mode, summary_type, enable_evidence],
                label="Quick Demo Example",
            )

        with gr.Column(scale=6):
            with gr.Tab("Executive Summary"):
                summary_output = gr.Markdown(value="Run the model to generate a structured summary.", elem_classes=["summary-markdown"])
            with gr.Tab("Evidence View"):
                overall_score = gr.Number(label="Overall Support Score", precision=4)
                evidence_table = gr.Dataframe(label="Claim-Level Evidence Table", interactive=False, wrap=True)
            with gr.Tab("Full Response JSON"):
                json_output = gr.JSON(label="Model Response")

    with gr.Accordion("What this demo proves", open=False):
        gr.Markdown(
            """
            - The app is backed by a deployed scientific summarization system, not a static mock.
            - Long scientific text is processed with section-aware and hierarchical logic.
            - Output claims can be inspected through evidence support scoring.
            - The same service powers the API, local app, and public Hugging Face Space.
            """
        )
        for note in MODEL_INFO.notes:
            gr.HTML(f'<span class="run-badge">{note}</span>')

    submit.click(
        run_demo,
        inputs=[title, paper_text, mode, summary_type, enable_evidence],
        outputs=[summary_output, evidence_table, overall_score, json_output],
    )
    load_example_btn.click(
        _load_example,
        inputs=None,
        outputs=[title, paper_text],
    )
    clear_btn.click(
        lambda: ("", "", "hierarchical", "structured", True, "Run the model to generate a structured summary.", pd.DataFrame(columns=["claim", "label", "support_score", "section"]), 0.0, {}),
        inputs=None,
        outputs=[title, paper_text, mode, summary_type, enable_evidence, summary_output, evidence_table, overall_score, json_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("SCISUM_DEMO_HOST", "127.0.0.1"),
        server_port=int(os.getenv("SCISUM_DEMO_PORT", "7860")),
        css=CUSTOM_CSS,
    )
