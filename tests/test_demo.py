from app.gradio_app import run_demo


def test_run_demo_returns_empty_state_for_blank_text() -> None:
    summary, evidence_df, overall_score, payload = run_demo("", "   ", "hierarchical", "structured", True)

    assert "Paste paper text" in summary
    assert evidence_df.empty
    assert overall_score == 0.0
    assert payload == {}


def test_run_demo_returns_summary_payload_and_dataframe() -> None:
    summary, evidence_df, overall_score, payload = run_demo(
        "Sample Paper",
        (
            "Introduction:\nThis paper studies scientific summarization.\n\n"
            "Method:\nThe method uses section-aware summarization.\n\n"
            "Results:\nIt improves ROUGE-L by 2.4 points.\n\n"
            "Conclusion:\nThe approach is effective."
        ),
        "hierarchical",
        "structured",
        True,
    )

    assert summary
    assert list(evidence_df.columns) == ["claim", "label", "support_score", "section"]
    assert overall_score >= 0.0
    assert payload["summary"]["raw_text"]
