from scisum_qwen.evidence.support_scorer import label_support, score_summary_support


def test_label_support_uses_expected_thresholds() -> None:
    assert label_support(0.8) == "supported"
    assert label_support(0.6) == "partially_supported"
    assert label_support(0.4) == "weakly_supported"


def test_score_summary_support_returns_ranked_evidence() -> None:
    source_text = (
        "Title: Sample\n\n"
        "Introduction:\nThis paper studies scientific summarization.\n\n"
        "Method:\nThe proposed method uses hierarchical section-aware summarization.\n\n"
        "Results:\nThe model improves ROUGE-L by 2.4 points over the baseline."
    )
    summary_text = "The proposed method uses hierarchical section-aware summarization. It improves ROUGE-L by 2.4 points."
    result = score_summary_support(
        paper_id="paper_1",
        summary_text=summary_text,
        source_text=source_text,
    )
    assert len(result.claims) == 2
    assert result.claims[0].top_evidence.chunk.section in {"method", "results", "introduction"}
    assert 0.0 <= result.overall_support_score <= 1.0

