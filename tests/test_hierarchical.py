from scisum_qwen.inference.hierarchical import HierarchicalSummarizer, build_structured_summary
from scisum_qwen.inference.token_budget import allocate_section_word_budgets, should_use_hierarchical


def test_should_use_hierarchical_respects_margin() -> None:
    assert should_use_hierarchical(5000, max_input_tokens=4096, safety_margin=0.9)
    assert not should_use_hierarchical(2000, max_input_tokens=4096, safety_margin=0.9)


def test_allocate_section_word_budgets_preserves_minimums() -> None:
    budgets = allocate_section_word_budgets({"introduction": 100, "method": 200}, total_word_budget=40)
    assert budgets["introduction"] >= 24
    assert budgets["method"] >= 24


def test_hierarchical_summarizer_returns_summary_and_metadata() -> None:
    record = {
        "paper_id": "paper_1",
        "title": "Hierarchical Summarization",
        "source_text": (
            "Title: Hierarchical Summarization\n\n"
            "Introduction:\nThis paper studies long-document summarization and proposes a section-aware pipeline.\n\n"
            "Method:\nThe approach summarizes each section separately before synthesizing a final summary.\n\n"
            "Results:\nThe hierarchical system preserves result details and improves ROUGE-L.\n\n"
            "Conclusion:\nThe method is effective for long scientific papers."
        ),
        "target_summary": "A section-aware summarization system improves long-document scientific summaries.",
    }
    summarizer = HierarchicalSummarizer()
    result = summarizer.summarize_record(record, summary_type="abstract")
    assert result.summary
    assert result.chunk_count >= 1
    assert result.section_count >= 3


def test_build_structured_summary_fills_expected_fields() -> None:
    structured = build_structured_summary(
        {"discussion": "The paper notes a limitation in data coverage."},
        [],
    )
    assert "TL;DR:" in structured
    assert "Limitations:" in structured

