from scisum_qwen.data.build_sft_dataset import build_messages, build_user_prompt, preprocess_record
from scisum_qwen.data.load_arxiv import FormattingConfig
from scisum_qwen.data.preprocess import PreprocessConfig


def test_build_messages_uses_chat_style() -> None:
    messages = build_messages("system", "user", "assistant")
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]


def test_build_user_prompt_places_instruction_before_paper() -> None:
    prompt = build_user_prompt("Summarize this paper.", "Paper body.")
    assert prompt.startswith("Summarize this paper.")
    assert "Paper:\nPaper body." in prompt


def test_preprocess_record_returns_sft_ready_example() -> None:
    formatting = FormattingConfig(
        include_title=True,
        include_section_headers=True,
        keep_sections=("introduction", "method", "results", "conclusion"),
        system_prompt="System prompt",
        user_instruction="Summarize the paper.",
    )
    record = preprocess_record(
        {
            "paper_id": "paper_1",
            "title": "Test Title",
            "article": """
            Introduction
            This paper studies scientific summarization.

            Method
            We fine-tune a model.

            Results
            The method improves ROUGE-L by 2 points.

            Conclusion
            The method is effective.
            """
            + (" extra text" * 300),
            "abstract": "We fine-tune a summarization model for scientific papers and improve results.",
        },
        preprocess_config=PreprocessConfig(min_article_chars=200, min_abstract_chars=20, max_article_chars=20000),
        formatting_config=formatting,
        fallback_chars_per_token=4.0,
    )
    assert record is not None
    assert record["paper_id"] == "paper_1"
    assert record["messages"][0]["role"] == "system"
    assert "Introduction:" in record["source_text"]
    assert record["article_token_estimate"] > record["summary_token_estimate"]
