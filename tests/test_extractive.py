from scisum_qwen.inference.extractive import sentence_similarity, split_sentences, textrank_summarize


def test_split_sentences_finds_multiple_sentences() -> None:
    sentences = split_sentences("Sentence one. Sentence two! Sentence three?")
    assert len(sentences) == 3


def test_sentence_similarity_is_higher_for_related_sentences() -> None:
    related = sentence_similarity("The model improves rouge score.", "The model improves score.")
    unrelated = sentence_similarity("The model improves rouge score.", "Completely different topic here.")
    assert related > unrelated


def test_textrank_returns_non_empty_summary() -> None:
    text = (
        "This paper studies summarization. "
        "The method uses a transformer architecture. "
        "Experiments show better ROUGE-L scores than the baseline. "
        "The model is efficient and accurate."
    )
    summary = textrank_summarize(text, target_word_budget=18)
    assert summary
    assert "transformer" in summary.lower() or "rouge" in summary.lower()

