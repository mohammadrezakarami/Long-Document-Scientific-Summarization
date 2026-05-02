from scisum_qwen.data.preprocess import PreprocessConfig, clean_abstract_text, clean_paper_text, is_usable_sample


def test_clean_paper_text_removes_reference_block_and_citations() -> None:
    raw_text = """
    Introduction
    We compare our system to prior work [1, 2].

    Results
    The model improves F1 by 3.2 points.

    References
    [1] Some citation
    [2] Another citation
    """
    cleaned = clean_paper_text(raw_text, config=PreprocessConfig())
    assert "References" not in cleaned
    assert "[1, 2]" not in cleaned
    assert "3.2" in cleaned


def test_clean_abstract_text_collapses_noise() -> None:
    raw_abstract = "This   is a  test.\\cite{foo}\n\n\nIt should clean up."
    cleaned = clean_abstract_text(raw_abstract)
    assert "\\cite" not in cleaned
    assert "This is a test." in cleaned


def test_is_usable_sample_applies_length_rules() -> None:
    article = "a" * 3000
    abstract = "b" * 150
    assert is_usable_sample(article, abstract, min_article_chars=2000, min_abstract_chars=120, max_article_chars=5000)
    assert not is_usable_sample(article[:100], abstract, min_article_chars=2000, min_abstract_chars=120)

