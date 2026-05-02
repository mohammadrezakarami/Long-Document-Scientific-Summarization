from scisum_qwen.data.section_parser import build_section_aware_text, parse_sections


def test_parse_sections_detects_common_scientific_headers() -> None:
    paper_text = """
    Abstract
    This is the abstract.

    Introduction
    This is the intro.

    Method
    This is the method.

    Results
    These are the results.

    Conclusion
    This is the end.
    """
    sections = parse_sections(paper_text)
    assert sections["abstract"].startswith("This is the abstract")
    assert sections["method"].startswith("This is the method")
    assert sections["results"].startswith("These are the results")


def test_build_section_aware_text_orders_sections() -> None:
    paper_text = """
    Method
    Method details.

    Introduction
    Intro details.

    Results
    Result details.
    """
    formatted = build_section_aware_text(title="Sample Paper", paper_text=paper_text)
    assert formatted.startswith("Title: Sample Paper")
    assert "Introduction:\nIntro details." in formatted
    assert formatted.index("Introduction:") < formatted.index("Method:")

