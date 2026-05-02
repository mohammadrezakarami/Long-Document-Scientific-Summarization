from scisum_qwen.evidence.chunker import chunk_section_aware_text, parse_section_aware_text


def test_parse_section_aware_text_reads_title_and_sections() -> None:
    source_text = (
        "Title: Sample Paper\n\n"
        "Introduction:\nA short intro.\n\n"
        "Method:\nMethod details.\n\n"
        "Results:\nResult details."
    )
    title, sections = parse_section_aware_text(source_text)
    assert title == "Sample Paper"
    assert sections["introduction"] == "A short intro."
    assert sections["method"] == "Method details."


def test_chunk_section_aware_text_creates_section_chunks() -> None:
    source_text = (
        "Title: Sample Paper\n\n"
        "Introduction:\n" + ("Intro sentence. " * 80) + "\n\n"
        "Method:\n" + ("Method sentence. " * 80)
    )
    _, chunks, sections = chunk_section_aware_text(
        paper_id="paper_1",
        source_text=source_text,
        max_tokens=80,
    )
    assert sections["introduction"]
    assert len(chunks) >= 2
    assert all(chunk.paper_id == "paper_1" for chunk in chunks)

