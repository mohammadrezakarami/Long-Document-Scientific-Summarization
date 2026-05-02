from scisum_qwen.inference.prompts import build_prompt_bundle, bundle_to_messages


def test_zero_shot_prompt_contains_paper_label() -> None:
    bundle = build_prompt_bundle(paper_text="Sample paper text.", title="Paper", style="zero_shot")
    assert "Summarize the following scientific paper." in bundle.user_prompt
    assert "Paper:\nSample paper text." in bundle.user_prompt


def test_structured_prompt_lists_required_fields() -> None:
    bundle = build_prompt_bundle(paper_text="Body", style="structured")
    assert "TL;DR:" in bundle.user_prompt
    assert "Problem:" in bundle.user_prompt
    assert "Limitations:" in bundle.user_prompt


def test_bundle_to_messages_keeps_two_turns() -> None:
    bundle = build_prompt_bundle(paper_text="Body")
    messages = bundle_to_messages(bundle)
    assert [item["role"] for item in messages] == ["system", "user"]

