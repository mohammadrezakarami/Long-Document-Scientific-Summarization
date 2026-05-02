from scisum_qwen.evidence.claim_extractor import extract_claims


def test_extract_claims_splits_summary_into_atomic_claims() -> None:
    claims = extract_claims("Problem: The paper studies summarization. Results: It improves ROUGE-L.")
    assert len(claims) == 2
    assert claims[0].text.startswith("The paper studies")
    assert claims[1].text.startswith("It improves")

