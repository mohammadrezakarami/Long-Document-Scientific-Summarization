from fastapi.testclient import TestClient

from scisum_qwen.api.main import app


client = TestClient(app)


def _sample_payload() -> dict:
    return {
        "title": "Sample Paper",
        "paper_text": (
            "Introduction:\nThis paper studies scientific summarization.\n\n"
            "Method:\nThe method uses section-aware summarization.\n\n"
            "Results:\nIt improves ROUGE-L by 2.4 points.\n\n"
            "Conclusion:\nThe approach is effective."
        ),
        "mode": "hierarchical",
        "summary_type": "structured",
        "enable_evidence_support": True,
    }


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint_returns_backend_metadata() -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert body["summary_backend"]


def test_summarize_endpoint_returns_summary_and_evidence() -> None:
    response = client.post("/summarize", json=_sample_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["raw_text"]
    assert body["metadata"]["mode"] == "hierarchical"
    assert body["evidence_support"]["claims"]


def test_evidence_support_endpoint_scores_claims() -> None:
    response = client.post(
        "/evidence-support",
        json={
            "paper_text": _sample_payload()["paper_text"],
            "summary_text": "The method uses section-aware summarization. It improves ROUGE-L by 2.4 points.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["overall_score"] >= 0.0
    assert len(body["claims"]) == 2


def test_summarize_pdf_endpoint_reuses_summarization_flow() -> None:
    response = client.post("/summarize/pdf", json=_sample_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["paper_id"]
    assert body["summary"]["raw_text"]
