from scisum_qwen.api.schemas import EvidenceSupportRequest, SummarizationRequest
from scisum_qwen.api.service import LLMBackendConfig, SummarizationService, service


class FakeGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate_from_bundle(self, bundle, decoding):
        del decoding

        class Result:
            text = f"Generated via {bundle.style}: {bundle.user_prompt[:48]}"
            metadata = {"device": "fake", "prompt_style": bundle.style}

        return Result()


def _paper_text() -> str:
    return (
        "Introduction:\nThis paper studies scientific summarization.\n\n"
        "Method:\nThe method uses section-aware summarization and hierarchical inference.\n\n"
        "Results:\nIt improves ROUGE-L by 2.4 points over the baseline.\n\n"
        "Conclusion:\nThe approach is practical for long documents."
    )


def test_service_summarize_returns_metadata() -> None:
    response = service.summarize(
        SummarizationRequest(
            title="Sample Paper",
            paper_text=_paper_text(),
            mode="hierarchical",
            summary_type="structured",
            enable_evidence_support=True,
        )
    )
    assert response.summary.raw_text
    assert response.metadata.chunk_count >= 1
    assert response.evidence_support is not None


def test_service_score_evidence_returns_claims() -> None:
    response = service.score_evidence(
        EvidenceSupportRequest(
            paper_text=_paper_text(),
            summary_text="The method uses section-aware summarization. It improves ROUGE-L by 2.4 points.",
        )
    )
    assert response.claims
    assert response.overall_score >= 0.0


def test_service_can_use_llm_backend_when_enabled() -> None:
    llm_service = SummarizationService(
        llm_backend=LLMBackendConfig(
            backend_mode="llm",
            model_name="Qwen/Qwen2.5-3B-Instruct",
            adapter_path=None,
            max_source_words=120,
            max_section_words=60,
        ),
        generator_factory=FakeGenerator,
    )
    response = llm_service.summarize(
        SummarizationRequest(
            title="Sample Paper",
            paper_text=_paper_text(),
            mode="hierarchical",
            summary_type="structured",
            enable_evidence_support=False,
        )
    )
    assert response.summary.raw_text
    assert response.metadata.backend.startswith("qwen-base") or response.metadata.backend.startswith("qwen-qlora")
