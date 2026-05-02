from __future__ import annotations

from fastapi import FastAPI

from scisum_qwen import __version__
from scisum_qwen.api.schemas import (
    EvidenceSupportRequest,
    EvidenceSupportResponse,
    HealthResponse,
    ModelInfoResponse,
    SummarizationRequest,
    SummarizationResponse,
)
from scisum_qwen.api.service import service

app = FastAPI(title="SciSum-Qwen API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return service.model_info()


@app.post("/summarize", response_model=SummarizationResponse)
def summarize(request: SummarizationRequest) -> SummarizationResponse:
    return service.summarize(request)


@app.post("/evidence-support", response_model=EvidenceSupportResponse)
def evidence_support(request: EvidenceSupportRequest) -> EvidenceSupportResponse:
    return service.score_evidence(request)


@app.post("/summarize/pdf", response_model=SummarizationResponse)
def summarize_pdf(request: SummarizationRequest) -> SummarizationResponse:
    return service.summarize(request)
