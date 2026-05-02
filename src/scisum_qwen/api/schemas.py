from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SummaryMode(str, Enum):
    single_pass = "single_pass"
    hierarchical = "hierarchical"


class SummaryType(str, Enum):
    abstract = "abstract"
    structured = "structured"


class SummarizationRequest(BaseModel):
    title: str | None = None
    paper_text: str
    mode: SummaryMode = SummaryMode.hierarchical
    summary_type: SummaryType = SummaryType.structured
    enable_evidence_support: bool = True


class EvidenceSupportRequest(BaseModel):
    paper_id: str | None = None
    summary_text: str
    paper_text: str


class SummaryFields(BaseModel):
    raw_text: str
    abstract: str | None = None
    tldr: str | None = None
    problem: str | None = None
    method: str | None = None
    contributions: list[str] = Field(default_factory=list)
    results: str | None = None
    limitations: str | None = None


class TopEvidencePayload(BaseModel):
    chunk_id: str
    section: str
    text: str
    score: float


class EvidenceCandidatePayload(BaseModel):
    chunk_id: str
    section: str
    score: float


class ClaimSupportPayload(BaseModel):
    claim_id: str
    claim: str
    label: str
    support_score: float
    top_evidence: TopEvidencePayload
    evidence_candidates: list[EvidenceCandidatePayload] = Field(default_factory=list)


class EvidenceSupportResponse(BaseModel):
    overall_score: float
    claims: list[ClaimSupportPayload] = Field(default_factory=list)


class SummarizationMetadata(BaseModel):
    backend: str
    mode: SummaryMode
    summary_type: SummaryType
    used_hierarchical: bool
    chunk_count: int
    section_count: int


class SummarizationResponse(BaseModel):
    paper_id: str
    title: str | None = None
    summary: SummaryFields
    metadata: SummarizationMetadata
    evidence_support: EvidenceSupportResponse | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


class ModelInfoResponse(BaseModel):
    base_model: str
    summary_backend: str
    evidence_backend: str
    local_mps_available: bool
    notes: list[str]

