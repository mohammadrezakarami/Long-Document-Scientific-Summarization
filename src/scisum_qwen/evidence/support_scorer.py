from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from scisum_qwen.evidence.chunker import PaperChunk, chunk_section_aware_text
from scisum_qwen.evidence.claim_extractor import Claim, extract_claims
from scisum_qwen.evidence.embedder import build_embedder
from scisum_qwen.evidence.index import EvidenceIndex, RetrievedEvidence


@dataclass(frozen=True)
class ClaimSupport:
    claim_id: str
    claim: str
    label: str
    support_score: float
    top_evidence: RetrievedEvidence
    evidence_candidates: tuple[RetrievedEvidence, ...]


@dataclass(frozen=True)
class EvidenceSupportResult:
    paper_id: str
    summary: str
    claims: tuple[ClaimSupport, ...]
    overall_support_score: float


def label_support(score: float) -> str:
    if score >= 0.75:
        return "supported"
    if score >= 0.55:
        return "partially_supported"
    return "weakly_supported"


def score_summary_support(
    *,
    paper_id: str,
    summary_text: str,
    source_text: str,
    preferred_embedder_model: str | None = None,
    top_k: int = 3,
    max_chunk_tokens: int = 900,
) -> EvidenceSupportResult:
    _, chunks, _ = chunk_section_aware_text(
        paper_id=paper_id,
        source_text=source_text,
        max_tokens=max_chunk_tokens,
    )
    if not chunks:
        return EvidenceSupportResult(
            paper_id=paper_id,
            summary=summary_text,
            claims=tuple(),
            overall_support_score=0.0,
        )

    claims = extract_claims(summary_text, claim_prefix=paper_id)
    if not claims:
        return EvidenceSupportResult(
            paper_id=paper_id,
            summary=summary_text,
            claims=tuple(),
            overall_support_score=0.0,
        )

    embedder = build_embedder(preferred_embedder_model)
    index = EvidenceIndex.from_chunks(chunks, embedder)

    supported_claims: list[ClaimSupport] = []
    for claim in claims:
        evidence = index.search(claim.text, embedder, top_k=top_k)
        top_evidence = evidence[0]
        score = round(top_evidence.score, 4)
        supported_claims.append(
            ClaimSupport(
                claim_id=claim.claim_id,
                claim=claim.text,
                label=label_support(score),
                support_score=score,
                top_evidence=top_evidence,
                evidence_candidates=tuple(evidence),
            )
        )

    overall_support = round(mean(item.support_score for item in supported_claims), 4) if supported_claims else 0.0
    return EvidenceSupportResult(
        paper_id=paper_id,
        summary=summary_text,
        claims=tuple(supported_claims),
        overall_support_score=overall_support,
    )

