"""Evidence scoring modules for SciSum-Qwen."""

from scisum_qwen.evidence.chunker import PaperChunk, chunk_section_aware_text, parse_section_aware_text
from scisum_qwen.evidence.claim_extractor import Claim, extract_claims
from scisum_qwen.evidence.support_scorer import EvidenceSupportResult, score_summary_support

__all__ = [
    "PaperChunk",
    "Claim",
    "EvidenceSupportResult",
    "chunk_section_aware_text",
    "parse_section_aware_text",
    "extract_claims",
    "score_summary_support",
]
