from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Callable

from scisum_qwen import __version__
from scisum_qwen.api.schemas import (
    ClaimSupportPayload,
    EvidenceCandidatePayload,
    EvidenceSupportRequest,
    EvidenceSupportResponse,
    ModelInfoResponse,
    SummarizationMetadata,
    SummarizationRequest,
    SummarizationResponse,
    SummaryFields,
    SummaryMode,
    SummaryType,
    TopEvidencePayload,
)
from scisum_qwen.evidence.chunker import DEFAULT_SECTION_ORDER
from scisum_qwen.evidence.chunker import parse_section_aware_text
from scisum_qwen.evidence.support_scorer import score_summary_support
from scisum_qwen.inference.decoding import DecodingConfig
from scisum_qwen.inference.extractive import split_sentences, textrank_summarize
from scisum_qwen.inference.generator import HuggingFaceGenerator
from scisum_qwen.inference.hierarchical import (
    HierarchicalSummarizer,
    SectionSummary,
    build_final_synthesis_prompt,
    build_section_summary_prompt,
)
from scisum_qwen.inference.prompts import DEFAULT_SYSTEM_PROMPT, PromptBundle, build_prompt_bundle
from scisum_qwen.inference.token_budget import TokenBudgetConfig, infer_target_word_budget

_HEADER_PATTERN = re.compile(r"^(TL;DR|Problem|Method|Key Contributions|Results|Limitations)\s*:\s*(.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class LLMBackendConfig:
    backend_mode: str = "auto"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    adapter_path: str | None = None
    device_map: str = "auto"
    max_source_words: int = 1200
    max_section_words: int = 450
    decoding: DecodingConfig = field(default_factory=lambda: DecodingConfig(max_new_tokens=160))


def load_llm_backend_config_from_env() -> LLMBackendConfig:
    return LLMBackendConfig(
        backend_mode=os.getenv("SCISUM_SUMMARY_BACKEND", "auto").strip().lower(),
        model_name=os.getenv("SCISUM_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct").strip(),
        adapter_path=os.getenv("SCISUM_ADAPTER_PATH") or None,
        device_map=os.getenv("SCISUM_DEVICE", "auto").strip(),
        max_source_words=int(os.getenv("SCISUM_MAX_SOURCE_WORDS", "1200")),
        max_section_words=int(os.getenv("SCISUM_MAX_SECTION_WORDS", "450")),
        decoding=DecodingConfig(
            max_new_tokens=int(os.getenv("SCISUM_MAX_NEW_TOKENS", "160")),
            temperature=float(os.getenv("SCISUM_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("SCISUM_TOP_P", "0.9")),
            repetition_penalty=float(os.getenv("SCISUM_REPETITION_PENALTY", "1.05")),
            do_sample=os.getenv("SCISUM_DO_SAMPLE", "false").strip().lower() == "true",
        ),
    )


def _build_paper_id(title: str | None, paper_text: str) -> str:
    digest = hashlib.sha1(f"{title or ''}\n{paper_text}".encode("utf-8")).hexdigest()
    return digest[:16]


def _normalize_input_text(title: str | None, paper_text: str) -> str:
    text = paper_text.strip()
    if title and not text.startswith("Title:"):
        return f"Title: {title.strip()}\n\n{text}"
    return text


def _first_sentence(text: str) -> str | None:
    sentences = split_sentences(text)
    return sentences[0] if sentences else None


def parse_structured_summary(raw_text: str) -> SummaryFields:
    values: dict[str, list[str] | str | None] = {
        "tldr": None,
        "problem": None,
        "method": None,
        "contributions": [],
        "results": None,
        "limitations": None,
    }
    current_field: str | None = None

    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        header_match = _HEADER_PATTERN.match(stripped)
        if header_match:
            current_field = header_match.group(1).lower().replace(" ", "_").replace(";", "")
            content = header_match.group(2).strip()
            if current_field == "key_contributions":
                current_field = "contributions"
                if content:
                    values["contributions"] = [content]
            elif content:
                values[current_field] = content
            continue

        if current_field == "contributions":
            contribution_line = re.sub(r"^(\d+\.|-|\*)\s*", "", stripped).strip()
            if contribution_line:
                cast_list = list(values["contributions"])  # type: ignore[arg-type]
                cast_list.append(contribution_line)
                values["contributions"] = cast_list
        elif current_field:
            existing = values[current_field]
            merged = f"{existing} {stripped}".strip() if existing else stripped
            values[current_field] = merged

    return SummaryFields(
        raw_text=raw_text,
        abstract=None,
        tldr=values["tldr"],  # type: ignore[arg-type]
        problem=values["problem"],  # type: ignore[arg-type]
        method=values["method"],  # type: ignore[arg-type]
        contributions=list(values["contributions"]),  # type: ignore[arg-type]
        results=values["results"],  # type: ignore[arg-type]
        limitations=values["limitations"],  # type: ignore[arg-type]
    )


def build_summary_fields(raw_text: str, summary_type: SummaryType) -> SummaryFields:
    if summary_type == SummaryType.structured:
        parsed = parse_structured_summary(raw_text)
        if any([parsed.tldr, parsed.problem, parsed.method, parsed.results, parsed.limitations, parsed.contributions]):
            return parsed
    return SummaryFields(
        raw_text=raw_text,
        abstract=raw_text,
        tldr=_first_sentence(raw_text),
    )


def build_evidence_payload(summary_text: str, paper_text: str, paper_id: str) -> EvidenceSupportResponse:
    support_result = score_summary_support(
        paper_id=paper_id,
        summary_text=summary_text,
        source_text=paper_text,
    )
    claims = [
        ClaimSupportPayload(
            claim_id=claim.claim_id,
            claim=claim.claim,
            label=claim.label,
            support_score=claim.support_score,
            top_evidence=TopEvidencePayload(
                chunk_id=claim.top_evidence.chunk.chunk_id,
                section=claim.top_evidence.chunk.section,
                text=claim.top_evidence.chunk.text,
                score=round(claim.top_evidence.score, 4),
            ),
            evidence_candidates=[
                EvidenceCandidatePayload(
                    chunk_id=item.chunk.chunk_id,
                    section=item.chunk.section,
                    score=round(item.score, 4),
                )
                for item in claim.evidence_candidates
            ],
        )
        for claim in support_result.claims
    ]
    return EvidenceSupportResponse(
        overall_score=support_result.overall_support_score,
        claims=claims,
    )


@dataclass
class SummarizationService:
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    llm_backend: LLMBackendConfig = field(default_factory=load_llm_backend_config_from_env)
    generator_factory: Callable[..., Any] = HuggingFaceGenerator

    def __post_init__(self) -> None:
        self.hierarchical_summarizer = HierarchicalSummarizer()
        self.single_pass_structured_summarizer = HierarchicalSummarizer(
            budget=TokenBudgetConfig(
                max_input_tokens=100_000,
                chunk_token_limit=100_000,
                safety_margin=1.0,
            )
        )
        self._generator = None

    def _adapter_exists(self) -> bool:
        return bool(self.llm_backend.adapter_path and Path(self.llm_backend.adapter_path).exists())

    def _use_llm_backend(self) -> bool:
        if self.llm_backend.backend_mode == "heuristic":
            return False
        if self.llm_backend.backend_mode == "llm":
            return True
        return self._adapter_exists()

    def _get_generator(self):
        if self._generator is None:
            adapter_path = self.llm_backend.adapter_path if self._adapter_exists() else None
            self._generator = self.generator_factory(
                model_name=self.llm_backend.model_name,
                adapter_path=adapter_path,
                device_map=self.llm_backend.device_map,
            )
        return self._generator

    def _truncate_words(self, text: str, max_words: int) -> str:
        if max_words <= 0:
            return text
        return " ".join(text.split()[:max_words])

    def _run_llm_bundle(self, bundle: PromptBundle) -> tuple[str, dict[str, Any]]:
        generator = self._get_generator()
        result = generator.generate_from_bundle(bundle, self.llm_backend.decoding)
        return result.text, result.metadata

    def _build_section_aware_text(self, title: str | None, sections: dict[str, str], *, max_words: int) -> str:
        if not sections:
            return ""
        blocks = []
        for section_name in DEFAULT_SECTION_ORDER:
            section_text = sections.get(section_name)
            if not section_text:
                continue
            truncated = self._truncate_words(section_text, self.llm_backend.max_section_words)
            blocks.append(f"{section_name.title()}:\n{truncated}")
        combined = "\n\n".join(blocks).strip()
        if title and not combined.startswith("Title:"):
            combined = f"Title: {title}\n\n{combined}" if combined else f"Title: {title}"
        return self._truncate_words(combined, max_words)

    def _summarize_with_llm(
        self,
        *,
        title: str | None,
        paper_id: str,
        normalized_text: str,
        request: SummarizationRequest,
        sections: dict[str, str],
    ) -> tuple[str, bool, int, int, str]:
        del paper_id
        style = "structured" if request.summary_type == SummaryType.structured else "faithful_abstract"

        if request.mode == SummaryMode.hierarchical and sections:
            section_summaries: list[SectionSummary] = []
            for section_name in DEFAULT_SECTION_ORDER:
                section_text = sections.get(section_name)
                if not section_text:
                    continue
                truncated_section = self._truncate_words(section_text, self.llm_backend.max_section_words)
                user_prompt = build_section_summary_prompt(section_name, truncated_section)
                bundle = PromptBundle(
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    style="zero_shot",
                )
                section_summary_text, _ = self._run_llm_bundle(bundle)
                section_summaries.append(
                    SectionSummary(
                        section=section_name,
                        summary=section_summary_text,
                        chunk_count=1,
                        source_token_count=len(truncated_section.split()),
                    )
                )

            if section_summaries:
                if request.summary_type == SummaryType.structured:
                    synthesis_prompt = build_final_synthesis_prompt(section_summaries, title=title)
                    synthesis_bundle = PromptBundle(
                        system_prompt=DEFAULT_SYSTEM_PROMPT,
                        user_prompt=synthesis_prompt,
                        style="structured",
                    )
                else:
                    combined_summaries = "\n\n".join(
                        f"{item.section.title()}: {item.summary}" for item in section_summaries
                    )
                    synthesis_bundle = build_prompt_bundle(
                        title=title,
                        paper_text=combined_summaries,
                        style="faithful_abstract",
                    )
                final_summary, _ = self._run_llm_bundle(synthesis_bundle)
                backend = "qwen-qlora-hierarchical" if self._adapter_exists() else "qwen-base-hierarchical"
                return final_summary, True, len(section_summaries), len(section_summaries), backend

        llm_source_text = self._build_section_aware_text(
            title,
            sections,
            max_words=self.llm_backend.max_source_words,
        ) or self._truncate_words(normalized_text, self.llm_backend.max_source_words)
        bundle = build_prompt_bundle(
            title=title,
            paper_text=llm_source_text,
            style=style,  # type: ignore[arg-type]
        )
        final_summary, _ = self._run_llm_bundle(bundle)
        backend = "qwen-qlora-single-pass" if self._adapter_exists() else "qwen-base-single-pass"
        return final_summary, False, 1, max(len(sections), 1), backend

    def summarize(self, request: SummarizationRequest) -> SummarizationResponse:
        normalized_text = _normalize_input_text(request.title, request.paper_text)
        paper_id = _build_paper_id(request.title, normalized_text)
        record = {
            "paper_id": paper_id,
            "title": request.title,
            "source_text": normalized_text,
            "target_summary": "",
        }
        parsed_title, sections = parse_section_aware_text(normalized_text)
        title = request.title or parsed_title
        target_budget = infer_target_word_budget(None)

        if self._use_llm_backend():
            raw_summary, used_hierarchical, chunk_count, section_count, backend = self._summarize_with_llm(
                title=title,
                paper_id=paper_id,
                normalized_text=normalized_text,
                request=request,
                sections=sections,
            )
        elif request.mode == SummaryMode.single_pass and request.summary_type == SummaryType.abstract:
            raw_summary = textrank_summarize(normalized_text, target_word_budget=target_budget, max_sentences=4)
            used_hierarchical = False
            chunk_count = 1
            section_count = len(sections)
            backend = "textrank-single-pass"
        else:
            summarizer = (
                self.hierarchical_summarizer
                if request.mode == SummaryMode.hierarchical
                else self.single_pass_structured_summarizer
            )
            result = summarizer.summarize_record(record, summary_type=request.summary_type.value, target_word_budget=target_budget)
            raw_summary = result.summary
            used_hierarchical = result.used_hierarchical if request.mode == SummaryMode.hierarchical else False
            chunk_count = result.chunk_count
            section_count = result.section_count
            backend = "textrank-hierarchical" if request.mode == SummaryMode.hierarchical else "textrank-structured-single-pass"

        summary_fields = build_summary_fields(raw_summary, request.summary_type)
        evidence_payload = None
        if request.enable_evidence_support:
            evidence_payload = build_evidence_payload(summary_fields.raw_text, normalized_text, paper_id)

        return SummarizationResponse(
            paper_id=paper_id,
            title=request.title,
            summary=summary_fields,
            metadata=SummarizationMetadata(
                backend=backend,
                mode=request.mode,
                summary_type=request.summary_type,
                used_hierarchical=used_hierarchical,
                chunk_count=chunk_count,
                section_count=section_count,
            ),
            evidence_support=evidence_payload,
        )

    def score_evidence(self, request: EvidenceSupportRequest) -> EvidenceSupportResponse:
        paper_id = request.paper_id or _build_paper_id(None, request.paper_text)
        return build_evidence_payload(request.summary_text, request.paper_text, paper_id)

    def model_info(self) -> ModelInfoResponse:
        try:
            import torch

            mps_available = torch.backends.mps.is_available()
            cuda_available = torch.cuda.is_available()
        except Exception:
            mps_available = False
            cuda_available = False

        backend_name = (
            "qwen-backed generation"
            if self._use_llm_backend()
            else "textrank + section-aware heuristics"
        )
        adapter_note = self.llm_backend.adapter_path or "not configured"

        return ModelInfoResponse(
            base_model=self.llm_backend.model_name,
            summary_backend=backend_name,
            evidence_backend="sentence-transformers fallback to TF-IDF retrieval",
            local_mps_available=mps_available,
            notes=[
                f"Backend mode: {self.llm_backend.backend_mode}",
                f"CUDA available: {cuda_available}",
                f"Adapter path: {adapter_note}",
                f"Adapter available locally: {self._adapter_exists()}",
                f"Service version: {__version__}",
            ],
        )


service = SummarizationService()
