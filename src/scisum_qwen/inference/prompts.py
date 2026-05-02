from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PromptStyle = Literal["zero_shot", "faithful_abstract", "structured"]


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    user_prompt: str
    style: PromptStyle


DEFAULT_SYSTEM_PROMPT = (
    "You are a scientific research assistant. Generate faithful, concise, and technically accurate summaries."
)


def build_prompt_bundle(
    *,
    paper_text: str,
    title: str | None = None,
    style: PromptStyle = "faithful_abstract",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> PromptBundle:
    prefix = f"Title: {title}\n\n" if title else ""
    if style == "zero_shot":
        user_prompt = f"Summarize the following scientific paper.\n\n{prefix}Paper:\n{paper_text}".strip()
    elif style == "structured":
        user_prompt = (
            "Generate a structured scientific summary with the following fields:\n"
            "TL;DR:\n"
            "Problem:\n"
            "Method:\n"
            "Key Contributions:\n"
            "Results:\n"
            "Limitations:\n"
            "Use only information supported by the paper.\n\n"
            f"{prefix}Paper:\n{paper_text}"
        ).strip()
    else:
        user_prompt = (
            "Generate a faithful abstract-style summary of the following scientific paper.\n"
            "Focus on:\n"
            "- research problem\n"
            "- proposed method\n"
            "- key findings\n"
            "- experimental results\n"
            "Do not invent numerical results.\n"
            "If a detail is not present in the paper, do not include it.\n\n"
            f"{prefix}Paper:\n{paper_text}"
        ).strip()
    return PromptBundle(system_prompt=system_prompt, user_prompt=user_prompt, style=style)


def bundle_to_messages(bundle: PromptBundle) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": bundle.system_prompt},
        {"role": "user", "content": bundle.user_prompt},
    ]

