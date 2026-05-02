from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodingConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    do_sample: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> "DecodingConfig":
        return cls(
            max_new_tokens=int(raw.get("max_new_tokens", 512)),
            temperature=float(raw.get("temperature", 0.0)),
            top_p=float(raw.get("top_p", 0.9)),
            repetition_penalty=float(raw.get("repetition_penalty", 1.05)),
            do_sample=bool(raw.get("do_sample", False)),
        )

