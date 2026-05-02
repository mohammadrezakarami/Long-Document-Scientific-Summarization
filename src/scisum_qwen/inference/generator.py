from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scisum_qwen.inference.decoding import DecodingConfig
from scisum_qwen.inference.prompts import PromptBundle, bundle_to_messages


@dataclass
class GenerationResult:
    text: str
    metadata: dict[str, Any]


class HuggingFaceGenerator:
    def __init__(
        self,
        *,
        model_name: str,
        adapter_path: str | None = None,
        device_map: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for LLM baseline generation.") from exc
        try:
            from peft import PeftModel
        except ImportError:
            PeftModel = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            model_device_map = "auto" if device_map == "auto" else device_map
            torch_dtype = torch.float16
            self.runtime_device = "cuda"
        elif torch.backends.mps.is_available():
            model_device_map = "mps"
            torch_dtype = torch.float16
            self.runtime_device = "mps"
        else:
            model_device_map = "cpu" if device_map == "auto" else device_map
            torch_dtype = torch.float32
            self.runtime_device = "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=model_device_map,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if adapter_path and Path(adapter_path).exists():
            if PeftModel is None:
                raise RuntimeError("peft is required to load a LoRA or QLoRA adapter.")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.model_device = model_device_map

    def generate_from_bundle(self, bundle: PromptBundle, decoding: DecodingConfig) -> GenerationResult:
        prompt_text = self.tokenizer.apply_chat_template(
            bundle_to_messages(bundle),
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(prompt_text, return_tensors="pt")
        if self.runtime_device in {"cuda", "mps"}:
            model_inputs = {key: value.to(self.runtime_device) for key, value in model_inputs.items()}
        generation_kwargs = {
            "max_new_tokens": decoding.max_new_tokens,
            "repetition_penalty": decoding.repetition_penalty,
            "do_sample": decoding.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if decoding.do_sample:
            generation_kwargs["temperature"] = decoding.temperature
            generation_kwargs["top_p"] = decoding.top_p

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for LLM generation.") from exc

        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                **generation_kwargs,
            )
        prompt_length = model_inputs["input_ids"].shape[-1]
        generated_tokens = generated[0][prompt_length:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            metadata={
                "model_name": self.model_name,
                "adapter_path": self.adapter_path,
                "device": self.model_device,
                "prompt_style": bundle.style,
                "prompt_length_tokens": prompt_length,
            },
        )
