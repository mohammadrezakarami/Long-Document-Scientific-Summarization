from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scisum_qwen.training.lora_config import TrainConfig, build_peft_config, load_train_config
from scisum_qwen.utils.io import read_jsonl


@dataclass(frozen=True)
class TokenizedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    prompt_length: int
    total_length: int


class WhitespaceChatTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1

    def apply_chat_template(self, messages: list[dict[str, str]], *, tokenize: bool = False, add_generation_prompt: bool = False):
        del tokenize
        parts = [f"{message['role'].upper()}: {message['content']}" for message in messages]
        if add_generation_prompt:
            parts.append("ASSISTANT:")
        return "\n".join(parts)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        token_count = max(1, len(text.split()))
        return {"input_ids": list(range(1, token_count + 1))}


class CausalLMDataCollator:
    def __init__(self, *, pad_token_id: int, label_pad_token_id: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required to collate training batches.") from exc

        max_length = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch_attention_mask.append(feature["attention_mask"] + [0] * padding)
            batch_labels.append(feature["labels"] + [self.label_pad_token_id] * padding)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def ensure_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if messages:
        return messages

    source_text = record.get("source_text", "")
    target_summary = record.get("target_summary", "")
    title = record.get("title")
    title_block = f"Title: {title}\n\n" if title else ""
    return [
        {
            "role": "system",
            "content": "You are a scientific research assistant. Generate faithful and concise scientific summaries.",
        },
        {
            "role": "user",
            "content": (
                "Summarize the following scientific paper into an abstract-style summary.\n\n"
                f"{title_block}Paper:\n{source_text}"
            ).strip(),
        },
        {
            "role": "assistant",
            "content": target_summary,
        },
    ]


def render_messages_as_text(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def tokenize_sft_record(tokenizer: Any, record: dict[str, Any], *, max_seq_length: int) -> TokenizedExample | None:
    messages = ensure_messages(record)
    prompt_messages = messages[:-1]
    full_text = render_messages_as_text(tokenizer, messages, add_generation_prompt=False)
    prompt_text = render_messages_as_text(tokenizer, prompt_messages, add_generation_prompt=True)

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    truncated_ids = full_ids[:max_seq_length]
    prompt_length = min(len(prompt_ids), len(truncated_ids))
    labels = truncated_ids.copy()
    for index in range(prompt_length):
        labels[index] = -100

    if all(label == -100 for label in labels):
        return None

    return TokenizedExample(
        input_ids=truncated_ids,
        attention_mask=[1] * len(truncated_ids),
        labels=labels,
        prompt_length=prompt_length,
        total_length=len(truncated_ids),
    )


def prepare_tokenized_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenized_records: list[dict[str, Any]] = []
    dropped_no_target = 0

    for record in records:
        tokenized = tokenize_sft_record(tokenizer, record, max_seq_length=max_seq_length)
        if tokenized is None:
            dropped_no_target += 1
            continue
        tokenized_records.append(
            {
                "input_ids": tokenized.input_ids,
                "attention_mask": tokenized.attention_mask,
                "labels": tokenized.labels,
                "paper_id": record.get("paper_id"),
                "prompt_length": tokenized.prompt_length,
                "total_length": tokenized.total_length,
            }
        )

    mean_sequence_length = (
        round(sum(item["total_length"] for item in tokenized_records) / len(tokenized_records), 2)
        if tokenized_records
        else 0.0
    )
    stats = {
        "count": len(tokenized_records),
        "dropped_no_target": dropped_no_target,
        "mean_sequence_length": mean_sequence_length,
        "max_sequence_length": max((item["total_length"] for item in tokenized_records), default=0),
    }
    return tokenized_records, stats


def load_jsonl_records(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    if limit is None or limit <= 0:
        return records
    return records[:limit]


def _import_training_stack() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "Full QLoRA training requires torch, datasets, transformers, and peft to be installed."
        ) from exc

    return {
        "torch": torch,
        "Dataset": Dataset,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def load_tokenizer(model_name: str, *, allow_fallback: bool = False):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        if allow_fallback:
            return WhitespaceChatTokenizer()
        raise RuntimeError("transformers is required to load the training tokenizer.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_quantization_config(config: TrainConfig, torch_module: Any, bits_and_bytes_cls: Any):
    if not config.model.load_in_4bit:
        return None
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("bitsandbytes is required for 4-bit QLoRA loading.") from exc

    compute_dtype = torch_module.bfloat16 if config.training.bf16 else torch_module.float16
    return bits_and_bytes_cls(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_model_and_tokenizer(config: TrainConfig):
    stack = _import_training_stack()
    tokenizer = load_tokenizer(config.model.name)
    quantization_config = _build_quantization_config(config, stack["torch"], stack["BitsAndBytesConfig"])

    model = stack["AutoModelForCausalLM"].from_pretrained(
        config.model.name,
        device_map=config.model.device_map,
        quantization_config=quantization_config,
    )
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if config.model.load_in_4bit:
        model = stack["prepare_model_for_kbit_training"](
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
        )

    model = stack["get_peft_model"](model, build_peft_config(config.lora))
    return model, tokenizer, stack


def build_training_arguments(config: TrainConfig, training_arguments_cls: Any):
    import inspect

    kwargs = {
        "output_dir": config.training.output_dir,
        "num_train_epochs": config.training.num_train_epochs,
        "learning_rate": config.training.learning_rate,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "per_device_eval_batch_size": config.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "warmup_ratio": config.training.warmup_ratio,
        "weight_decay": config.training.weight_decay,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "logging_steps": config.training.logging_steps,
        "save_strategy": config.training.save_strategy,
        "save_total_limit": config.training.save_total_limit,
        "bf16": config.training.bf16,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "report_to": [],
        "seed": config.training.seed,
    }

    signature = inspect.signature(training_arguments_cls.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = config.training.evaluation_strategy
    else:
        kwargs["evaluation_strategy"] = config.training.evaluation_strategy

    return training_arguments_cls(**kwargs)


def write_training_report(path: str | Path, *, config: TrainConfig, train_stats: dict[str, Any], eval_stats: dict[str, Any], dry_run: bool, trainer_metrics: dict[str, Any] | None = None) -> None:
    lines = [
        "# Training Log",
        "",
        f"- Model: `{config.model.name}`",
        f"- Load in 4-bit: `{config.model.load_in_4bit}`",
        f"- LoRA rank: `{config.lora.r}`",
        f"- LoRA alpha: `{config.lora.alpha}`",
        f"- Target modules: `{', '.join(config.lora.target_modules)}`",
        f"- Max sequence length: `{config.training.max_seq_length}`",
        f"- Dry run: `{dry_run}`",
        "",
        "## Dataset Stats",
        "",
        f"- Train examples: `{train_stats['count']}`",
        f"- Train dropped after masking: `{train_stats['dropped_no_target']}`",
        f"- Train mean sequence length: `{train_stats['mean_sequence_length']}`",
        f"- Validation examples: `{eval_stats['count']}`",
        f"- Validation dropped after masking: `{eval_stats['dropped_no_target']}`",
        f"- Validation mean sequence length: `{eval_stats['mean_sequence_length']}`",
    ]
    if trainer_metrics:
        lines.extend(
            [
                "",
                "## Training Metrics",
                "",
            ]
        )
        for key, value in trainer_metrics.items():
            lines.append(f"- {key}: `{value}`")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training(
    config: TrainConfig,
    *,
    dry_run: bool = False,
    skip_model_load: bool = False,
    train_file_override: str | None = None,
    validation_file_override: str | None = None,
) -> dict[str, Any]:
    train_file = train_file_override or config.training.train_file
    validation_file = validation_file_override or config.training.validation_file

    train_records = load_jsonl_records(train_file, limit=config.training.max_train_samples)
    eval_records = load_jsonl_records(validation_file, limit=config.training.max_eval_samples)
    tokenizer = load_tokenizer(config.model.name, allow_fallback=dry_run or skip_model_load)
    tokenized_train, train_stats = prepare_tokenized_records(
        train_records,
        tokenizer,
        max_seq_length=config.training.max_seq_length,
    )
    tokenized_eval, eval_stats = prepare_tokenized_records(
        eval_records,
        tokenizer,
        max_seq_length=config.training.max_seq_length,
    )

    if dry_run or skip_model_load:
        write_training_report(
            config.training.report_path,
            config=config,
            train_stats=train_stats,
            eval_stats=eval_stats,
            dry_run=True,
        )
        return {
            "mode": "dry_run",
            "train_stats": train_stats,
            "eval_stats": eval_stats,
        }

    model, tokenizer, stack = build_model_and_tokenizer(config)
    train_dataset = stack["Dataset"].from_list(tokenized_train)
    eval_dataset = stack["Dataset"].from_list(tokenized_eval)
    training_arguments = build_training_arguments(config, stack["TrainingArguments"])
    collator = CausalLMDataCollator(pad_token_id=tokenizer.pad_token_id)

    import inspect

    trainer_kwargs = {
        "model": model,
        "args": training_arguments,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
    trainer_signature = inspect.signature(stack["Trainer"].__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = stack["Trainer"](**trainer_kwargs)
    train_result = trainer.train()
    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)
    metrics = dict(train_result.metrics)
    write_training_report(
        config.training.report_path,
        config=config,
        train_stats=train_stats,
        eval_stats=eval_stats,
        dry_run=False,
        trainer_metrics=metrics,
    )
    return {
        "mode": "train",
        "train_stats": train_stats,
        "eval_stats": eval_stats,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen with QLoRA on processed summarization data.")
    parser.add_argument("--config", type=str, default="configs/train_qlora.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-model-load", action="store_true")
    parser.add_argument("--train-file", type=str, default=None)
    parser.add_argument("--validation-file", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config)
    run_training(
        config,
        dry_run=args.dry_run,
        skip_model_load=args.skip_model_load,
        train_file_override=args.train_file,
        validation_file_override=args.validation_file,
    )


if __name__ == "__main__":
    main()
