from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelLoadConfig:
    name: str
    load_in_4bit: bool = True
    device_map: str = "auto"


@dataclass(frozen=True)
class LoRAConfigSpec:
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(frozen=True)
class TrainingRunConfig:
    train_file: str
    validation_file: str
    output_dir: str
    report_path: str = "reports/training_log.md"
    num_train_epochs: int = 2
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 4096
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
    model: ModelLoadConfig
    lora: LoRAConfigSpec
    training: TrainingRunConfig


def load_train_config(config_path: str | Path) -> TrainConfig:
    import yaml

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model = ModelLoadConfig(
        name=raw["model"]["name"],
        load_in_4bit=bool(raw["model"].get("load_in_4bit", True)),
        device_map=raw["model"].get("device_map", "auto"),
    )
    lora = LoRAConfigSpec(
        r=int(raw["lora"]["r"]),
        alpha=int(raw["lora"]["alpha"]),
        dropout=float(raw["lora"]["dropout"]),
        target_modules=tuple(raw["lora"]["target_modules"]),
        bias=raw["lora"].get("bias", "none"),
        task_type=raw["lora"].get("task_type", "CAUSAL_LM"),
    )
    training = TrainingRunConfig(
        train_file=raw["training"]["train_file"],
        validation_file=raw["training"]["validation_file"],
        output_dir=raw["training"]["output_dir"],
        report_path=raw["training"].get("report_path", "reports/training_log.md"),
        num_train_epochs=int(raw["training"].get("num_train_epochs", 2)),
        learning_rate=float(raw["training"].get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(raw["training"].get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(raw["training"].get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(raw["training"].get("gradient_accumulation_steps", 8)),
        max_seq_length=int(raw["training"].get("max_seq_length", 4096)),
        warmup_ratio=float(raw["training"].get("warmup_ratio", 0.03)),
        weight_decay=float(raw["training"].get("weight_decay", 0.01)),
        lr_scheduler_type=raw["training"].get("lr_scheduler_type", "cosine"),
        bf16=bool(raw["training"].get("bf16", True)),
        gradient_checkpointing=bool(raw["training"].get("gradient_checkpointing", True)),
        logging_steps=int(raw["training"].get("logging_steps", 10)),
        evaluation_strategy=raw["training"].get("evaluation_strategy", "epoch"),
        save_strategy=raw["training"].get("save_strategy", "epoch"),
        save_total_limit=int(raw["training"].get("save_total_limit", 2)),
        max_train_samples=raw["training"].get("max_train_samples"),
        max_eval_samples=raw["training"].get("max_eval_samples"),
        seed=int(raw["training"].get("seed", 42)),
    )
    return TrainConfig(model=model, lora=lora, training=training)


def build_peft_config(spec: LoRAConfigSpec):
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise RuntimeError("peft is required to build the LoRA configuration.") from exc

    return LoraConfig(
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        bias=spec.bias,
        task_type=spec.task_type,
        target_modules=list(spec.target_modules),
    )

