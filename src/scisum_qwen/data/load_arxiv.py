from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    article_field: str
    abstract_field: str
    title_field: str | None
    paper_id_field: str | None
    cache_dir: str | None
    use_hf_splits: bool
    streaming: bool
    subset_limits: dict[str, int]


@dataclass(frozen=True)
class FormattingConfig:
    include_title: bool
    include_section_headers: bool
    keep_sections: tuple[str, ...]
    system_prompt: str
    user_instruction: str


@dataclass(frozen=True)
class TokenizationConfig:
    tokenizer_name: str | None
    use_fast: bool
    fallback_chars_per_token: float


@dataclass(frozen=True)
class OutputConfig:
    processed_dir: str
    report_path: str
    split_filenames: dict[str, str]


@dataclass(frozen=True)
class DataConfig:
    dataset: DatasetConfig
    preprocessing: dict[str, Any]
    formatting: FormattingConfig
    tokenization: TokenizationConfig
    outputs: OutputConfig


def load_data_config(config_path: str | Path) -> DataConfig:
    import yaml

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    dataset = DatasetConfig(
        name=raw["dataset"]["name"],
        article_field=raw["dataset"]["article_field"],
        abstract_field=raw["dataset"]["abstract_field"],
        title_field=raw["dataset"].get("title_field") or None,
        paper_id_field=raw["dataset"].get("paper_id_field") or None,
        cache_dir=raw["dataset"].get("cache_dir") or None,
        use_hf_splits=bool(raw["dataset"].get("use_hf_splits", True)),
        streaming=bool(raw["dataset"].get("streaming", False)),
        subset_limits={key: int(value) for key, value in raw["dataset"].get("subset_limits", {}).items()},
    )
    formatting = FormattingConfig(
        include_title=bool(raw["formatting"].get("include_title", True)),
        include_section_headers=bool(raw["formatting"].get("include_section_headers", True)),
        keep_sections=tuple(raw["formatting"].get("keep_sections", [])),
        system_prompt=raw["formatting"]["system_prompt"],
        user_instruction=raw["formatting"]["user_instruction"],
    )
    tokenization = TokenizationConfig(
        tokenizer_name=raw["tokenization"].get("tokenizer_name") or None,
        use_fast=bool(raw["tokenization"].get("use_fast", True)),
        fallback_chars_per_token=float(raw["tokenization"].get("fallback_chars_per_token", 4.0)),
    )
    outputs = OutputConfig(
        processed_dir=raw["outputs"]["processed_dir"],
        report_path=raw["outputs"]["report_path"],
        split_filenames=dict(raw["outputs"].get("split_filenames", {})),
    )
    return DataConfig(
        dataset=dataset,
        preprocessing=dict(raw["preprocessing"]),
        formatting=formatting,
        tokenization=tokenization,
        outputs=outputs,
    )


def load_arxiv_dataset(config: DatasetConfig):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load the arXiv summarization corpus.") from exc

    dataset = load_dataset(config.name, cache_dir=config.cache_dir, streaming=config.streaming)
    if config.use_hf_splits:
        return dataset
    return dataset["train"]


def load_arxiv_split(config: DatasetConfig, split_name: str):
    dataset = load_arxiv_dataset(config)
    return dataset[split_name]


def limit_split(dataset_split, limit: int | None):
    if limit is None or limit <= 0:
        return dataset_split
    if hasattr(dataset_split, "take"):
        return dataset_split.take(limit)
    if len(dataset_split) <= limit:
        return dataset_split
    return dataset_split.select(range(limit))


def iter_split_records(dataset_split):
    if hasattr(dataset_split, "__iter__"):
        yield from dataset_split
        return
    raise TypeError("The provided dataset split is not iterable.")


def raw_record_to_example(raw_record: dict[str, Any], config: DatasetConfig, index: int) -> dict[str, Any]:
    return {
        "paper_id": raw_record.get(config.paper_id_field) if config.paper_id_field else f"paper_{index:07d}",
        "title": raw_record.get(config.title_field) if config.title_field else None,
        "article": raw_record[config.article_field],
        "abstract": raw_record[config.abstract_field],
    }
