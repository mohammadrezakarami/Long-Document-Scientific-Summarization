from __future__ import annotations

import argparse
import random
from pathlib import Path

from scisum_qwen.utils.io import read_jsonl, write_jsonl


def sample_records(input_path: str | Path, sample_size: int, seed: int) -> list[dict]:
    records = read_jsonl(input_path)
    if sample_size <= 0 or sample_size >= len(records):
        return records
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), sample_size))
    return [records[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build small Colab-ready JSONL subsets from processed splits.")
    parser.add_argument("--train-input", default="data/processed/train.jsonl")
    parser.add_argument("--valid-input", default="data/processed/valid.jsonl")
    parser.add_argument("--test-input", default="data/processed/test.jsonl")
    parser.add_argument("--output-dir", default="data/colab")
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--valid-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = sample_records(args.train_input, args.train_size, args.seed)
    valid_records = sample_records(args.valid_input, args.valid_size, args.seed + 1)
    test_records = sample_records(args.test_input, args.test_size, args.seed + 2)

    write_jsonl(output_dir / "train_subset.jsonl", train_records)
    write_jsonl(output_dir / "valid_subset.jsonl", valid_records)
    write_jsonl(output_dir / "test_subset.jsonl", test_records)

    summary = [
        "# Colab Subset Report",
        "",
        f"- Train subset: `{len(train_records)}`",
        f"- Validation subset: `{len(valid_records)}`",
        f"- Test subset: `{len(test_records)}`",
        f"- Seed: `{args.seed}`",
    ]
    (output_dir / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
