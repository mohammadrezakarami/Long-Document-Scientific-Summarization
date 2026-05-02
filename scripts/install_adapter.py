from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Install a trained QLoRA adapter into the local models directory.")
    parser.add_argument("source", help="Path to the downloaded adapter directory")
    parser.add_argument("--target", default="models/qwen-arxiv-qlora-colab", help="Destination path inside the repo")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing destination")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    target = Path(args.target).resolve()

    if not source.exists():
        raise SystemExit(f"Source adapter directory does not exist: {source}")
    if not source.is_dir():
        raise SystemExit(f"Source must be a directory: {source}")

    required = [
        source / "adapter_config.json",
        source / "adapter_model.safetensors",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required adapter files: {missing}")

    if target.exists():
        if not args.force:
            raise SystemExit(f"Target already exists: {target}. Use --force to replace it.")
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    print(target)


if __name__ == "__main__":
    main()
