from __future__ import annotations

import argparse
from pathlib import Path


IGNORE_PATTERNS = [
    ".git/*",
    ".venv/*",
    ".venv_py314_backup/*",
    ".DS_Store",
    "**/.DS_Store",
    "__pycache__/*",
    "*.pyc",
    ".pytest_cache/*",
    "data/processed/*",
    "data/raw/*",
    "data/colab/*",
    "dist/*",
    "reports/real_runs/*",
    "reports/longdoc/*",
    "models/*/checkpoint-*",
    "models/*/checkpoint-*/*",
    "models/*/optimizer.pt",
    "models/*/scheduler.pt",
    "models/*/rng_state.pth",
    "models/*/training_args.bin",
    "models/*/README.md",
    "models/*/tokenizer.json",
    "models/*/tokenizer_config.json",
    "models/*/chat_template.jinja",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or update a Hugging Face Gradio Space from this repo.")
    parser.add_argument("--repo-id", required=True, help="Target repo id, e.g. username/scisum-qwen")
    parser.add_argument("--space-name", default="", help="Optional human-friendly Space title")
    parser.add_argument("--private", action="store_true", help="Create the Space as private")
    parser.add_argument("--root", default=".", help="Project root to upload")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required for deployment.") from exc

    root = Path(args.root).resolve()
    api = HfApi()

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=str(root),
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Deploy SciSum-Qwen Space",
    )

    title = args.space_name or args.repo_id
    print(f"Deployed {title}: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
