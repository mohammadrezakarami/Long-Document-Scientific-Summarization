from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile


DEFAULT_INCLUDES = [
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
    ".env.example",
    "src",
    "app",
    "configs",
    "tests",
    "reports",
    "data/colab",
    "data/samples",
    "scripts",
    "colab",
]


def should_skip(path: Path) -> bool:
    if path.name == ".DS_Store":
        return True
    if path.suffix == ".pyc":
        return True
    if "__pycache__" in path.parts:
        return True
    if ".pytest_cache" in path.parts:
        return True
    return False


def iter_files(root: Path, include: str):
    target = root / include
    if target.is_file():
        if not should_skip(target):
            yield target
        return
    if target.is_dir():
        for path in sorted(target.rglob("*")):
            if path.is_file() and not should_skip(path):
                yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a lightweight Colab-ready project archive.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="dist/scisum-qwen-colab-ready.zip")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for include in DEFAULT_INCLUDES:
            for path in iter_files(root, include):
                zf.write(path, path.relative_to(root).as_posix())

    notebook_source = root / "colab" / "scisum_qwen_colab.ipynb"
    notebook_copy = output.parent / "scisum_qwen_colab.ipynb"
    shutil.copy2(notebook_source, notebook_copy)

    print(output)


if __name__ == "__main__":
    main()
