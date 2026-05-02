from __future__ import annotations

import os
import runpy
from pathlib import Path
import sys


if __name__ == "__main__":
    src_path = Path(__file__).resolve().parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    app_path = Path(__file__).resolve().parent / "app" / "gradio_app.py"
    os.environ.setdefault("SCISUM_DEMO_HOST", "0.0.0.0")
    os.environ.setdefault("SCISUM_DEMO_PORT", os.getenv("PORT", "7860"))
    runpy.run_path(str(app_path), run_name="__main__")
