from __future__ import annotations

import argparse
import json


def _sample_payload() -> dict:
    return {
        "title": "Sample Paper",
        "paper_text": (
            "Introduction:\nThis paper studies scientific summarization.\n\n"
            "Method:\nThe method uses section-aware summarization and hierarchical inference.\n\n"
            "Results:\nIt improves ROUGE-L by 2.4 points over the baseline.\n\n"
            "Conclusion:\nThe approach is practical for long documents."
        ),
        "mode": "hierarchical",
        "summary_type": "structured",
        "enable_evidence_support": True,
    }


def run_local_smoke() -> None:
    from fastapi.testclient import TestClient

    from scisum_qwen.api.main import app

    client = TestClient(app)
    health = client.get("/health")
    model_info = client.get("/model-info")
    summarize = client.post("/summarize", json=_sample_payload())

    print("health:", health.status_code, health.json())
    print("model-info:", model_info.status_code, model_info.json())
    print("summarize:", summarize.status_code)
    print(json.dumps(summarize.json(), indent=2)[:1600])


def run_remote_smoke(base_url: str) -> None:
    import requests

    health = requests.get(f"{base_url}/health", timeout=60)
    model_info = requests.get(f"{base_url}/model-info", timeout=60)
    summarize = requests.post(f"{base_url}/summarize", json=_sample_payload(), timeout=300)

    print("health:", health.status_code, health.json())
    print("model-info:", model_info.status_code, model_info.json())
    print("summarize:", summarize.status_code)
    print(json.dumps(summarize.json(), indent=2)[:1600])


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the SciSum-Qwen API locally or remotely.")
    parser.add_argument("--base-url", default="")
    args = parser.parse_args()

    if args.base_url:
        run_remote_smoke(args.base_url.rstrip("/"))
    else:
        run_local_smoke()


if __name__ == "__main__":
    main()
