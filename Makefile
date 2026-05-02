PYTHON ?= .venv/bin/python
SCISUM_API_HOST ?= 127.0.0.1
SCISUM_API_PORT ?= 8000
SCISUM_DEMO_HOST ?= 127.0.0.1
SCISUM_DEMO_PORT ?= 7860

.PHONY: test preprocess baseline train evaluate compare errors longdoc evidence api demo docker-api docker-demo build-colab-subset package-colab smoke-api install-adapter deploy-space test-public-space

test:
	$(PYTHON) -m pytest

preprocess:
	PYTHONPATH=src $(PYTHON) -m scisum_qwen.data.build_sft_dataset --config configs/data.yaml

baseline:
	PYTHONPATH=src $(PYTHON) -m scisum_qwen.evaluation.run_baselines --input data/samples/sample_processed.jsonl --output-dir reports --skip-llm

train:
	PYTHONPATH=src $(PYTHON) -m scisum_qwen.training.train_qlora --config configs/train_qlora.yaml --dry-run --skip-model-load --train-file data/samples/sample_processed.jsonl --validation-file data/samples/sample_processed.jsonl

evaluate:
	PYTHONPATH=src $(PYTHON) -m scisum_qwen.evaluation.run_eval --input-dir reports --output-csv reports/baseline_metrics.csv --output-md reports/baseline_eval.md --manual-review-csv reports/manual_review_template.csv

compare:
	PYTHONPATH=src $(PYTHON) -m scisum_qwen.evaluation.compare_models --input-dir reports --pattern "*_outputs.jsonl" --output-csv reports/model_comparison.csv --output-md reports/experiment_report.md

errors:
	PYTHONPATH=src $(PYTHON) -m scisum_qwen.evaluation.error_analysis --input-dir reports --pattern "*_outputs.jsonl" --output-md reports/error_analysis.md

longdoc:
	PYTHONPATH=src $(PYTHON) -m scisum_qwen.evaluation.long_document_eval --input data/samples/sample_processed.jsonl --output-dir reports/longdoc --summary-type abstract

evidence:
	PYTHONPATH=src $(PYTHON) -m scisum_qwen.evaluation.evidence_support_eval --predictions reports/longdoc/hierarchical_outputs.jsonl --output-jsonl reports/evidence_support.jsonl --output-md reports/evidence_examples.md

build-colab-subset:
	PYTHONPATH=src $(PYTHON) scripts/build_colab_subset.py

package-colab: build-colab-subset
	PYTHONPATH=src $(PYTHON) scripts/package_colab.py

smoke-api:
	PYTHONPATH=src $(PYTHON) scripts/smoke_test_api.py

install-adapter:
	@echo "Usage: $(PYTHON) scripts/install_adapter.py /path/to/qwen-arxiv-qlora-colab --force"

deploy-space:
	@echo "Usage: HF_TOKEN=... $(PYTHON) scripts/deploy_hf_space.py --repo-id username/scisum-qwen"

test-public-space:
	PYTHONPATH=src $(PYTHON) scripts/test_public_space.py

api:
	PYTHONPATH=src SCISUM_API_HOST=$(SCISUM_API_HOST) SCISUM_API_PORT=$(SCISUM_API_PORT) $(PYTHON) -m uvicorn scisum_qwen.api.main:app --reload --host $(SCISUM_API_HOST) --port $(SCISUM_API_PORT)

demo:
	PYTHONPATH=src SCISUM_DEMO_HOST=$(SCISUM_DEMO_HOST) SCISUM_DEMO_PORT=$(SCISUM_DEMO_PORT) $(PYTHON) app/gradio_app.py

docker-api:
	docker compose up --build scisum-api

docker-demo:
	docker compose up --build scisum-demo
