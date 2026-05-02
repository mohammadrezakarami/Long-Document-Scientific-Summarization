# Experiment Report

## Model Comparison

| Run | Method | Prompt Style | Inference Mode | Context | Evidence Prompt | Count | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | Format Validity | Field Completion |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_qlora | qwen_qlora | faithful_abstract |  |  |  | 10 | 0.3265 | 0.0993 | 0.191 | None | 0.0 | 0.0 |

## Highlights

- Best ROUGE-L so far: `baseline_qlora` with `0.191`.
- Best structured format completion so far: `baseline_qlora` with `0.0`.
- Only one run is available so far; add zero-shot, prompted, and QLoRA outputs for a fuller comparison.
