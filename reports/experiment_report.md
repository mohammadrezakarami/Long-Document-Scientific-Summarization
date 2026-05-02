# Experiment Report

## Model Comparison

| Run | Method | Prompt Style | Inference Mode | Context | Evidence Prompt | Count | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | Format Validity | Field Completion |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_textrank | textrank |  |  |  |  | 2 | 0.3591 | 0.1688 | 0.3364 | None | 0.0 | 0.0833 |

## Highlights

- Best ROUGE-L so far: `baseline_textrank` with `0.3364`.
- Best structured format completion so far: `baseline_textrank` with `0.0833`.
- Only one run is available so far; add zero-shot, prompted, and QLoRA outputs for a fuller comparison.
