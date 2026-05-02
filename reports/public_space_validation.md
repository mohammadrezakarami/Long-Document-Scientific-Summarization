# Public Space Validation

Public deployment target:

- Space repo: `mokarami/scisum-qwen`
- App URL: `https://mokarami-scisum-qwen.hf.space`

Final public checks completed:

1. Space repository created successfully on Hugging Face Spaces.
2. Runtime errors fixed:
   - `ModuleNotFoundError: scisum_qwen`
   - Gradio `show_api` compatibility mismatch
3. Public app returned `HTTP 200`.
4. Public Gradio API exposed `/run_demo`.
5. End-to-end prediction test succeeded against the deployed Space.

Public prediction snapshot:

- `paper_id`: `40ba8ecc8f69c561`
- `overall_score`: `0.75`
- `evidence_rows`: `8`

Summary preview:

```text
TL;DR: The method uses section-aware summarization and hierarchical inference.
The approach is practical for long documents.

Problem: This paper studies scientific summarization.
Method: The method uses section-aware summarization and hierarchical inference.
Key Contributions:
1. It improves ROUGE-L by 2.4 points over the baseline.
Results: It improves ROUGE-L by 2.4 points over the baseline.
Limitations: Not specified
```

This validation confirms that the deployed public demo is not a static mock. It runs the deployed summarization workflow and returns structured summaries with evidence support outputs.
