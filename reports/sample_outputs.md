# Sample Outputs

## Baseline Extractive Output

- Source: `reports/baseline_outputs.md`
- Mode: `single_pass`
- Backend: `textrank`

### sample_0001

- Title: `Efficient Scientific Summarization`
- Generated:

```text
Results:
The proposed system improves ROUGE-L by 2.4 points over a prompt-only baseline and produces fewer unsupported claims.
```

- Reference:

```text
We present an adapter-based scientific summarization system that uses section-aware prompting to improve faithfulness and ROUGE-L over prompt-only baselines.
```

## Hierarchical Structured Output

- Source: `reports/longdoc/hierarchical_outputs.jsonl`
- Mode: `hierarchical`
- Backend: `section-aware heuristic synthesis`

### sample_0002

- Title: `Hierarchical Long-Document Summarization`
- Generated characteristics:
  - longer than the single-pass baseline
  - preserves multi-section context
  - suitable for downstream claim-level evidence scoring

## Evidence Support Example

- Source: `reports/evidence_examples.md`

### sample_0001 claim

- Claim: `The proposed system improves ROUGE-L by 2.4 points over a prompt-only baseline and produces fewer unsupported claims.`
- Label: `supported`
- Section: `results`
- Support score: `1.0`
