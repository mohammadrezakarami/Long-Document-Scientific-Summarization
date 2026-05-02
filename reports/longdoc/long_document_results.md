# Long-Document Results

| Inference Mode | Count | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg Pred Words | Avg Ref Words | Avg Compression | Format Validity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single_pass | 2 | 0.3591 | 0.1688 | 0.3364 | 17.5 | 22.0 | 0.7899 | 0.0 |
| hierarchical | 2 | 0.3846 | 0.1628 | 0.2291 | 43.5 | 22.0 | 1.9741 | 0.0 |

## Notes

- ROUGE-L delta (`hierarchical - single_pass`): `-0.1073`.
- Use these outputs as a lightweight proxy before running the heavier LLM-backed hierarchical path.
