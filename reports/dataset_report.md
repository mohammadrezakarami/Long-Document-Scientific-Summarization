# Dataset Report

This report summarizes the processed arXiv scientific summarization dataset.

| Split | Count | Avg Article Tokens | Avg Summary Tokens | Avg Compression Ratio | Filtered Samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 4745 | 7436.22 | 370.17 | 0.1235 | 255 |
| validation | 486 | 7678.47 | 234.7 | 0.0429 | 14 |
| test | 481 | 7529.06 | 239.82 | 0.0419 | 19 |

## Notes

- Preprocessing preserves scientific numbers and metric-heavy sentences whenever possible.
- Reference sections are removed conservatively using heading detection.
- Token counts are approximate until a model tokenizer is wired into the preprocessing stage.
