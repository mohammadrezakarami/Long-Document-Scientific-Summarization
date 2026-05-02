# Model Card

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Adaptation: `QLoRA adapter`
- Task: scientific paper summarization
- Training target: `paper/sections -> abstract`
- Intended use: research paper triage and summarization
- Intended outputs: abstract-style summaries, structured summaries, and claim-level evidence support reports
- Training data: `ccdv/arxiv-summarization` as the main corpus, with `SciTLDR` reserved for complementary evaluation
- Not intended for: replacing expert peer review, unsupported scientific claims, or medical/legal decision-making
- Limitation: evidence support scores are not formal verification
- Limitation: local lightweight inference currently uses extractive and heuristic components for the product layer
- Future work: full QLoRA-backed local serving, numeric consistency checking, and stronger verification models
