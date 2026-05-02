# Training Log

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Load in 4-bit: `True`
- LoRA rank: `16`
- LoRA alpha: `32`
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Max sequence length: `4096`
- Dry run: `True`

## Dataset Stats

- Train examples: `2`
- Train dropped after masking: `0`
- Train mean sequence length: `106.0`
- Validation examples: `2`
- Validation dropped after masking: `0`
- Validation mean sequence length: `106.0`
