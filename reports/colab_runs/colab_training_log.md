# Training Log

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Load in 4-bit: `True`
- LoRA rank: `8`
- LoRA alpha: `16`
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Max sequence length: `2048`
- Dry run: `False`

## Dataset Stats

- Train examples: `16`
- Train dropped after masking: `284`
- Train mean sequence length: `1818.69`
- Validation examples: `2`
- Validation dropped after masking: `48`
- Validation mean sequence length: `1453.0`

## Training Metrics

- train_runtime: `97.3213`
- train_samples_per_second: `0.164`
- train_steps_per_second: `0.041`
- total_flos: `487071960539136.0`
- train_loss: `2.6168301105499268`
- epoch: `1.0`
