import pytest

from scisum_qwen.training.train_qlora import CausalLMDataCollator, prepare_tokenized_records, tokenize_sft_record


class FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for message in messages:
            parts.append(f"{message['role'].upper()}: {message['content']}")
        if add_generation_prompt:
            parts.append("ASSISTANT:")
        return "\n".join(parts)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        token_count = max(1, len(text.split()))
        return {"input_ids": list(range(1, token_count + 1))}


def test_tokenize_sft_record_masks_prompt_tokens() -> None:
    tokenizer = FakeTokenizer()
    record = {
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Prompt content"},
            {"role": "assistant", "content": "Target summary"},
        ]
    }
    tokenized = tokenize_sft_record(tokenizer, record, max_seq_length=64)
    assert tokenized is not None
    assert tokenized.prompt_length > 0
    assert all(label == -100 for label in tokenized.labels[: tokenized.prompt_length])
    assert any(label != -100 for label in tokenized.labels[tokenized.prompt_length :])


def test_prepare_tokenized_records_returns_stats() -> None:
    tokenizer = FakeTokenizer()
    records = [
        {
            "paper_id": "paper_1",
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Prompt"},
                {"role": "assistant", "content": "Summary"},
            ],
        }
    ]
    tokenized_records, stats = prepare_tokenized_records(records, tokenizer, max_seq_length=64)
    assert len(tokenized_records) == 1
    assert stats["count"] == 1
    assert stats["mean_sequence_length"] > 0


def test_causal_lm_data_collator_pads_inputs() -> None:
    torch = pytest.importorskip("torch")
    assert torch is not None
    collator = CausalLMDataCollator(pad_token_id=0)
    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [10, 11]},
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [10, 11, 12]},
        ]
    )
    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert batch["input_ids"][0, -1].item() == 0
    assert batch["labels"][0, -1].item() == -100
