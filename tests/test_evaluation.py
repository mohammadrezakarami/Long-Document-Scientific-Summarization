from scisum_qwen.evaluation.format_eval import evaluate_structured_format
from scisum_qwen.evaluation.length_analysis import compute_length_stats
from scisum_qwen.evaluation.rouge_bertscore import aggregate_rouge, rouge_l, rouge_n


def test_format_eval_detects_missing_fields() -> None:
    text = "TL;DR: x\nProblem: y\nMethod: z"
    evaluation = evaluate_structured_format(text)
    assert evaluation.format_valid == 0
    assert "Results" in evaluation.missing_fields


def test_length_stats_returns_positive_word_counts() -> None:
    stats = compute_length_stats(["short summary"], ["a slightly longer reference summary"])
    assert stats["avg_prediction_words"] > 0
    assert stats["avg_reference_words"] > stats["avg_prediction_words"]


def test_rouge_scores_are_bounded() -> None:
    rouge1 = rouge_n("the model improves scores", "the model improves results", 1)
    rouge_l_score = rouge_l("the model improves scores", "the model improves results")
    assert 0.0 <= rouge1["f1"] <= 1.0
    assert 0.0 <= rouge_l_score["f1"] <= 1.0


def test_aggregate_rouge_rewards_identical_predictions() -> None:
    scores = aggregate_rouge(["exact match"], ["exact match"])
    assert scores["rouge1"] == 1.0
    assert scores["rougeL"] == 1.0

