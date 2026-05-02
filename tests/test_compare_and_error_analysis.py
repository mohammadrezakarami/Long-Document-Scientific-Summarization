from scisum_qwen.evaluation.compare_models import infer_run_tags, load_comparison_rows
from scisum_qwen.evaluation.error_analysis import detect_error_flags, primary_error
from scisum_qwen.utils.io import write_jsonl


def test_infer_run_tags_reads_filename_signals() -> None:
    tags = infer_run_tags("reports/qlora_hierarchical_8k_evidence_outputs.jsonl", [{"metadata": {}}])
    assert tags["context_window"] == "8k"
    assert tags["inference_mode"] == "hierarchical"
    assert tags["evidence_aware_prompt"] == "on"


def test_detect_error_flags_catches_section_confusion_and_missing_contribution() -> None:
    flags = detect_error_flags(
        {
            "generated_summary": "Results: The system improves performance.",
            "source_text": "The paper proposes a hierarchical summarization model and reports better performance.",
            "reference_summary": "We propose a hierarchical summarization model that improves performance.",
            "prompt_style": "",
        }
    )
    assert "section_confusion" in flags
    assert "missing_key_contribution" in flags
    assert primary_error(flags) in flags


def test_load_comparison_rows_reads_prediction_outputs(tmp_path) -> None:
    output_file = tmp_path / "sample_outputs.jsonl"
    write_jsonl(
        output_file,
        [
            {
                "paper_id": "paper_1",
                "method": "textrank",
                "prompt_style": "",
                "generated_summary": "A short summary.",
                "reference_summary": "A short summary.",
                "source_text": "A short source.",
                "metadata": {},
            }
        ],
    )
    rows = load_comparison_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["rougeL"] == 1.0

