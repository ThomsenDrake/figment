import json
from pathlib import Path

from figment.config import FigmentConfig
from scripts import run_eval


INITIAL_CASES = Path("data/eval/initial_handwritten_cases.jsonl")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_canned_eval_runner_keeps_fallback_out_of_model_competence(tmp_path: Path) -> None:
    output_path = tmp_path / "eval-results.jsonl"

    summary = run_eval.run_eval(
        case_paths=[INITIAL_CASES],
        output_path=output_path,
        config=FigmentConfig(model_backend="canned"),
    )

    records = _jsonl(output_path)
    assert summary["total_cases"] == 10
    assert len(records) == 10
    assert summary["raw_configured_model_successes"] == 0
    assert summary["repair_successes"] == 0
    assert summary["canned_fallback_successes"] == 10
    assert summary["competence_successes"] == 0
    assert summary["final_validation_successes"] == 10
    assert "expected_label_successes" in summary
    assert "expected_label_check_successes" in summary

    first = records[0]
    assert first["case_id"] == "initial-ams-confusion-001"
    assert first["model_backend"] == "canned"
    assert first["model_stack"] == "omni_native"
    assert first["active_model_id"]
    assert first["fallback_tier"] == "canned"
    assert first["fallback_reason"] == "canned_backend"
    assert first["raw_configured_model_attempted"] is False
    assert first["raw_configured_model_success"] is False
    assert first["repair_attempted"] is False
    assert first["repair_success"] is False
    assert first["canned_fallback_used"] is True
    assert first["canned_fallback_success"] is True
    assert first["competence_success"] is False
    assert first["final_validation"]["passed"] is True
    assert first["expected_source_card_ids"] == [
        "AMS-RED-FLAGS-v1",
        "SAFETY-BOUNDARIES-v1",
        "REFERRAL-SBAR-v1",
    ]
    assert first["expected_missing_observations"]
    assert first["forbidden_behavior"]
    assert first["actual_protocol_urgency"] == first["final_output"]["protocol_urgency"]
    assert first["actual_source_card_ids"] == first["final_output"]["source_cards"]
    assert "expected_label_score" in first
    assert first["expected_label_score"]["red_flags_match"] is True
    assert first["expected_label_score"]["min_urgency_met"] is True
    assert first["field_provenance"]["protocol_urgency"] == "deterministic_fallback"
    assert summary["records_with_field_provenance"] == 10
    assert summary["model_field_pass_rate"] == 0.0
    assert summary["model_visible_fields_retained"] == 0.0
    assert summary["deterministic_patch_count"] == len(first["field_provenance"]) * 10
    assert first["latency_ms"] >= 0
    assert isinstance(first["trace_hash"], str)
    assert len(first["trace_hash"]) >= 12
    assert first["raw_model_output"] is None
    assert first["repaired_output"] is None
    assert isinstance(first["fallback_output"], dict)


def test_eval_cli_runs_initial_cases_against_canned_without_network(tmp_path: Path) -> None:
    output_path = tmp_path / "cli-results.jsonl"

    exit_code = run_eval.main(
        [
            "--backend",
            "canned",
            "--cases",
            str(INITIAL_CASES),
            "--output",
            str(output_path),
        ]
    )

    records = _jsonl(output_path)
    assert exit_code == 0
    assert len(records) == 10
    assert {record["raw_configured_model_success"] for record in records} == {False}
    assert {record["canned_fallback_used"] for record in records} == {True}
    assert {record["final_validation"]["passed"] for record in records} == {True}
    assert {record["field_provenance"]["source_cards"] for record in records} == {"deterministic_fallback"}
    assert all("expected_label_score" in record for record in records)


def test_llama_eval_summary_describes_real_eval_evidence_scope(tmp_path: Path) -> None:
    summary = run_eval._summarize(
        [
            {
                "raw_configured_model_success": True,
                "repair_success": False,
                "canned_fallback_used": False,
                "canned_fallback_success": False,
                "competence_success": True,
                "final_validation": {"passed": True},
            }
        ],
        FigmentConfig(model_backend="llama_cpp", model_stack="local_4b_parakeet"),
        [INITIAL_CASES],
        tmp_path / "local-eval.jsonl",
    )

    assert summary["local_llm_evidence"]["proof_status"] == "eval_records_summarized"
    assert summary["local_llm_evidence"]["model_backend"] == "llama_cpp"
    assert summary["local_llm_evidence"]["counts_as_50_case_local_llm_competence"] is False
    assert summary["local_llm_evidence"]["competence_successes"] == 1
    assert "MODEL_BACKEND=llama_cpp" in summary["local_llm_evidence"]["real_eval_command"]
