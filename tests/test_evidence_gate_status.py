import json
from pathlib import Path

from scripts import evidence_gate_status


def test_current_repo_report_keeps_remaining_external_gates_incomplete() -> None:
    report = evidence_gate_status.build_report()

    assert report["status"] == "incomplete"
    assert report["ready_for_badge_claims"] is False
    assert report["gates"]["claim_audit"]["passed"] is True
    assert report["gates"]["hosted_omni_eval"]["passed"] is True
    assert report["gates"]["local_4b_50_case_eval"]["passed"] is False
    assert report["gates"]["no_cloud_route"]["passed"] is False
    assert report["gates"]["llama_champion_route"]["passed"] is False
    assert report["gates"]["local_asr_provider_proof"]["passed"] is False
    assert report["gates"]["trained_responder_user_test"]["passed"] is False
    assert "local_4b_50_case_eval" in report["missing_gate_keys"]
    assert "no_cloud_route" in report["missing_gate_keys"]
    assert "llama_champion_route" in report["missing_gate_keys"]
    assert "local_asr_provider_proof" in report["missing_gate_keys"]
    assert "local full-weight endpoint" in report["gates"]["local_4b_50_case_eval"]["next_action"]
    assert "real local Parakeet provider payload" in report["gates"]["local_asr_provider_proof"]["next_action"]


def test_report_uses_local_evidence_artifacts_when_present(tmp_path: Path) -> None:
    local_4b_dir = tmp_path / "traces/local_4b_evidence_20260607T000000Z"
    local_4b_dir.mkdir(parents=True)
    (local_4b_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "eval_completed",
                "total_cases": 50,
                "counts_as_no_cloud_route_proof": True,
                "counts_as_50_case_local_llm_competence": True,
            }
        ),
        encoding="utf-8",
    )
    (local_4b_dir / "eval_evidence_manifest.json").write_text("{}", encoding="utf-8")
    asr_dir = tmp_path / "traces/local_asr_parakeet_evidence_20260607T000000Z"
    asr_dir.mkdir(parents=True)
    (asr_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "local_asr_proof_passed",
                "counts_as_local_asr_artifact": True,
                "counts_as_local_asr_proof": True,
            }
        ),
        encoding="utf-8",
    )
    (asr_dir / "asr_evidence_manifest.json").write_text("{}", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "submission_checklist.md").write_text(
        "\n".join(
            [
                "| Demo video | Proof needed | Pending |",
                "| Social post | Proof needed | Pending |",
            ]
        ),
        encoding="utf-8",
    )
    (docs / "user_test_notes.md").write_text(
        "Status: template only. No completed user-test results are recorded here yet.",
        encoding="utf-8",
    )
    (docs / "model_parameter_evidence_ledger.md").write_text(
        "4B Figment adapter stretch: Not trained, published, or measured in this ledger.",
        encoding="utf-8",
    )

    report = evidence_gate_status.build_report(tmp_path)

    assert report["gates"]["local_4b_50_case_eval"]["passed"] is True
    assert report["gates"]["no_cloud_route"]["passed"] is True
    assert report["gates"]["llama_champion_route"]["passed"] is True
    assert report["gates"]["local_asr_provider_proof"]["passed"] is True
    assert str(local_4b_dir / "eval_evidence_manifest.json") in report["gates"]["local_4b_50_case_eval"]["evidence_paths"]
    assert str(asr_dir / "asr_evidence_manifest.json") in report["gates"]["local_asr_provider_proof"]["evidence_paths"]
    assert report["status"] == "incomplete"
    assert "trained_responder_user_test" in report["missing_gate_keys"]
