import json
from pathlib import Path

from scripts.export_v4_training_seeds import export_v4_training_seeds


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_exports_failed_holdout_rows_as_non_direct_v4_seeds(tmp_path: Path) -> None:
    cases = tmp_path / "field_workflow_holdout_v1.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    output = tmp_path / "seeds.jsonl"
    _write_jsonl(
        cases,
        [
            {
                "case_id": "holdout-1",
                "dataset_version": "field_workflow_holdout_v1",
                "workflow_category": "radio_handoff",
                "structured_intake": {"chief_concern": "radio handoff", "confirmed": True},
                "target_protocol_card_id": "REFERRAL-SBAR-v1",
                "expected_red_flag_rule_ids": ["RED-1"],
                "expected_min_protocol_urgency": "urgent",
                "expected_source_card_ids": ["REFERRAL-SBAR-v1"],
                "expected_candidate_pathway_card_ids": ["REFERRAL-SBAR-v1"],
            }
        ],
    )
    _write_jsonl(
        eval_path,
        [
            {
                "case_id": "holdout-1",
                "case_path": str(cases),
                "case_line": 1,
                "target_protocol_card_id": "REFERRAL-SBAR-v1",
                "expected_red_flag_rule_ids": ["RED-1"],
                "actual_red_flag_rule_ids": ["RED-1"],
                "expected_min_protocol_urgency": "urgent",
                "expected_source_card_ids": ["REFERRAL-SBAR-v1"],
                "expected_candidate_pathway_card_ids": ["REFERRAL-SBAR-v1"],
                "expected_handoff_cues": ["red flags already fired"],
                "actual_protocol_urgency": "urgent",
                "actual_source_card_ids": ["REFERRAL-SBAR-v1"],
                "actual_candidate_pathway_card_ids": ["REFERRAL-SBAR-v1"],
                "final_validation": {"passed": True, "failures": []},
                "final_output": {
                    "protocol_urgency": "urgent",
                    "source_cards": ["REFERRAL-SBAR-v1"],
                    "candidate_protocol_pathways": [{"card_id": "REFERRAL-SBAR-v1"}],
                    "missing_info_to_collect": [],
                    "next_observations_to_collect": [],
                    "handoff_note_sbar": {
                        "situation": "Needs handoff.",
                        "background": "Known background.",
                        "assessment_observations_only": "Observed concern.",
                        "handoff_request": "Request review.",
                    },
                },
            }
        ],
    )

    manifest = export_v4_training_seeds(eval_path=eval_path, output_path=output)
    seeds = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert manifest["seed_count"] == 1
    assert seeds[0]["seed_type"] == "v4_failure_seed"
    assert seeds[0]["direct_training_allowed"] is False
    assert seeds[0]["repair_scopes"] == ["handoff_note_sbar"]
    assert seeds[0]["structured_intake"]["chief_concern"] == "radio handoff"
    assert "Do not copy" in seeds[0]["teacher_instruction"]


def test_can_export_high_quality_replay_candidates_when_requested(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    output = tmp_path / "seeds.jsonl"
    _write_jsonl(
        cases,
        [
            {
                "case_id": "case-1",
                "dataset_version": "figment_sft_v3",
                "structured_intake": {"chief_concern": "wound", "confirmed": True},
                "target_protocol_card_id": "WOUND-INFECTION-ESCALATION-v1",
            }
        ],
    )
    _write_jsonl(
        eval_path,
        [
            {
                "case_id": "case-1",
                "case_path": str(cases),
                "case_line": 1,
                "target_protocol_card_id": "WOUND-INFECTION-ESCALATION-v1",
                "expected_min_protocol_urgency": "urgent",
                "expected_red_flag_rule_ids": [],
                "actual_red_flag_rule_ids": [],
                "expected_source_card_ids": ["WOUND-INFECTION-ESCALATION-v1"],
                "expected_candidate_pathway_card_ids": ["WOUND-INFECTION-ESCALATION-v1"],
                "actual_protocol_urgency": "urgent",
                "actual_source_card_ids": ["WOUND-INFECTION-ESCALATION-v1"],
                "actual_candidate_pathway_card_ids": ["WOUND-INFECTION-ESCALATION-v1"],
                "final_validation": {"passed": True, "failures": []},
                "final_output": {
                    "protocol_urgency": "urgent",
                    "source_cards": ["WOUND-INFECTION-ESCALATION-v1"],
                    "candidate_protocol_pathways": [{"card_id": "WOUND-INFECTION-ESCALATION-v1"}],
                    "missing_info_to_collect": [],
                    "next_observations_to_collect": [],
                    "handoff_note_sbar": {
                        "situation": "Wound concern.",
                        "background": "Known background.",
                        "assessment_observations_only": "Observed wound concern.",
                        "handoff_request": "Request review.",
                    },
                },
            }
        ],
    )

    manifest = export_v4_training_seeds(eval_path=eval_path, output_path=output, include_passing=True)
    seeds = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert manifest["replay_seed_count"] == 1
    assert seeds[0]["seed_type"] == "v4_replay_candidate"
    assert seeds[0]["direct_training_allowed"] is True


def test_harness_only_evidence_miss_is_not_a_model_failure_seed(tmp_path: Path) -> None:
    cases = tmp_path / "field_workflow_holdout_v1.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    output = tmp_path / "seeds.jsonl"
    _write_jsonl(
        cases,
        [
            {
                "case_id": "holdout-2",
                "dataset_version": "field_workflow_holdout_v1",
                "workflow_category": "rural_clinic_intake",
                "structured_intake": {
                    "chief_concern": "wound check",
                    "confirmed": True,
                    "workflow_category": "rural_clinic_intake",
                },
                "target_protocol_card_id": "WOUND-INFECTION-ESCALATION-v1",
                "expected_red_flag_rule_ids": [],
                "expected_min_protocol_urgency": "urgent",
                "expected_source_card_ids": ["WOUND-INFECTION-ESCALATION-v1"],
                "expected_candidate_pathway_card_ids": ["WOUND-INFECTION-ESCALATION-v1"],
                "expected_missing_observations": [
                    "wound redness or swelling extent",
                    "manual correction status for audio-derived fields",
                ],
            }
        ],
    )
    _write_jsonl(
        eval_path,
        [
            {
                "case_id": "holdout-2",
                "case_path": str(cases),
                "case_line": 1,
                "target_protocol_card_id": "WOUND-INFECTION-ESCALATION-v1",
                "expected_red_flag_rule_ids": [],
                "actual_red_flag_rule_ids": [],
                "expected_min_protocol_urgency": "urgent",
                "expected_source_card_ids": ["WOUND-INFECTION-ESCALATION-v1"],
                "expected_candidate_pathway_card_ids": ["WOUND-INFECTION-ESCALATION-v1"],
                "expected_missing_observations": [
                    "wound redness or swelling extent",
                    "manual correction status for audio-derived fields",
                ],
                "actual_protocol_urgency": "urgent",
                "actual_source_card_ids": ["WOUND-INFECTION-ESCALATION-v1"],
                "actual_candidate_pathway_card_ids": ["WOUND-INFECTION-ESCALATION-v1"],
                "final_validation": {"passed": True, "failures": []},
                "final_output": {
                    "protocol_urgency": "urgent",
                    "source_cards": ["WOUND-INFECTION-ESCALATION-v1"],
                    "candidate_protocol_pathways": [{"card_id": "WOUND-INFECTION-ESCALATION-v1"}],
                    "missing_info_to_collect": ["wound redness or swelling extent"],
                    "next_observations_to_collect": ["wound redness or swelling extent"],
                    "handoff_note_sbar": {
                        "situation": "Wound concern.",
                        "background": "Known background.",
                        "assessment_observations_only": "Observed wound concern.",
                        "handoff_request": "Request review.",
                    },
                },
            }
        ],
    )

    manifest = export_v4_training_seeds(eval_path=eval_path, output_path=output, include_passing=True)
    seeds = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert manifest["failure_seed_count"] == 0
    assert manifest["replay_seed_count"] == 1
    assert manifest["harness_only_score_failure_count"] == 1
    assert seeds[0]["seed_type"] == "v4_replay_candidate"
    assert seeds[0]["model_training_failed"] is False
    assert seeds[0]["harness_only_score_failure"] is True
    assert seeds[0]["repair_scopes"] == []
    assert seeds[0]["workflow_category"] == "rural_clinic_intake"
