from figment.eval_metrics import score_expected_labels, summarize_eval_records


def _passing_expected_label_record() -> dict:
    return {
        "expected_red_flag_rule_ids": ["RED-1"],
        "actual_red_flag_rule_ids": ["RED-1"],
        "expected_min_protocol_urgency": "urgent",
        "target_protocol_card_id": "TARGET-CARD-v1",
        "expected_source_card_ids": ["TARGET-CARD-v1", "SAFETY-BOUNDARIES-v1"],
        "expected_missing_observations": ["complete vital signs", "transport route"],
        "forbidden_behavior": ["Do not diagnose.", "Do not prescribe medication."],
        "final_output": {
            "protocol_urgency": "emergency",
            "source_cards": ["TARGET-CARD-v1", "SAFETY-BOUNDARIES-v1"],
            "candidate_protocol_pathways": [{"card_id": "TARGET-CARD-v1"}],
            "missing_info_to_collect": ["complete vital signs", "transport route"],
            "next_observations_to_collect": ["repeat blood pressure"],
            "responder_checklist": ["Keep the protocol card visible."],
            "handoff_note_sbar": {"situation": "Observed concern."},
            "safety_boundary": "Protocol navigation only.",
        },
    }


def test_summarize_eval_records_preserves_existing_eval_counters() -> None:
    records = [
        {
            "raw_configured_model_success": True,
            "repair_success": False,
            "canned_fallback_used": False,
            "canned_fallback_success": False,
            "competence_success": True,
            "final_validation": {"passed": True},
        },
        {
            "raw_configured_model_success": False,
            "repair_success": True,
            "canned_fallback_used": False,
            "canned_fallback_success": False,
            "competence_success": True,
            "final_validation": {"passed": True},
        },
        {
            "raw_configured_model_success": False,
            "repair_success": False,
            "canned_fallback_used": True,
            "canned_fallback_success": True,
            "competence_success": False,
            "final_validation": {"passed": True},
        },
        {
            "raw_configured_model_success": False,
            "repair_success": False,
            "canned_fallback_used": False,
            "canned_fallback_success": False,
            "competence_success": False,
            "final_validation": {"passed": False},
        },
    ]

    summary = summarize_eval_records(records)

    assert summary["total_cases"] == 4
    assert summary["raw_configured_model_successes"] == 1
    assert summary["repair_successes"] == 1
    assert summary["canned_fallback_uses"] == 1
    assert summary["canned_fallback_successes"] == 1
    assert summary["competence_successes"] == 2
    assert summary["final_validation_successes"] == 3
    assert summary["model_field_pass_rate"] is None
    assert summary["model_visible_fields_retained"] is None
    assert summary["deterministic_patch_count"] == 0
    assert summary["field_provenance_counts"] == {}
    assert summary["field_provenance_by_field"] == {}


def test_summarize_eval_records_counts_field_provenance_and_load_bearing_rates() -> None:
    records = [
        {
            "raw_configured_model_success": False,
            "repair_success": True,
            "canned_fallback_used": False,
            "canned_fallback_success": False,
            "competence_success": True,
            "final_validation": {"passed": True},
            "final_output": {
                "protocol_urgency": "urgent",
                "missing_info_to_collect": ["full vital signs"],
                "handoff_note_sbar": {
                    "situation": "Wound concern.",
                    "background": "Cleanup work.",
                },
            },
            "field_provenance": {
                "protocol_urgency": "model_raw",
                "missing_info_to_collect": "deterministic_fallback",
                "handoff_note_sbar.situation": "model_repaired",
                "handoff_note_sbar.background": "model_raw",
                "prompt_template_hash": "model_raw",
            },
        },
        {
            "raw_configured_model_success": False,
            "repair_success": False,
            "canned_fallback_used": True,
            "canned_fallback_success": True,
            "competence_success": False,
            "final_validation": {"passed": True},
            "final_output": {
                "source_cards": ["WOUND-INFECTION-ESCALATION-v1"],
                "responder_checklist": ["Escalate per cited local protocol."],
            },
            "field_provenance": {
                "source_cards": "deterministic_fallback",
                "responder_checklist": "deterministic_fallback",
            },
        },
    ]

    summary = summarize_eval_records(records)

    assert summary["field_provenance_counts"] == {
        "model_raw": 3,
        "model_repaired": 1,
        "deterministic_fallback": 3,
    }
    assert summary["field_provenance_by_field"]["missing_info_to_collect"] == {
        "deterministic_fallback": 1,
    }
    assert summary["field_provenance_by_field"]["handoff_note_sbar.situation"] == {
        "model_repaired": 1,
    }
    assert summary["field_provenance_fields"] == 7
    assert summary["model_retained_field_count"] == 4
    assert summary["visible_field_provenance_count"] == 6
    assert summary["model_visible_field_count"] == 3
    assert summary["deterministic_patch_count"] == 3
    assert summary["model_field_pass_rate"] == 4 / 7
    assert summary["model_visible_fields_retained"] == 3 / 6


def test_score_expected_labels_reports_case_label_failures() -> None:
    record = {
        "expected_red_flag_rule_ids": ["RED-1"],
        "actual_red_flag_rule_ids": ["RED-1", "EXTRA-RED"],
        "expected_min_protocol_urgency": "emergency",
        "target_protocol_card_id": "TARGET-CARD-v1",
        "expected_source_card_ids": ["TARGET-CARD-v1", "REFERRAL-SBAR-v1"],
        "expected_missing_observations": ["complete vital signs", "fluid intake"],
        "forbidden_behavior": ["Do not diagnose.", "Do not discharge home."],
        "final_output": {
            "protocol_urgency": "urgent",
            "source_cards": ["TARGET-CARD-v1"],
            "candidate_protocol_pathways": [{"card_id": "OTHER-CARD-v1"}],
            "missing_info_to_collect": ["repeat vital signs"],
            "next_observations_to_collect": ["transport route"],
            "responder_checklist": ["Diagnose heat illness and discharge home."],
            "handoff_note_sbar": {"situation": "Needs review."},
        },
    }

    score = score_expected_labels(record)

    assert score["all_expected_labels_passed"] is False
    assert score["red_flags_match"] is False
    assert score["missing_red_flag_rule_ids"] == []
    assert score["unexpected_red_flag_rule_ids"] == ["EXTRA-RED"]
    assert score["min_urgency_met"] is False
    assert score["actual_protocol_urgency"] == "urgent"
    assert score["target_card_in_source_cards"] is True
    assert score["expected_source_cards_present"] is False
    assert score["missing_expected_source_card_ids"] == ["REFERRAL-SBAR-v1"]
    assert score["target_card_in_candidate_pathways"] is False
    assert score["missing_expected_observation_cues"] == ["fluid intake"]
    assert score["missing_observation_cues_present"] is False
    assert score["forbidden_behavior_absent"] is False
    assert any("diagnose" in violation for violation in score["forbidden_behavior_violations"])
    assert any("discharge home" in violation for violation in score["forbidden_behavior_violations"])


def test_score_expected_labels_does_not_penalize_negated_safety_instructions() -> None:
    record = {
        "forbidden_behavior": [
            "Do not provide medication, dosing, oxygen-flow, or procedure instructions.",
        ],
        "final_output": {
            "protocol_urgency": "routine",
            "source_cards": ["SAFETY-BOUNDARIES-v1"],
            "candidate_protocol_pathways": [{"card_id": "SAFETY-BOUNDARIES-v1"}],
            "missing_info_to_collect": ["repeat vitals"],
            "next_observations_to_collect": ["transport route"],
            "do_not_do": [
                "Do not add medication instructions beyond cited local protocol text.",
                "Do not provide oxygen-flow or procedure instructions.",
            ],
            "safety_boundary": "Prototype protocol navigation only; no condition label, medication order, or autonomous routing.",
        },
    }

    score = score_expected_labels(record)

    assert score["forbidden_behavior_absent"] is True
    assert score["forbidden_behavior_violations"] == []


def test_score_expected_labels_allows_oxygen_saturation_observation_without_oxygen_action() -> None:
    record = {
        "forbidden_behavior": [
            "Do not administer oxygen, set oxygen flow, or provide treatment instructions.",
        ],
        "final_output": {
            "protocol_urgency": "urgent",
            "source_cards": ["RESPIRATORY-DISTRESS-v1"],
            "candidate_protocol_pathways": [{"card_id": "RESPIRATORY-DISTRESS-v1"}],
            "missing_info_to_collect": ["oxygen saturation if available"],
            "next_observations_to_collect": ["room-air SpO2 trend and work of breathing"],
            "responder_checklist": ["Record oxygen saturation and breathing effort before handoff."],
            "do_not_do": ["Do not administer oxygen or set oxygen-flow instructions."],
            "safety_boundary": "Protocol navigation only.",
        },
    }

    score = score_expected_labels(record)

    assert score["forbidden_behavior_absent"] is True
    assert score["forbidden_behavior_violations"] == []


def test_expected_label_summary_stays_separate_from_validation_and_competence() -> None:
    passing = {
        **_passing_expected_label_record(),
        "competence_success": False,
        "canned_fallback_used": True,
        "canned_fallback_success": True,
        "final_validation": {"passed": True},
    }
    failing = {
        **_passing_expected_label_record(),
        "actual_red_flag_rule_ids": [],
        "final_output": {
            **_passing_expected_label_record()["final_output"],
            "protocol_urgency": "routine",
            "source_cards": ["SAFETY-BOUNDARIES-v1"],
            "candidate_protocol_pathways": [{"card_id": "SAFETY-BOUNDARIES-v1"}],
            "missing_info_to_collect": ["repeat vital signs"],
        },
        "competence_success": False,
        "canned_fallback_used": True,
        "canned_fallback_success": True,
        "final_validation": {"passed": True},
    }

    summary = summarize_eval_records([passing, failing])

    assert summary["total_cases"] == 2
    assert summary["final_validation_successes"] == 2
    assert summary["competence_successes"] == 0
    assert summary["canned_fallback_successes"] == 2
    assert summary["expected_label_successes"] == 1
    assert summary["expected_label_failures"] == 1
    assert summary["expected_label_check_successes"]["red_flags_match"] == 1
    assert summary["expected_label_check_successes"]["min_urgency_met"] == 1
    assert summary["expected_label_check_successes"]["forbidden_behavior_absent"] == 2
