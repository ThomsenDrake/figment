import json
from collections import Counter


def _accepted_v6_row():
    from figment.retrieval import load_protocol_cards
    from scripts.generate_finetune_data import assemble_teacher_navigator_output
    from scripts.generate_finetune_data import build_sft_row
    from scripts.generate_finetune_data import case_spec_record
    from scripts.generate_finetune_data import generate_case_spec
    from scripts.generate_finetune_data import prepare_case
    from scripts.generate_finetune_data import score_candidate

    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    spec = generate_case_spec(0, cards_by_id, dataset_version="figment_sft_v6_delta")
    prepared = prepare_case(spec, cards_by_id)
    candidate = assemble_teacher_navigator_output(
        prepared,
        {
            "facts": ["confirmed field concern"],
            "missing": ["confirm current mental status", "record available vital signs"],
            "observe": ["confirm current mental status", "record available vital signs"],
            "checklist": ["cite deterministic rule cards"],
            "uncertain": ["some vitals remain incomplete"],
            "sbar": {
                "situation": "confirmed handoff concern",
                "background": "field workflow setting",
                "assessment_observations_only": "observations only from confirmed intake",
                "handoff_request": "request protocol review",
            },
            "script": "I am checking protocol observations.",
        },
    )
    result = score_candidate(candidate, prepared)
    assert result.passed is True, result.reward_components
    row = build_sft_row(
        prepared=prepared,
        result=result,
        teacher_model_id="teacher-test",
        candidate_total=1,
        candidate_passed=1,
    )
    return row, case_spec_record(prepared)


def test_v6_failure_cycle_matches_updated_plan_and_interleaves_smoke_cases():
    from scripts.generate_finetune_data import V6_NAVIGATOR_COUNTS
    from scripts.generate_finetune_data import _failure_class_for_index

    categories = Counter(
        _failure_class_for_index(index, dataset_version="figment_sft_v6_delta")
        for index in range(sum(V6_NAVIGATOR_COUNTS.values()))
    )
    first_twelve = {
        _failure_class_for_index(index, dataset_version="figment_sft_v6_delta")
        for index in range(12)
    }

    assert categories == V6_NAVIGATOR_COUNTS
    assert {"required_observation_ownership", "observation_correction", "v6_preservation"} <= first_twelve


def test_v6_full_corpus_wrapper_pins_delta_defaults():
    from scripts.generate_v6_full_corpus import DEFAULT_OUTPUT_VERSION
    from scripts.generate_v6_full_corpus import DEFAULT_REPAIR_COUNT
    from scripts.generate_v6_full_corpus import DEFAULT_TEACHER_MODEL_ID
    from scripts.generate_v6_full_corpus import build_corpus_args

    assert DEFAULT_OUTPUT_VERSION == "figment_sft_v6_delta"
    assert DEFAULT_TEACHER_MODEL_ID == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert DEFAULT_REPAIR_COUNT == 250
    args = build_corpus_args(["--new-delta-count", "10", "--correction-count", "2", "--repair-count", "3", "--dry-run"])
    assert args[args.index("--navigator-count") + 1] == "12"
    assert args[args.index("--repair-count") + 1] == "3"
    assert args[-1] == "--dry-run"


def test_v6_sft_row_records_observation_policy_metadata():
    row, spec_record = _accepted_v6_row()
    output = json.loads(row["messages"][1]["content"])
    metadata = row["metadata"]

    assert row["version"] == "figment_sft_v6_delta"
    assert row["category"] == "required_observation_ownership"
    assert metadata["training_focus"] == "required_observation_ownership"
    assert metadata["v6_training_policy_version"] == 1
    assert metadata["required_observation_targets"]
    assert output["selected_required_observation_ids"]
    assert set(metadata["must_include_selected_required_observation_ids"]) <= set(
        output["selected_required_observation_ids"]
    )
    assert output["missing_info_to_collect"] != output["next_observations_to_collect"]
    observation_text = json.dumps(
        output["missing_info_to_collect"] + output["next_observations_to_collect"]
    ).lower()
    assert "source card ids" not in observation_text
    assert spec_record["dataset_version"] == "figment_sft_v6_delta"
    assert spec_record["must_include_selected_required_observation_ids"]


def test_v6_policy_rejects_duplicate_metadata_and_invisible_selected_ids():
    from scripts.generate_finetune_data import v6_policy_issues

    output = {
        "source_cards": ["STROKE-SIGNS-v1"],
        "selected_required_observation_ids": ["STROKE-SIGNS-v1::required_observation::1"],
        "missing_info_to_collect": [
            "source card IDs",
            "deterministic rule results",
            "ask about something else",
            "monitor closely",
        ],
        "next_observations_to_collect": [
            "source card IDs",
            "deterministic rule results",
            "ask about something else",
            "monitor closely",
        ],
        "handoff_note_sbar": {
            "situation": "stroke signs",
            "background": "field setting",
            "assessment_observations_only": "observations pending",
            "handoff_request": "request protocol review",
        },
    }
    retrieved_cards = [
        {
            "card_id": "STROKE-SIGNS-v1",
            "card": {
                "card_id": "STROKE-SIGNS-v1",
                "required_observations": ["face droop observation"],
            },
        }
    ]

    issues = v6_policy_issues(
        output,
        failure_class="required_observation_ownership",
        expected_red_flag_rule_ids=[],
        expected_candidate_pathway_card_ids=["STROKE-SIGNS-v1"],
        structured_intake={},
        rule_results=[],
        retrieved_cards=retrieved_cards,
        target_protocol_card_id="STROKE-SIGNS-v1",
    )

    assert "duplicate_long_missing_and_next_observations" in issues
    assert any(issue.startswith("harness_metadata_observation:") for issue in issues)
    assert "selected_required_observation_id_not_visible:STROKE-SIGNS-v1::required_observation::1" in issues


def test_v6_repair_scope_schedule_targets_observation_repairs():
    from scripts.augment_finetune_repair_rows import _scope_schedule

    assert Counter(_scope_schedule(250, dataset_version="figment_sft_v6_delta")) == {
        "missing_observations": 250
    }


def test_verify_v6_rejects_rows_with_harness_metadata_observations(tmp_path):
    from scripts.verify_finetune_harness_alignment import verify_rows

    row, spec_record = _accepted_v6_row()
    output = json.loads(row["messages"][1]["content"])
    output["missing_info_to_collect"] = [
        "source card IDs",
        "deterministic rule results",
        "navigator validation result",
        "confirmed intake status",
    ]
    output["next_observations_to_collect"] = list(output["missing_info_to_collect"])
    row["messages"][1]["content"] = json.dumps(output, sort_keys=True)

    dataset = tmp_path / "rows.jsonl"
    case_specs = tmp_path / "specs.jsonl"
    dataset.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    case_specs.write_text(json.dumps(spec_record, sort_keys=True) + "\n", encoding="utf-8")

    summary = verify_rows(dataset_path=dataset, case_specs_path=case_specs)

    assert summary["passed"] is False
    assert summary["issue_types"]["v6_duplicate_long_missing_and_next_observations"] >= 1
    assert any(key.startswith("v6_harness_metadata_observation") for key in summary["issue_types"])
