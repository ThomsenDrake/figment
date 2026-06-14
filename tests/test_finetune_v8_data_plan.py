import json
from collections import Counter


def _accepted_v8_row():
    from figment.retrieval import load_protocol_cards
    from scripts.generate_finetune_data import assemble_teacher_navigator_output
    from scripts.generate_finetune_data import build_sft_row
    from scripts.generate_finetune_data import case_spec_record
    from scripts.generate_finetune_data import generate_case_spec
    from scripts.generate_finetune_data import prepare_case
    from scripts.generate_finetune_data import score_candidate

    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    spec = generate_case_spec(0, cards_by_id, dataset_version="figment_sft_v8_delta")
    prepared = prepare_case(spec, cards_by_id)
    candidate = assemble_teacher_navigator_output(
        prepared,
        {
            "facts": ["postpartum fever confirmed", "temperature elevated"],
            "missing": ["pregnancy or postpartum status", "bleeding report", "temperature if available"],
            "observe": ["temperature if available", "bleeding report", "mental status"],
            "checklist": ["cite fever and pregnancy cards"],
            "uncertain": ["blood pressure pending"],
            "sbar": {
                "situation": "postpartum fever",
                "background": "field workflow setting",
                "assessment_observations_only": "fever and postpartum context confirmed",
                "handoff_request": "request protocol review",
            },
            "script": "I am checking fever and postpartum observations.",
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


def test_v8_failure_cycle_targets_multirule_observation_ownership():
    from scripts.generate_finetune_data import V8_NAVIGATOR_COUNTS
    from scripts.generate_finetune_data import _failure_class_for_index

    categories = Counter(
        _failure_class_for_index(index, dataset_version="figment_sft_v8_delta")
        for index in range(sum(V8_NAVIGATOR_COUNTS.values()))
    )
    first_twelve = {
        _failure_class_for_index(index, dataset_version="figment_sft_v8_delta")
        for index in range(12)
    }

    assert categories == V8_NAVIGATOR_COUNTS
    assert {"multi_rule_observation_ownership", "multi_rule_candidate_focus"} <= first_twelve


def test_v8_sft_row_requires_all_fired_clinical_observation_ids():
    row, spec_record = _accepted_v8_row()
    output = json.loads(row["messages"][1]["content"])
    metadata = row["metadata"]
    selected_ids = set(output["selected_required_observation_ids"])

    assert row["version"] == "figment_sft_v8_delta"
    assert spec_record["dataset_version"] == "figment_sft_v8_delta"
    assert spec_record["target_protocol_card_id"] == "FEVER-RED-FLAGS-v1"
    assert {"PREG-001", "FEVER-001"} <= set(spec_record["expected_red_flag_rule_ids"])
    assert "PREG-DANGER-SIGNS-v1" in output["source_cards"]
    candidate_ids = [item["card_id"] for item in output["candidate_protocol_pathways"]]
    assert candidate_ids == ["FEVER-RED-FLAGS-v1", "PREG-DANGER-SIGNS-v1"]
    assert any(item.startswith("FEVER-RED-FLAGS-v1::required_observation::") for item in selected_ids)
    assert any(item.startswith("PREG-DANGER-SIGNS-v1::required_observation::") for item in selected_ids)
    assert set(metadata["must_include_selected_required_observation_ids"]) <= selected_ids
    observation_text = json.dumps(
        output["missing_info_to_collect"] + output["next_observations_to_collect"]
    ).lower()
    for cue in ("temperature if available", "pregnancy or postpartum status", "bleeding report", "fever report"):
        assert cue in observation_text
    assert metadata["v7_training_policy_version"] == 1


def test_v8_full_corpus_wrapper_pins_delta_defaults():
    from scripts.generate_v8_full_corpus import DEFAULT_NAVIGATOR_COUNT
    from scripts.generate_v8_full_corpus import DEFAULT_OUTPUT_VERSION
    from scripts.generate_v8_full_corpus import DEFAULT_REPAIR_COUNT
    from scripts.generate_v8_full_corpus import DEFAULT_TEACHER_MODEL_ID
    from scripts.generate_v8_full_corpus import build_corpus_args

    assert DEFAULT_OUTPUT_VERSION == "figment_sft_v8_delta"
    assert DEFAULT_TEACHER_MODEL_ID == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert DEFAULT_NAVIGATOR_COUNT == 400
    assert DEFAULT_REPAIR_COUNT == 0
    args = build_corpus_args(["--navigator-count", "8", "--repair-count", "0", "--dry-run"])
    assert args[args.index("--navigator-count") + 1] == "8"
    assert args[args.index("--repair-count") + 1] == "0"
    assert args[-1] == "--dry-run"
