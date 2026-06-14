import json
from collections import Counter


def _accepted_v11_row():
    from figment.retrieval import load_protocol_cards
    from scripts.generate_finetune_data import assemble_teacher_navigator_output
    from scripts.generate_finetune_data import build_sft_row
    from scripts.generate_finetune_data import case_spec_record
    from scripts.generate_finetune_data import generate_case_spec
    from scripts.generate_finetune_data import prepare_case
    from scripts.generate_finetune_data import score_candidate

    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    spec = generate_case_spec(0, cards_by_id, dataset_version="figment_sft_v11_delta")
    prepared = prepare_case(spec, cards_by_id)
    candidate = assemble_teacher_navigator_output(
        prepared,
        {
            "facts": ["postpartum fever confirmed", "temperature elevated", "blood pressure pending"],
            "missing": [
                "temperature if available",
                "age or pregnancy status",
                "mental status",
                "neck stiffness report",
                "rash report",
                "hydration observations",
                "available vital signs",
                "pregnancy or postpartum status",
                "bleeding report",
                "abdominal pain report",
                "headache or vision symptoms",
                "seizure or fainting report",
                "fever report",
            ],
            "observe": ["temperature if available", "bleeding report", "abdominal pain report"],
            "checklist": ["cite fever and pregnancy danger-sign cards"],
            "uncertain": ["blood pressure pending"],
            "sbar": {
                "situation": "postpartum fever",
                "background": "postpartum field intake",
                "assessment_observations_only": "fever and postpartum context confirmed",
                "handoff_request": "request protocol review",
            },
            "script": "I am checking fever and postpartum danger-sign observations.",
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


def test_v11_failure_cycle_targets_v10_visible_observation_gap():
    from scripts.generate_finetune_data import V11_NAVIGATOR_COUNTS
    from scripts.generate_finetune_data import _failure_class_for_index

    categories = Counter(
        _failure_class_for_index(index, dataset_version="figment_sft_v11_delta")
        for index in range(sum(V11_NAVIGATOR_COUNTS.values()))
    )
    first_twelve = {
        _failure_class_for_index(index, dataset_version="figment_sft_v11_delta")
        for index in range(12)
    }

    assert categories == V11_NAVIGATOR_COUNTS
    assert {
        "postpartum_fever_required_obs_visible_dual_field_holdout_shape",
        "postpartum_fever_required_obs_dual_field_closure",
        "postpartum_fever_required_obs_candidate_focus",
    } <= first_twelve


def test_v11_sft_row_front_loads_pregnancy_danger_sign_text_in_both_fields():
    row, spec_record = _accepted_v11_row()
    output = json.loads(row["messages"][1]["content"])
    metadata = row["metadata"]
    selected_ids = set(output["selected_required_observation_ids"])

    assert row["version"] == "figment_sft_v11_delta"
    assert spec_record["dataset_version"] == "figment_sft_v11_delta"
    assert spec_record["failure_class"] == "postpartum_fever_required_obs_visible_dual_field_holdout_shape"
    assert "PREG-DANGER-SIGNS-v1" in output["source_cards"]
    assert "FEVER-RED-FLAGS-v1" in output["source_cards"]
    assert set(metadata["must_include_selected_required_observation_ids"]) <= selected_ids

    missing_text = json.dumps(output["missing_info_to_collect"]).lower()
    observe_text = json.dumps(output["next_observations_to_collect"]).lower()
    for cue in (
        "pregnancy or postpartum status",
        "bleeding report",
        "abdominal pain report",
        "headache or vision symptoms",
        "seizure or fainting report",
        "fever report",
        "temperature if available",
        "age or pregnancy status",
        "mental status",
        "neck stiffness report",
        "rash report",
        "hydration observations",
        "available vital signs",
    ):
        assert cue in missing_text
        assert cue in observe_text

    first_missing = json.dumps(output["missing_info_to_collect"][:6]).lower()
    assert "bleeding report" in first_missing
    assert "seizure or fainting report" in first_missing


def test_v11_full_corpus_wrapper_pins_delta_defaults():
    from scripts.generate_v11_full_corpus import DEFAULT_NAVIGATOR_COUNT
    from scripts.generate_v11_full_corpus import DEFAULT_OUTPUT_VERSION
    from scripts.generate_v11_full_corpus import DEFAULT_REPAIR_COUNT
    from scripts.generate_v11_full_corpus import DEFAULT_TEACHER_MODEL_ID
    from scripts.generate_v11_full_corpus import build_corpus_args
    from scripts.merge_v11_training_corpus import build_merge_args

    assert DEFAULT_OUTPUT_VERSION == "figment_sft_v11_delta"
    assert DEFAULT_TEACHER_MODEL_ID == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert DEFAULT_NAVIGATOR_COUNT == 800
    assert DEFAULT_REPAIR_COUNT == 0
    args = build_corpus_args(["--navigator-count", "8", "--repair-count", "0", "--dry-run"])
    assert args[args.index("--navigator-count") + 1] == "8"
    assert args[args.index("--repair-count") + 1] == "0"
    assert args[-1] == "--dry-run"

    merge_args = build_merge_args(["--skip-verify"])
    assert merge_args[merge_args.index("--dataset-version") + 1] == "figment_sft_v11"
    assert merge_args[merge_args.index("--base") + 1] == "data/finetune/figment_sft_v10.jsonl"
    assert "--skip-verify" in merge_args
