import json
from collections import Counter


def _first_index_for_failure_class(failure_class: str) -> int:
    from scripts.generate_finetune_data import V12_NAVIGATOR_COUNTS
    from scripts.generate_finetune_data import _failure_class_for_index

    for index in range(sum(V12_NAVIGATOR_COUNTS.values())):
        if _failure_class_for_index(index, dataset_version="figment_sft_v12_delta") == failure_class:
            return index
    raise AssertionError(f"missing v12 failure class: {failure_class}")


def _accepted_v12_row(failure_class: str):
    from figment.retrieval import load_protocol_cards
    from scripts.generate_finetune_data import assemble_teacher_navigator_output
    from scripts.generate_finetune_data import build_sft_row
    from scripts.generate_finetune_data import case_spec_record
    from scripts.generate_finetune_data import generate_case_spec
    from scripts.generate_finetune_data import prepare_case
    from scripts.generate_finetune_data import score_candidate

    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    index = _first_index_for_failure_class(failure_class)
    spec = generate_case_spec(index, cards_by_id, dataset_version="figment_sft_v12_delta")
    prepared = prepare_case(spec, cards_by_id)
    candidate = assemble_teacher_navigator_output(
        prepared,
        {
            "facts": [
                str(prepared.spec.structured_intake.get("chief_concern") or "field concern"),
                str(prepared.spec.structured_intake.get("vitals") or "vitals pending"),
            ],
            "missing": prepared.expected_missing_observations,
            "observe": prepared.expected_missing_observations,
            "checklist": ["cite source cards", "collect required observations", "prepare grounded handoff"],
            "uncertain": ["incomplete observations require local protocol review"],
            "sbar": {
                "situation": str(prepared.spec.structured_intake.get("chief_concern") or "field concern"),
                "background": str(prepared.spec.structured_intake.get("setting") or "field intake"),
                "assessment_observations_only": "deterministic red flags and cited observations only",
                "handoff_request": "request protocol review per cited source cards",
            },
            "script": "I am checking cited protocol observations.",
        },
    )
    result = score_candidate(candidate, prepared)
    assert result.passed is True, result.reward_components
    assert not result.patched_fields
    row = build_sft_row(
        prepared=prepared,
        result=result,
        teacher_model_id="teacher-test",
        candidate_total=1,
        candidate_passed=1,
    )
    return row, case_spec_record(prepared), result


def test_v12_failure_cycle_mixes_targeted_fix_and_replay_rows():
    from scripts.generate_finetune_data import V12_NAVIGATOR_COUNTS
    from scripts.generate_finetune_data import _failure_class_for_index

    categories = Counter(
        _failure_class_for_index(index, dataset_version="figment_sft_v12_delta")
        for index in range(sum(V12_NAVIGATOR_COUNTS.values()))
    )

    assert categories == V12_NAVIGATOR_COUNTS
    assert set(categories) == {
        "postpartum_fever_required_obs_dual_card_selected_ids_visible_fields",
        "postpartum_fever_required_obs_candidate_and_source_closure",
        "wound_source_card_schema_replay",
        "referral_candidate_pathway_replay",
    }


def test_v12_postpartum_row_front_loads_preg_and_fever_cues_without_scaffold_fill():
    row, spec_record, result = _accepted_v12_row(
        "postpartum_fever_required_obs_dual_card_selected_ids_visible_fields"
    )
    output = json.loads(row["messages"][1]["content"])
    selected_ids = set(output["selected_required_observation_ids"])

    assert row["version"] == "figment_sft_v12_delta"
    assert spec_record["dataset_version"] == "figment_sft_v12_delta"
    assert "PREG-DANGER-SIGNS-v1" in output["source_cards"]
    assert "FEVER-RED-FLAGS-v1" in output["source_cards"]
    assert [item["card_id"] for item in output["candidate_protocol_pathways"]] == [
        "FEVER-RED-FLAGS-v1",
        "PREG-DANGER-SIGNS-v1",
    ]
    assert any(item.startswith("FEVER-RED-FLAGS-v1::required_observation::") for item in selected_ids)
    assert any(item.startswith("PREG-DANGER-SIGNS-v1::required_observation::") for item in selected_ids)
    assert not result.filled_required_observation_ids

    for field in ("missing_info_to_collect", "next_observations_to_collect"):
        observation_text = json.dumps(output[field]).lower()
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
            assert cue in observation_text


def test_v12_wound_replay_preserves_schema_source_cards_and_grounding():
    row, spec_record, result = _accepted_v12_row("wound_source_card_schema_replay")
    output = json.loads(row["messages"][1]["content"])
    output_text = json.dumps(output).lower()

    assert spec_record["target_protocol_card_id"] == "WOUND-INFECTION-ESCALATION-v1"
    assert {"WOUND-INFECTION-ESCALATION-v1", "SAFETY-BOUNDARIES-v1", "REFERRAL-SBAR-v1"} <= set(
        output["source_cards"]
    )
    assert [item["card_id"] for item in output["candidate_protocol_pathways"]] == [
        "WOUND-INFECTION-ESCALATION-v1"
    ]
    assert any(
        item.startswith("WOUND-INFECTION-ESCALATION-v1::required_observation::")
        for item in output["selected_required_observation_ids"]
    )
    assert "pregnan" not in output_text
    assert not result.filled_required_observation_ids


def test_v12_referral_replay_keeps_target_and_fired_clinical_candidate_paths():
    row, spec_record, result = _accepted_v12_row("referral_candidate_pathway_replay")
    output = json.loads(row["messages"][1]["content"])
    candidate_ids = [item["card_id"] for item in output["candidate_protocol_pathways"]]

    assert spec_record["target_protocol_card_id"] == "REFERRAL-SBAR-v1"
    assert candidate_ids == ["REFERRAL-SBAR-v1", "PREG-DANGER-SIGNS-v1", "FEVER-RED-FLAGS-v1"]
    assert {"REFERRAL-SBAR-v1", "SAFETY-BOUNDARIES-v1", "PREG-DANGER-SIGNS-v1", "FEVER-RED-FLAGS-v1"} <= set(
        output["source_cards"]
    )
    assert not result.filled_required_observation_ids


def test_v12_full_corpus_wrapper_pins_delta_defaults():
    from scripts.generate_v12_full_corpus import DEFAULT_NAVIGATOR_COUNT
    from scripts.generate_v12_full_corpus import DEFAULT_OUTPUT_VERSION
    from scripts.generate_v12_full_corpus import DEFAULT_REPAIR_COUNT
    from scripts.generate_v12_full_corpus import DEFAULT_TEACHER_MODEL_ID
    from scripts.generate_v12_full_corpus import build_corpus_args
    from scripts.merge_v12_training_corpus import build_merge_args

    assert DEFAULT_OUTPUT_VERSION == "figment_sft_v12_delta"
    assert DEFAULT_TEACHER_MODEL_ID == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert DEFAULT_NAVIGATOR_COUNT == 560
    assert DEFAULT_REPAIR_COUNT == 0
    args = build_corpus_args(["--navigator-count", "8", "--repair-count", "0", "--dry-run"])
    assert args[args.index("--navigator-count") + 1] == "8"
    assert args[args.index("--repair-count") + 1] == "0"
    assert args[-1] == "--dry-run"

    merge_args = build_merge_args(["--skip-verify"])
    assert merge_args[merge_args.index("--dataset-version") + 1] == "figment_sft_v12"
    assert merge_args[merge_args.index("--base") + 1] == "data/finetune/figment_sft_v10.jsonl"
    assert "--skip-verify" in merge_args
