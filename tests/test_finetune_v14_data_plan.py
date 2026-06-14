import json
from collections import Counter


def _first_index_for_failure_class(failure_class: str) -> int:
    from scripts.generate_finetune_data import V14_NAVIGATOR_COUNTS
    from scripts.generate_finetune_data import _failure_class_for_index

    for index in range(sum(V14_NAVIGATOR_COUNTS.values())):
        if _failure_class_for_index(index, dataset_version="figment_sft_v14_delta") == failure_class:
            return index
    raise AssertionError(f"missing v14 failure class: {failure_class}")


def _accepted_v14_row(failure_class: str):
    from figment.retrieval import load_protocol_cards
    from scripts.generate_finetune_data import assemble_teacher_navigator_output
    from scripts.generate_finetune_data import build_sft_row
    from scripts.generate_finetune_data import case_spec_record
    from scripts.generate_finetune_data import generate_case_spec
    from scripts.generate_finetune_data import prepare_case
    from scripts.generate_finetune_data import score_candidate

    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    index = _first_index_for_failure_class(failure_class)
    spec = generate_case_spec(index, cards_by_id, dataset_version="figment_sft_v14_delta")
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


def test_v14_failure_cycle_expands_v13_delta_with_wound_replay_boost():
    from scripts.generate_finetune_data import V14_NAVIGATOR_COUNTS
    from scripts.generate_finetune_data import _failure_class_for_index

    categories = Counter(
        _failure_class_for_index(index, dataset_version="figment_sft_v14_delta")
        for index in range(sum(V14_NAVIGATOR_COUNTS.values()))
    )

    assert categories == V14_NAVIGATOR_COUNTS
    assert categories["wound_source_card_schema_replay"] == 200
    assert sum(categories.values()) == 1120


def test_v14_postpartum_rows_make_preg_cues_visible_without_scaffold_fill():
    for failure_class in (
        "postpartum_fever_required_obs_visible_preg_source_card_cue_closure",
        "postpartum_fever_required_obs_visible_preg_candidate_pathway_closure",
        "postpartum_fever_required_obs_selected_id_compressed_field_repair",
    ):
        row, spec_record, result = _accepted_v14_row(failure_class)
        output = json.loads(row["messages"][1]["content"])
        selected_ids = set(output["selected_required_observation_ids"])

        assert row["version"] == "figment_sft_v14_delta"
        assert spec_record["dataset_version"] == "figment_sft_v14_delta"
        assert "v14" in spec_record["tags"]
        assert "PREG-DANGER-SIGNS-v1" in output["source_cards"]
        assert "FEVER-RED-FLAGS-v1" in output["source_cards"]
        assert [item["card_id"] for item in output["candidate_protocol_pathways"]] == [
            "FEVER-RED-FLAGS-v1",
            "PREG-DANGER-SIGNS-v1",
        ]
        assert any(item.startswith("FEVER-RED-FLAGS-v1::required_observation::") for item in selected_ids)
        assert any(item.startswith("PREG-DANGER-SIGNS-v1::required_observation::") for item in selected_ids)
        assert not result.filled_required_observation_ids

        observation_text = json.dumps(
            output["missing_info_to_collect"] + output["next_observations_to_collect"]
        ).lower()
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


def test_v14_wound_replay_preserves_source_cards_schema_and_no_pregnancy_bleedthrough():
    row, spec_record, result = _accepted_v14_row("wound_source_card_schema_replay")
    output = json.loads(row["messages"][1]["content"])
    output_text = json.dumps(output).lower()

    assert "v14" in spec_record["tags"]
    assert spec_record["target_protocol_card_id"] == "WOUND-INFECTION-ESCALATION-v1"
    assert {"WOUND-INFECTION-ESCALATION-v1", "SAFETY-BOUNDARIES-v1", "REFERRAL-SBAR-v1"} <= set(
        output["source_cards"]
    )
    assert [item["card_id"] for item in output["candidate_protocol_pathways"]] == [
        "WOUND-INFECTION-ESCALATION-v1"
    ]
    assert "pregnan" not in output_text
    assert not result.filled_required_observation_ids


def test_v14_full_corpus_wrapper_pins_delta_defaults():
    from scripts.generate_v14_full_corpus import DEFAULT_NAVIGATOR_COUNT
    from scripts.generate_v14_full_corpus import DEFAULT_OUTPUT_VERSION
    from scripts.generate_v14_full_corpus import DEFAULT_REPAIR_COUNT
    from scripts.generate_v14_full_corpus import DEFAULT_TEACHER_MODEL_ID
    from scripts.generate_v14_full_corpus import build_corpus_args
    from scripts.merge_v14_training_corpus import build_merge_args

    assert DEFAULT_OUTPUT_VERSION == "figment_sft_v14_delta"
    assert DEFAULT_TEACHER_MODEL_ID == "nvidia/nemotron-3-ultra-550b-a55b"
    assert DEFAULT_NAVIGATOR_COUNT == 1120
    assert DEFAULT_REPAIR_COUNT == 0
    args = build_corpus_args(["--navigator-count", "8", "--repair-count", "0", "--dry-run"])
    assert args[args.index("--navigator-count") + 1] == "8"
    assert args[args.index("--repair-count") + 1] == "0"
    assert "--no-teacher-worker" in args
    assert args[-1] == "--dry-run"

    merge_args = build_merge_args(["--skip-verify"])
    assert merge_args[merge_args.index("--dataset-version") + 1] == "figment_sft_v14"
    assert merge_args[merge_args.index("--base") + 1] == "data/finetune/figment_sft_v10.jsonl"
    assert "--skip-verify" in merge_args
