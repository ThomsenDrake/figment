import json
from collections import Counter
from pathlib import Path


def _accepted_v5_row():
    from figment.observation_targets import required_observation_targets
    from figment.retrieval import load_protocol_cards
    from scripts.generate_finetune_data import assemble_teacher_navigator_output
    from scripts.generate_finetune_data import build_sft_row
    from scripts.generate_finetune_data import case_spec_record
    from scripts.generate_finetune_data import generate_case_spec
    from scripts.generate_finetune_data import prepare_case
    from scripts.generate_finetune_data import score_candidate
    from scripts.generate_finetune_data import v5_required_selected_observation_ids

    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    spec = generate_case_spec(0, cards_by_id, dataset_version="figment_sft_v5")
    prepared = prepare_case(spec, cards_by_id)
    candidate = assemble_teacher_navigator_output(
        prepared,
        {
            "facts": ["confirmed field concern"],
            "missing": ["highest-value observation pending"],
            "observe": ["highest-value observation pending"],
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
    selected_ids = v5_required_selected_observation_ids(
        source_card_ids=[str(card_id) for card_id in candidate.get("source_cards", [])],
        retrieved_cards=prepared.retrieved_cards,
    )
    required_targets_by_id = {str(target["id"]): target for target in required_observation_targets(prepared.retrieved_cards)}
    required_observation_text = [
        str(required_targets_by_id[selected_id]["display_text"])
        for selected_id in selected_ids
        if selected_id in required_targets_by_id
    ]
    candidate["selected_required_observation_ids"] = selected_ids
    candidate["missing_info_to_collect"] = required_observation_text + list(candidate["missing_info_to_collect"])
    candidate["next_observations_to_collect"] = required_observation_text
    result = score_candidate(candidate, prepared)
    assert result.passed is True
    row = build_sft_row(
        prepared=prepared,
        result=result,
        teacher_model_id="teacher-test",
        candidate_total=1,
        candidate_passed=1,
    )
    return row, case_spec_record(prepared)


def test_v5_failure_distribution_matches_focused_plan():
    from scripts.generate_finetune_data import V5_FOCUSED_COUNTS
    from scripts.generate_finetune_data import _failure_class_for_index

    categories = Counter(_failure_class_for_index(index, dataset_version="figment_sft_v5") for index in range(1100))

    assert categories == V5_FOCUSED_COUNTS


def test_v5_full_corpus_wrapper_pins_v5_defaults():
    from scripts.generate_v5_full_corpus import DEFAULT_ARGS
    from scripts.generate_v5_full_corpus import DEFAULT_COUNTS
    from scripts.generate_v5_full_corpus import DEFAULT_NAVIGATOR_COUNT
    from scripts.generate_v5_full_corpus import DEFAULT_OUTPUT_VERSION
    from scripts.generate_v5_full_corpus import DEFAULT_TEACHER_MODEL_ID
    from scripts.generate_v5_full_corpus import build_corpus_args

    assert DEFAULT_OUTPUT_VERSION == "figment_sft_v5"
    assert DEFAULT_TEACHER_MODEL_ID == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert DEFAULT_COUNTS == {
        "sbar_observation_ownership": 350,
        "required_observation_id_selection": 250,
        "source_card_invariant": 150,
        "noisy_field_audio_style": 100,
        "general_regression": 250,
    }
    assert DEFAULT_NAVIGATOR_COUNT == 1100
    assert DEFAULT_ARGS[DEFAULT_ARGS.index("--dataset-version") + 1] == "figment_sft_v5"
    assert DEFAULT_ARGS[DEFAULT_ARGS.index("--teacher-model-id") + 1] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert DEFAULT_ARGS[DEFAULT_ARGS.index("--navigator-count") + 1] == "1100"
    assert DEFAULT_ARGS[DEFAULT_ARGS.index("--repair-count") + 1] == "200"
    assert DEFAULT_ARGS[DEFAULT_ARGS.index("--output") + 1] == "data/finetune/figment_sft_v5.jsonl"
    assert DEFAULT_ARGS[DEFAULT_ARGS.index("--modal-output-dir") + 1] == "data/finetune/modal/figment_sft_v5"
    args = build_corpus_args(["--navigator-count", "2", "--output", "tmp/v5_smoke.jsonl"])
    assert args[-4:] == ["--navigator-count", "2", "--output", "tmp/v5_smoke.jsonl"]
    dry_run_args = build_corpus_args(["--navigator-count", "2", "--dry-run"])
    assert dry_run_args[-3:] == ["--navigator-count", "2", "--dry-run"]


def test_v5_sft_row_records_training_focus_and_required_observation_ids():
    row, spec_record = _accepted_v5_row()
    output = json.loads(row["messages"][1]["content"])
    metadata = row["metadata"]

    assert row["version"] == "figment_sft_v5"
    assert row["category"] == "sbar_observation_ownership"
    assert metadata["training_focus"] == "sbar_observation_ownership"
    assert metadata["excluded_eval_case_ids"] == [
        "field_workflow_holdout_v1-000054",
        "field_workflow_holdout_v1-000099",
    ]
    assert metadata["must_include_source_cards"]
    assert set(metadata["must_include_source_cards"]) <= set(output["source_cards"])
    assert output["selected_required_observation_ids"]
    assert set(metadata["must_include_selected_required_observation_ids"]) <= set(
        output["selected_required_observation_ids"]
    )
    assert spec_record["dataset_version"] == "figment_sft_v5"
    assert spec_record["workflow_category"] == "sbar_observation_ownership"


def test_v5_policy_rejects_missing_fired_card_selected_ids_and_generic_observations():
    from scripts.generate_finetune_data import v5_policy_issues

    output = {
        "source_cards": ["SAFETY-BOUNDARIES-v1"],
        "missing_info_to_collect": ["repeat vitals"],
        "next_observations_to_collect": ["monitor closely"],
        "handoff_note_sbar": {
            "situation": "",
            "background": "",
            "assessment_observations_only": "",
            "handoff_request": "",
        },
    }
    retrieved_cards = [
        {
            "card_id": "STROKE-SIGNS-v1",
            "card": {
                "card_id": "STROKE-SIGNS-v1",
                "required_observations": ["time last known well"],
            },
        }
    ]

    issues = v5_policy_issues(
        output,
        failure_class="source_card_invariant",
        expected_red_flag_rule_ids=["STROKE-001"],
        expected_candidate_pathway_card_ids=["STROKE-SIGNS-v1"],
        structured_intake={},
        rule_results=[{"rule_id": "STROKE-001", "card_id": "STROKE-SIGNS-v1"}],
        retrieved_cards=retrieved_cards,
        target_protocol_card_id="STROKE-SIGNS-v1",
    )

    assert "fired_rule_source_card_missing:STROKE-SIGNS-v1" in issues
    assert "generic_observation_phrase:repeat_vitals" in issues
    assert "generic_observation_phrase:monitor_closely" in issues


def test_verify_v5_rejects_rows_without_selected_ids(tmp_path):
    from scripts.verify_finetune_harness_alignment import verify_rows

    row, spec_record = _accepted_v5_row()
    output = json.loads(row["messages"][1]["content"])
    output.pop("selected_required_observation_ids", None)
    output["missing_info_to_collect"] = ["repeat vitals"]
    output["next_observations_to_collect"] = ["monitor closely"]
    row["messages"][1]["content"] = json.dumps(output, sort_keys=True)

    dataset = tmp_path / "rows.jsonl"
    case_specs = tmp_path / "specs.jsonl"
    dataset.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    case_specs.write_text(json.dumps(spec_record, sort_keys=True) + "\n", encoding="utf-8")

    summary = verify_rows(dataset_path=dataset, case_specs_path=case_specs)

    assert summary["passed"] is False
    assert summary["issue_types"]["v5_selected_required_observation_ids_missing"] >= 1
    assert summary["issue_types"]["v5_generic_observation_phrase:repeat_vitals"] >= 1


def test_v5_repair_scope_schedule_targets_observation_and_handoff_repairs():
    from scripts.augment_finetune_repair_rows import _scope_schedule

    counts = Counter(_scope_schedule(200, dataset_version="figment_sft_v5"))

    assert counts == {
        "missing_observations": 55,
        "handoff_note_sbar": 45,
        "citations_and_pathways": 35,
        "forbidden_clinical_language": 25,
        "protocol_urgency": 20,
        "schema": 20,
    }
