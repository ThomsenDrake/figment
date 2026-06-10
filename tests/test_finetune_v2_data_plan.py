import json
import argparse
from collections import Counter
from pathlib import Path

import pytest


def test_dataset_paths_are_versioned():
    from scripts.generate_finetune_data import dataset_paths

    paths = dataset_paths("figment_sft_v2")

    assert paths["output"] == Path("data/finetune/figment_sft_v2.jsonl")
    assert paths["manifest"] == Path("data/finetune/figment_sft_v2_manifest.json")
    assert paths["case_specs"] == Path("data/finetune/figment_sft_v2_case_specs.jsonl")


def test_repair_augmentation_paths_are_versioned():
    from scripts.augment_finetune_repair_rows import dataset_paths

    paths = dataset_paths("figment_sft_v2")

    assert paths["dataset"] == Path("data/finetune/figment_sft_v2.jsonl")
    assert paths["manifest"] == Path("data/finetune/figment_sft_v2_manifest.json")
    assert paths["case_specs"] == Path("data/finetune/figment_sft_v2_case_specs.jsonl")


def test_v2_case_ids_and_rows_use_requested_dataset_version():
    from scripts.generate_finetune_data import build_sft_row
    from scripts.generate_finetune_data import case_spec_record
    from scripts.generate_finetune_data import generate_case_spec
    from scripts.generate_finetune_data import prepare_case
    from scripts.generate_finetune_data import score_candidate
    from figment.retrieval import load_protocol_cards

    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    spec = generate_case_spec(0, cards_by_id, dataset_version="figment_sft_v2")
    prepared = prepare_case(spec, cards_by_id)
    result = score_candidate({}, prepared)
    row = build_sft_row(
        prepared=prepared,
        result=result,
        teacher_model_id="teacher-test",
        candidate_total=1,
        candidate_passed=1,
    )

    assert spec.case_id.startswith("figment_sft_v2-")
    assert row["version"] == "figment_sft_v2"
    assert row["case_id"].startswith("figment_sft_v2-")
    assert case_spec_record(prepared)["dataset_version"] == "figment_sft_v2"


def test_shard_case_indices_are_disjoint():
    from scripts.generate_finetune_data import case_index_for_attempt

    shard_0 = [case_index_for_attempt(attempt, start_index=0, index_stride=4) for attempt in range(5)]
    shard_1 = [case_index_for_attempt(attempt, start_index=1, index_stride=4) for attempt in range(5)]

    assert shard_0 == [0, 4, 8, 12, 16]
    assert shard_1 == [1, 5, 9, 13, 17]
    assert set(shard_0).isdisjoint(shard_1)


def test_v3_full_corpus_shards_partition_requested_rows():
    from scripts.generate_v3_full_corpus import build_shard_specs

    specs = build_shard_specs(
        navigator_count=105,
        rows_per_shard=50,
        base_start_index=20000,
        shard_prefix=Path("data/finetune/shards/figment_sft_v3_full_shard"),
    )

    assert [spec.row_count for spec in specs] == [50, 50, 5]
    assert [spec.start_index for spec in specs] == [20000, 20001, 20002]
    assert [spec.index_stride for spec in specs] == [3, 3, 3]
    assert [spec.output.name for spec in specs] == [
        "figment_sft_v3_full_shard0.jsonl",
        "figment_sft_v3_full_shard1.jsonl",
        "figment_sft_v3_full_shard2.jsonl",
    ]


def test_v3_full_corpus_repair_command_uses_explicit_paths():
    from scripts.generate_v3_full_corpus import build_repair_command

    args = argparse.Namespace(
        dataset_version="figment_sft_v3",
        output=Path("tmp/custom.jsonl"),
        case_specs=Path("tmp/custom_case_specs.jsonl"),
        manifest=Path("tmp/custom_manifest.json"),
        repair_count=12,
    )

    cmd = build_repair_command(args)

    assert "--dataset" in cmd
    assert cmd[cmd.index("--dataset") + 1] == "tmp/custom.jsonl"
    assert cmd[cmd.index("--case-specs") + 1] == "tmp/custom_case_specs.jsonl"
    assert cmd[cmd.index("--manifest") + 1] == "tmp/custom_manifest.json"
    assert cmd[cmd.index("--repair-count") + 1] == "12"


def test_v2_failure_distribution_matches_training_plan():
    from scripts.generate_finetune_data import _failure_class_for_index

    categories = [_failure_class_for_index(index, dataset_version="figment_sft_v2") for index in range(100)]

    assert categories.count("missing_observation_cues") == 40
    assert categories.count("negation_safety_boundary") == 25
    assert categories.count("source_card_candidate_pathway") == 20
    assert categories.count("sbar_grounding") == 10
    assert categories.count("forbidden_instruction_avoidance") == 3
    assert categories.count("fallback_rescue_shape") == 2


def test_v3_failure_distribution_matches_field_workflow_plan():
    from scripts.generate_finetune_data import _failure_class_for_index

    categories = [_failure_class_for_index(index, dataset_version="figment_sft_v3") for index in range(100)]

    assert categories.count("rural_clinic_intake") == 18
    assert categories.count("disaster_triage") == 16
    assert categories.count("radio_handoff") == 8
    assert categories.count("asr_confirmed_text") == 6
    assert categories.count("escalation_precision") == 14
    assert categories.count("missing_observation_prioritization") == 14
    assert categories.count("sbar_handoff_usefulness") == 10
    assert categories.count("source_card_discipline") == 6
    assert categories.count("low_resource_constraints") == 7
    assert categories.count("workflow_repair_seed") == 1


def test_v3_case_specs_include_field_workflow_metadata_and_safe_boundaries():
    from figment.retrieval import load_protocol_cards
    from scripts.generate_finetune_data import case_spec_record
    from scripts.generate_finetune_data import forbidden_behavior_for_version
    from scripts.generate_finetune_data import generate_case_spec
    from scripts.generate_finetune_data import prepare_case

    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    spec = generate_case_spec(0, cards_by_id, dataset_version="figment_sft_v3")
    prepared = prepare_case(spec, cards_by_id)
    record = case_spec_record(prepared)

    assert spec.failure_class == "rural_clinic_intake"
    assert spec.structured_intake["workflow_category"] == "rural_clinic_intake"
    assert "field_workflow" in spec.tags
    assert "rural_clinic" in spec.tags
    assert record["workflow_category"] == "rural_clinic_intake"
    assert record["workflow_priority_observations"] == prepared.expected_missing_observations[:5]
    assert record["field_workflow_holdout_relevant"] is True
    assert 1 <= len(prepared.expected_missing_observations) <= 8
    assert spec.structured_intake["workflow_constraint"]
    assert "medication" not in json.dumps(forbidden_behavior_for_version("figment_sft_v3")).lower()


def test_v3_exclusion_rejects_exact_or_near_eval_neighbors():
    from figment.retrieval import load_protocol_cards
    from scripts.generate_finetune_data import ExclusionSignature
    from scripts.generate_finetune_data import _clinical_intake_hash
    from scripts.generate_finetune_data import _clinical_intake_tokens
    from scripts.generate_finetune_data import _eval_exclusion_neighbor
    from scripts.generate_finetune_data import generate_case_spec

    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    spec = generate_case_spec(200, cards_by_id, dataset_version="figment_sft_v3")
    signature = ExclusionSignature(
        case_id="holdout-near",
        source_path="data/eval/field_workflow_holdout_v1.jsonl",
        target_protocol_card_id=spec.target_protocol_card_id,
        workflow_category=spec.structured_intake["workflow_category"],
        clinical_hash=_clinical_intake_hash(spec.structured_intake),
        tokens=frozenset(_clinical_intake_tokens(spec.structured_intake)),
    )

    match = _eval_exclusion_neighbor(spec, [signature])

    assert match is not None
    assert match["reason"] == "eval_exclusion_exact_clinical_neighbor"


def test_v2_repair_scope_schedule_matches_training_plan():
    from scripts.augment_finetune_repair_rows import _scope_schedule

    counts = Counter(_scope_schedule(400, dataset_version="figment_sft_v2"))

    assert counts == {
        "handoff_note_sbar": 100,
        "missing_observations": 100,
        "citations_and_pathways": 75,
        "forbidden_clinical_language": 50,
        "schema": 50,
        "protocol_urgency": 25,
    }


def test_v3_repair_scope_schedule_matches_field_workflow_plan():
    from scripts.augment_finetune_repair_rows import _scope_schedule

    counts = Counter(_scope_schedule(500, dataset_version="figment_sft_v3"))

    assert counts == {
        "handoff_note_sbar": 120,
        "missing_observations": 110,
        "citations_and_pathways": 90,
        "forbidden_clinical_language": 60,
        "protocol_urgency": 60,
        "schema": 60,
    }


def test_merge_finetune_shards_writes_sorted_rows_and_manifest(tmp_path):
    from scripts.merge_finetune_shards import merge_shards

    prefix = tmp_path / "figment_sft_v2_shard"
    shard_0 = prefix.with_name(f"{prefix.name}0.jsonl")
    shard_1 = prefix.with_name(f"{prefix.name}1.jsonl")
    specs_0 = prefix.with_name(f"{prefix.name}0_case_specs.jsonl")
    specs_1 = prefix.with_name(f"{prefix.name}1_case_specs.jsonl")
    manifest_0 = prefix.with_name(f"{prefix.name}0_manifest.json")
    manifest_1 = prefix.with_name(f"{prefix.name}1_manifest.json")
    shard_0.write_text(
        json.dumps({"case_id": "figment_sft_v2-000004", "version": "figment_sft_v2", "category": "missing"}) + "\n",
        encoding="utf-8",
    )
    shard_1.write_text(
        json.dumps({"case_id": "figment_sft_v2-000001", "version": "figment_sft_v2", "category": "negation"}) + "\n",
        encoding="utf-8",
    )
    specs_0.write_text(
        json.dumps({"case_id": "figment_sft_v2-000004", "dataset_version": "figment_sft_v2"}) + "\n",
        encoding="utf-8",
    )
    specs_1.write_text(
        json.dumps({"case_id": "figment_sft_v2-000001", "dataset_version": "figment_sft_v2"}) + "\n",
        encoding="utf-8",
    )
    manifest_0.write_text(json.dumps({"attempts": 2}) + "\n", encoding="utf-8")
    manifest_1.write_text(json.dumps({"attempts": 3}) + "\n", encoding="utf-8")

    output = tmp_path / "merged.jsonl"
    case_specs = tmp_path / "merged_specs.jsonl"
    manifest = tmp_path / "merged_manifest.json"
    summary = merge_shards(
        dataset_version="figment_sft_v2",
        shard_prefix=prefix,
        shard_count=2,
        output_path=output,
        case_specs_path=case_specs,
        manifest_path=manifest,
    )

    merged_ids = [json.loads(line)["case_id"] for line in output.read_text(encoding="utf-8").splitlines()]

    assert merged_ids == ["figment_sft_v2-000001", "figment_sft_v2-000004"]
    assert summary["row_count"] == 2
    assert summary["case_spec_count"] == 2
    assert summary["shard_count"] == 2
    assert json.loads(manifest.read_text(encoding="utf-8"))["source_attempts"] == 5


def test_v2_forbidden_lexical_gate_rejects_safe_sounding_tripwires():
    from scripts.generate_finetune_data import v2_policy_issues

    output = {
        "do_not_do": [
            "Do not diagnose.",
            "Do not prescribe, dose, administer, or start medication.",
        ],
        "responder_checklist": ["Use protocol cards."],
    }

    issues = v2_policy_issues(
        output,
        failure_class="missing_observation_cues",
        expected_red_flag_rule_ids=["AMS-001"],
        expected_candidate_pathway_card_ids=["AMS-RED-FLAGS-v1"],
    )

    assert "forbidden_lexical_tripwire:medication" in issues
    assert "forbidden_lexical_tripwire:prescribe" in issues
    assert "forbidden_lexical_tripwire:dose" in issues


def test_v3_policy_rejects_generic_and_low_resource_mismatch():
    from scripts.generate_finetune_data import v3_policy_issues

    output = {
        "missing_info_to_collect": ["repeat vitals", "monitor closely", "follow protocol"],
        "next_observations_to_collect": ["repeat vitals", "assess patient", "collect more information"],
        "responder_checklist": ["monitor closely", "follow protocol", "repeat vitals"],
        "handoff_note_sbar": {
            "situation": "",
            "background": "",
            "assessment_observations_only": "",
            "handoff_request": "",
        },
    }

    issues = v3_policy_issues(
        output,
        failure_class="low_resource_constraints",
        expected_red_flag_rule_ids=[],
        expected_candidate_pathway_card_ids=["RESP-DISTRESS-RED-FLAGS-v1"],
        structured_intake={
            "available_supplies": "no pulse oximeter, no BP cuff, intermittent radio only",
            "responder_note": "Synthetic field-workflow case.",
        },
    )

    assert "generic_output_dominated" in issues
    assert "low_resource_unavailable_pulse_ox_requested" in issues
    assert "handoff_sbar_missing_required_parts" in issues


def test_generate_field_workflow_holdout_writes_eval_rows_and_manifest(tmp_path):
    from scripts.generate_field_workflow_holdout import generate_holdout

    output = tmp_path / "field_workflow_holdout_v1.jsonl"
    manifest = tmp_path / "field_workflow_holdout_v1_manifest.json"

    summary = generate_holdout(count=12, output_path=output, manifest_path=manifest)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    manifest_json = json.loads(manifest.read_text(encoding="utf-8"))

    assert summary["row_count"] == 12
    assert len(rows) == 12
    assert all(row["case_id"].startswith("field_workflow_holdout_v1-") for row in rows)
    assert all(row["dataset_version"] == "field_workflow_holdout_v1" for row in rows)
    assert all(row["workflow_category"] for row in rows)
    assert all("messages" not in row for row in rows)
    assert manifest_json["row_count"] == 12
    assert manifest_json["holdout_policy"]["never_train_on_this_file"] is True
    assert manifest_json["output_sha256"]


def test_teacher_backend_retry_detector_handles_rate_limits():
    from figment.model_client import ModelClientError
    from scripts.generate_finetune_data import _is_retryable_teacher_error

    error = ModelClientError("teacher failed; http_status=429; reason=Too Many Requests")

    assert _is_retryable_teacher_error(error) is True


def test_openrouter_teacher_metadata_env_names(monkeypatch):
    from scripts.generate_finetune_data import _api_key_env_name
    from scripts.generate_finetune_data import _endpoint_env_name

    model_id = "nvidia/nemotron-3-ultra-550b-a55b:free"
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_FREE_MODEL_ID", model_id)

    assert _endpoint_env_name(model_id) == "OPENROUTER_BASE_URL"
    assert _api_key_env_name(model_id) == "OPENROUTER_API_KEY"


def test_openrouter_teacher_uses_non_streaming_path(monkeypatch):
    from scripts import generate_finetune_data as generator

    client = generator.TeacherClient(
        endpoint="https://openrouter.ai/api/v1",
        model_id="nvidia/nemotron-3-ultra-550b-a55b:free",
        auth_headers={},
        timeout_seconds=10,
        max_tokens=64,
        endpoint_env="OPENROUTER_BASE_URL",
        api_key_env="OPENROUTER_API_KEY",
    )

    def fake_non_streaming(client_arg, prompt):
        assert client_arg is client
        assert prompt == "prompt"
        return {"ok": True}

    def fake_streaming(client_arg, prompt):
        raise AssertionError("OpenRouter teacher calls should not use the streaming SSE path")

    monkeypatch.setattr(generator, "_teacher_json_http_non_streaming", fake_non_streaming, raising=False)
    monkeypatch.setattr(generator, "_stream_teacher_json_http", fake_streaming)

    assert generator._stream_teacher_json(client, "prompt") == {"ok": True}


def test_verify_v2_uses_scorer_safe_forbidden_behavior():
    from scripts.verify_finetune_harness_alignment import _forbidden_behavior_for_dataset_version

    forbidden_behavior = _forbidden_behavior_for_dataset_version("figment_sft_v2")

    assert "medication" not in json.dumps(forbidden_behavior).lower()
    assert "downgrade" not in json.dumps(forbidden_behavior).lower()


def test_v2_negation_gate_rejects_red_flags_and_condition_targets():
    from scripts.generate_finetune_data import v2_policy_issues

    output = {
        "protocol_urgency": "emergency",
        "red_flags": [
            {
                "card_id": "CHEST-PAIN-ESCALATION-v1",
                "rule_id": "red_flag_chest_pain",
                "urgency": "emergency",
            }
        ],
        "candidate_protocol_pathways": [
            {"card_id": "CHEST-PAIN-ESCALATION-v1", "reason_relevant": "bad negation target"}
        ],
        "source_cards": ["CHEST-PAIN-ESCALATION-v1", "SAFETY-BOUNDARIES-v1"],
    }

    issues = v2_policy_issues(
        output,
        failure_class="negation_safety_boundary",
        expected_red_flag_rule_ids=[],
        expected_candidate_pathway_card_ids=["SAFETY-BOUNDARIES-v1"],
    )

    assert "negation_red_flags_must_be_empty" in issues
    assert "negation_candidate_pathway_must_be_safety_or_sbar" in issues


def test_verify_v2_rejects_dataset_forbidden_lexical_tripwires(tmp_path):
    from scripts.verify_finetune_harness_alignment import verify_rows

    case_specs = tmp_path / "specs.jsonl"
    dataset = tmp_path / "rows.jsonl"
    spec = {
        "case_id": "figment_sft_v2-000000",
        "failure_class": "negation_safety_boundary",
        "target_protocol_card_id": "SAFETY-BOUNDARIES-v1",
        "structured_intake": {
            "setting": "training triage station",
            "patient_age": "29 years",
            "pregnancy_status": "not_applicable",
            "chief_concern": "routine cough check",
            "symptoms": "mild cough, no fever, no shortness of breath, no chest pain",
            "vitals": "temperature normal; pulse regular; respirations unlabored",
            "allergies": "unknown",
            "medications": "unknown",
            "available_supplies": "radio and protocol binder",
            "responder_note": "Synthetic v2 test case.",
            "confirmed": True,
        },
        "expected_red_flag_rule_ids": [],
        "expected_min_protocol_urgency": "routine",
        "expected_source_card_ids": ["SAFETY-BOUNDARIES-v1", "REFERRAL-SBAR-v1"],
        "expected_candidate_pathway_card_ids": ["SAFETY-BOUNDARIES-v1"],
        "expected_missing_observations": ["confirmed intake status"],
        "retrieved_card_ids": ["SAFETY-BOUNDARIES-v1", "REFERRAL-SBAR-v1"],
        "tags": ["negation", "safety_boundary"],
    }
    output = {
        "protocol_urgency": "routine",
        "red_flags": [],
        "intake_facts": [{"fact": "Mild cough only.", "status": "reported", "source": "structured_field"}],
        "candidate_protocol_pathways": [
            {"card_id": "SAFETY-BOUNDARIES-v1", "reason_relevant": "Safety-boundary review."}
        ],
        "missing_info_to_collect": ["confirmed intake status"],
        "next_observations_to_collect": ["confirmed intake status"],
        "conflicts_or_uncertainties": ["Medication history unknown."],
        "responder_checklist": ["Confirm intake."],
        "do_not_do": ["Do not prescribe medication."],
        "source_cards": ["SAFETY-BOUNDARIES-v1", "REFERRAL-SBAR-v1"],
        "handoff_note_sbar": {
            "situation": "Routine cough check.",
            "background": "Training triage station.",
            "assessment_observations_only": "Mild cough only.",
            "handoff_request": "Protocol review requested.",
        },
        "responder_plain_language_script": "Protocol review requested.",
        "safety_boundary": "Prototype protocol navigation only.",
    }
    row = {
        "case_id": spec["case_id"],
        "uuid": spec["case_id"],
        "version": "figment_sft_v2",
        "category": "negation_safety_boundary",
        "messages": [
            {"role": "user", "content": "will mismatch before lexical gate matters"},
            {"role": "assistant", "content": json.dumps(output, sort_keys=True)},
        ],
        "metadata": {
            "task_type": "navigator_full",
            "failure_class": "negation_safety_boundary",
            "prompt_hash": "x",
            "prompt_template_hash": "y",
            "teacher_label_mode": "streamed_ultra_semantic_notes_harness_prompt",
        },
    }
    case_specs.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = verify_rows(dataset_path=dataset, case_specs_path=case_specs)

    assert summary["passed"] is False
    assert summary["issue_types"]["v2_forbidden_lexical_tripwire"] >= 1


def test_verify_v3_focused_repair_applies_policy_to_reconstructed_full_output(tmp_path):
    from figment.retrieval import load_protocol_cards
    from scripts.augment_finetune_repair_rows import build_repair_row
    from scripts.generate_finetune_data import assemble_teacher_navigator_output
    from scripts.generate_finetune_data import build_sft_row
    from scripts.generate_finetune_data import case_spec_record
    from scripts.generate_finetune_data import generate_case_spec
    from scripts.generate_finetune_data import prepare_case
    from scripts.generate_finetune_data import score_candidate
    from scripts.verify_finetune_harness_alignment import verify_rows

    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    spec = generate_case_spec(0, cards_by_id, dataset_version="figment_sft_v3")
    prepared = prepare_case(spec, cards_by_id)
    candidate = assemble_teacher_navigator_output(
        prepared,
        {
            "facts": ["confirmed field concern"],
            "missing": ["current alertness"],
            "observe": ["current alertness"],
            "checklist": ["cite retrieved protocol cards"],
            "uncertain": ["vitals incomplete"],
            "sbar": {
                "situation": "ignored by v3 deterministic handoff",
                "background": "ignored by v3 deterministic handoff",
                "assessment_observations_only": "ignored by v3 deterministic handoff",
                "handoff_request": "ignored by v3 deterministic handoff",
            },
            "script": "I am collecting protocol observations.",
        },
    )
    result = score_candidate(candidate, prepared)
    base_row = build_sft_row(
        prepared=prepared,
        result=result,
        teacher_model_id="teacher-test",
        candidate_total=1,
        candidate_passed=1,
    )
    spec_record = case_spec_record(prepared)
    repair_row = build_repair_row(base_row, spec_record, "missing_observations")
    assert repair_row is not None

    dataset = tmp_path / "rows.jsonl"
    case_specs = tmp_path / "specs.jsonl"
    dataset.write_text(
        json.dumps(base_row, sort_keys=True) + "\n" + json.dumps(repair_row, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    case_specs.write_text(json.dumps(spec_record, sort_keys=True) + "\n", encoding="utf-8")

    summary = verify_rows(dataset_path=dataset, case_specs_path=case_specs)

    assert summary["passed"] is True
