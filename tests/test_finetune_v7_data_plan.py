import json
from pathlib import Path


def _accepted_v7_row():
    from figment.retrieval import load_protocol_cards
    from scripts.generate_finetune_data import assemble_teacher_navigator_output
    from scripts.generate_finetune_data import build_sft_row
    from scripts.generate_finetune_data import case_spec_record
    from scripts.generate_finetune_data import generate_case_spec
    from scripts.generate_finetune_data import prepare_case
    from scripts.generate_finetune_data import score_candidate

    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    spec = generate_case_spec(6, cards_by_id, dataset_version="figment_sft_v7_delta")
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


def test_v7_source_card_closure_policy_flags_missing_support_and_target_cards():
    from scripts.generate_finetune_data import v7_source_card_closure_issues

    output = {
        "source_cards": ["STROKE-SIGNS-v1"],
        "safety_boundary": "Use local protocol and do not provide treatment instructions.",
        "do_not_do": ["Do not diagnose."],
        "handoff_note_sbar": {
            "situation": "stroke signs",
            "background": "field setting",
            "assessment_observations_only": "face droop observed",
            "handoff_request": "request protocol review",
        },
    }

    issues = v7_source_card_closure_issues(output, target_protocol_card_id="CHEST-PAIN-ESCALATION-v1")

    assert "missing_target_source_card:CHEST-PAIN-ESCALATION-v1" in issues
    assert "missing_referral_sbar_source_card" in issues
    assert "missing_safety_boundaries_source_card" in issues


def test_verify_v7_rejects_missing_referral_sbar_source_card(tmp_path):
    from scripts.verify_finetune_harness_alignment import verify_rows

    row, spec_record = _accepted_v7_row()
    output = json.loads(row["messages"][1]["content"])
    output["source_cards"] = [card_id for card_id in output["source_cards"] if card_id != "REFERRAL-SBAR-v1"]
    row["messages"][1]["content"] = json.dumps(output, sort_keys=True)

    dataset = tmp_path / "rows.jsonl"
    case_specs = tmp_path / "specs.jsonl"
    dataset.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    case_specs.write_text(json.dumps(spec_record, sort_keys=True) + "\n", encoding="utf-8")

    summary = verify_rows(dataset_path=dataset, case_specs_path=case_specs)

    assert summary["passed"] is False
    assert summary["issue_types"]["v7_missing_referral_sbar_source_card"] >= 1


def test_verify_v7_rejects_missing_safety_boundaries_source_card(tmp_path):
    from scripts.verify_finetune_harness_alignment import verify_rows

    row, spec_record = _accepted_v7_row()
    output = json.loads(row["messages"][1]["content"])
    output["source_cards"] = [card_id for card_id in output["source_cards"] if card_id != "SAFETY-BOUNDARIES-v1"]
    row["messages"][1]["content"] = json.dumps(output, sort_keys=True)

    dataset = tmp_path / "rows.jsonl"
    case_specs = tmp_path / "specs.jsonl"
    dataset.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    case_specs.write_text(json.dumps(spec_record, sort_keys=True) + "\n", encoding="utf-8")

    summary = verify_rows(dataset_path=dataset, case_specs_path=case_specs)

    assert summary["passed"] is False
    assert summary["issue_types"]["v7_missing_safety_boundaries_source_card"] >= 1


def test_verify_v7_rejects_missing_target_source_card(tmp_path):
    from scripts.verify_finetune_harness_alignment import verify_rows

    row, spec_record = _accepted_v7_row()
    output = json.loads(row["messages"][1]["content"])
    target_card_id = spec_record["target_protocol_card_id"]
    output["source_cards"] = [card_id for card_id in output["source_cards"] if card_id != target_card_id]
    row["messages"][1]["content"] = json.dumps(output, sort_keys=True)

    dataset = tmp_path / "rows.jsonl"
    case_specs = tmp_path / "specs.jsonl"
    dataset.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    case_specs.write_text(json.dumps(spec_record, sort_keys=True) + "\n", encoding="utf-8")

    summary = verify_rows(dataset_path=dataset, case_specs_path=case_specs)

    assert summary["passed"] is False
    assert summary["issue_types"][f"v7_missing_target_source_card:{target_card_id}"] >= 1


def _replay_row(
    *,
    case_id: str,
    output: dict,
    category: str = "general_regression",
    dataset_version: str = "figment_sft_v6_delta",
    task_type: str | None = None,
) -> dict:
    metadata = {
        "dataset_version": dataset_version,
        "validator_passed": True,
        "validation_result": {"passed": True, "failures": []},
        "expected_label_score": {
            "all_expected_labels_passed": True,
            "forbidden_behavior_absent": True,
        },
        "must_include_source_cards": ["STROKE-SIGNS-v1", "SAFETY-BOUNDARIES-v1", "REFERRAL-SBAR-v1"],
    }
    if task_type:
        metadata["task_type"] = task_type
    return {
        "case_id": case_id,
        "category": category,
        "version": dataset_version,
        "metadata": metadata,
        "messages": [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": json.dumps(output, sort_keys=True)},
        ],
    }


def test_v7_replay_audit_rejects_full_rows_missing_support_cards():
    from scripts.build_v7_replay_corpus import audit_row

    row = _replay_row(
        case_id="missing-support",
        output={
            "protocol_urgency": "urgent",
            "source_cards": ["STROKE-SIGNS-v1"],
            "safety_boundary": "Use local protocol and do not provide treatment instructions.",
            "handoff_note_sbar": {
                "situation": "stroke signs",
                "background": "field setting",
                "assessment_observations_only": "face droop observed",
                "handoff_request": "request protocol review",
            },
            "missing_info_to_collect": ["confirm current alertness"],
            "next_observations_to_collect": ["confirm current alertness"],
        },
    )

    result = audit_row(row)

    assert result.accepted is False
    assert "missing_referral_sbar_source_card" in result.reasons
    assert "missing_safety_boundaries_source_card" in result.reasons


def test_build_v7_replay_corpus_selects_target_buckets_and_reversions_rows(tmp_path: Path):
    from scripts.build_v7_replay_corpus import build_replay_corpus

    clean_full = _replay_row(
        case_id="clean-full",
        output={
            "protocol_urgency": "urgent",
            "source_cards": ["STROKE-SIGNS-v1", "SAFETY-BOUNDARIES-v1", "REFERRAL-SBAR-v1"],
            "safety_boundary": "Use local protocol and do not provide treatment instructions.",
            "handoff_note_sbar": {
                "situation": "stroke signs",
                "background": "field setting",
                "assessment_observations_only": "face droop observed",
                "handoff_request": "request protocol review",
            },
            "missing_info_to_collect": ["confirm current alertness"],
            "next_observations_to_collect": ["confirm current alertness"],
        },
    )
    clean_repair = _replay_row(
        case_id="clean-repair",
        output={"source_cards": ["STROKE-SIGNS-v1", "SAFETY-BOUNDARIES-v1", "REFERRAL-SBAR-v1"]},
        category="focused_repair:citations_and_pathways",
        dataset_version="figment_sft_v3",
        task_type="focused_repair",
    )
    delta_path = tmp_path / "figment_sft_v6_delta.jsonl"
    replay_path = tmp_path / "figment_sft_v6_replay.jsonl"
    delta_path.write_text(json.dumps(clean_full, sort_keys=True) + "\n", encoding="utf-8")
    replay_path.write_text(json.dumps(clean_repair, sort_keys=True) + "\n", encoding="utf-8")

    output_path = tmp_path / "selected.jsonl"
    manifest_path = tmp_path / "manifest.json"
    summary = build_replay_corpus(
        input_paths=[delta_path, replay_path],
        output_path=output_path,
        manifest_path=manifest_path,
        targets={"figment_sft_v6_delta": 1, "figment_sft_v6_replay": 1},
        seed="test",
    )
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert summary["selected_rows"] == 2
    assert summary["selected_by_source_bucket"] == {
        "figment_sft_v6_delta": 1,
        "figment_sft_v6_replay": 1,
    }
    assert {row["version"] for row in rows} == {"figment_sft_v7_replay"}
    assert all(row["metadata"]["v7_replay_audit"]["accepted"] is True for row in rows)


def test_v7_failure_cycle_matches_planned_navigator_counts():
    from scripts.generate_finetune_data import V7_NAVIGATOR_COUNTS
    from scripts.generate_finetune_data import _failure_class_for_index

    counts = {
        name: sum(
            1
            for index in range(sum(V7_NAVIGATOR_COUNTS.values()))
            if _failure_class_for_index(index, dataset_version="figment_sft_v7_delta") == name
        )
        for name in V7_NAVIGATOR_COUNTS
    }

    assert counts == V7_NAVIGATOR_COUNTS
    assert {
        _failure_class_for_index(index, dataset_version="figment_sft_v7_delta")
        for index in range(20)
    } == set(V7_NAVIGATOR_COUNTS)


def test_v7_prepare_case_retrieves_mandatory_support_cards():
    from figment.retrieval import load_protocol_cards
    from scripts.generate_finetune_data import SAFETY_CARD_ID
    from scripts.generate_finetune_data import SBAR_CARD_ID
    from scripts.generate_finetune_data import generate_case_spec
    from scripts.generate_finetune_data import prepare_case

    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    spec = generate_case_spec(0, cards_by_id, dataset_version="figment_sft_v7_delta")
    prepared = prepare_case(spec, cards_by_id)

    assert spec.failure_class == "source_card_closure"
    assert SAFETY_CARD_ID in prepared.retrieved_ids
    assert SBAR_CARD_ID in prepared.retrieved_ids
    assert SAFETY_CARD_ID in prepared.expected_source_card_ids
    assert SBAR_CARD_ID in prepared.expected_source_card_ids


def test_v7_repair_scope_schedule_matches_planned_counts():
    from collections import Counter

    from scripts.augment_finetune_repair_rows import V7_REPAIR_SCOPE_DISTRIBUTION
    from scripts.augment_finetune_repair_rows import _scope_schedule

    schedule = _scope_schedule(240, dataset_version="figment_sft_v7_delta")

    assert Counter(schedule) == dict(V7_REPAIR_SCOPE_DISTRIBUTION)


def test_generate_v7_full_corpus_defaults_and_overrides():
    from scripts.generate_v7_full_corpus import DEFAULT_NAVIGATOR_COUNT
    from scripts.generate_v7_full_corpus import build_corpus_args

    defaults = build_corpus_args([])
    overridden = build_corpus_args(["--navigator-count", "12", "--repair-count", "5", "--dry-run"])

    assert defaults[defaults.index("--dataset-version") + 1] == "figment_sft_v7_delta"
    assert defaults[defaults.index("--navigator-count") + 1] == str(DEFAULT_NAVIGATOR_COUNT)
    assert defaults[defaults.index("--output") + 1] == "data/finetune/figment_sft_v7_delta.jsonl"
    assert overridden[overridden.index("--navigator-count") + 1] == "12"
    assert overridden[overridden.index("--repair-count") + 1] == "5"
    assert "--dry-run" in overridden


def test_merge_v7_corpus_resolves_replay_case_specs(tmp_path: Path, monkeypatch):
    import scripts.merge_v7_training_corpus as merge_v7

    delta_row = _replay_row(
        case_id="figment_sft_v7_delta-080000",
        output={
            "protocol_urgency": "urgent",
            "source_cards": ["STROKE-SIGNS-v1", "SAFETY-BOUNDARIES-v1", "REFERRAL-SBAR-v1"],
        },
        category="source_card_closure",
        dataset_version="figment_sft_v7_delta",
    )
    replay_row = _replay_row(
        case_id="figment_sft_v6_delta-070000",
        output={
            "protocol_urgency": "urgent",
            "source_cards": ["STROKE-SIGNS-v1", "SAFETY-BOUNDARIES-v1", "REFERRAL-SBAR-v1"],
        },
        category="required_observation_ownership",
        dataset_version="figment_sft_v7_replay",
    )
    replay_row["metadata"]["v7_replay_audit"] = {
        "accepted": True,
        "original_source_dataset_version": "figment_sft_v6_delta",
        "source_bucket": "figment_sft_v6_delta",
    }

    delta_path = tmp_path / "delta.jsonl"
    replay_path = tmp_path / "replay.jsonl"
    delta_specs = tmp_path / "delta_specs.jsonl"
    output = tmp_path / "merged.jsonl"
    specs = tmp_path / "merged_specs.jsonl"
    manifest = tmp_path / "manifest.json"
    source_specs = tmp_path / "v6_delta_specs.jsonl"
    delta_path.write_text(json.dumps(delta_row, sort_keys=True) + "\n", encoding="utf-8")
    replay_path.write_text(json.dumps(replay_row, sort_keys=True) + "\n", encoding="utf-8")
    delta_specs.write_text(
        json.dumps({"case_id": "figment_sft_v7_delta-080000", "structured_intake": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_specs.write_text(
        json.dumps({"case_id": "figment_sft_v6_delta-070000", "structured_intake": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(merge_v7.SOURCE_CASE_SPECS, "figment_sft_v6_delta", source_specs)

    summary = merge_v7.merge_v7_corpus(
        delta_path=delta_path,
        delta_case_specs_path=delta_specs,
        replay_path=replay_path,
        output_path=output,
        case_specs_path=specs,
        manifest_path=manifest,
        dataset_version="figment_sft_v7",
    )

    assert summary["row_count"] == 2
    assert summary["replay_source_counts"] == {"figment_sft_v6_delta": 1}
