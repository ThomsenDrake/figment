import json
from pathlib import Path


def _row(
    *,
    case_id: str = "case-1",
    category: str = "focused_repair:handoff_note_sbar",
    output: dict | None = None,
    dataset_version: str = "figment_sft_v5",
    metadata: dict | None = None,
) -> dict:
    assistant_output = output or {
        "handoff_note_sbar": {
            "situation": "confirmed field handoff",
            "background": "low-resource setting",
            "assessment_observations_only": "observations only",
            "handoff_request": "request protocol review",
        }
    }
    merged_metadata = {
        "dataset_version": dataset_version,
        "task_type": "focused_repair" if "protocol_urgency" not in assistant_output else "navigator_full",
        "validator_passed": True,
        "validation_result": {"passed": True, "failures": []},
        "expected_label_score": {
            "all_expected_labels_passed": True,
            "forbidden_behavior_absent": True,
        },
    }
    if metadata:
        merged_metadata.update(metadata)
    return {
        "case_id": case_id,
        "category": category,
        "version": dataset_version,
        "metadata": merged_metadata,
        "messages": [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": json.dumps(assistant_output, sort_keys=True)},
        ],
    }


def test_v6_replay_audit_rejects_duplicate_long_observation_lists():
    from scripts.build_v6_replay_corpus import audit_row

    row = _row(
        output={
            "protocol_urgency": "urgent",
            "missing_info_to_collect": ["a", "b", "c", "d"],
            "next_observations_to_collect": ["a", "b", "c", "d"],
        },
        category="sbar_observation_ownership",
    )

    result = audit_row(row)

    assert result.accepted is False
    assert "duplicate_long_missing_and_next_observations" in result.reasons


def test_v6_replay_audit_rejects_harness_metadata_in_observation_fields():
    from scripts.build_v6_replay_corpus import audit_row

    row = _row(
        output={
            "protocol_urgency": "urgent",
            "missing_info_to_collect": ["retrieve source protocol card IDs"],
            "next_observations_to_collect": ["count respiratory rate"],
        },
        category="general_regression",
    )

    result = audit_row(row)

    assert result.accepted is False
    assert "harness_metadata_observation:source_protocol_card_ids" in result.reasons


def test_v6_replay_audit_requires_selected_ids_for_observation_focused_full_rows():
    from scripts.build_v6_replay_corpus import audit_row

    row = _row(
        output={
            "protocol_urgency": "urgent",
            "missing_info_to_collect": ["count respiratory rate"],
            "next_observations_to_collect": ["count respiratory rate"],
        },
        category="required_observation_id_selection",
    )

    result = audit_row(row)

    assert result.accepted is False
    assert "observation_focused_row_missing_selected_required_observation_ids" in result.reasons


def test_v6_replay_audit_accepts_clean_non_observation_repair_row():
    from scripts.build_v6_replay_corpus import audit_row

    result = audit_row(_row())

    assert result.accepted is True
    assert result.reasons == ()


def test_build_v6_replay_corpus_writes_only_clean_rows(tmp_path: Path):
    from scripts.build_v6_replay_corpus import build_replay_corpus

    input_path = tmp_path / "rows.jsonl"
    clean = _row(case_id="clean")
    bad = _row(
        case_id="bad",
        output={
            "protocol_urgency": "urgent",
            "missing_info_to_collect": ["source protocol card IDs"],
            "next_observations_to_collect": ["source protocol card IDs"],
        },
        category="general_regression",
    )
    input_path.write_text(
        json.dumps(clean, sort_keys=True) + "\n" + json.dumps(bad, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "selected.jsonl"
    manifest_path = tmp_path / "manifest.json"

    summary = build_replay_corpus(
        input_paths=[input_path],
        output_path=output_path,
        manifest_path=manifest_path,
        targets={"figment_sft_v5": 2},
        seed="test",
    )

    selected_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert summary["selected_rows"] == 1
    assert len(selected_rows) == 1
    assert selected_rows[0]["case_id"] == "clean"
    assert selected_rows[0]["metadata"]["v6_replay_audit"]["accepted"] is True
    assert summary["shortage_by_source_dataset_version"] == {"figment_sft_v5": 1}
    assert summary["rejected_reason_counts"] == {
        "figment_sft_v5:harness_metadata_observation:source_protocol_card_ids": 1
    }

