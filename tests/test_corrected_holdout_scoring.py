import json
from pathlib import Path

from scripts.build_corrected_field_workflow_holdout import build_corrected_view


def test_corrected_holdout_scoring_removes_negated_chest_pain_labels(tmp_path: Path) -> None:
    output = tmp_path / "field_workflow_holdout_v1_corrected_scoring.jsonl"
    manifest = tmp_path / "field_workflow_holdout_v1_corrected_scoring_manifest.json"

    result = build_corrected_view(
        input_path=Path("data/eval/field_workflow_holdout_v1.jsonl"),
        output_path=output,
        manifest_path=manifest,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    by_id = {row["case_id"]: row for row in rows}

    assert result["row_count"] == 150
    assert by_id["field_workflow_holdout_v1-000050"]["expected_red_flag_rule_ids"] == []
    assert by_id["field_workflow_holdout_v1-000050"]["expected_min_protocol_urgency"] == "routine"
    assert "CHEST-PAIN-ESCALATION-v1" not in by_id["field_workflow_holdout_v1-000050"]["expected_source_card_ids"]
    assert "red_flag_chest_pain" not in by_id["field_workflow_holdout_v1-000019"]["expected_red_flag_rule_ids"]
    assert manifest.exists()
