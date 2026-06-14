"""Build a corrected scoring view for the frozen field-workflow holdout.

The original holdout file remains frozen. This script preserves case IDs and
structured intakes, then recomputes expected labels with the current rule and
prompt-building code.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figment.prompt_builder import build_prompt  # noqa: E402
from figment.retrieval import query_from_intake  # noqa: E402
from figment.retrieval import load_protocol_cards  # noqa: E402
from figment.retrieval import search_protocol_cards  # noqa: E402
from figment.rules import run_red_flag_checks  # noqa: E402
from figment.validators import urgency_floor_from_rules  # noqa: E402
from scripts.generate_finetune_data import stable_hash  # noqa: E402


BASE_VERSION = "field_workflow_holdout_v1"
DERIVED_VERSION = "field_workflow_holdout_v1_corrected_scoring"
DEFAULT_INPUT_PATH = Path("data/eval/field_workflow_holdout_v1.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/eval/field_workflow_holdout_v1_corrected_scoring.jsonl")
DEFAULT_MANIFEST_PATH = Path("data/eval/field_workflow_holdout_v1_corrected_scoring_manifest.json")
RULE_CARD_IDS = {
    "AMS-001": "AMS-RED-FLAGS-v1",
    "RESP-001": "RESP-DISTRESS-RED-FLAGS-v1",
    "PREG-001": "PREG-DANGER-SIGNS-v1",
    "red_flag_chest_pain": "CHEST-PAIN-ESCALATION-v1",
    "STROKE-001": "STROKE-SIGNS-v1",
    "PED-DEHYD-001": "PED-DEHYD-RED-FLAGS-v1",
    "FEVER-001": "FEVER-RED-FLAGS-v1",
    "WOUND-001": "WOUND-INFECTION-ESCALATION-v1",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args(argv)

    manifest = build_corrected_view(input_path=args.input, output_path=args.output, manifest_path=args.manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def build_corrected_view(*, input_path: Path, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    input_rows = _load_jsonl(input_path)
    output_rows: list[dict[str, Any]] = []
    row_hashes: list[dict[str, str]] = []
    changed_cases: list[dict[str, Any]] = []

    for row in input_rows:
        corrected = _correct_row(row, cards_by_id)
        output_rows.append(corrected)
        row_hashes.append({"case_id": corrected["case_id"], "sha256": _json_sha256(corrected)})
        diff = _label_diff(row, corrected)
        if diff:
            changed_cases.append({"case_id": corrected["case_id"], "changes": diff})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )

    manifest = {
        "dataset_version": DERIVED_VERSION,
        "derived_from_dataset_version": BASE_VERSION,
        "source_path": str(input_path),
        "source_sha256": _file_sha256(input_path),
        "output_path": str(output_path),
        "output_sha256": _file_sha256(output_path),
        "row_count": len(output_rows),
        "changed_case_count": len(changed_cases),
        "changed_cases": changed_cases,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_generator": "scripts/build_corrected_field_workflow_holdout.py",
        "policy": {
            "preserve_frozen_holdout_file": True,
            "preserve_case_ids": True,
            "recompute_expected_labels_with_current_rules": True,
        },
        "row_hashes": row_hashes,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _correct_row(row: dict[str, Any], cards_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    intake = dict(row["structured_intake"])
    rule_results = [rule.to_dict() for rule in run_red_flag_checks(intake)]
    urgency_floor = urgency_floor_from_rules(rule_results)
    retrieved = search_protocol_cards(query_from_intake(intake), limit=6)
    retrieved_ids = [str(item.get("card_id", "")) for item in retrieved if item.get("card_id")]
    prompt, prompt_template_hash = build_prompt(intake, retrieved, rule_results, urgency_floor)

    expected_red_flags = [str(rule["rule_id"]) for rule in rule_results]
    current_rule_ids = {str(rule.get("rule_id")) for rule in rule_results if rule.get("rule_id")}
    removed_rule_cards = _removed_rule_cards(row, current_rule_ids)
    expected_source_cards = _correct_expected_source_cards(
        row,
        rule_results,
        retrieved_ids,
        removed_rule_cards=removed_rule_cards,
    )
    expected_missing = _remove_required_observations_for_removed_rule_cards(
        row,
        expected_source_cards=expected_source_cards,
        cards_by_id=cards_by_id,
        removed_rule_cards=removed_rule_cards,
    )

    corrected = dict(row)
    corrected.update(
        {
            "dataset_version": DERIVED_VERSION,
            "base_dataset_version": str(row.get("dataset_version") or BASE_VERSION),
            "expected_red_flag_rule_ids": expected_red_flags,
            "expected_min_protocol_urgency": urgency_floor,
            "expected_source_card_ids": expected_source_cards,
            "expected_missing_observations": expected_missing,
            "retrieved_card_ids": retrieved_ids,
            "workflow_priority_observations": expected_missing[:5],
            "prompt_hash": stable_hash(prompt),
            "prompt_template_hash": prompt_template_hash,
        }
    )
    for key in (
        "expected_model_observation_cues",
        "expected_handoff_cues",
        "expected_harness_evidence_cues",
    ):
        corrected.pop(key, None)
    return corrected


def _correct_expected_source_cards(
    row: dict[str, Any],
    rule_results: list[dict[str, Any]],
    retrieved_ids: list[str],
    *,
    removed_rule_cards: set[str],
) -> list[str]:
    expected = [str(card_id) for card_id in row.get("expected_source_card_ids", []) if str(card_id)]
    target_card_id = str(row.get("target_protocol_card_id") or "")
    current_rule_cards = {str(rule.get("card_id")) for rule in rule_results if rule.get("card_id")}

    corrected = [
        card_id
        for card_id in expected
        if card_id not in removed_rule_cards or card_id == target_card_id or card_id in current_rule_cards
    ]
    for card_id in current_rule_cards:
        if card_id in retrieved_ids and card_id not in corrected:
            corrected.append(card_id)
    return corrected


def _remove_required_observations_for_removed_rule_cards(
    row: dict[str, Any],
    *,
    expected_source_cards: list[str],
    cards_by_id: dict[str, dict[str, Any]],
    removed_rule_cards: set[str],
) -> list[str]:
    if not removed_rule_cards:
        return [str(item) for item in row.get("expected_missing_observations", []) if str(item)]

    removed_required: set[str] = set()
    for card_id in removed_rule_cards:
        removed_required.update(_required_observations(cards_by_id.get(card_id, {})))

    remaining_required: set[str] = set()
    for card_id in expected_source_cards:
        remaining_required.update(_required_observations(cards_by_id.get(card_id, {})))

    corrected: list[str] = []
    for cue in [str(item) for item in row.get("expected_missing_observations", []) if str(item)]:
        if cue in removed_required and cue not in remaining_required:
            continue
        corrected.append(cue)
    return corrected


def _removed_rule_cards(row: dict[str, Any], current_rule_ids: set[str]) -> set[str]:
    old_rule_ids = {str(item) for item in row.get("expected_red_flag_rule_ids", [])}
    return _removed_rule_cards_from_ids(old_rule_ids=old_rule_ids, new_rule_ids=current_rule_ids)


def _removed_rule_cards_from_ids(*, old_rule_ids: set[str], new_rule_ids: set[str]) -> set[str]:
    return {RULE_CARD_IDS[rule_id] for rule_id in old_rule_ids - new_rule_ids if rule_id in RULE_CARD_IDS}


def _required_observations(card: dict[str, Any]) -> set[str]:
    values = card.get("required_observations")
    if not isinstance(values, list):
        return set()
    return {str(item) for item in values if str(item)}


def _label_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    keys = (
        "expected_red_flag_rule_ids",
        "expected_min_protocol_urgency",
        "expected_source_card_ids",
        "expected_candidate_pathway_card_ids",
        "expected_missing_observations",
    )
    for key in keys:
        if before.get(key) != after.get(key):
            diff[key] = {"before": before.get(key), "after": after.get(key)}
    return diff


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_sha256(value: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
