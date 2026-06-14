"""Add focused-repair SFT rows that match Figment's local 4B harness."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figment.focused_repair import build_focused_repair_prompts  # noqa: E402
from figment.observation_targets import required_observation_targets  # noqa: E402
from figment.prompt_builder import build_prompt  # noqa: E402
from figment.retrieval import known_card_ids  # noqa: E402
from figment.retrieval import load_protocol_cards  # noqa: E402
from figment.retrieval import query_from_intake  # noqa: E402
from figment.retrieval import search_protocol_cards  # noqa: E402
from figment.rules import run_red_flag_checks  # noqa: E402
from figment.trace import stable_hash  # noqa: E402
from figment.validators import urgency_floor_from_rules  # noqa: E402
from figment.validators import validate_navigator_output  # noqa: E402
from scripts.generate_finetune_data import SAFETY_CARD_ID  # noqa: E402
from scripts.generate_finetune_data import SBAR_CARD_ID  # noqa: E402
from scripts.generate_finetune_data import _required_retrieved_ids  # noqa: E402
from scripts.generate_finetune_data import ensure_retrieved_cards  # noqa: E402
from scripts.generate_finetune_data import uses_v7_source_card_policy  # noqa: E402
from scripts.generate_finetune_data import v7_source_card_closure_issues  # noqa: E402


DATASET_PATH = Path("data/finetune/figment_sft_v1.jsonl")
CASE_SPEC_PATH = Path("data/finetune/figment_sft_v1_case_specs.jsonl")
MANIFEST_PATH = Path("data/finetune/figment_sft_v1_manifest.json")
REPAIR_SCOPES = (
    "missing_observations",
    "citations_and_pathways",
    "handoff_note_sbar",
    "forbidden_clinical_language",
    "protocol_urgency",
    "schema",
)
V2_REPAIR_SCOPE_DISTRIBUTION = (
    ("handoff_note_sbar", 100),
    ("missing_observations", 100),
    ("citations_and_pathways", 75),
    ("forbidden_clinical_language", 50),
    ("schema", 50),
    ("protocol_urgency", 25),
)
V3_REPAIR_SCOPE_DISTRIBUTION = (
    ("handoff_note_sbar", 120),
    ("missing_observations", 110),
    ("citations_and_pathways", 90),
    ("forbidden_clinical_language", 60),
    ("protocol_urgency", 60),
    ("schema", 60),
)
V4_REPAIR_SCOPE_DISTRIBUTION = (
    ("handoff_note_sbar", 45),
    ("citations_and_pathways", 25),
    ("missing_observations", 15),
    ("forbidden_clinical_language", 5),
    ("protocol_urgency", 5),
    ("schema", 5),
)
V5_REPAIR_SCOPE_DISTRIBUTION = (
    ("missing_observations", 55),
    ("handoff_note_sbar", 45),
    ("citations_and_pathways", 35),
    ("forbidden_clinical_language", 25),
    ("protocol_urgency", 20),
    ("schema", 20),
)
V6_REPAIR_SCOPE_DISTRIBUTION = (
    ("missing_observations", 250),
)
V7_REPAIR_SCOPE_DISTRIBUTION = (
    ("source_card_closure", 160),
    ("source_card_negative_correction", 50),
    ("observation_patch_repair", 30),
)
V7_SOURCE_REPAIR_SCOPES = {"source_card_closure", "source_card_negative_correction"}


def dataset_paths(dataset_version: str) -> dict[str, Path]:
    root = Path("data/finetune")
    return {
        "dataset": root / f"{dataset_version}.jsonl",
        "case_specs": root / f"{dataset_version}_case_specs.jsonl",
        "manifest": root / f"{dataset_version}_manifest.json",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default="figment_sft_v1")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--case-specs", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--repair-count", type=int, default=60)
    args = parser.parse_args(argv)
    paths = dataset_paths(args.dataset_version)
    args.dataset = args.dataset or paths["dataset"]
    args.case_specs = args.case_specs or paths["case_specs"]
    args.manifest = args.manifest or paths["manifest"]

    rows = _read_jsonl(args.dataset)
    specs = {str(item["case_id"]): item for item in _read_jsonl(args.case_specs)}
    existing_ids = {str(row.get("case_id")) for row in rows}
    base_rows = [row for row in rows if row.get("metadata", {}).get("task_type", "navigator_full") == "navigator_full"]
    repair_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for scope_name in _scope_schedule(args.repair_count, dataset_version=args.dataset_version):
        created = None
        for base_row in base_rows:
            candidate = build_repair_row(base_row, specs[str(base_row["case_id"])], scope_name)
            if candidate is None:
                continue
            if candidate["case_id"] in existing_ids:
                continue
            created = candidate
            break
        if created is None:
            skipped[scope_name] += 1
            continue
        repair_rows.append(created)
        rows.append(created)
        existing_ids.add(created["case_id"])

    args.dataset.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    update_manifest(args.manifest, args.dataset, args.case_specs, rows, repair_rows, skipped)
    print(
        json.dumps(
            {
                "base_rows": len(base_rows),
                "repair_rows_added": len(repair_rows),
                "total_rows": len(rows),
                "repair_scope_counts": dict(Counter(row["metadata"]["repair_scope"] for row in repair_rows)),
                "skipped": dict(skipped),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_repair_row(base_row: dict[str, Any], spec: dict[str, Any], scope_name: str) -> dict[str, Any] | None:
    intake = spec["structured_intake"]
    rule_results = [rule.to_dict() for rule in run_red_flag_checks(intake)]
    floor = urgency_floor_from_rules(rule_results)
    retrieved = search_protocol_cards(query_from_intake(intake), limit=6)
    spec_dataset_version = str(spec.get("dataset_version") or base_row.get("version") or "")
    if uses_v7_source_card_policy(spec_dataset_version):
        cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
        synthetic_spec = type(
            "SyntheticSpecForRepairGeneration",
            (),
            {
                "target_protocol_card_id": str(spec.get("target_protocol_card_id") or ""),
                "dataset_version": spec_dataset_version,
            },
        )()
        retrieved = ensure_retrieved_cards(
            retrieved,
            required_ids=_required_retrieved_ids(synthetic_spec, rule_results),
            cards_by_id=cards_by_id,
            limit=6,
        )
    retrieved_ids = [str(item.get("card_id", "")) for item in retrieved if item.get("card_id")]
    original_prompt, prompt_hash = build_prompt(intake, retrieved, rule_results, floor)
    gold_output = json.loads(base_row["messages"][1]["content"])
    previous_output = _corrupt_output(gold_output, scope_name, floor)
    if previous_output is None:
        return None
    validation = validate_navigator_output(
        previous_output,
        known_card_ids(),
        urgency_floor=floor,
        confirmed_intake=intake,
        rule_results=rule_results,
        retrieved_card_ids=set(retrieved_ids),
        retrieved_cards=retrieved,
        strict_schema=True,
    ).to_dict()
    failures = validation.get("failures") or []
    extra_failures = _extra_failures_for_scope(previous_output, spec, scope_name)
    failures = list(failures) + extra_failures
    if validation.get("passed") is True and not extra_failures:
        return None
    if not failures:
        return None
    focused_prompts = build_focused_repair_prompts(
        original_prompt=original_prompt,
        previous_output=previous_output,
        failures=failures,
        urgency_floor=floor,
        required_observation_targets=required_observation_targets(retrieved),
    )
    focused_prompt = next((item for item in focused_prompts if item.scope.name == scope_name), None)
    if focused_prompt is None:
        return None
    target = {field: gold_output[field] for field in focused_prompt.scope.fields if field in gold_output}
    if set(target) != set(focused_prompt.scope.fields):
        return None
    base_case_id = str(base_row["case_id"])
    case_id = f"{base_case_id}--repair-{scope_name}"
    return {
        "case_id": case_id,
        "uuid": case_id,
        "license": base_row.get("license", "synthetic internal training data"),
        "generator": base_row.get("generator"),
        "version": base_row.get("version", "figment_sft_v1"),
        "category": f"focused_repair:{scope_name}",
        "reasoning": "off",
        "messages": [
            {"role": "user", "content": focused_prompt.prompt},
            {"role": "assistant", "content": json.dumps(target, sort_keys=True)},
        ],
        "tags": sorted(set(base_row.get("tags", [])) | {"focused_repair", scope_name}),
        "metadata": {
            **base_row.get("metadata", {}),
            "task_type": "focused_repair",
            "base_case_id": base_case_id,
            "base_failure_class": base_row.get("metadata", {}).get("failure_class", base_row.get("category")),
            "failure_class": f"focused_repair:{scope_name}",
            "repair_scope": scope_name,
            "repair_fields": list(focused_prompt.scope.fields),
            "validation_failures": failures,
            "previous_output_hash": stable_hash(previous_output),
            "prompt_hash": stable_hash(focused_prompt.prompt),
            "prompt_template_hash": prompt_hash,
            "retrieved_card_ids": retrieved_ids,
            "teacher_label_mode": "teacher_gold_subset_focused_repair_harness_prompt",
            "expected_action": {
                "repair_scope": scope_name,
                "repair_fields": list(focused_prompt.scope.fields),
                "base_case_id": base_case_id,
            },
            "reward_components": {
                "repair_prompt_matches_harness": 1,
                "allowed_fields_only": 1,
                "target_fields_from_teacher_gold": 1,
                "no_visible_reasoning": 1,
            },
            "pass_rate_total": 1,
            "pass_rate_passed": 1,
            "raw_teacher_output_hash": base_row.get("metadata", {}).get("raw_teacher_output_hash"),
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }


def _corrupt_output(output: dict[str, Any], scope_name: str, floor: str) -> dict[str, Any] | None:
    corrupted = json.loads(json.dumps(output))
    if scope_name == "missing_observations":
        corrupted["missing_info_to_collect"] = ["repeat vitals"]
        corrupted["next_observations_to_collect"] = ["repeat vitals"]
    elif scope_name == "citations_and_pathways":
        corrupted["source_cards"] = []
        corrupted["candidate_protocol_pathways"] = [{"card_id": "NOT-A-CARD", "reason_relevant": "bad citation"}]
    elif scope_name == "handoff_note_sbar":
        corrupted["handoff_note_sbar"] = {
            "situation": "unsupported rash oxygen pregnancy finding",
            "background": "",
            "assessment_observations_only": "",
            "handoff_request": "",
        }
    elif scope_name == "forbidden_clinical_language":
        corrupted["responder_checklist"] = ["Diagnose the condition, prescribe medication, and discharge home."]
    elif scope_name == "protocol_urgency":
        if floor == "routine":
            return None
        corrupted["protocol_urgency"] = "routine"
    elif scope_name == "schema":
        corrupted.pop("safety_boundary", None)
        corrupted.pop("responder_plain_language_script", None)
    elif scope_name == "source_card_closure":
        source_cards = [str(card_id) for card_id in corrupted.get("source_cards", []) if str(card_id)]
        corrupted["source_cards"] = [card_id for card_id in source_cards if card_id not in {SAFETY_CARD_ID, SBAR_CARD_ID}]
    elif scope_name == "source_card_negative_correction":
        source_cards = [str(card_id) for card_id in corrupted.get("source_cards", []) if str(card_id)]
        for distractor in ("PED-DEHYD-RED-FLAGS-v1", "WOUND-INFECTION-ESCALATION-v1", "FEVER-RED-FLAGS-v1"):
            if distractor not in source_cards:
                source_cards.append(distractor)
                break
        corrupted["source_cards"] = source_cards
    elif scope_name == "observation_patch_repair":
        corrupted["missing_info_to_collect"] = ["source card ids", "deterministic rule results", "navigator validation result"]
        corrupted["next_observations_to_collect"] = list(corrupted["missing_info_to_collect"])
    else:
        return None
    return corrupted


def _extra_failures_for_scope(previous_output: dict[str, Any], spec: dict[str, Any], scope_name: str) -> list[str]:
    if scope_name == "source_card_closure":
        target = str(spec.get("target_protocol_card_id") or "")
        issues = v7_source_card_closure_issues(previous_output, target_protocol_card_id=target)
        return [f"source_card_closure:{issue}" for issue in issues]
    if scope_name == "source_card_negative_correction":
        return ["source_card_negative_correction:remove irrelevant or disallowed source card"]
    if scope_name == "observation_patch_repair":
        return ["observation_patch_repair:replace scaffold-owned observation fields"]
    return []


def _scope_schedule(count: int, *, dataset_version: str = "figment_sft_v1") -> list[str]:
    if dataset_version.startswith("figment_sft_v7"):
        return _weighted_scope_schedule(count, V7_REPAIR_SCOPE_DISTRIBUTION)
    if dataset_version.startswith("figment_sft_v6"):
        return _weighted_scope_schedule(count, V6_REPAIR_SCOPE_DISTRIBUTION)
    if dataset_version.startswith("figment_sft_v5"):
        return _weighted_scope_schedule(count, V5_REPAIR_SCOPE_DISTRIBUTION)
    if dataset_version.startswith("figment_sft_v4"):
        return _weighted_scope_schedule(count, V4_REPAIR_SCOPE_DISTRIBUTION)
    if dataset_version.startswith("figment_sft_v3"):
        return _weighted_scope_schedule(count, V3_REPAIR_SCOPE_DISTRIBUTION)
    if dataset_version == "figment_sft_v2":
        return _weighted_scope_schedule(count, V2_REPAIR_SCOPE_DISTRIBUTION)
    schedule = []
    while len(schedule) < count:
        schedule.extend(REPAIR_SCOPES)
    return schedule[:count]


def _weighted_scope_schedule(count: int, distribution: tuple[tuple[str, int], ...]) -> list[str]:
    if count <= 0:
        return []
    total_weight = sum(weight for _, weight in distribution)
    targets: dict[str, int] = {}
    fractions: list[tuple[float, int, str]] = []
    for order, (scope, weight) in enumerate(distribution):
        exact = count * weight / total_weight
        targets[scope] = int(exact)
        fractions.append((exact - int(exact), -order, scope))
    for _, _, scope in sorted(fractions, reverse=True)[: count - sum(targets.values())]:
        targets[scope] += 1

    produced: Counter[str] = Counter()
    schedule: list[str] = []
    scope_order = {scope: index for index, (scope, _) in enumerate(distribution)}
    while len(schedule) < count:
        remaining_scopes = [scope for scope, target in targets.items() if produced[scope] < target]
        scope = max(
            remaining_scopes,
            key=lambda item: (
                (targets[item] - produced[item]) / targets[item],
                -scope_order[item],
            ),
        )
        schedule.append(scope)
        produced[scope] += 1
    return schedule


def update_manifest(
    manifest_path: Path,
    dataset_path: Path,
    case_specs_path: Path,
    rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
    skipped: Counter[str],
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    task_counts = Counter(row.get("metadata", {}).get("task_type", "navigator_full") for row in rows)
    category_counts = Counter(str(row.get("category", "")) for row in rows)
    repair_scope_counts = Counter(
        row.get("metadata", {}).get("repair_scope")
        for row in rows
        if row.get("metadata", {}).get("task_type") == "focused_repair"
    )
    manifest.update(
        {
            "row_count": len(rows),
            "output_sha256": _file_sha256(dataset_path),
            "case_specs_sha256": _file_sha256(case_specs_path),
            "task_type_counts": dict(sorted(task_counts.items())),
            "category_counts": dict(sorted(category_counts.items())),
            "repair_scope_counts": {
                str(key): value for key, value in sorted(repair_scope_counts.items()) if key
            },
            "focused_repair_rows_added": len(repair_rows),
            "focused_repair_skipped": dict(skipped),
            "harness_task_coverage": {
                "navigator_full": task_counts.get("navigator_full", 0),
                "focused_repair": task_counts.get("focused_repair", 0),
                "audio_field_draft": "not a 4B chat-completion task in this harness; Parakeet/provider payload plus deterministic field drafting",
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
