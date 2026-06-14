"""Merge Figment v8 delta rows with the v7 corpus and prepare Modal splits."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figment.focused_repair import build_focused_repair_prompts  # noqa: E402
from figment.eval_metrics import bucket_expected_observation_cues  # noqa: E402
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
from scripts.augment_finetune_repair_rows import _corrupt_output  # noqa: E402
from scripts.augment_finetune_repair_rows import _extra_failures_for_scope  # noqa: E402
from scripts.generate_finetune_data import _required_retrieved_ids  # noqa: E402
from scripts.generate_finetune_data import _expected_candidate_cards  # noqa: E402
from scripts.generate_finetune_data import _expected_missing_observations  # noqa: E402
from scripts.generate_finetune_data import _expected_source_cards  # noqa: E402
from scripts.generate_finetune_data import ensure_retrieved_cards  # noqa: E402
from scripts.generate_finetune_data import uses_v7_source_card_policy  # noqa: E402


DEFAULT_BASE = Path("data/finetune/figment_sft_v7.jsonl")
DEFAULT_BASE_CASE_SPECS = Path("data/finetune/figment_sft_v7_case_specs.jsonl")
DEFAULT_DELTA = Path("data/finetune/figment_sft_v8_delta.jsonl")
DEFAULT_DELTA_CASE_SPECS = Path("data/finetune/figment_sft_v8_delta_case_specs.jsonl")
DEFAULT_OUTPUT = Path("data/finetune/figment_sft_v8.jsonl")
DEFAULT_CASE_SPECS = Path("data/finetune/figment_sft_v8_case_specs.jsonl")
DEFAULT_MANIFEST = Path("data/finetune/figment_sft_v8_manifest.json")
DEFAULT_MODAL_DIR = Path("data/finetune/modal/figment_sft_v8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--base-case-specs", type=Path, default=DEFAULT_BASE_CASE_SPECS)
    parser.add_argument("--delta", type=Path, default=DEFAULT_DELTA)
    parser.add_argument("--delta-case-specs", type=Path, default=DEFAULT_DELTA_CASE_SPECS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-specs", type=Path, default=DEFAULT_CASE_SPECS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--modal-output-dir", type=Path, default=DEFAULT_MODAL_DIR)
    parser.add_argument("--dataset-version", default="figment_sft_v8")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", default="figment-modal-sft-v8")
    parser.add_argument("--min-validation-group-size", type=int, default=5)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-modal-prep", action="store_true")
    args = parser.parse_args(argv)

    summary = merge_v8_corpus(
        base_path=args.base,
        base_case_specs_path=args.base_case_specs,
        delta_path=args.delta,
        delta_case_specs_path=args.delta_case_specs,
        output_path=args.output,
        case_specs_path=args.case_specs,
        dataset_version=args.dataset_version,
    )

    verify_summary = None
    if not args.skip_verify:
        verify_summary = _run_json_command(
            [
                sys.executable,
                "scripts/verify_finetune_harness_alignment.py",
                "--dataset",
                str(args.output),
                "--case-specs",
                str(args.case_specs),
            ]
        )
        if verify_summary.get("passed") is not True:
            raise SystemExit(f"harness verification failed: {json.dumps(verify_summary, sort_keys=True)}")

    modal_summary = None
    if not args.skip_modal_prep:
        modal_summary = _run_json_command(
            [
                sys.executable,
                "scripts/prepare_modal_finetune_dataset.py",
                "--dataset",
                str(args.output),
                "--dataset-version",
                args.dataset_version,
                "--output-dir",
                str(args.modal_output_dir),
                "--validation-fraction",
                str(args.validation_fraction),
                "--seed",
                args.seed,
                "--min-validation-group-size",
                str(args.min_validation_group_size),
            ]
        )

    summary["verify"] = verify_summary
    summary["modal"] = modal_summary
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def merge_v8_corpus(
    *,
    base_path: Path,
    base_case_specs_path: Path,
    delta_path: Path,
    delta_case_specs_path: Path,
    output_path: Path,
    case_specs_path: Path,
    dataset_version: str,
) -> dict[str, Any]:
    base_rows = _read_jsonl(base_path)
    delta_rows = _read_jsonl(delta_path)
    rows = _dedupe_rows(base_rows + delta_rows)
    specs = _case_specs_for_rows(
        rows,
        source_case_specs=[base_case_specs_path, delta_case_specs_path],
    )
    spec_refresh_summary = _refresh_case_specs_for_current_harness(specs)
    refresh_summary = _refresh_prompts_for_current_harness(rows, specs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    case_specs_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, rows)
    _write_jsonl(case_specs_path, specs)

    return {
        "dataset_version": dataset_version,
        "merged_at": datetime.now(UTC).isoformat(),
        "row_count": len(rows),
        "base_rows": len(base_rows),
        "delta_rows": len(delta_rows),
        "case_spec_count": len(specs),
        "output_path": str(output_path),
        "case_specs_path": str(case_specs_path),
        "output_sha256": _sha256_path(output_path),
        "case_specs_sha256": _sha256_path(case_specs_path),
        "task_type_counts": dict(sorted(Counter(_task_type(row) for row in rows).items())),
        "category_counts": dict(sorted(Counter(str(row.get("category") or "unknown") for row in rows).items())),
        "case_spec_refresh": spec_refresh_summary,
        "prompt_refresh": refresh_summary,
    }


def _refresh_case_specs_for_current_harness(specs: list[dict[str, Any]]) -> dict[str, Any]:
    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    changed: Counter[str] = Counter()
    changed_specs = 0
    for spec in specs:
        spec_changed = False
        harness = _harness_for_spec(spec, {"version": spec.get("dataset_version")}, cards_by_id)
        synthetic_spec = _synthetic_spec_for_case_spec(spec)
        expected_source = _expected_source_cards(
            synthetic_spec,
            harness["rule_results"],
            harness["retrieved_ids"],
        )
        expected_candidates = _expected_candidate_cards(synthetic_spec, harness["rule_results"])
        expected_missing = _expected_missing_observations(
            synthetic_spec,
            [card_id for card_id in expected_source if card_id in harness["retrieved_ids"]],
            cards_by_id,
        )
        updates = {
            "expected_red_flag_rule_ids": [str(rule["rule_id"]) for rule in harness["rule_results"]],
            "expected_min_protocol_urgency": harness["urgency_floor"],
            "expected_source_card_ids": expected_source,
            "expected_candidate_pathway_card_ids": expected_candidates,
            "expected_missing_observations": expected_missing,
            "retrieved_card_ids": harness["retrieved_ids"],
        }
        cue_buckets = bucket_expected_observation_cues(expected_missing)
        updates.update(
            {
                "expected_model_observation_cues": cue_buckets["model"],
                "expected_handoff_cues": cue_buckets["handoff"],
                "expected_harness_evidence_cues": cue_buckets["harness"],
            }
        )
        for key, value in updates.items():
            if spec.get(key) != value:
                changed[key] += 1
                spec[key] = value
                spec_changed = True
        if spec_changed:
            changed_specs += 1
    return {
        "policy_version": 1,
        "updated_field_counts": dict(sorted(changed.items())),
        "updated_specs": changed_specs,
    }


def _synthetic_spec_for_case_spec(spec: dict[str, Any]) -> Any:
    return type(
        "SyntheticSpecForV8MergeCaseSpecRefresh",
        (),
        {
            "target_protocol_card_id": str(spec.get("target_protocol_card_id") or ""),
            "dataset_version": str(spec.get("dataset_version") or ""),
            "failure_class": str(spec.get("failure_class") or ""),
        },
    )()


def _refresh_prompts_for_current_harness(rows: list[dict[str, Any]], specs: list[dict[str, Any]]) -> dict[str, Any]:
    specs_by_id = {str(spec.get("case_id") or ""): spec for spec in specs}
    rows_by_id = {str(row.get("case_id") or row.get("uuid") or ""): row for row in rows}
    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    refreshed: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for row in rows:
        row_id = str(row.get("case_id") or row.get("uuid") or "")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        base_case_id = str(metadata.get("base_case_id") or row_id)
        spec = specs_by_id.get(base_case_id)
        messages = row.get("messages")
        if spec is None or not isinstance(messages, list) or len(messages) < 2:
            skipped["missing_spec_or_messages"] += 1
            continue

        harness = _harness_for_spec(spec, row, cards_by_id)
        task_type = str(metadata.get("task_type") or "navigator_full")
        if task_type == "focused_repair":
            prompt = _focused_repair_prompt_for_row(
                row=row,
                rows_by_id=rows_by_id,
                spec=spec,
                harness=harness,
            )
            if prompt is None:
                skipped["focused_repair_prompt_not_refreshed"] += 1
                continue
            prompt_text = prompt
        else:
            prompt_text = harness["prompt"]

        if not isinstance(row.get("metadata"), dict):
            row["metadata"] = metadata
        if not isinstance(messages[0], dict):
            skipped["bad_user_message_shape"] += 1
            continue
        messages[0]["content"] = prompt_text
        metadata["prompt_hash"] = stable_hash(prompt_text)
        metadata["prompt_template_hash"] = harness["prompt_template_hash"]
        refreshed[task_type] += 1

    return {
        "refreshed_rows": sum(refreshed.values()),
        "refreshed_by_task_type": dict(sorted(refreshed.items())),
        "skipped": dict(sorted(skipped.items())),
        "policy_version": 1,
    }


def _harness_for_spec(spec: dict[str, Any], row: dict[str, Any], cards_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    intake = spec["structured_intake"]
    rule_results = [rule.to_dict() for rule in run_red_flag_checks(intake)]
    floor = urgency_floor_from_rules(rule_results)
    retrieved = search_protocol_cards(query_from_intake(intake), limit=6)
    spec_dataset_version = str(spec.get("dataset_version") or row.get("version") or "")
    if uses_v7_source_card_policy(spec_dataset_version):
        synthetic_spec = type(
            "SyntheticSpecForV8MergeRefresh",
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
    prompt, prompt_template_hash = build_prompt(intake, retrieved, rule_results, floor)
    return {
        "prompt": prompt,
        "prompt_template_hash": prompt_template_hash,
        "rule_results": rule_results,
        "urgency_floor": floor,
        "retrieved": retrieved,
        "retrieved_ids": [str(item.get("card_id", "")) for item in retrieved if item.get("card_id")],
    }


def _focused_repair_prompt_for_row(
    *,
    row: dict[str, Any],
    rows_by_id: dict[str, dict[str, Any]],
    spec: dict[str, Any],
    harness: dict[str, Any],
) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    base_case_id = str(metadata.get("base_case_id") or "")
    base_row = rows_by_id.get(base_case_id)
    if base_row is None:
        return None
    repair_scope = str(metadata.get("repair_scope") or "")
    if not repair_scope:
        return None
    try:
        base_gold = json.loads(base_row["messages"][1]["content"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    previous_output = _corrupt_output(base_gold, repair_scope, str(harness["urgency_floor"]))
    if previous_output is None:
        return None
    validation = validate_navigator_output(
        previous_output,
        known_card_ids=known_card_ids(),
        urgency_floor=str(harness["urgency_floor"]),
        confirmed_intake=spec["structured_intake"],
        rule_results=harness["rule_results"],
        retrieved_card_ids=set(harness["retrieved_ids"]),
        retrieved_cards=harness["retrieved"],
        strict_schema=True,
    ).to_dict()
    focused_prompt = next(
        (
            item
            for item in build_focused_repair_prompts(
                original_prompt=str(harness["prompt"]),
                previous_output=previous_output,
                failures=list(validation.get("failures", []))
                + _extra_failures_for_scope(previous_output, spec, repair_scope),
                urgency_floor=str(harness["urgency_floor"]),
                required_observation_targets=required_observation_targets(harness["retrieved"]),
            )
            if item.scope.name == repair_scope
        ),
        None,
    )
    return focused_prompt.prompt if focused_prompt is not None else None


def _case_specs_for_rows(rows: list[dict[str, Any]], *, source_case_specs: list[Path]) -> list[dict[str, Any]]:
    specs_by_id: dict[str, dict[str, Any]] = {}
    for path in source_case_specs:
        for spec in _read_jsonl(path):
            case_id = str(spec.get("case_id") or "")
            if case_id:
                specs_by_id[case_id] = spec

    needed_ids = {_row_case_id(row) for row in rows}
    missing = sorted(needed_ids - set(specs_by_id))
    if missing:
        raise ValueError(f"missing case specs for rows: {', '.join(missing[:10])}")
    return [specs_by_id[case_id] for case_id in sorted(needed_ids)]


def _row_case_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("base_case_id") or row.get("case_id") or row.get("uuid") or "")


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("case_id") or row.get("uuid") or "")
        if not row_id:
            raise ValueError("row missing case_id/uuid")
        if row_id in rows_by_id:
            raise ValueError(f"duplicate row id: {row_id}")
        rows_by_id[row_id] = row
    return [rows_by_id[row_id] for row_id in sorted(rows_by_id)]


def _task_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("task_type") or "navigator_full")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_json_command(cmd: list[str]) -> dict[str, Any]:
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
