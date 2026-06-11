"""Export v4 teacher/repair seeds from updated Figment eval failures."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figment.eval_metrics import score_expected_labels  # noqa: E402


DEFAULT_OUTPUT = Path("data/finetune/v4_seed_exports/figment_sft_v4_failure_seeds.jsonl")
MODEL_TRAINING_CHECKS = (
    "red_flags_match",
    "min_urgency_met",
    "target_card_in_source_cards",
    "expected_source_cards_present",
    "target_card_in_candidate_pathways",
    "expected_candidate_pathways_present",
    "missing_observation_cues_present",
    "model_observation_cues_present",
    "handoff_cues_present",
    "handoff_readiness_passed",
    "forbidden_behavior_absent",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, required=True, help="Scored eval JSONL to export from.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-passing", action="store_true", help="Also emit high-quality replay candidates.")
    args = parser.parse_args(argv)

    manifest = export_v4_training_seeds(
        eval_path=args.eval,
        output_path=args.output,
        include_passing=args.include_passing,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def export_v4_training_seeds(*, eval_path: Path, output_path: Path, include_passing: bool = False) -> dict[str, Any]:
    records = _read_jsonl(eval_path)
    case_cache: dict[str, list[dict[str, Any]]] = {}
    seeds = []
    for record in records:
        score = score_expected_labels(record)
        failed = _model_training_failed(score, record)
        if not failed and not include_passing:
            continue
        source_case = _source_case_for_record(record, case_cache)
        seed = _seed_from_record(record, score, source_case, failed=failed)
        if seed:
            seeds.append(seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(seed, sort_keys=True) + "\n" for seed in seeds), encoding="utf-8")
    manifest = {
        "source_eval_path": str(eval_path),
        "output_path": str(output_path),
        "source_records": len(records),
        "seed_count": len(seeds),
        "failure_seed_count": sum(1 for seed in seeds if seed["seed_type"] == "v4_failure_seed"),
        "replay_seed_count": sum(1 for seed in seeds if seed["seed_type"] == "v4_replay_candidate"),
        "harness_only_score_failure_count": sum(1 for seed in seeds if seed.get("harness_only_score_failure")),
        "repair_scope_counts": _scope_counts(seeds),
        "generated_at": datetime.now(UTC).isoformat(),
        "holdout_policy": {
            "holdout_rows_are_not_training_rows": True,
            "teacher_must_generate_synthetic_siblings_or_repairs": True,
            "copying_source_case_or_close_paraphrase_allowed": False,
        },
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _seed_from_record(
    record: dict[str, Any],
    score: dict[str, Any],
    source_case: dict[str, Any],
    *,
    failed: bool,
) -> dict[str, Any] | None:
    repair_scopes = _repair_scopes_for_score(score, record)
    if failed and not repair_scopes:
        repair_scopes = ["responder_checklist"]
    if not failed and not _high_quality_replay(record, score):
        return None
    case_id = str(record.get("case_id") or source_case.get("case_id") or "")
    dataset_version = str(source_case.get("dataset_version") or "")
    direct_training_allowed = dataset_version not in {"field_workflow_holdout_v1"} and not str(
        record.get("case_path") or ""
    ).endswith("field_workflow_holdout_v1.jsonl")
    return {
        "seed_id": f"v4-seed-{case_id}",
        "seed_type": "v4_failure_seed" if failed else "v4_replay_candidate",
        "source_case_id": case_id,
        "source_case_path": record.get("case_path"),
        "source_case_line": record.get("case_line"),
        "source_trace_hash": record.get("trace_hash"),
        "workflow_category": _workflow_category(record, source_case),
        "target_protocol_card_id": record.get("target_protocol_card_id") or source_case.get("target_protocol_card_id"),
        "repair_scopes": repair_scopes,
        "score_failed_checks": _score_failed_checks(score),
        "model_training_failed": failed,
        "harness_only_score_failure": _harness_only_score_failure(score, record),
        "expected_label_score": score,
        "final_validation": record.get("final_validation") or record.get("validation_result"),
        "field_provenance": record.get("field_provenance"),
        "model_route": record.get("model_route"),
        "harness_evidence": record.get("harness_evidence") or record.get("final_output", {}).get("harness_evidence"),
        "structured_intake": source_case.get("structured_intake"),
        "expected_labels": {
            "expected_red_flag_rule_ids": source_case.get("expected_red_flag_rule_ids")
            or record.get("expected_red_flag_rule_ids", []),
            "expected_min_protocol_urgency": source_case.get("expected_min_protocol_urgency")
            or record.get("expected_min_protocol_urgency"),
            "expected_source_card_ids": source_case.get("expected_source_card_ids")
            or record.get("expected_source_card_ids", []),
            "expected_candidate_pathway_card_ids": source_case.get("expected_candidate_pathway_card_ids")
            or record.get("expected_candidate_pathway_card_ids", []),
            "expected_model_observation_cues": score.get("expected_model_observation_cues", []),
            "expected_handoff_cues": score.get("expected_handoff_cues", []),
            "expected_harness_evidence_cues": score.get("expected_harness_evidence_cues", []),
        },
        "previous_output": record.get("final_output"),
        "teacher_instruction": _teacher_instruction(repair_scopes, direct_training_allowed),
        "direct_training_allowed": direct_training_allowed,
        "anti_overfit_policy": {
            "do_not_copy_source_case": True,
            "do_not_create_close_paraphrase": True,
            "use_as_failure_pattern_or_repair_seed": True,
        },
    }


def _repair_scopes_for_score(score: dict[str, Any], record: dict[str, Any]) -> list[str]:
    scopes: list[str] = []
    if score.get("red_flags_match") is False or score.get("min_urgency_met") is False:
        scopes.append("safety_boundary")
    if score.get("handoff_readiness_passed") is False or score.get("handoff_cues_present") is False:
        scopes.append("handoff_note_sbar")
    if score.get("expected_source_cards_present") is False or score.get("target_card_in_source_cards") is False:
        scopes.append("source_cards")
    if (
        score.get("expected_candidate_pathways_present") is False
        or score.get("target_card_in_candidate_pathways") is False
    ):
        scopes.append("candidate_protocol_pathways")
    if score.get("model_observation_cues_present") is False or score.get("missing_observation_cues_present") is False:
        scopes.append("missing_observations")
    if score.get("forbidden_behavior_absent") is False:
        scopes.append("safety_boundary")
    validation = record.get("final_validation") or record.get("validation_result")
    if isinstance(validation, dict) and validation.get("passed") is False:
        scopes.append("validation_failure")
    return _ordered_unique(scopes)


def _teacher_instruction(repair_scopes: list[str], direct_training_allowed: bool) -> str:
    if direct_training_allowed:
        return (
            "Generate a JSON-only Figment navigator target or focused repair row matching the current harness. "
            "Improve only the listed repair scopes while preserving deterministic red flags, urgency floor, "
            "retrieved-card discipline, and protocol-navigation safety."
        )
    return (
        "This source is an eval/holdout seed. Do not copy it or make a close paraphrase. Generate a synthetic "
        "sibling or repair pattern that exercises the same failure scopes: "
        f"{', '.join(repair_scopes) or 'replay'}."
    )


def _high_quality_replay(record: dict[str, Any], score: dict[str, Any]) -> bool:
    validation = record.get("final_validation") or record.get("validation_result")
    return (
        isinstance(validation, dict)
        and validation.get("passed") is True
        and _model_training_passed(score, record)
        and not record.get("canned_fallback_used")
        and not record.get("fallback_reason")
    )


def _model_training_failed(score: dict[str, Any], record: dict[str, Any]) -> bool:
    validation = record.get("final_validation") or record.get("validation_result")
    if isinstance(validation, dict) and validation.get("passed") is False:
        return True
    return any(score.get(check) is False for check in MODEL_TRAINING_CHECKS)


def _model_training_passed(score: dict[str, Any], record: dict[str, Any]) -> bool:
    validation = record.get("final_validation") or record.get("validation_result")
    if isinstance(validation, dict) and validation.get("passed") is not True:
        return False
    return not _model_training_failed(score, record)


def _score_failed_checks(score: dict[str, Any]) -> list[str]:
    return [key for key, value in score.items() if isinstance(value, bool) and value is False]


def _harness_only_score_failure(score: dict[str, Any], record: dict[str, Any]) -> bool:
    return (
        score.get("all_expected_labels_passed") is False
        and not _model_training_failed(score, record)
        and score.get("harness_evidence_cues_visible") is False
    )


def _workflow_category(record: dict[str, Any], source_case: dict[str, Any]) -> str | None:
    structured_intake = source_case.get("structured_intake")
    if isinstance(structured_intake, dict) and structured_intake.get("workflow_category"):
        return str(structured_intake["workflow_category"])
    for payload in (source_case, record):
        if payload.get("workflow_category"):
            return str(payload["workflow_category"])
    return None


def _source_case_for_record(record: dict[str, Any], case_cache: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    path = record.get("case_path")
    line = record.get("case_line")
    if not path or not line:
        return {}
    path_text = str(path)
    if path_text not in case_cache:
        case_cache[path_text] = _read_jsonl(Path(path_text))
    index = int(line) - 1
    cases = case_cache[path_text]
    if index < 0 or index >= len(cases):
        return {}
    return cases[index]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ordered_unique(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _scope_counts(seeds: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for seed in seeds:
        for scope in seed.get("repair_scopes", []):
            counts[scope] = counts.get(scope, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
