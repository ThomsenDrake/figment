"""Verify Figment SFT rows match the local 4B navigator harness."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figment.focused_repair import build_focused_repair_prompts  # noqa: E402
from figment.observation_targets import required_observation_targets  # noqa: E402
from figment.eval_metrics import score_expected_labels  # noqa: E402
from figment.prompt_builder import build_prompt  # noqa: E402
from figment.retrieval import known_card_ids  # noqa: E402
from figment.retrieval import query_from_intake  # noqa: E402
from figment.retrieval import search_protocol_cards  # noqa: E402
from figment.rules import run_red_flag_checks  # noqa: E402
from figment.validators import urgency_floor_from_rules  # noqa: E402
from figment.validators import validate_navigator_output  # noqa: E402
from scripts.augment_finetune_repair_rows import _corrupt_output  # noqa: E402
from scripts.generate_finetune_data import forbidden_behavior_for_version  # noqa: E402
from scripts.generate_finetune_data import v2_policy_issues  # noqa: E402
from scripts.generate_finetune_data import v3_policy_issues  # noqa: E402


DEFAULT_DATASET = Path("data/finetune/figment_sft_v1.jsonl")
DEFAULT_CASE_SPECS = Path("data/finetune/figment_sft_v1_case_specs.jsonl")
ALLOWED_FACT_SOURCES = {"structured_field", "responder_note", "protocol_card"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--case-specs", type=Path, default=DEFAULT_CASE_SPECS)
    args = parser.parse_args(argv)

    summary = verify_rows(dataset_path=args.dataset, case_specs_path=args.case_specs)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


def verify_rows(*, dataset_path: Path, case_specs_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(dataset_path)
    specs = {str(item["case_id"]): item for item in _read_jsonl(case_specs_path)}
    rows_by_id = {str(row.get("case_id")): row for row in rows}
    issues: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    task_types: Counter[str] = Counter()

    for row_number, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id", ""))
        categories[str(row.get("category", ""))] += 1
        task_type = str(row.get("metadata", {}).get("task_type", "navigator_full"))
        task_types[task_type] += 1
        base_case_id = str(row.get("metadata", {}).get("base_case_id") or case_id)
        spec = specs.get(base_case_id)
        if spec is None:
            issues.append(_issue(row_number, case_id, "missing_case_spec"))
            continue
        dataset_version = str(row.get("version") or spec.get("dataset_version") or "figment_sft_v1")

        messages = row.get("messages")
        if not isinstance(messages, list) or [item.get("role") for item in messages] != ["user", "assistant"]:
            issues.append(_issue(row_number, case_id, "messages_must_match_llama_cpp_user_assistant_shape"))
            continue

        intake = spec["structured_intake"]
        rule_results = [rule.to_dict() for rule in run_red_flag_checks(intake)]
        floor = urgency_floor_from_rules(rule_results)
        retrieved = search_protocol_cards(query_from_intake(intake), limit=6)
        retrieved_ids = [str(item.get("card_id", "")) for item in retrieved if item.get("card_id")]
        harness_prompt, prompt_hash = build_prompt(intake, retrieved, rule_results, floor)
        expected_user_prompt = harness_prompt

        try:
            output = json.loads(str(messages[1].get("content", "")))
        except json.JSONDecodeError as exc:
            issues.append(_issue(row_number, case_id, "assistant_content_is_not_json", error=str(exc)))
            continue

        output_text = json.dumps(output, sort_keys=True)
        lowered_output_text = output_text.lower()
        if "<think" in lowered_output_text or "</think" in lowered_output_text:
            issues.append(_issue(row_number, case_id, "assistant_contains_visible_reasoning_tag"))
        if "teacher" in lowered_output_text:
            issues.append(_issue(row_number, case_id, "assistant_contains_teacher_artifact"))
        if dataset_version == "figment_sft_v2":
            for issue in v2_policy_issues(
                output,
                failure_class=str(row.get("category") or row.get("metadata", {}).get("failure_class") or spec.get("failure_class") or ""),
                expected_red_flag_rule_ids=[str(item) for item in spec.get("expected_red_flag_rule_ids", [])],
                expected_candidate_pathway_card_ids=[
                    str(item) for item in spec.get("expected_candidate_pathway_card_ids", [])
                ],
            ):
                issue_type = "v2_forbidden_lexical_tripwire" if issue.startswith("forbidden_lexical_tripwire:") else f"v2_{issue}"
                issues.append(_issue(row_number, case_id, issue_type, policy_issue=issue))
        if dataset_version.startswith("figment_sft_v3") and task_type != "focused_repair":
            for issue in v3_policy_issues(
                output,
                failure_class=str(row.get("category") or row.get("metadata", {}).get("failure_class") or spec.get("failure_class") or ""),
                expected_red_flag_rule_ids=[str(item) for item in spec.get("expected_red_flag_rule_ids", [])],
                expected_candidate_pathway_card_ids=[
                    str(item) for item in spec.get("expected_candidate_pathway_card_ids", [])
                ],
                structured_intake=intake,
            ):
                issue_type = "v3_forbidden_lexical_tripwire" if issue.startswith("forbidden_lexical_tripwire:") else f"v3_{issue}"
                issues.append(_issue(row_number, case_id, issue_type, policy_issue=issue))

        for fact in output.get("intake_facts", []):
            if isinstance(fact, dict) and fact.get("source") not in ALLOWED_FACT_SOURCES:
                issues.append(
                    _issue(
                        row_number,
                        case_id,
                        "intake_fact_source_not_in_harness_schema",
                        source=fact.get("source"),
                    )
                )

        if task_type == "focused_repair":
            base_row = rows_by_id.get(base_case_id)
            if base_row is None:
                issues.append(_issue(row_number, case_id, "focused_repair_missing_base_row", base_case_id=base_case_id))
                continue
            repair_scope = str(row.get("metadata", {}).get("repair_scope", ""))
            base_gold = json.loads(base_row["messages"][1]["content"])
            previous_output = _corrupt_output(base_gold, repair_scope, floor)
            if previous_output is None:
                issues.append(_issue(row_number, case_id, "focused_repair_previous_output_could_not_be_rebuilt"))
                continue
            previous_validation = validate_navigator_output(
                previous_output,
                known_card_ids=known_card_ids(),
                urgency_floor=floor,
                confirmed_intake=intake,
                rule_results=rule_results,
                retrieved_card_ids=set(retrieved_ids),
                retrieved_cards=retrieved,
                strict_schema=True,
            ).to_dict()
            focused_prompt = next(
                (
                    item
                    for item in build_focused_repair_prompts(
                        original_prompt=harness_prompt,
                        previous_output=previous_output,
                        failures=previous_validation.get("failures", []),
                        urgency_floor=floor,
                        required_observation_targets=required_observation_targets(retrieved),
                    )
                    if item.scope.name == repair_scope
                ),
                None,
            )
            if focused_prompt is None:
                issues.append(_issue(row_number, case_id, "focused_repair_prompt_not_generated", repair_scope=repair_scope))
                continue
            expected_user_prompt = focused_prompt.prompt
            expected_fields = set(focused_prompt.scope.fields)
            if set(output) != expected_fields:
                issues.append(
                    _issue(
                        row_number,
                        case_id,
                        "focused_repair_output_keys_do_not_match_scope",
                        expected=sorted(expected_fields),
                        actual=sorted(output),
                    )
                )
            for field in expected_fields:
                if output.get(field) != base_gold.get(field):
                    issues.append(_issue(row_number, case_id, "focused_repair_field_not_from_base_gold", field=field))
            if dataset_version.startswith("figment_sft_v3"):
                reconstructed = json.loads(json.dumps(base_gold))
                reconstructed.update(output)
                for issue in v3_policy_issues(
                    reconstructed,
                    failure_class=str(spec.get("failure_class") or base_row.get("category") or ""),
                    expected_red_flag_rule_ids=[str(item) for item in spec.get("expected_red_flag_rule_ids", [])],
                    expected_candidate_pathway_card_ids=[
                        str(item) for item in spec.get("expected_candidate_pathway_card_ids", [])
                    ],
                    structured_intake=intake,
                ):
                    issue_type = (
                        "v3_forbidden_lexical_tripwire"
                        if issue.startswith("forbidden_lexical_tripwire:")
                        else f"v3_{issue}"
                    )
                    issues.append(_issue(row_number, case_id, issue_type, policy_issue=issue))
            if row.get("metadata", {}).get("prompt_hash") != _stable_hash_content(expected_user_prompt):
                issues.append(_issue(row_number, case_id, "metadata_prompt_hash_mismatch"))
            if messages[0].get("content") != expected_user_prompt:
                issues.append(_issue(row_number, case_id, "focused_repair_prompt_does_not_match_harness"))
            continue

        if messages[0].get("content") != expected_user_prompt:
            issues.append(
                _issue(
                    row_number,
                    case_id,
                    "user_prompt_does_not_equal_harness_build_prompt",
                    retrieved_card_ids=retrieved_ids,
                )
            )

        validation = validate_navigator_output(
            output,
            known_card_ids=known_card_ids(),
            urgency_floor=floor,
            confirmed_intake=intake,
            rule_results=rule_results,
            retrieved_card_ids=set(retrieved_ids),
            retrieved_cards=retrieved,
            strict_schema=True,
        ).to_dict()
        if validation.get("passed") is not True:
            issues.append(_issue(row_number, case_id, "validator_failed_under_harness", validation=validation))

        expected_score = score_expected_labels(
            {
                "case_id": case_id,
                "target_protocol_card_id": spec.get("target_protocol_card_id"),
                "expected_min_protocol_urgency": spec.get("expected_min_protocol_urgency"),
                "expected_red_flag_rule_ids": spec.get("expected_red_flag_rule_ids", []),
                "expected_source_card_ids": spec.get("expected_source_card_ids", []),
                "expected_candidate_pathway_card_ids": spec.get("expected_candidate_pathway_card_ids", []),
                "expected_missing_observations": spec.get("expected_missing_observations", []),
                "forbidden_behavior": _forbidden_behavior_for_dataset_version(dataset_version),
                "actual_red_flag_rule_ids": [str(rule["rule_id"]) for rule in rule_results],
                "actual_protocol_urgency": output.get("protocol_urgency"),
                "actual_source_card_ids": _string_list(output.get("source_cards")),
                "actual_candidate_pathway_card_ids": _candidate_ids(output.get("candidate_protocol_pathways")),
                "final_output": output,
                "final_validation": validation,
            }
        )
        if expected_score.get("all_expected_labels_passed") is not True:
            issues.append(_issue(row_number, case_id, "expected_label_score_failed", expected_score=expected_score))

        metadata = row.get("metadata", {})
        if metadata.get("prompt_hash") != _stable_hash_content(harness_prompt):
            issues.append(_issue(row_number, case_id, "metadata_prompt_hash_mismatch"))
        if metadata.get("prompt_template_hash") != prompt_hash:
            issues.append(_issue(row_number, case_id, "metadata_prompt_template_hash_mismatch"))
        if "harness" not in str(metadata.get("teacher_label_mode", "")):
            issues.append(_issue(row_number, case_id, "teacher_label_mode_does_not_record_harness_alignment"))

    return {
        "rows": len(rows),
        "case_specs": len(specs),
        "passed": not issues,
        "issue_count": len(issues),
        "issue_types": dict(Counter(item["type"] for item in issues)),
        "categories": dict(sorted(categories.items())),
        "task_types": dict(sorted(task_types.items())),
        "issues_sample": issues[:20],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _issue(row_number: int, case_id: str, issue_type: str, **details: Any) -> dict[str, Any]:
    return {"row_number": row_number, "case_id": case_id, "type": issue_type, **details}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _candidate_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    ids = []
    for item in value:
        if isinstance(item, dict) and item.get("card_id"):
            ids.append(str(item["card_id"]))
    return ids


def _forbidden_behavior_for_dataset_version(dataset_version: str) -> list[str]:
    return forbidden_behavior_for_version(dataset_version)


def _stable_hash_content(value: str) -> str:
    from figment.trace import stable_hash

    return stable_hash(value)


if __name__ == "__main__":
    raise SystemExit(main())
