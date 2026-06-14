"""Summarize v6 eval failures that should shape the Figment v7 corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize_eval(args.eval_jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def summarize_eval(eval_jsonl: Path) -> dict[str, Any]:
    records = _read_jsonl(eval_jsonl)
    deterministic_patch_field_counts: Counter[str] = Counter()
    expected_label_check_failures: Counter[str] = Counter()
    competence_failure_case_ids: list[str] = []
    expected_label_failure_case_ids: list[str] = []
    missing_source_card_ids_by_case: dict[str, list[str]] = {}
    deterministic_patch_fields_by_case: dict[str, list[str]] = {}
    actual_source_card_sets_for_failures: dict[str, list[str]] = {}

    for record in records:
        case_id = str(record.get("case_id") or "")
        patch_fields = _string_list(record.get("deterministic_scaffold_patched_fields"))
        for field in patch_fields:
            deterministic_patch_field_counts[field] += 1
        if patch_fields:
            deterministic_patch_fields_by_case[case_id] = patch_fields

        expected_score = record.get("expected_label_score") if isinstance(record.get("expected_label_score"), dict) else {}
        for key, value in expected_score.items():
            if isinstance(value, bool) and value is False:
                expected_label_check_failures[key] += 1

        if record.get("competence_success") is not True:
            competence_failure_case_ids.append(case_id)
            actual_source_card_sets_for_failures[case_id] = _string_list(record.get("actual_source_card_ids"))

        if expected_score.get("all_expected_labels_passed") is not True:
            expected_label_failure_case_ids.append(case_id)
            missing_source_card_ids = _string_list(expected_score.get("missing_expected_source_card_ids"))
            if missing_source_card_ids:
                missing_source_card_ids_by_case[case_id] = missing_source_card_ids
            actual_source_card_sets_for_failures[case_id] = _string_list(record.get("actual_source_card_ids"))

    return {
        "total_cases": len(records),
        "competence_failure_case_ids": competence_failure_case_ids,
        "competence_failure_count": len(competence_failure_case_ids),
        "expected_label_failure_case_ids": expected_label_failure_case_ids,
        "expected_label_failure_count": len(expected_label_failure_case_ids),
        "expected_label_check_failures": dict(sorted(expected_label_check_failures.items())),
        "missing_source_card_ids_by_case": missing_source_card_ids_by_case,
        "missing_source_card_id_counts": dict(
            sorted(Counter(card_id for cards in missing_source_card_ids_by_case.values() for card_id in cards).items())
        ),
        "deterministic_patch_fields_by_case": deterministic_patch_fields_by_case,
        "deterministic_patch_field_counts": dict(sorted(deterministic_patch_field_counts.items())),
        "actual_source_card_sets_for_failures": actual_source_card_sets_for_failures,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


if __name__ == "__main__":
    raise SystemExit(main())
