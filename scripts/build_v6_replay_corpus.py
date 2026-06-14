"""Audit v3-v5 SFT rows and select clean replay rows for Figment v6."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUTS = [
    Path("data/finetune/figment_sft_v5.jsonl"),
    Path("data/finetune/figment_sft_v4.jsonl"),
    Path("data/finetune/figment_sft_v3.jsonl"),
]
DEFAULT_OUTPUT = Path("data/finetune/figment_sft_v6_replay.jsonl")
DEFAULT_MANIFEST = Path("data/finetune/figment_sft_v6_replay_manifest.json")
DEFAULT_TARGETS = {
    "figment_sft_v5": 450,
    "figment_sft_v4": 300,
    "figment_sft_v3": 150,
}
DEFAULT_SEED = "figment-sft-v6-replay-selection"

HARNESS_METADATA_CUES = {
    "source card ids",
    "source protocol card ids",
    "retrieved protocol card ids",
    "deterministic rule results",
    "navigator validation result",
    "confirmed intake status",
    "manual correction status for audio-derived fields",
}

GENERIC_OBSERVATION_PHRASES = {
    "monitor closely",
    "collect more information",
    "follow up",
    "assess patient",
    "repeat vitals",
}

UNSAFE_OBSERVATION_FRAGMENTS = {
    "diagnose",
    "prescribe",
    "dose",
    "discharge",
    "send home",
    "clear for discharge",
    "start medication",
}

OBSERVATION_FOCUSED_CATEGORIES = {
    "missing_observation_prioritization",
    "required_observation_id_selection",
    "focused_repair:missing_observations",
}

CATEGORY_PRIORITY = {
    "focused_repair:handoff_note_sbar": 100,
    "focused_repair:citations_and_pathways": 95,
    "focused_repair:protocol_urgency": 90,
    "focused_repair:schema": 85,
    "source_card_invariant": 80,
    "sbar_observation_ownership": 78,
    "general_regression": 76,
    "noisy_field_audio_style": 74,
    "radio_handoff": 72,
    "sbar_handoff_usefulness": 70,
    "source_card_discipline": 68,
    "low_resource_constraints": 66,
    "rural_clinic_intake": 64,
    "disaster_triage": 62,
}


@dataclass(frozen=True)
class AuditResult:
    accepted: bool
    reasons: tuple[str, ...]
    score: int


@dataclass(frozen=True)
class Candidate:
    row: dict[str, Any]
    source_path: str
    source_dataset_version: str
    category: str
    task_type: str
    audit: AuditResult
    ordinal: int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=None, help="Input JSONL corpus path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    for dataset_version, target in DEFAULT_TARGETS.items():
        parser.add_argument(f"--{dataset_version.replace('_', '-')}-target", type=int, default=target)
    args = parser.parse_args(argv)

    input_paths = args.input or DEFAULT_INPUTS
    targets = {
        "figment_sft_v5": args.figment_sft_v5_target,
        "figment_sft_v4": args.figment_sft_v4_target,
        "figment_sft_v3": args.figment_sft_v3_target,
    }
    summary = build_replay_corpus(
        input_paths=input_paths,
        output_path=args.output,
        manifest_path=args.manifest,
        targets=targets,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["selected_rows"] > 0 else 1


def build_replay_corpus(
    *,
    input_paths: list[Path],
    output_path: Path,
    manifest_path: Path,
    targets: dict[str, int],
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    candidates: list[Candidate] = []
    rejected: Counter[str] = Counter()
    source_row_counts: Counter[str] = Counter()
    input_hashes: dict[str, str] = {}

    for input_path in input_paths:
        input_hashes[str(input_path)] = _sha256_path(input_path)
        for ordinal, row in enumerate(_read_jsonl(input_path), start=1):
            source_version = _source_dataset_version(row, input_path)
            source_row_counts[source_version] += 1
            audit = audit_row(row)
            category = _category(row)
            task_type = _task_type(row)
            if audit.accepted:
                candidates.append(
                    Candidate(
                        row=row,
                        source_path=str(input_path),
                        source_dataset_version=source_version,
                        category=category,
                        task_type=task_type,
                        audit=audit,
                        ordinal=ordinal,
                    )
                )
            else:
                for reason in audit.reasons:
                    rejected[f"{source_version}:{reason}"] += 1

    selected = select_candidates(candidates, targets=targets, seed=seed)
    rows = [_annotate_row(candidate) for candidate in selected]
    _write_jsonl(output_path, rows)

    manifest = {
        "dataset_version": "figment_sft_v6_replay",
        "selection_policy_version": 1,
        "seed": seed,
        "input_paths": [str(path) for path in input_paths],
        "input_sha256": input_hashes,
        "source_row_counts": dict(sorted(source_row_counts.items())),
        "target_rows_by_source_dataset_version": dict(sorted(targets.items())),
        "accepted_candidate_rows": len(candidates),
        "selected_rows": len(selected),
        "selected_sha256": _sha256_path(output_path),
        "output_path": str(output_path),
        "rejected_reason_counts": dict(sorted(rejected.items())),
        "selected_by_source_dataset_version": dict(
            sorted(Counter(candidate.source_dataset_version for candidate in selected).items())
        ),
        "selected_by_category": dict(sorted(Counter(candidate.category for candidate in selected).items())),
        "selected_by_task_type": dict(sorted(Counter(candidate.task_type for candidate in selected).items())),
        "shortage_by_source_dataset_version": _shortages(selected, targets),
        "policy_notes": [
            "Rows are direct replay candidates only; rejected rows should not be used without rewriting.",
            "Duplicate long missing/next observation lists are rejected.",
            "Harness metadata cues in observation fields are rejected.",
            "Observation-focused full rows must carry selected_required_observation_ids.",
            "The selector does not fill quotas with rows that fail v6 replay policy.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def audit_row(row: dict[str, Any]) -> AuditResult:
    reasons: list[str] = []
    score = CATEGORY_PRIORITY.get(_category(row), 20)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    output = _assistant_output(row)

    if output is None:
        return AuditResult(False, ("assistant_content_not_json",), 0)

    if metadata.get("validator_passed") is False:
        reasons.append("validator_failed")
    validation = metadata.get("validation_result")
    if isinstance(validation, dict) and validation.get("passed") is False:
        reasons.append("validation_result_failed")
    expected_score = metadata.get("expected_label_score")
    if isinstance(expected_score, dict):
        if expected_score.get("all_expected_labels_passed") is False:
            reasons.append("expected_labels_failed")
        if expected_score.get("forbidden_behavior_absent") is False:
            reasons.append("forbidden_behavior_present")

    output_text = json.dumps(output, sort_keys=True).lower()
    if "teacher" in output_text:
        reasons.append("teacher_artifact")
    if "<think" in output_text or "</think" in output_text:
        reasons.append("visible_reasoning_tag")

    observation_texts = _observation_texts(output)
    missing = _string_list(output.get("missing_info_to_collect"))
    next_obs = _string_list(output.get("next_observations_to_collect"))
    if missing and missing == next_obs and len(missing) > 3:
        reasons.append("duplicate_long_missing_and_next_observations")
    for cue in sorted(HARNESS_METADATA_CUES):
        if _contains_phrase(observation_texts, cue):
            reasons.append(f"harness_metadata_observation:{cue.replace(' ', '_')}")
    for phrase in sorted(GENERIC_OBSERVATION_PHRASES):
        if _contains_exact_item(missing + next_obs, phrase):
            reasons.append(f"generic_observation_phrase:{phrase.replace(' ', '_')}")
    for fragment in sorted(UNSAFE_OBSERVATION_FRAGMENTS):
        if _contains_phrase(observation_texts, fragment):
            reasons.append(f"unsafe_observation_fragment:{fragment.replace(' ', '_')}")

    category = _category(row)
    task_type = _task_type(row)
    selected_ids = _string_list(output.get("selected_required_observation_ids"))
    if task_type == "navigator_full" and category in OBSERVATION_FOCUSED_CATEGORIES:
        if not selected_ids:
            reasons.append("observation_focused_row_missing_selected_required_observation_ids")
    invalid_selected = _string_list(metadata.get("invalid_selected_required_observation_ids"))
    if invalid_selected:
        reasons.append("invalid_selected_required_observation_ids")
    must_include_selected = _string_list(metadata.get("must_include_selected_required_observation_ids"))
    if selected_ids and must_include_selected:
        missing_required = sorted(set(must_include_selected) - set(selected_ids))
        if missing_required:
            reasons.append("selected_required_observation_ids_missing_required")

    patched = set(_string_list(metadata.get("deterministic_scaffold_patched_fields")))
    if {"missing_info_to_collect", "next_observations_to_collect"} & patched:
        score -= 20
    if selected_ids:
        score += 10
    if task_type == "focused_repair":
        score += 15
    if not reasons:
        score += 25

    return AuditResult(not reasons, tuple(reasons), score)


def select_candidates(candidates: list[Candidate], *, targets: dict[str, int], seed: str) -> list[Candidate]:
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    by_source: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in shuffled:
        by_source[candidate.source_dataset_version].append(candidate)

    selected: list[Candidate] = []
    selected_keys: set[str] = set()
    for source_version, target in targets.items():
        selected.extend(_take_ranked(by_source.get(source_version, []), target, selected_keys))

    target_total = sum(targets.values())
    if len(selected) < target_total:
        remaining = [candidate for candidate in shuffled if _candidate_key(candidate) not in selected_keys]
        selected.extend(_take_ranked(remaining, target_total - len(selected), selected_keys))

    return sorted(selected, key=lambda candidate: (candidate.source_dataset_version, candidate.category, candidate.ordinal))


def _take_ranked(candidates: list[Candidate], count: int, selected_keys: set[str]) -> list[Candidate]:
    ranked = sorted(candidates, key=lambda candidate: (-candidate.audit.score, candidate.category, candidate.ordinal))
    chosen: list[Candidate] = []
    for candidate in ranked:
        if len(chosen) >= count:
            break
        key = _candidate_key(candidate)
        if key in selected_keys:
            continue
        selected_keys.add(key)
        chosen.append(candidate)
    return chosen


def _annotate_row(candidate: Candidate) -> dict[str, Any]:
    row = json.loads(json.dumps(candidate.row, sort_keys=True))
    metadata = row.setdefault("metadata", {})
    metadata["v6_replay_audit"] = {
        "accepted": True,
        "audit_score": candidate.audit.score,
        "replay_reason": _replay_reason(candidate),
        "source_dataset_version": candidate.source_dataset_version,
        "source_path": candidate.source_path,
        "selection_policy_version": 1,
    }
    return row


def _replay_reason(candidate: Candidate) -> str:
    if candidate.task_type == "focused_repair":
        return f"clean_{candidate.category}_focused_repair"
    return f"clean_{candidate.category}_navigator_replay"


def _shortages(selected: list[Candidate], targets: dict[str, int]) -> dict[str, int]:
    counts = Counter(candidate.source_dataset_version for candidate in selected)
    return {
        source_version: max(0, target - counts.get(source_version, 0))
        for source_version, target in sorted(targets.items())
    }


def _candidate_key(candidate: Candidate) -> str:
    case_id = str(candidate.row.get("case_id", ""))
    return f"{candidate.source_path}:{case_id}:{candidate.ordinal}"


def _assistant_output(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    content = messages[-1].get("content") if isinstance(messages[-1], dict) else None
    if not isinstance(content, str):
        return None
    try:
        output = json.loads(content)
    except json.JSONDecodeError:
        return None
    return output if isinstance(output, dict) else None


def _observation_texts(output: dict[str, Any]) -> list[str]:
    return _string_list(output.get("missing_info_to_collect")) + _string_list(
        output.get("next_observations_to_collect")
    )


def _contains_phrase(items: list[str], phrase: str) -> bool:
    normalized_phrase = _normalize_text(phrase)
    return any(normalized_phrase in _normalize_text(item) for item in items)


def _contains_exact_item(items: list[str], phrase: str) -> bool:
    normalized_phrase = _normalize_text(phrase)
    return any(normalized_phrase == _normalize_text(item) for item in items)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_text(value: str) -> str:
    return " ".join(str(value).lower().replace("-", " ").replace("_", " ").split())


def _category(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("category") or metadata.get("category") or metadata.get("failure_class") or "missing")


def _task_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    task_type = metadata.get("task_type")
    if task_type:
        return str(task_type)
    output = _assistant_output(row) or {}
    return "navigator_full" if "protocol_urgency" in output else "focused_repair"


def _source_dataset_version(row: dict[str, Any], input_path: Path) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    version = row.get("version") or metadata.get("dataset_version")
    if version:
        return str(version)
    stem = input_path.stem
    if stem.startswith("figment_sft_"):
        return stem
    return "unknown"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

