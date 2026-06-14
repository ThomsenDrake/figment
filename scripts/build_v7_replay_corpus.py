"""Audit existing SFT rows and select clean replay rows for Figment v7."""

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

from scripts.build_v6_replay_corpus import audit_row as audit_v6_replay_row  # noqa: E402
from scripts.generate_finetune_data import CLINICAL_CARD_IDS  # noqa: E402
from scripts.generate_finetune_data import SAFETY_CARD_ID  # noqa: E402
from scripts.generate_finetune_data import SBAR_CARD_ID  # noqa: E402
from scripts.generate_finetune_data import v7_source_card_closure_issues  # noqa: E402


DEFAULT_INPUTS = [
    Path("data/finetune/figment_sft_v6_delta.jsonl"),
    Path("data/finetune/figment_sft_v6_replay.jsonl"),
    Path("data/finetune/figment_sft_v5.jsonl"),
    Path("data/finetune/figment_sft_v4.jsonl"),
    Path("data/finetune/figment_sft_v3.jsonl"),
]
DEFAULT_OUTPUT = Path("data/finetune/figment_sft_v7_replay.jsonl")
DEFAULT_MANIFEST = Path("data/finetune/figment_sft_v7_replay_manifest.json")
DEFAULT_TARGETS = {
    "figment_sft_v6_delta": 1430,
    "figment_sft_v6_replay": 570,
    "figment_sft_v5": 0,
    "figment_sft_v4": 0,
    "figment_sft_v3": 0,
}
DEFAULT_SEED = "figment-sft-v7-replay-selection"
V7_REPLAY_VERSION = "figment_sft_v7_replay"


@dataclass(frozen=True)
class AuditResult:
    accepted: bool
    reasons: tuple[str, ...]
    score: int


@dataclass(frozen=True)
class Candidate:
    row: dict[str, Any]
    source_path: str
    source_bucket: str
    original_source_dataset_version: str
    category: str
    task_type: str
    audit: AuditResult
    ordinal: int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--fill-shortage-from-any", action="store_true")
    for source_bucket, target in DEFAULT_TARGETS.items():
        parser.add_argument(f"--{source_bucket.replace('_', '-')}-target", type=int, default=target)
    args = parser.parse_args(argv)

    targets = {
        "figment_sft_v6_delta": args.figment_sft_v6_delta_target,
        "figment_sft_v6_replay": args.figment_sft_v6_replay_target,
        "figment_sft_v5": args.figment_sft_v5_target,
        "figment_sft_v4": args.figment_sft_v4_target,
        "figment_sft_v3": args.figment_sft_v3_target,
    }
    summary = build_replay_corpus(
        input_paths=args.input or DEFAULT_INPUTS,
        output_path=args.output,
        manifest_path=args.manifest,
        targets=targets,
        seed=args.seed,
        fill_shortage_from_any=args.fill_shortage_from_any,
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
    fill_shortage_from_any: bool = False,
) -> dict[str, Any]:
    candidates: list[Candidate] = []
    rejected: Counter[str] = Counter()
    source_row_counts: Counter[str] = Counter()
    accepted_by_bucket: Counter[str] = Counter()
    input_hashes: dict[str, str] = {}

    for input_path in input_paths:
        source_bucket = _source_bucket(input_path)
        input_hashes[str(input_path)] = _sha256_path(input_path)
        for ordinal, row in enumerate(_read_jsonl(input_path), start=1):
            source_row_counts[source_bucket] += 1
            audit = audit_row(row)
            if audit.accepted:
                accepted_by_bucket[source_bucket] += 1
                candidates.append(
                    Candidate(
                        row=row,
                        source_path=str(input_path),
                        source_bucket=source_bucket,
                        original_source_dataset_version=_original_source_dataset_version(row, input_path),
                        category=_category(row),
                        task_type=_task_type(row),
                        audit=audit,
                        ordinal=ordinal,
                    )
                )
            else:
                for reason in audit.reasons:
                    rejected[f"{source_bucket}:{reason}"] += 1

    selected = select_candidates(
        candidates,
        targets=targets,
        seed=seed,
        fill_shortage_from_any=fill_shortage_from_any,
    )
    rows = [_annotate_row(candidate) for candidate in selected]
    _write_jsonl(output_path, rows)

    manifest = {
        "dataset_version": V7_REPLAY_VERSION,
        "selection_policy_version": 1,
        "seed": seed,
        "fill_shortage_from_any": fill_shortage_from_any,
        "input_paths": [str(path) for path in input_paths],
        "input_sha256": input_hashes,
        "source_row_counts": dict(sorted(source_row_counts.items())),
        "accepted_candidate_rows": len(candidates),
        "accepted_by_source_bucket": dict(sorted(accepted_by_bucket.items())),
        "target_rows_by_source_bucket": dict(sorted(targets.items())),
        "selected_rows": len(selected),
        "selected_sha256": _sha256_path(output_path),
        "output_path": str(output_path),
        "rejected_reason_counts": dict(sorted(rejected.items())),
        "selected_by_source_bucket": dict(sorted(Counter(candidate.source_bucket for candidate in selected).items())),
        "selected_by_original_source_dataset_version": dict(
            sorted(Counter(candidate.original_source_dataset_version for candidate in selected).items())
        ),
        "selected_by_category": dict(sorted(Counter(candidate.category for candidate in selected).items())),
        "selected_by_task_type": dict(sorted(Counter(candidate.task_type for candidate in selected).items())),
        "shortage_by_source_bucket": _shortages(selected, targets),
        "policy_notes": [
            "Rows are direct replay candidates only; rejected rows should not be used without rewriting.",
            "V7 replay applies the v6 replay cleanliness policy first.",
            "Full navigator rows must pass v7 source-card closure checks.",
            "Historical v3-v5 rows are audited and counted but not selected by default.",
            "Selected rows are re-versioned as figment_sft_v7_replay while preserving original provenance.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def audit_row(row: dict[str, Any]) -> AuditResult:
    v6_audit = audit_v6_replay_row(row)
    reasons = list(v6_audit.reasons)
    output = _assistant_output(row)
    if output is None:
        return AuditResult(False, tuple(reasons or ["assistant_content_not_json"]), 0)

    if _task_type(row) == "navigator_full":
        for issue in v7_source_card_closure_issues(
            output,
            target_protocol_card_id=_target_protocol_card_id(row, output),
        ):
            reasons.append(issue)

    score = int(v6_audit.score)
    source_cards = _string_list(output.get("source_cards"))
    source_card_set = set(source_cards)
    if {SAFETY_CARD_ID, SBAR_CARD_ID} <= source_card_set:
        score += 15
    if 3 <= len(source_cards) <= 5:
        score += 8
    if _task_type(row) == "navigator_full":
        score += 5
    if not reasons:
        score += 25
    return AuditResult(not reasons, tuple(_dedupe(reasons)), score)


def select_candidates(
    candidates: list[Candidate],
    *,
    targets: dict[str, int],
    seed: str,
    fill_shortage_from_any: bool,
) -> list[Candidate]:
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    by_bucket: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in shuffled:
        by_bucket[candidate.source_bucket].append(candidate)

    selected: list[Candidate] = []
    selected_keys: set[str] = set()
    for source_bucket, target in targets.items():
        selected.extend(_take_ranked(by_bucket.get(source_bucket, []), target, selected_keys))

    target_total = sum(targets.values())
    if fill_shortage_from_any and len(selected) < target_total:
        remaining = [candidate for candidate in shuffled if _candidate_key(candidate) not in selected_keys]
        selected.extend(_take_ranked(remaining, target_total - len(selected), selected_keys))

    return sorted(selected, key=lambda candidate: (candidate.source_bucket, candidate.category, candidate.ordinal))


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
    metadata["dataset_version"] = V7_REPLAY_VERSION
    metadata["v7_replay_audit"] = {
        "accepted": True,
        "audit_score": candidate.audit.score,
        "original_source_dataset_version": candidate.original_source_dataset_version,
        "replay_reason": _replay_reason(candidate),
        "selection_policy_version": 1,
        "source_bucket": candidate.source_bucket,
        "source_path": candidate.source_path,
    }
    row["version"] = V7_REPLAY_VERSION
    return row


def _replay_reason(candidate: Candidate) -> str:
    if candidate.task_type == "focused_repair":
        return f"clean_{candidate.category}_focused_repair"
    return f"clean_{candidate.category}_navigator_replay"


def _shortages(selected: list[Candidate], targets: dict[str, int]) -> dict[str, int]:
    counts = Counter(candidate.source_bucket for candidate in selected)
    return {source_bucket: max(0, target - counts.get(source_bucket, 0)) for source_bucket, target in sorted(targets.items())}


def _candidate_key(candidate: Candidate) -> str:
    case_id = str(candidate.row.get("case_id", ""))
    return f"{candidate.source_path}:{case_id}:{candidate.ordinal}"


def _target_protocol_card_id(row: dict[str, Any], output: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for card_id in _string_list(metadata.get("must_include_source_cards")):
        if card_id in CLINICAL_CARD_IDS:
            return card_id
    for card_id in _string_list(output.get("source_cards")):
        if card_id in CLINICAL_CARD_IDS:
            return card_id
    return ""


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


def _source_bucket(input_path: Path) -> str:
    stem = input_path.stem
    if stem.startswith("figment_sft_"):
        return stem
    return "unknown"


def _original_source_dataset_version(row: dict[str, Any], input_path: Path) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    replay_audit = metadata.get("v6_replay_audit")
    if isinstance(replay_audit, dict) and replay_audit.get("source_dataset_version"):
        return str(replay_audit["source_dataset_version"])
    version = row.get("version") or metadata.get("dataset_version")
    if version:
        return str(version)
    return _source_bucket(input_path)


def _category(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("category") or metadata.get("category") or metadata.get("failure_class") or "missing")


def _task_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if metadata.get("task_type"):
        return str(metadata["task_type"])
    output = _assistant_output(row) or {}
    return "navigator_full" if "protocol_urgency" in output else "focused_repair"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


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
