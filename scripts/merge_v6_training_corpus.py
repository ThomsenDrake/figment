"""Merge Figment v6 delta rows with audited replay rows and prepare Modal splits."""

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


DEFAULT_DELTA = Path("data/finetune/figment_sft_v6_delta.jsonl")
DEFAULT_DELTA_CASE_SPECS = Path("data/finetune/figment_sft_v6_delta_case_specs.jsonl")
DEFAULT_REPLAY = Path("data/finetune/figment_sft_v6_replay.jsonl")
DEFAULT_OUTPUT = Path("data/finetune/figment_sft_v6.jsonl")
DEFAULT_CASE_SPECS = Path("data/finetune/figment_sft_v6_case_specs.jsonl")
DEFAULT_MANIFEST = Path("data/finetune/figment_sft_v6_manifest.json")
DEFAULT_MODAL_DIR = Path("data/finetune/modal/figment_sft_v6")
SOURCE_CASE_SPECS = {
    "figment_sft_v3": Path("data/finetune/figment_sft_v3_case_specs.jsonl"),
    "figment_sft_v4": Path("data/finetune/figment_sft_v4_case_specs.jsonl"),
    "figment_sft_v5": Path("data/finetune/figment_sft_v5_case_specs.jsonl"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Path, default=DEFAULT_DELTA)
    parser.add_argument("--delta-case-specs", type=Path, default=DEFAULT_DELTA_CASE_SPECS)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-specs", type=Path, default=DEFAULT_CASE_SPECS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--modal-output-dir", type=Path, default=DEFAULT_MODAL_DIR)
    parser.add_argument("--dataset-version", default="figment_sft_v6")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", default="figment-modal-sft-v6")
    parser.add_argument("--min-validation-group-size", type=int, default=5)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-modal-prep", action="store_true")
    args = parser.parse_args(argv)

    summary = merge_v6_corpus(
        delta_path=args.delta,
        delta_case_specs_path=args.delta_case_specs,
        replay_path=args.replay,
        output_path=args.output,
        case_specs_path=args.case_specs,
        manifest_path=args.manifest,
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
    args.manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def merge_v6_corpus(
    *,
    delta_path: Path,
    delta_case_specs_path: Path,
    replay_path: Path,
    output_path: Path,
    case_specs_path: Path,
    manifest_path: Path,
    dataset_version: str,
) -> dict[str, Any]:
    delta_rows = _read_jsonl(delta_path)
    replay_rows = _read_jsonl(replay_path)
    rows = _dedupe_rows(delta_rows + replay_rows)
    specs = _case_specs_for_rows(delta_case_specs_path, rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    case_specs_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, rows)
    _write_jsonl(case_specs_path, specs)

    return {
        "dataset_version": dataset_version,
        "merged_at": datetime.now(UTC).isoformat(),
        "row_count": len(rows),
        "delta_rows": len(delta_rows),
        "replay_rows": len(replay_rows),
        "case_spec_count": len(specs),
        "output_path": str(output_path),
        "case_specs_path": str(case_specs_path),
        "output_sha256": _sha256_path(output_path),
        "case_specs_sha256": _sha256_path(case_specs_path),
        "task_type_counts": dict(sorted(Counter(_task_type(row) for row in rows).items())),
        "category_counts": dict(sorted(Counter(str(row.get("category") or "unknown") for row in rows).items())),
        "replay_source_counts": dict(sorted(Counter(_replay_source(row) for row in replay_rows).items())),
    }


def _case_specs_for_rows(delta_case_specs_path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs_by_id = {str(item.get("case_id")): item for item in _read_jsonl(delta_case_specs_path)}
    source_specs_cache: dict[str, dict[str, dict[str, Any]]] = {}
    needed_ids = {str(row.get("metadata", {}).get("base_case_id") or row.get("case_id")) for row in rows}
    for row in rows:
        base_id = str(row.get("metadata", {}).get("base_case_id") or row.get("case_id"))
        if base_id in specs_by_id:
            continue
        source_version = _source_dataset_version(row)
        source_path = SOURCE_CASE_SPECS.get(source_version)
        if source_path is None:
            continue
        if source_version not in source_specs_cache:
            source_specs_cache[source_version] = {
                str(item.get("case_id")): item for item in _read_jsonl(source_path)
            }
        source_spec = source_specs_cache[source_version].get(base_id)
        if source_spec is not None:
            specs_by_id[base_id] = source_spec

    missing = sorted(needed_ids - set(specs_by_id))
    if missing:
        raise ValueError(f"missing case specs for rows: {', '.join(missing[:10])}")
    return [specs_by_id[case_id] for case_id in sorted(needed_ids)]


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


def _source_dataset_version(row: dict[str, Any]) -> str:
    replay_audit = row.get("metadata", {}).get("v6_replay_audit")
    if isinstance(replay_audit, dict) and replay_audit.get("source_dataset_version"):
        return str(replay_audit["source_dataset_version"])
    return str(row.get("version") or row.get("metadata", {}).get("dataset_version") or "")


def _replay_source(row: dict[str, Any]) -> str:
    return _source_dataset_version(row) or "unknown"


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
