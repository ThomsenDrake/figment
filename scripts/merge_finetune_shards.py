"""Merge disjoint Figment SFT teacher-generation shards."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--shard-prefix", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-specs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = merge_shards(
        dataset_version=args.dataset_version,
        shard_prefix=args.shard_prefix,
        shard_count=args.shard_count,
        output_path=args.output,
        case_specs_path=args.case_specs,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def merge_shards(
    *,
    dataset_version: str,
    shard_prefix: Path,
    shard_count: int,
    output_path: Path,
    case_specs_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")

    rows_by_id: dict[str, dict[str, Any]] = {}
    specs_by_id: dict[str, dict[str, Any]] = {}
    shard_summaries: list[dict[str, Any]] = []
    source_attempts = 0
    anti_overfit_eval_paths: set[str] = set()
    anti_overfit_signature_count = 0
    anti_overfit_enabled = False

    for shard_index in range(shard_count):
        dataset_shard, spec_shard, manifest_shard = shard_paths(shard_prefix, shard_index)
        rows = _read_jsonl(dataset_shard)
        specs = _read_jsonl(spec_shard)
        source_manifest = _read_json(manifest_shard) if manifest_shard.exists() else {}
        source_attempts += int(source_manifest.get("attempts") or 0)
        source_exclusions = source_manifest.get("anti_overfit_exclusions")
        if isinstance(source_exclusions, dict):
            anti_overfit_enabled = anti_overfit_enabled or bool(source_exclusions.get("enabled"))
            anti_overfit_eval_paths.update(str(path) for path in source_exclusions.get("eval_paths", []) if str(path))
            anti_overfit_signature_count = max(
                anti_overfit_signature_count,
                int(source_exclusions.get("signature_count") or 0),
            )

        for row in rows:
            row_id = str(row.get("case_id") or row.get("uuid") or "")
            if not row_id:
                raise ValueError(f"{dataset_shard}: row missing case_id")
            if str(row.get("version")) != dataset_version:
                raise ValueError(f"{dataset_shard}: {row_id} version does not match {dataset_version}")
            if row_id in rows_by_id:
                raise ValueError(f"duplicate row id across shards: {row_id}")
            rows_by_id[row_id] = row

        for spec in specs:
            spec_id = str(spec.get("case_id") or "")
            if not spec_id:
                raise ValueError(f"{spec_shard}: spec missing case_id")
            if str(spec.get("dataset_version") or dataset_version) != dataset_version:
                raise ValueError(f"{spec_shard}: {spec_id} dataset_version does not match {dataset_version}")
            if spec_id in specs_by_id:
                raise ValueError(f"duplicate case spec id across shards: {spec_id}")
            specs_by_id[spec_id] = spec

        shard_summaries.append(
            {
                "index": shard_index,
                "dataset_path": str(dataset_shard),
                "case_specs_path": str(spec_shard),
                "manifest_path": str(manifest_shard),
                "row_count": len(rows),
                "case_spec_count": len(specs),
                "attempts": int(source_manifest.get("attempts") or 0),
                "accepted_by_failure_class": source_manifest.get("accepted_by_failure_class", {}),
                "rejection_reasons": source_manifest.get("rejection_reasons", {}),
            }
        )

    missing_specs = sorted(set(rows_by_id) - set(specs_by_id))
    orphan_specs = sorted(set(specs_by_id) - set(rows_by_id))
    if missing_specs:
        raise ValueError(f"missing case specs for rows: {', '.join(missing_specs[:10])}")
    if orphan_specs:
        raise ValueError(f"case specs without rows: {', '.join(orphan_specs[:10])}")

    rows = [rows_by_id[row_id] for row_id in sorted(rows_by_id)]
    specs = [specs_by_id[case_id] for case_id in sorted(specs_by_id)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    case_specs_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, rows)
    _write_jsonl(case_specs_path, specs)

    task_counts = Counter(_task_type(row) for row in rows)
    category_counts = Counter(str(row.get("category") or "unknown") for row in rows)
    summary = {
        "dataset_version": dataset_version,
        "merged_at": datetime.now(UTC).isoformat(),
        "row_count": len(rows),
        "case_spec_count": len(specs),
        "shard_count": shard_count,
        "source_attempts": source_attempts,
        "output_path": str(output_path),
        "case_specs_path": str(case_specs_path),
        "output_sha256": _sha256_path(output_path),
        "case_specs_sha256": _sha256_path(case_specs_path),
        "task_type_counts": dict(sorted(task_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "anti_overfit_exclusions": {
            "enabled": anti_overfit_enabled,
            "eval_paths": sorted(anti_overfit_eval_paths),
            "signature_count": anti_overfit_signature_count,
        },
        "shards": shard_summaries,
    }
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def shard_paths(shard_prefix: Path, shard_index: int) -> tuple[Path, Path, Path]:
    base = f"{shard_prefix}{shard_index}"
    return (
        Path(f"{base}.jsonl"),
        Path(f"{base}_case_specs.jsonl"),
        Path(f"{base}_manifest.json"),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(item)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    item = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(item, dict):
        raise ValueError(f"{path}: expected JSON object")
    return item


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _task_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("task_type") or "navigator_full")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
