"""Prepare Figment SFT JSONL for Modal fine-tuning.

The generated SFT rows are already harness-aligned. This script keeps that
shape intact, validates the two-message chat contract, and writes a small
train/validation split that a Modal job can stage into a Volume.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
from datetime import UTC
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_DATASET_VERSION = "figment_sft_v1"
DEFAULT_DATASET = Path("data/finetune/figment_sft_v1.jsonl")
DEFAULT_OUTPUT_ROOT = Path("data/finetune/modal")


class DatasetPrepError(ValueError):
    """Raised when an SFT row is not safe to hand to the Modal trainer."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", default="figment-modal-sft-v1")
    parser.add_argument("--min-validation-group-size", type=int, default=5)
    args = parser.parse_args(argv)

    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / args.dataset_version
    manifest = prepare_dataset(
        dataset_path=args.dataset,
        output_dir=output_dir,
        dataset_version=args.dataset_version,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        min_validation_group_size=args.min_validation_group_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def prepare_dataset(
    *,
    dataset_path: Path,
    output_dir: Path,
    dataset_version: str,
    validation_fraction: float = 0.1,
    seed: str = "figment-modal-sft-v1",
    min_validation_group_size: int = 5,
) -> dict[str, Any]:
    if not dataset_path.exists():
        raise DatasetPrepError(f"dataset does not exist: {dataset_path}")
    if not 0 <= validation_fraction < 1:
        raise DatasetPrepError("--validation-fraction must be in [0, 1)")
    if min_validation_group_size < 2:
        raise DatasetPrepError("--min-validation-group-size must be at least 2")

    rows = _read_jsonl(dataset_path)
    if not rows:
        raise DatasetPrepError(f"dataset is empty: {dataset_path}")

    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        _validate_row(row, row_number)
        row_id = _row_id(row)
        if row_id in seen_ids:
            raise DatasetPrepError(f"row {row_number}: duplicate row id {row_id!r}")
        seen_ids.add(row_id)

    train_rows, validation_rows = _split_rows(
        rows,
        validation_fraction=validation_fraction,
        seed=seed,
        min_validation_group_size=min_validation_group_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    manifest_path = output_dir / "manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)

    manifest = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_dataset": str(dataset_path),
        "row_count": len(rows),
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "task_type_counts": dict(sorted(Counter(_task_type(row) for row in rows).items())),
        "group_counts": dict(sorted(Counter(_group_key(row) for row in rows).items())),
        "train_group_counts": dict(sorted(Counter(_group_key(row) for row in train_rows).items())),
        "validation_group_counts": dict(sorted(Counter(_group_key(row) for row in validation_rows).items())),
        "validation_fraction": validation_fraction,
        "seed": seed,
        "min_validation_group_size": min_validation_group_size,
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "train_sha256": _sha256_path(train_path),
        "validation_sha256": _sha256_path(validation_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetPrepError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise DatasetPrepError(f"{path}:{line_number}: expected object row")
        rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _validate_row(row: dict[str, Any], row_number: int) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or [message.get("role") for message in messages] != ["user", "assistant"]:
        raise DatasetPrepError(f"row {row_number}: expected user/assistant messages")
    for message_index, message in enumerate(messages, start=1):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise DatasetPrepError(f"row {row_number}: message {message_index} has empty content")
    if not _row_id(row):
        raise DatasetPrepError(f"row {row_number}: missing uuid or case_id")


def _split_rows(
    rows: list[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: str,
    min_validation_group_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)

    validation_ids: set[str] = set()
    for group_rows in groups.values():
        if validation_fraction == 0 or len(group_rows) < min_validation_group_size:
            continue
        validation_count = max(1, round(len(group_rows) * validation_fraction))
        validation_count = min(validation_count, len(group_rows) - 1)
        ranked = sorted(group_rows, key=lambda row: _split_key(row, seed))
        validation_ids.update(_row_id(row) for row in ranked[:validation_count])

    train_rows = [row for row in rows if _row_id(row) not in validation_ids]
    validation_rows = [row for row in rows if _row_id(row) in validation_ids]
    return train_rows, validation_rows


def _split_key(row: dict[str, Any], seed: str) -> str:
    return hashlib.sha256(f"{seed}:{_row_id(row)}".encode("utf-8")).hexdigest()


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("uuid") or row.get("case_id") or "")


def _group_key(row: dict[str, Any]) -> str:
    task_type = _task_type(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if task_type == "focused_repair":
        return f"focused_repair:{metadata.get('repair_scope') or row.get('category') or 'unknown'}"
    return f"{task_type}:{row.get('category') or 'unknown'}"


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
