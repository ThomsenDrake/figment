"""Generate, merge, verify, and Modal-prep the full Figment v3 SFT corpus."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


DATASET_VERSION = "figment_sft_v3"
DEFAULT_SHARD_PREFIX = Path("data/finetune/shards/figment_sft_v3_full_shard")
DEFAULT_OUTPUT = Path("data/finetune/figment_sft_v3.jsonl")
DEFAULT_CASE_SPECS = Path("data/finetune/figment_sft_v3_case_specs.jsonl")
DEFAULT_MANIFEST = Path("data/finetune/figment_sft_v3_manifest.json")
DEFAULT_MODAL_DIR = Path("data/finetune/modal/figment_sft_v3")
DEFAULT_TEACHER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


@dataclass(frozen=True)
class ShardSpec:
    index: int
    row_count: int
    start_index: int
    index_stride: int
    output: Path
    case_specs: Path
    manifest: Path
    log: Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--navigator-count", type=int, default=2500)
    parser.add_argument("--repair-count", type=int, default=500)
    parser.add_argument("--rows-per-shard", type=int, default=50)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--base-start-index", type=int, default=20000)
    parser.add_argument("--shard-prefix", type=Path, default=DEFAULT_SHARD_PREFIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-specs", type=Path, default=DEFAULT_CASE_SPECS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--modal-output-dir", type=Path, default=DEFAULT_MODAL_DIR)
    parser.add_argument("--teacher-model-id", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--teacher-max-tokens", type=int, default=700)
    parser.add_argument("--teacher-error-retries", type=int, default=2)
    parser.add_argument("--teacher-error-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--max-attempts-multiplier", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", default="figment-modal-sft-v3")
    parser.add_argument("--min-validation-group-size", type=int, default=5)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    parser.add_argument("--only-generate", action="store_true")
    parser.add_argument("--log-rejections", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic fallback rows instead of teacher calls.")
    args = parser.parse_args(argv)

    if args.navigator_count <= 0:
        raise SystemExit("--navigator-count must be positive")
    if args.repair_count < 0:
        raise SystemExit("--repair-count must be non-negative")
    if args.rows_per_shard <= 0:
        raise SystemExit("--rows-per-shard must be positive")
    if args.parallelism <= 0:
        raise SystemExit("--parallelism must be positive")
    if args.max_attempts_multiplier < 1:
        raise SystemExit("--max-attempts-multiplier must be at least 1")

    shard_specs = build_shard_specs(
        navigator_count=args.navigator_count,
        rows_per_shard=args.rows_per_shard,
        base_start_index=args.base_start_index,
        shard_prefix=args.shard_prefix,
    )
    generation_results = []
    if not args.skip_generation:
        generation_results = generate_shards(args, shard_specs)

    if args.only_generate:
        print(json.dumps({"shards": [spec.__dict__ | {"output": str(spec.output)} for spec in shard_specs]}, default=str))
        return 0

    merge_cmd = [
        sys.executable,
        "scripts/merge_finetune_shards.py",
        "--dataset-version",
        args.dataset_version,
        "--shard-prefix",
        str(args.shard_prefix),
        "--shard-count",
        str(len(shard_specs)),
        "--output",
        str(args.output),
        "--case-specs",
        str(args.case_specs),
        "--manifest",
        str(args.manifest),
    ]
    merge_summary = _run_json_command(merge_cmd)

    repair_summary = None
    if not args.skip_repair and args.repair_count:
        repair_summary = _run_json_command(build_repair_command(args))

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

    summary = {
        "dataset_version": args.dataset_version,
        "navigator_count": args.navigator_count,
        "repair_count": args.repair_count,
        "shard_count": len(shard_specs),
        "generation": generation_results,
        "merge": merge_summary,
        "repair": repair_summary,
        "verify": verify_summary,
        "modal": modal_summary,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


def build_shard_specs(
    *,
    navigator_count: int,
    rows_per_shard: int,
    base_start_index: int,
    shard_prefix: Path,
) -> list[ShardSpec]:
    shard_count = math.ceil(navigator_count / rows_per_shard)
    specs: list[ShardSpec] = []
    remaining = navigator_count
    for index in range(shard_count):
        row_count = min(rows_per_shard, remaining)
        base = Path(f"{shard_prefix}{index}")
        specs.append(
            ShardSpec(
                index=index,
                row_count=row_count,
                start_index=base_start_index + index,
                index_stride=shard_count,
                output=Path(f"{base}.jsonl"),
                case_specs=Path(f"{base}_case_specs.jsonl"),
                manifest=Path(f"{base}_manifest.json"),
                log=Path(f"{base}.log"),
            )
        )
        remaining -= row_count
    return specs


def generate_shards(args: argparse.Namespace, shard_specs: list[ShardSpec]) -> list[dict[str, Any]]:
    for spec in shard_specs:
        spec.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        futures = [executor.submit(generate_one_shard, args, spec) for spec in shard_specs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"shard_complete": result}, sort_keys=True), flush=True)
    return sorted(results, key=lambda item: int(item["index"]))


def generate_one_shard(args: argparse.Namespace, spec: ShardSpec) -> dict[str, Any]:
    existing = _read_manifest(spec.manifest)
    if existing and int(existing.get("row_count") or 0) >= spec.row_count:
        return {"index": spec.index, "status": "skipped_existing", "rows": int(existing.get("row_count") or 0)}

    cmd = [
        sys.executable,
        "scripts/generate_finetune_data.py",
        "--dataset-version",
        args.dataset_version,
        "--count",
        str(spec.row_count),
        "--output",
        str(spec.output),
        "--case-specs",
        str(spec.case_specs),
        "--manifest",
        str(spec.manifest),
        "--teacher-model-id",
        args.teacher_model_id,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--teacher-max-tokens",
        str(args.teacher_max_tokens),
        "--teacher-error-retries",
        str(args.teacher_error_retries),
        "--teacher-error-sleep-seconds",
        str(args.teacher_error_sleep_seconds),
        "--candidate-count",
        "1",
        "--high-risk-candidate-count",
        "1",
        "--max-attempts",
        str(spec.row_count * args.max_attempts_multiplier),
        "--start-index",
        str(spec.start_index),
        "--index-stride",
        str(spec.index_stride),
        "--resume",
    ]
    if args.log_rejections:
        cmd.append("--log-rejections")
    if args.dry_run:
        cmd.append("--dry-run")

    spec.log.parent.mkdir(parents=True, exist_ok=True)
    with spec.log.open("a", encoding="utf-8") as log:
        log.write("\n=== command ===\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"shard {spec.index} failed with exit {completed.returncode}; see {spec.log}")
    manifest = _read_manifest(spec.manifest)
    return {
        "index": spec.index,
        "status": "generated",
        "rows": int(manifest.get("row_count") or 0),
        "attempts": int(manifest.get("attempts") or 0),
        "manifest": str(spec.manifest),
        "log": str(spec.log),
    }


def build_repair_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/augment_finetune_repair_rows.py",
        "--dataset-version",
        args.dataset_version,
        "--dataset",
        str(args.output),
        "--case-specs",
        str(args.case_specs),
        "--manifest",
        str(args.manifest),
        "--repair-count",
        str(args.repair_count),
    ]


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _run_json_command(cmd: list[str]) -> dict[str, Any]:
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
