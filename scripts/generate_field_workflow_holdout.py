"""Generate the frozen v3 field-workflow holdout eval cases."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_finetune_data import case_spec_record  # noqa: E402
from scripts.generate_finetune_data import forbidden_behavior_for_version  # noqa: E402
from scripts.generate_finetune_data import generate_case_spec  # noqa: E402
from scripts.generate_finetune_data import prepare_case  # noqa: E402
from scripts.generate_finetune_data import safety_boundary_for_version  # noqa: E402
from scripts.generate_finetune_data import stable_hash  # noqa: E402
from figment.eval_metrics import bucket_expected_observation_cues  # noqa: E402
from figment.retrieval import load_protocol_cards  # noqa: E402


HOLDOUT_VERSION = "field_workflow_holdout_v1"
OUTPUT_PATH = Path("data/eval/field_workflow_holdout_v1.jsonl")
MANIFEST_PATH = Path("data/eval/field_workflow_holdout_v1_manifest.json")
SOURCE_DATASET_VERSION = "figment_sft_v3_holdout_source"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args(argv)

    manifest = generate_holdout(count=args.count, output_path=args.output, manifest_path=args.manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def generate_holdout(*, count: int, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")

    cards_by_id = {str(card["card_id"]): card for card in load_protocol_cards()}
    rows: list[dict[str, Any]] = []
    row_hashes: list[dict[str, str]] = []
    attempts = 0
    max_attempts = count * 8

    while len(rows) < count and attempts < max_attempts:
        source_spec = generate_case_spec(attempts, cards_by_id, dataset_version=SOURCE_DATASET_VERSION)
        attempts += 1
        prepared = prepare_case(source_spec, cards_by_id)
        record = case_spec_record(prepared)
        holdout_case_id = f"{HOLDOUT_VERSION}-{len(rows):06d}"
        row = _holdout_row(
            record,
            holdout_case_id=holdout_case_id,
            source_case_id=source_spec.case_id,
            prompt_hash=stable_hash(prepared.prompt),
            prompt_template_hash=prepared.prompt_hash,
        )
        row_hashes.append({"case_id": holdout_case_id, "sha256": _json_sha256(row)})
        rows.append(row)

    if len(rows) < count:
        raise RuntimeError(f"could only generate {len(rows)} holdout rows after {attempts} attempts")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    manifest = {
        "dataset_version": HOLDOUT_VERSION,
        "source_dataset_version": SOURCE_DATASET_VERSION,
        "row_count": len(rows),
        "output_path": str(output_path),
        "output_sha256": _file_sha256(output_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "source_generator": "scripts/generate_field_workflow_holdout.py",
        "source_generator_prompt_family": "figment_sft_v3_field_workflow",
        "category_counts": dict(sorted(Counter(row["workflow_category"] for row in rows).items())),
        "row_hashes": row_hashes,
        "holdout_policy": {
            "never_train_on_this_file": True,
            "never_copy_close_paraphrases_into_training": True,
            "freeze_case_ids": True,
            "primary_v3_success_surface": True,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _holdout_row(
    record: dict[str, Any],
    *,
    holdout_case_id: str,
    source_case_id: str,
    prompt_hash: str,
    prompt_template_hash: str,
) -> dict[str, Any]:
    workflow_category = str(record.get("workflow_category") or record.get("failure_class") or "field_workflow")
    cue_buckets = bucket_expected_observation_cues(record["expected_missing_observations"])
    return {
        "case_id": holdout_case_id,
        "dataset_version": HOLDOUT_VERSION,
        "source_generator_case_id": source_case_id,
        "workflow_category": workflow_category,
        "target_protocol_card_id": record["target_protocol_card_id"],
        "structured_intake": record["structured_intake"],
        "expected_red_flag_rule_ids": record["expected_red_flag_rule_ids"],
        "expected_min_protocol_urgency": record["expected_min_protocol_urgency"],
        "expected_source_card_ids": record["expected_source_card_ids"],
        "expected_candidate_pathway_card_ids": record["expected_candidate_pathway_card_ids"],
        "expected_missing_observations": record["expected_missing_observations"],
        "expected_model_observation_cues": cue_buckets["model"],
        "expected_handoff_cues": cue_buckets["handoff"],
        "expected_harness_evidence_cues": cue_buckets["harness"],
        "workflow_priority_observations": record.get("workflow_priority_observations", []),
        "retrieved_card_ids": record["retrieved_card_ids"],
        "tags": record.get("tags", []),
        "safety_notes": safety_boundary_for_version(HOLDOUT_VERSION),
        "forbidden_behavior": forbidden_behavior_for_version(HOLDOUT_VERSION),
        "prompt_hash": prompt_hash,
        "prompt_template_hash": prompt_template_hash,
    }


def _json_sha256(value: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
