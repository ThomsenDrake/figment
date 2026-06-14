"""Merge Figment v10 delta rows with the v9 corpus and prepare Modal splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_v8_training_corpus import main as _merge_main  # noqa: E402


DEFAULT_BASE = "data/finetune/figment_sft_v9.jsonl"
DEFAULT_BASE_CASE_SPECS = "data/finetune/figment_sft_v9_case_specs.jsonl"
DEFAULT_DELTA = "data/finetune/figment_sft_v10_delta.jsonl"
DEFAULT_DELTA_CASE_SPECS = "data/finetune/figment_sft_v10_delta_case_specs.jsonl"
DEFAULT_OUTPUT = "data/finetune/figment_sft_v10.jsonl"
DEFAULT_CASE_SPECS = "data/finetune/figment_sft_v10_case_specs.jsonl"
DEFAULT_MANIFEST = "data/finetune/figment_sft_v10_manifest.json"
DEFAULT_MODAL_DIR = "data/finetune/modal/figment_sft_v10"


def build_merge_args(argv: list[str] | None = None) -> list[str]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--base-case-specs", default=DEFAULT_BASE_CASE_SPECS)
    parser.add_argument("--delta", default=DEFAULT_DELTA)
    parser.add_argument("--delta-case-specs", default=DEFAULT_DELTA_CASE_SPECS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--case-specs", default=DEFAULT_CASE_SPECS)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--modal-output-dir", default=DEFAULT_MODAL_DIR)
    parser.add_argument("--dataset-version", default="figment_sft_v10")
    parsed, remaining = parser.parse_known_args(raw_args)
    return [
        "--base",
        parsed.base,
        "--base-case-specs",
        parsed.base_case_specs,
        "--delta",
        parsed.delta,
        "--delta-case-specs",
        parsed.delta_case_specs,
        "--output",
        parsed.output,
        "--case-specs",
        parsed.case_specs,
        "--manifest",
        parsed.manifest,
        "--modal-output-dir",
        parsed.modal_output_dir,
        "--dataset-version",
        parsed.dataset_version,
        *remaining,
    ]


def main(argv: list[str] | None = None) -> int:
    return _merge_main(build_merge_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
