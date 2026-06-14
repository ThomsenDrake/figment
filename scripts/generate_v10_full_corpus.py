"""Generate, verify, and Modal-prep the Figment v10 delta SFT corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_finetune_data import V10_NAVIGATOR_COUNTS  # noqa: E402
from scripts.generate_v3_full_corpus import main as _run_full_corpus  # noqa: E402


DEFAULT_OUTPUT_VERSION = "figment_sft_v10_delta"
DEFAULT_TEACHER_MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_NAVIGATOR_COUNT = sum(V10_NAVIGATOR_COUNTS.values())
DEFAULT_REPAIR_COUNT = 0
DEFAULT_ARGS = [
    "--dataset-version",
    DEFAULT_OUTPUT_VERSION,
    "--navigator-count",
    str(DEFAULT_NAVIGATOR_COUNT),
    "--repair-count",
    str(DEFAULT_REPAIR_COUNT),
    "--teacher-model-id",
    DEFAULT_TEACHER_MODEL_ID,
    "--base-start-index",
    "120000",
    "--rows-per-shard",
    "40",
    "--shard-prefix",
    "data/finetune/shards/figment_sft_v10_delta_full_shard",
    "--output",
    "data/finetune/figment_sft_v10_delta.jsonl",
    "--case-specs",
    "data/finetune/figment_sft_v10_delta_case_specs.jsonl",
    "--manifest",
    "data/finetune/figment_sft_v10_delta_manifest.json",
    "--modal-output-dir",
    "data/finetune/modal/figment_sft_v10_delta",
    "--seed",
    "figment-modal-sft-v10-delta",
]


def build_corpus_args(argv: list[str] | None = None) -> list[str]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--navigator-count", type=int, default=DEFAULT_NAVIGATOR_COUNT)
    parser.add_argument("--repair-count", type=int, default=DEFAULT_REPAIR_COUNT)
    parsed, remaining = parser.parse_known_args(raw_args)
    defaults = list(DEFAULT_ARGS)
    defaults[defaults.index("--navigator-count") + 1] = str(parsed.navigator_count)
    defaults[defaults.index("--repair-count") + 1] = str(parsed.repair_count)
    return defaults + remaining


def main(argv: list[str] | None = None) -> int:
    return _run_full_corpus(build_corpus_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
