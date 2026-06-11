"""Generate, verify, and Modal-prep the full Figment v5 SFT corpus."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_finetune_data import V5_FOCUSED_COUNTS  # noqa: E402
from scripts.generate_v3_full_corpus import main as _run_full_corpus  # noqa: E402


DEFAULT_OUTPUT_VERSION = "figment_sft_v5"
DEFAULT_TEACHER_MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_COUNTS = dict(V5_FOCUSED_COUNTS)
DEFAULT_NAVIGATOR_COUNT = sum(DEFAULT_COUNTS.values())

DEFAULT_ARGS = [
    "--dataset-version",
    DEFAULT_OUTPUT_VERSION,
    "--navigator-count",
    str(DEFAULT_NAVIGATOR_COUNT),
    "--repair-count",
    "200",
    "--teacher-model-id",
    DEFAULT_TEACHER_MODEL_ID,
    "--base-start-index",
    "60000",
    "--shard-prefix",
    "data/finetune/shards/figment_sft_v5_full_shard",
    "--output",
    "data/finetune/figment_sft_v5.jsonl",
    "--case-specs",
    "data/finetune/figment_sft_v5_case_specs.jsonl",
    "--manifest",
    "data/finetune/figment_sft_v5_manifest.json",
    "--modal-output-dir",
    "data/finetune/modal/figment_sft_v5",
    "--seed",
    "figment-modal-sft-v5",
]


def build_corpus_args(argv: list[str] | None = None) -> list[str]:
    return DEFAULT_ARGS + list(sys.argv[1:] if argv is None else argv)


def main(argv: list[str] | None = None) -> int:
    return _run_full_corpus(build_corpus_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
