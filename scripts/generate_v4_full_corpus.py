"""Generate, verify, and Modal-prep the full Figment v4 SFT corpus."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_v3_full_corpus import main as _run_full_corpus  # noqa: E402


DEFAULT_ARGS = [
    "--dataset-version",
    "figment_sft_v4",
    "--navigator-count",
    "1500",
    "--repair-count",
    "150",
    "--base-start-index",
    "40000",
    "--shard-prefix",
    "data/finetune/shards/figment_sft_v4_full_shard",
    "--output",
    "data/finetune/figment_sft_v4.jsonl",
    "--case-specs",
    "data/finetune/figment_sft_v4_case_specs.jsonl",
    "--manifest",
    "data/finetune/figment_sft_v4_manifest.json",
    "--modal-output-dir",
    "data/finetune/modal/figment_sft_v4",
    "--seed",
    "figment-modal-sft-v4",
]


def build_corpus_args(argv: list[str] | None = None) -> list[str]:
    return DEFAULT_ARGS + list(sys.argv[1:] if argv is None else argv)


def main(argv: list[str] | None = None) -> int:
    return _run_full_corpus(build_corpus_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
