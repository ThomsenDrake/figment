"""Generate Figment local-4B supervised fine-tuning data.

The generator creates synthetic, de-identified protocol-navigation cases,
asks the Ultra teacher for candidate gold navigator JSON, validates candidates
with Figment's deterministic gates, and writes accepted SFT rows plus a
manifest. It intentionally does not copy locked eval rows or NVIDIA dataset
rows.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from dataclasses import replace
from datetime import UTC
from datetime import datetime
import hashlib
import httpx
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from time import sleep
from typing import Any
import urllib.parse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figment.config import NVIDIA_API_BASE_URL  # noqa: E402
from figment.config import load_config  # noqa: E402
from figment.eval_metrics import bucket_expected_observation_cues, score_expected_labels  # noqa: E402
from figment.harness_evidence import build_harness_evidence  # noqa: E402
from figment.model_client import ModelClientError  # noqa: E402
from figment.model_client import canned_navigator_output  # noqa: E402
from figment.observation_targets import CARD_IDS_EXEMPT_FROM_OBSERVATION_TARGETS  # noqa: E402
from figment.observation_targets import TRACE_ONLY_REQUIRED_OBSERVATION_IDS_KEY  # noqa: E402
from figment.observation_targets import apply_navigation_scaffolding  # noqa: E402
from figment.observation_targets import required_observation_targets  # noqa: E402
from figment.prompt_builder import REQUIRED_JSON_SKELETON  # noqa: E402
from figment.prompt_builder import SYSTEM_PROMPT  # noqa: E402
from figment.prompt_builder import build_prompt  # noqa: E402
from figment.retrieval import load_protocol_cards  # noqa: E402
from figment.retrieval import query_from_intake  # noqa: E402
from figment.retrieval import search_protocol_cards  # noqa: E402
from figment.rules import run_red_flag_checks  # noqa: E402
from figment.trace import stable_hash  # noqa: E402
from figment.validators import urgency_floor_from_rules  # noqa: E402
from figment.validators import validate_navigator_output  # noqa: E402


TEACHER_MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b"
DATASET_VERSION = "figment_sft_v1"
OUTPUT_PATH = Path("data/finetune/figment_sft_v1.jsonl")
MANIFEST_PATH = Path("data/finetune/figment_sft_v1_manifest.json")
CASE_SPEC_PATH = Path("data/finetune/figment_sft_v1_case_specs.jsonl")
CLINICAL_CARD_IDS = (
    "AMS-RED-FLAGS-v1",
    "CHEST-PAIN-ESCALATION-v1",
    "PED-DEHYD-RED-FLAGS-v1",
    "FEVER-RED-FLAGS-v1",
    "PREG-DANGER-SIGNS-v1",
    "RESP-DISTRESS-RED-FLAGS-v1",
    "STROKE-SIGNS-v1",
    "WOUND-INFECTION-ESCALATION-v1",
)
SAFETY_CARD_ID = "SAFETY-BOUNDARIES-v1"
SBAR_CARD_ID = "REFERRAL-SBAR-v1"
FAILURE_DISTRIBUTION = (
    ("missing_observation_cues", 40),
    ("negation_safety_boundary", 20),
    ("source_card_candidate_pathway", 15),
    ("sbar_grounding", 15),
    ("forbidden_instruction_avoidance", 5),
    ("fallback_rescue_shape", 5),
)
V2_FAILURE_DISTRIBUTION = (
    ("missing_observation_cues", 40),
    ("negation_safety_boundary", 25),
    ("source_card_candidate_pathway", 20),
    ("sbar_grounding", 10),
    ("forbidden_instruction_avoidance", 3),
    ("fallback_rescue_shape", 2),
)
V3_FAILURE_DISTRIBUTION = (
    ("rural_clinic_intake", 18),
    ("disaster_triage", 16),
    ("radio_handoff", 8),
    ("asr_confirmed_text", 6),
    ("escalation_precision", 14),
    ("missing_observation_prioritization", 14),
    ("sbar_handoff_usefulness", 10),
    ("source_card_discipline", 6),
    ("low_resource_constraints", 7),
    ("workflow_repair_seed", 1),
)
V4_FAILURE_DISTRIBUTION = (
    ("radio_handoff", 25),
    ("sbar_handoff_usefulness", 22),
    ("source_card_discipline", 14),
    ("low_resource_constraints", 10),
    ("missing_observation_prioritization", 10),
    ("workflow_repair_seed", 7),
    ("rural_clinic_intake", 4),
    ("disaster_triage", 3),
    ("escalation_precision", 5),
)
V5_FOCUSED_COUNTS = {
    "sbar_observation_ownership": 350,
    "required_observation_id_selection": 250,
    "source_card_invariant": 150,
    "noisy_field_audio_style": 100,
    "general_regression": 250,
}
V5_FAILURE_DISTRIBUTION = tuple(V5_FOCUSED_COUNTS.items())
V5_EXCLUDED_EVAL_CASE_IDS = ("field_workflow_holdout_v1-000054", "field_workflow_holdout_v1-000099")
FORBIDDEN_BEHAVIOR = [
    "Do not diagnose.",
    "Do not prescribe, dose, administer, or start medication.",
    "Do not discharge, send home, or replace local protocol.",
    "Do not downgrade deterministic red flags.",
]
SCORER_SAFE_FORBIDDEN_BEHAVIOR = [
    "Use protocol navigation only.",
    "Do not provide clinical orders or treatment instructions.",
    "Do not replace local protocol.",
    "Keep deterministic red flags visible.",
]
V2_FORBIDDEN_LEXICAL_PATTERNS = {
    "diagnose": re.compile(r"\bdiagnos(?:e|is|ed)\b", re.IGNORECASE),
    "prescribe": re.compile(r"\bprescrib(?:e|ing|ed)\b", re.IGNORECASE),
    "dose": re.compile(r"\b(?:dose|dosing)\b|\bgive \d+(?:\.\d+)?\s*(?:mg|mcg|ml)\b", re.IGNORECASE),
    "administer": re.compile(r"\b(?:administer|inject|dispense)\b", re.IGNORECASE),
    "medication": re.compile(r"\b(?:aspirin|antibiotic|opioid|insulin|medication|medications|drug)\b", re.IGNORECASE),
    "discharge": re.compile(r"\b(?:discharge(?: home)?|send (?:them|the patient|the person|him|her)?\s*home)\b", re.IGNORECASE),
}
V3_GENERIC_OUTPUT_PATTERNS = (
    re.compile(r"\brepeat\s+vitals?\b", re.IGNORECASE),
    re.compile(r"\bmonitor\s+closely\b", re.IGNORECASE),
    re.compile(r"\bfollow\s+protocol\b", re.IGNORECASE),
    re.compile(r"\bassess\s+(?:the\s+)?patient\b", re.IGNORECASE),
    re.compile(r"\bcollect\s+more\s+information\b", re.IGNORECASE),
    re.compile(r"\bcontinue\s+to\s+observe\b", re.IGNORECASE),
)
V5_GENERIC_OBSERVATION_PATTERNS = V3_GENERIC_OUTPUT_PATTERNS + (
    re.compile(r"\bask\s+(?:anything|everything)\s+else\b", re.IGNORECASE),
    re.compile(r"\bkeep\s+monitoring\b", re.IGNORECASE),
    re.compile(r"\bcheck\s+vitals?\b", re.IGNORECASE),
)
V3_SBAR_FAILURE_CLASSES = {"radio_handoff", "sbar_handoff_usefulness", "workflow_repair_seed"}
TEACHER_NOTE_MAX_TOKENS = 320


def dataset_paths(dataset_version: str) -> dict[str, Path]:
    """Return the default local artifact paths for a fine-tune dataset version."""

    root = Path("data/finetune")
    return {
        "output": root / f"{dataset_version}.jsonl",
        "manifest": root / f"{dataset_version}_manifest.json",
        "case_specs": root / f"{dataset_version}_case_specs.jsonl",
    }


def _exclusion_paths_for_generation(dataset_version: str, configured_paths: list[Path] | None) -> list[Path]:
    if configured_paths:
        return [path for path in configured_paths if path.exists()]
    if not uses_v3_field_workflow_policy(dataset_version) or dataset_version.startswith("field_workflow_holdout"):
        return []
    candidates = [
        Path("data/eval/initial_handwritten_cases.jsonl"),
        Path("data/eval/adversarial_strict_cases.jsonl"),
        Path("data/eval/comprehensive_hosted_cases.jsonl"),
        Path("data/eval/field_workflow_holdout_v1.jsonl"),
    ]
    return [path for path in candidates if path.exists()]


def load_exclusion_signatures(paths: list[Path]) -> list[ExclusionSignature]:
    signatures: list[ExclusionSignature] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("structured_intake"), dict):
                continue
            signatures.append(
                ExclusionSignature(
                    case_id=str(item.get("case_id") or f"{path}:{line_number}"),
                    source_path=str(path),
                    target_protocol_card_id=str(item.get("target_protocol_card_id") or ""),
                    workflow_category=str(item.get("workflow_category") or item.get("structured_intake", {}).get("workflow_category") or ""),
                    clinical_hash=_clinical_intake_hash(item["structured_intake"]),
                    tokens=frozenset(_clinical_intake_tokens(item["structured_intake"])),
                )
            )
    return signatures


def case_index_for_attempt(attempt_index: int, *, start_index: int = 0, index_stride: int = 1) -> int:
    """Map a local attempt number to a globally unique synthetic case index."""

    return start_index + (attempt_index * index_stride)


def uses_v3_field_workflow_policy(dataset_version: str) -> bool:
    """Return whether a dataset version should use v3 field-workflow behavior."""

    return (
        dataset_version.startswith("figment_sft_v3")
        or dataset_version.startswith("figment_sft_v4")
        or uses_v5_focused_policy(dataset_version)
        or dataset_version.startswith("field_workflow_holdout")
    )


def uses_v5_focused_policy(dataset_version: str) -> bool:
    """Return whether a dataset version should use v5 focused-training behavior."""

    return dataset_version.startswith("figment_sft_v5")


def forbidden_behavior_for_version(dataset_version: str) -> list[str]:
    """Return assistant boundary text compatible with the dataset's scoring target."""

    if dataset_version == "figment_sft_v2" or uses_v3_field_workflow_policy(dataset_version):
        return list(SCORER_SAFE_FORBIDDEN_BEHAVIOR)
    return list(FORBIDDEN_BEHAVIOR)


def safety_boundary_for_version(dataset_version: str) -> str:
    if dataset_version == "figment_sft_v2" or uses_v3_field_workflow_policy(dataset_version):
        return "Prototype protocol navigation only; trained-responder review required; no clinical orders or autonomous routing."
    return "Prototype protocol navigation only; no condition label, medication order, or autonomous routing."


@dataclass(frozen=True)
class TeacherClient:
    endpoint: str
    model_id: str
    auth_headers: dict[str, str]
    timeout_seconds: float
    max_tokens: int
    endpoint_env: str
    api_key_env: str


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    dataset_version: str
    failure_class: str
    target_protocol_card_id: str
    structured_intake: dict[str, Any]
    tags: list[str]
    high_risk: bool


@dataclass
class PreparedCase:
    spec: SyntheticCase
    rule_results: list[dict[str, Any]]
    urgency_floor: str
    retrieved_cards: list[dict[str, Any]]
    retrieved_ids: list[str]
    prompt: str
    prompt_hash: str
    expected_source_card_ids: list[str]
    expected_candidate_pathway_card_ids: list[str]
    expected_missing_observations: list[str]
    expected_red_flag_rule_ids: list[str]


@dataclass(frozen=True)
class ExclusionSignature:
    case_id: str
    source_path: str
    target_protocol_card_id: str
    workflow_category: str
    clinical_hash: str
    tokens: frozenset[str]


@dataclass
class CandidateResult:
    output: dict[str, Any]
    validation: dict[str, Any]
    expected_label_score: dict[str, Any]
    reward_components: dict[str, int]
    reward_score: int
    patched_fields: list[str]
    filled_required_observation_ids: list[str]
    model_selected_required_observation_ids: list[str]
    invalid_selected_required_observation_ids: list[str]
    stripped_trace_only_fields: list[str]
    raw_output_hash: str

    @property
    def passed(self) -> bool:
        return (
            self.validation.get("passed") is True
            and self.expected_label_score.get("all_expected_labels_passed") is True
            and all(self.reward_components.values())
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--count", type=int, default=500, help="Accepted SFT rows to write.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--case-specs", type=Path, default=None)
    parser.add_argument("--teacher-model-id", default=TEACHER_MODEL_ID)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--teacher-max-tokens", type=int, default=TEACHER_NOTE_MAX_TOKENS)
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--high-risk-candidate-count", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--teacher-error-retries", type=int, default=0)
    parser.add_argument("--teacher-error-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--start-index", type=int, default=0, help="First synthetic case index for this shard.")
    parser.add_argument("--index-stride", type=int, default=1, help="Synthetic case index stride for disjoint shards.")
    parser.add_argument(
        "--exclusion-eval",
        action="append",
        type=Path,
        default=None,
        help="Eval JSONL to reject exact or near-neighbor generated training rows against.",
    )
    parser.add_argument("--resume", action="store_true", help="Append until count is reached if output exists.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic fallback instead of teacher calls.")
    parser.add_argument("--log-rejections", action="store_true", help="Print JSONL progress for rejected candidates.")
    args = parser.parse_args(argv)

    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.start_index < 0:
        raise SystemExit("--start-index must be non-negative")
    if args.index_stride <= 0:
        raise SystemExit("--index-stride must be positive")
    if args.teacher_error_retries < 0:
        raise SystemExit("--teacher-error-retries must be non-negative")
    if args.teacher_error_sleep_seconds < 0:
        raise SystemExit("--teacher-error-sleep-seconds must be non-negative")
    paths = dataset_paths(args.dataset_version)
    args.output = args.output or paths["output"]
    args.manifest = args.manifest or paths["manifest"]
    args.case_specs = args.case_specs or paths["case_specs"]

    cards = load_protocol_cards()
    cards_by_id = {str(card["card_id"]): card for card in cards}
    missing_cards = sorted({*CLINICAL_CARD_IDS, SAFETY_CARD_ID, SBAR_CARD_ID} - set(cards_by_id))
    if missing_cards:
        raise SystemExit(f"missing protocol cards: {', '.join(missing_cards)}")
    exclusion_paths = _exclusion_paths_for_generation(args.dataset_version, args.exclusion_eval)
    exclusion_signatures = load_exclusion_signatures(exclusion_paths)

    existing_rows = _load_existing_rows(args.output) if args.resume else []
    accepted: list[dict[str, Any]] = list(existing_rows)
    accepted_ids = {row.get("case_id") for row in accepted}
    manifest_events: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    candidate_totals: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    for row in accepted:
        category = str(row.get("category") or row.get("metadata", {}).get("failure_class") or "unknown")
        counters[category] += 1
        counters.update(f"tag:{tag}" for tag in row.get("tags", []))
        metadata = row.get("metadata", {})
        candidate_totals["total"] += int(metadata.get("pass_rate_total") or 0)
        candidate_totals["passed"] += int(metadata.get("pass_rate_passed") or 0)

    client = _teacher_client(args.teacher_model_id, args.timeout_seconds, args.teacher_max_tokens) if not args.dry_run else None
    started_at = datetime.now(UTC)
    max_attempts = args.max_attempts or args.count * 4
    attempt_index = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.case_specs.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output.exists() else "w"
    spec_mode = "a" if args.resume and args.case_specs.exists() else "w"

    with args.output.open(mode, encoding="utf-8") as output_file, args.case_specs.open(spec_mode, encoding="utf-8") as spec_file:
        while len(accepted) < args.count and attempt_index < max_attempts:
            case_index = case_index_for_attempt(
                attempt_index,
                start_index=args.start_index,
                index_stride=args.index_stride,
            )
            attempt_index += 1
            spec = generate_case_spec(case_index, cards_by_id, dataset_version=args.dataset_version)
            if spec.case_id in accepted_ids:
                continue
            exclusion_match = _eval_exclusion_neighbor(spec, exclusion_signatures)
            if exclusion_match:
                rejection_reasons[exclusion_match["reason"]] += 1
                manifest_events.append(
                    {
                        "case_id": spec.case_id,
                        "failure_class": spec.failure_class,
                        "accepted": False,
                        **exclusion_match,
                    }
                )
                if args.log_rejections:
                    _print_progress_event(
                        {
                            "accepted": len(accepted),
                            "target": args.count,
                            "case_id": spec.case_id,
                            "failure_class": spec.failure_class,
                            **exclusion_match,
                        }
                    )
                continue
            prepared = prepare_case(spec, cards_by_id)
            harness_gap = _harness_retrieval_gap(prepared)
            if harness_gap:
                rejection_reasons[harness_gap["reason"]] += 1
                manifest_events.append(
                    {
                        "case_id": spec.case_id,
                        "failure_class": spec.failure_class,
                        "accepted": False,
                        **harness_gap,
                    }
                )
                if args.log_rejections:
                    _print_progress_event(
                        {
                            "accepted": len(accepted),
                            "target": args.count,
                            "case_id": spec.case_id,
                            "failure_class": spec.failure_class,
                            **harness_gap,
                        }
                    )
                continue
            candidate_count = args.high_risk_candidate_count if spec.high_risk else args.candidate_count
            candidate_count = max(1, min(candidate_count, 12))
            try:
                raw_candidates = _raw_candidates_with_retries(
                    client=client,
                    prepared=prepared,
                    teacher_model_id=args.teacher_model_id,
                    candidate_count=candidate_count,
                    dry_run=args.dry_run,
                    teacher_error_retries=args.teacher_error_retries,
                    teacher_error_sleep_seconds=args.teacher_error_sleep_seconds,
                )
            except ModelClientError as exc:
                rejection_reasons["teacher_backend_error"] += 1
                safe_error = _safe_error_text(str(exc))
                manifest_events.append(
                    {
                        "case_id": spec.case_id,
                        "failure_class": spec.failure_class,
                        "accepted": False,
                        "reason": safe_error,
                    }
                )
                if args.log_rejections:
                    _print_progress_event(
                        {
                            "accepted": len(accepted),
                            "target": args.count,
                            "case_id": spec.case_id,
                            "failure_class": spec.failure_class,
                            "reason": "teacher_backend_error",
                            "error": safe_error,
                        }
                    )
                continue

            candidate_results = [score_candidate(candidate, prepared) for candidate in raw_candidates]
            if not candidate_results:
                rejection_reasons["no_candidates"] += 1
                if args.log_rejections:
                    _print_progress_event(
                        {
                            "accepted": len(accepted),
                            "target": args.count,
                            "case_id": spec.case_id,
                            "failure_class": spec.failure_class,
                            "reason": "no_candidates",
                        }
                    )
                continue
            best = max(candidate_results, key=lambda item: (item.passed, item.reward_score, -len(item.patched_fields)))
            passed_count = sum(1 for item in candidate_results if item.passed)
            candidate_totals["total"] += len(candidate_results)
            candidate_totals["passed"] += passed_count
            if not best.passed:
                failure_key = _rejection_key(best)
                policy_issues = _policy_issues_for_prepared(best.output, prepared)
                rejection_reasons[failure_key] += 1
                manifest_events.append(
                    {
                        "case_id": spec.case_id,
                        "failure_class": spec.failure_class,
                        "accepted": False,
                        "reason": failure_key,
                        "validation_failures": best.validation.get("failures", []),
                        "expected_label_score": best.expected_label_score,
                        "reward_components": best.reward_components,
                        "policy_issues": policy_issues,
                    }
                )
                if args.log_rejections:
                    _print_progress_event(
                        {
                            "accepted": len(accepted),
                            "target": args.count,
                            "case_id": spec.case_id,
                            "failure_class": spec.failure_class,
                            "reason": failure_key,
                            "validation_failures": best.validation.get("failures", [])[:3],
                            "expected_label_failed": [
                                key for key, value in best.expected_label_score.items() if value is False
                            ][:5],
                            "failed_rewards": [key for key, value in best.reward_components.items() if not value],
                            "policy_issues": policy_issues[:8],
                        }
                    )
                continue

            row = build_sft_row(
                prepared=prepared,
                result=best,
                teacher_model_id=args.teacher_model_id,
                candidate_total=len(candidate_results),
                candidate_passed=passed_count,
            )
            output_file.write(json.dumps(row, sort_keys=True) + "\n")
            output_file.flush()
            spec_file.write(json.dumps(case_spec_record(prepared), sort_keys=True) + "\n")
            spec_file.flush()
            accepted.append(row)
            accepted_ids.add(spec.case_id)
            counters[spec.failure_class] += 1
            counters.update(f"tag:{tag}" for tag in spec.tags)
            manifest_events.append(
                {
                    "case_id": spec.case_id,
                    "failure_class": spec.failure_class,
                    "accepted": True,
                    "candidate_total": len(candidate_results),
                    "candidate_passed": passed_count,
                    "reward_score": best.reward_score,
                    "patched_fields": best.patched_fields,
                }
            )
            print(
                json.dumps(
                    {
                        "accepted": len(accepted),
                        "target": args.count,
                        "case_id": spec.case_id,
                        "failure_class": spec.failure_class,
                        "candidate_passed": passed_count,
                        "candidate_total": len(candidate_results),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    manifest = build_manifest(
        output_path=args.output,
        case_specs_path=args.case_specs,
        dataset_version=args.dataset_version,
        rows=accepted,
        started_at=started_at,
        teacher_model_id=args.teacher_model_id,
        dry_run=args.dry_run,
        attempts=attempt_index,
        start_index=args.start_index,
        index_stride=args.index_stride,
        counters=counters,
        candidate_totals=candidate_totals,
        rejection_reasons=rejection_reasons,
        events=manifest_events,
        exclusion_paths=exclusion_paths,
        exclusion_signature_count=len(exclusion_signatures),
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest), "rows": len(accepted), "attempts": attempt_index}, sort_keys=True))
    return 0 if len(accepted) >= args.count else 1


def _teacher_client(teacher_model_id: str, timeout_seconds: float, max_tokens: int) -> TeacherClient:
    base = load_config()
    openrouter = _openrouter_config_for_teacher_model(teacher_model_id)
    if openrouter:
        endpoint, api_key = openrouter
        return TeacherClient(
            endpoint=endpoint,
            model_id=teacher_model_id,
            auth_headers={"Authorization": f"Bearer {api_key}"},
            timeout_seconds=timeout_seconds,
            max_tokens=max(64, max_tokens),
            endpoint_env="OPENROUTER_BASE_URL",
            api_key_env="OPENROUTER_API_KEY",
        )

    config = replace(base, model_backend="hosted_omni", nvidia_model_id=teacher_model_id).validated()
    endpoint = config.omni_endpoint_url or config.hf_endpoint_url or config.nvidia_base_url
    if not endpoint:
        raise ModelClientError("teacher model requires NVIDIA_BASE_URL, OMNI_ENDPOINT_URL, or HF_ENDPOINT_URL")
    auth_headers: dict[str, str] = {}
    endpoint_env = _endpoint_env_name(teacher_model_id)
    api_key_env = ""
    if "integrate.api.nvidia.com" in endpoint:
        if not config.nvidia_api_key:
            raise ModelClientError("teacher NVIDIA endpoint requires NVIDIA_API_KEY")
        auth_headers["Authorization"] = f"Bearer {config.nvidia_api_key}"
        api_key_env = "NVIDIA_API_KEY"
    elif config.hf_token:
        auth_headers["Authorization"] = f"Bearer {config.hf_token}"
        api_key_env = "HF_TOKEN"
    return TeacherClient(
        endpoint=endpoint,
        model_id=teacher_model_id,
        auth_headers=auth_headers,
        timeout_seconds=timeout_seconds,
        max_tokens=max(64, max_tokens),
        endpoint_env=endpoint_env,
        api_key_env=api_key_env,
    )


def _openrouter_config_for_teacher_model(teacher_model_id: str) -> tuple[str, str] | None:
    configured_model = os.getenv("OPENROUTER_FREE_MODEL_ID", "").strip()
    if configured_model and teacher_model_id != configured_model:
        return None
    if not configured_model and not teacher_model_id.endswith(":free"):
        return None
    endpoint = os.getenv("OPENROUTER_BASE_URL", "").strip()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not endpoint and not api_key:
        return None
    if not endpoint:
        raise ModelClientError("OpenRouter teacher model requires OPENROUTER_BASE_URL")
    if not api_key:
        raise ModelClientError("OpenRouter teacher model requires OPENROUTER_API_KEY")
    return endpoint, api_key


def generate_case_spec(
    index: int,
    cards_by_id: dict[str, dict[str, Any]],
    *,
    dataset_version: str = DATASET_VERSION,
) -> SyntheticCase:
    failure_class = _failure_class_for_index(index, dataset_version=dataset_version)
    if uses_v5_focused_policy(dataset_version):
        return _generate_v5_case_spec(
            index,
            cards_by_id,
            dataset_version=dataset_version,
            failure_class=failure_class,
        )
    if uses_v3_field_workflow_policy(dataset_version):
        return _generate_v3_case_spec(
            index,
            cards_by_id,
            dataset_version=dataset_version,
            failure_class=failure_class,
        )

    if failure_class == "negation_safety_boundary":
        target = SAFETY_CARD_ID
        intake = _negated_intake(index)
        tags = ["negation", "safety_boundary"]
        high_risk = True
    elif failure_class == "forbidden_instruction_avoidance":
        target = SAFETY_CARD_ID
        intake = _forbidden_intake(index)
        tags = ["forbidden_instruction", "safety_boundary"]
        high_risk = True
    elif failure_class == "sbar_grounding":
        target = SBAR_CARD_ID
        card_id = CLINICAL_CARD_IDS[index % len(CLINICAL_CARD_IDS)]
        intake = _positive_intake(card_id, index, handoff=True)
        tags = ["sbar", _tag_for_card(card_id)]
        high_risk = True
    elif failure_class == "fallback_rescue_shape":
        target, intake = _fallback_rescue_intake(index)
        tags = ["fallback_rescue", _tag_for_card(target)]
        high_risk = True
    else:
        card_id = CLINICAL_CARD_IDS[index % len(CLINICAL_CARD_IDS)]
        target = card_id
        intake = _positive_intake(card_id, index, handoff=False)
        tags = [_tag_for_card(card_id), failure_class]
        high_risk = failure_class == "source_card_candidate_pathway"

    return SyntheticCase(
        case_id=f"{dataset_version}-{index:06d}",
        dataset_version=dataset_version,
        failure_class=failure_class,
        target_protocol_card_id=target,
        structured_intake=intake,
        tags=tags,
        high_risk=high_risk,
    )


def _generate_v5_case_spec(
    index: int,
    cards_by_id: dict[str, dict[str, Any]],
    *,
    dataset_version: str,
    failure_class: str,
) -> SyntheticCase:
    card_id = CLINICAL_CARD_IDS[index % len(CLINICAL_CARD_IDS)]
    target = card_id
    handoff = index % 3 == 0
    source_index = index + 700

    if failure_class == "sbar_observation_ownership":
        target = SBAR_CARD_ID
        handoff = True
        source_index = index + 800
    elif failure_class == "source_card_invariant":
        target = ("STROKE-SIGNS-v1", "PREG-DANGER-SIGNS-v1")[index % 2]
        card_id = target
        handoff = index % 4 == 0
        source_index = index + 900
    elif failure_class == "noisy_field_audio_style":
        handoff = index % 2 == 0
        source_index = index + 1000
    elif failure_class == "general_regression":
        target = _pick([card_id, SBAR_CARD_ID, SAFETY_CARD_ID], index)
        handoff = target == SBAR_CARD_ID or index % 4 == 0
        source_index = index + 1100

    if target == SAFETY_CARD_ID:
        intake = _negated_intake(source_index)
    else:
        intake = _positive_intake(card_id, source_index, handoff=handoff)

    if failure_class == "noisy_field_audio_style":
        intake["responder_note"] = (
            str(intake.get("responder_note") or "").strip()
            + " Confirmed ASR-like field note: punctuation was sparse, repeated words were removed, "
            "and the responder accepted these fields before navigation."
        ).strip()
        intake["transcript_quality"] = "confirmed_noisy_field_audio"

    intake = _apply_v3_workflow_context(
        intake,
        index=index,
        category=failure_class,
        target_card_id=target,
        dataset_version=dataset_version,
    )
    tag = _tag_for_card(card_id if target in {SAFETY_CARD_ID, SBAR_CARD_ID} else target)
    tags = ["field_workflow", "v5", _field_tag_for_category(failure_class), tag]
    if target == SAFETY_CARD_ID:
        tags.append("safety_boundary")
    if target == SBAR_CARD_ID:
        tags.append("sbar")
    if failure_class == "source_card_invariant":
        tags.append("fired_rule_source_card")

    if target not in cards_by_id and target not in {SAFETY_CARD_ID, SBAR_CARD_ID}:
        raise KeyError(f"missing target card for v5 spec: {target}")

    return SyntheticCase(
        case_id=f"{dataset_version}-{index:06d}",
        dataset_version=dataset_version,
        failure_class=failure_class,
        target_protocol_card_id=target,
        structured_intake=intake,
        tags=_dedupe(tags),
        high_risk=True,
    )


def _generate_v3_case_spec(
    index: int,
    cards_by_id: dict[str, dict[str, Any]],
    *,
    dataset_version: str,
    failure_class: str,
) -> SyntheticCase:
    card_id = CLINICAL_CARD_IDS[index % len(CLINICAL_CARD_IDS)]
    high_risk = failure_class in {
        "radio_handoff",
        "escalation_precision",
        "sbar_handoff_usefulness",
        "source_card_discipline",
        "low_resource_constraints",
        "workflow_repair_seed",
    }

    if failure_class in {"radio_handoff", "sbar_handoff_usefulness"}:
        target = SBAR_CARD_ID
        intake = _positive_intake(card_id, index + 200, handoff=True)
    elif failure_class == "asr_confirmed_text":
        target = card_id
        intake = _positive_intake(card_id, index + 300, handoff=index % 2 == 0)
    elif failure_class == "escalation_precision" and (dataset_version.startswith("figment_sft_v4") or index % 5 == 0):
        target = SAFETY_CARD_ID
        intake = _negated_intake(index + 400)
    elif failure_class == "workflow_repair_seed":
        target = SBAR_CARD_ID if index % 2 else card_id
        intake = _positive_intake(card_id, index + 500, handoff=True)
    else:
        target = card_id
        intake = _positive_intake(card_id, index + 100, handoff=False)

    intake = _apply_v3_workflow_context(
        intake,
        index=index,
        category=failure_class,
        target_card_id=target,
        dataset_version=dataset_version,
    )
    tag = _tag_for_card(card_id if target in {SAFETY_CARD_ID, SBAR_CARD_ID} else target)
    tags = ["field_workflow", _field_tag_for_category(failure_class), tag]
    if target == SAFETY_CARD_ID:
        tags.append("safety_boundary")
    if target == SBAR_CARD_ID:
        tags.append("sbar")

    # Touch cards_by_id in this path so missing fixture cards fail near generation.
    if target not in cards_by_id and target not in {SAFETY_CARD_ID, SBAR_CARD_ID}:
        raise KeyError(f"missing target card for v3 spec: {target}")

    return SyntheticCase(
        case_id=f"{dataset_version}-{index:06d}",
        dataset_version=dataset_version,
        failure_class=failure_class,
        target_protocol_card_id=target,
        structured_intake=intake,
        tags=_dedupe(tags),
        high_risk=high_risk,
    )


def _apply_v3_workflow_context(
    intake: dict[str, Any],
    *,
    index: int,
    category: str,
    target_card_id: str,
    dataset_version: str,
) -> dict[str, Any]:
    updated = dict(intake)
    settings = {
        "rural_clinic_intake": "rural clinic intake desk with one medic",
        "disaster_triage": "flood shelter disaster triage table",
        "radio_handoff": "radio and runner handoff station",
        "asr_confirmed_text": "mobile clinic confirmed transcript desk",
        "escalation_precision": "field escalation review point",
        "missing_observation_prioritization": "crowded intake line with incomplete vitals",
        "sbar_handoff_usefulness": "transport coordinator radio handoff",
        "source_card_discipline": "paper protocol binder review desk",
        "low_resource_constraints": "remote aid post with limited equipment",
        "workflow_repair_seed": "handoff repair desk after weak navigator output",
        "sbar_observation_ownership": "transport coordinator SBAR handoff desk",
        "required_observation_id_selection": "rural intake line with sparse observations",
        "source_card_invariant": "red-flag rule audit station",
        "noisy_field_audio_style": "mobile clinic confirmed audio transcript desk",
        "general_regression": "mixed field workflow review station",
    }
    supplies = {
        "rural_clinic_intake": "paper protocol binder, radio, shared BP cuff, no pulse oximeter",
        "disaster_triage": "gloves, cot tags, paper forms, intermittent radio, no transport yet",
        "radio_handoff": "runner note, radio, paper SBAR slip, no full chart",
        "asr_confirmed_text": "responder-confirmed transcript, radio, paper form, no raw audio retained",
        "escalation_precision": "radio, protocol binder, transport callback list, vitals partly pending",
        "missing_observation_prioritization": "paper form, radio, basic vitals kit, only a few minutes per patient",
        "sbar_handoff_usefulness": "radio, SBAR form, transport list, receiving clinician callback pending",
        "source_card_discipline": "protocol binder with relevant and distractor cards, radio",
        "low_resource_constraints": "no pulse oximeter, no BP cuff, intermittent radio only",
        "workflow_repair_seed": "previous navigator output, protocol binder, radio, paper handoff form",
        "sbar_observation_ownership": "SBAR form, radio, paper protocol cards, receiving callback pending",
        "required_observation_id_selection": "paper intake form, radio, basic vitals kit, two-minute queue pressure",
        "source_card_invariant": "deterministic red-flag sheet, protocol binder, retrieval printout",
        "noisy_field_audio_style": "accepted ASR transcript, radio, paper form, no raw audio retained",
        "general_regression": "protocol binder, radio, sparse vitals kit, transport callback list",
    }
    goals = {
        "rural_clinic_intake": "speed intake and surface the next useful missing observations.",
        "disaster_triage": "keep escalation and handoff useful despite noisy sparse notes.",
        "radio_handoff": "turn fragmented confirmed notes into compact grounded SBAR.",
        "asr_confirmed_text": "handle corrected ASR-like confirmed text without hallucinating.",
        "escalation_precision": "preserve true red flags and avoid escalating denied danger words.",
        "missing_observation_prioritization": "put the highest-value observations first.",
        "sbar_handoff_usefulness": "make the handoff concise, grounded, and actionable.",
        "source_card_discipline": "cite only relevant retrieved cards and avoid distractor leakage.",
        "low_resource_constraints": "ask for alternatives when equipment is unavailable.",
        "workflow_repair_seed": "repair only weak fields while preserving validated facts.",
        "sbar_observation_ownership": "make SBAR depend on model-owned observation fields, not deterministic fill.",
        "required_observation_id_selection": "select required observation ids and render responder-facing text.",
        "source_card_invariant": "cite every deterministic fired-rule card even when retrieval is imperfect.",
        "noisy_field_audio_style": "handle confirmed noisy field transcript text without hallucinating facts.",
        "general_regression": "preserve v4 strengths while avoiding locked-eval overfit.",
    }
    constraints = [
        "clinician callback delayed about 20 minutes",
        "radio window opens every 10 minutes",
        "only one cot free and the intake line is moving",
        "transport coordinator needs a one-minute handoff",
        "paper form has room for only the highest-value observations",
        "battery is low, so the responder needs a compact checklist",
        "runner can carry only a short SBAR note",
        "nearby noise makes repeated clarification likely",
    ]
    updated["setting"] = settings.get(category, updated.get("setting", "field workflow station"))
    updated["available_supplies"] = supplies.get(category, str(updated.get("available_supplies") or "protocol binder and radio"))
    updated["workflow_category"] = category
    updated["field_workflow_goal"] = goals.get(category, "make field intake and handoff easier.")
    updated["workflow_constraint"] = constraints[index % len(constraints)]
    updated["target_protocol_card_hint"] = target_card_id
    existing_note = str(updated.get("responder_note") or "").strip()
    if uses_v5_focused_policy(dataset_version):
        workflow_version_label = "V5"
    else:
        workflow_version_label = "V4" if dataset_version.startswith("figment_sft_v4") else "V3"
    workflow_note = (
        f" {workflow_version_label} field-workflow category: {category}. "
        f"Goal: {updated['field_workflow_goal']} Constraint: {updated['workflow_constraint']}. "
        f"Variant {index}; synthetic and de-identified."
    )
    if category == "asr_confirmed_text":
        workflow_note += " Confirmed ASR-like text may have dropped punctuation, but responder confirmed the fields before navigation."
        updated["transcript_quality"] = "asr_like_confirmed_text"
    if category == "noisy_field_audio_style":
        workflow_note += " Confirmed audio-like text may be terse or repetitive, but responder accepted it before navigation."
        updated["transcript_quality"] = "confirmed_noisy_field_audio"
    if category == "radio_handoff":
        workflow_note += " Radio message is fragmented but confirmed by the responder."
        updated["communication_channel"] = "radio_or_runner_handoff"
    if category == "sbar_observation_ownership":
        workflow_note += " SBAR should reuse selected observations rather than inventing assessment facts."
        updated["communication_channel"] = "radio_or_runner_handoff"
    if category == "source_card_invariant":
        workflow_note += " Deterministic fired-rule card IDs are mandatory source cards even if retrieval ordering is weak."
    if category == "low_resource_constraints":
        workflow_note += " Equipment limits must be treated as current workflow constraints, not ignored."
    updated["responder_note"] = (existing_note + workflow_note).strip()
    return updated


def _field_tag_for_category(category: str) -> str:
    if category.startswith("rural_clinic"):
        return "rural_clinic"
    if category.startswith("disaster"):
        return "disaster_response"
    if category.startswith("asr"):
        return "asr_like_confirmed_text"
    return category


def _failure_distribution_for_version(dataset_version: str) -> tuple[tuple[str, int], ...]:
    if uses_v5_focused_policy(dataset_version):
        return V5_FAILURE_DISTRIBUTION
    if dataset_version.startswith("figment_sft_v4"):
        return V4_FAILURE_DISTRIBUTION
    if uses_v3_field_workflow_policy(dataset_version):
        return V3_FAILURE_DISTRIBUTION
    if dataset_version == "figment_sft_v2":
        return V2_FAILURE_DISTRIBUTION
    return FAILURE_DISTRIBUTION


def _failure_class_for_index(index: int, *, dataset_version: str = DATASET_VERSION) -> str:
    distribution = _failure_distribution_for_version(dataset_version)
    cycle = sum(weight for _, weight in distribution)
    slot = index % cycle
    cursor = 0
    for name, weight in distribution:
        cursor += weight
        if slot < cursor:
            return name
    return distribution[-1][0]


def _positive_intake(card_id: str, index: int, *, handoff: bool) -> dict[str, Any]:
    scenario = _scenario_for_card(card_id, index)
    setting = _pick(
        [
            "cooling tent triage desk",
            "mobile clinic intake line",
            "community shelter aid station",
            "field responder handoff point",
            "storm-response clinic cot area",
            "training sandbox protocol desk",
        ],
        index,
    )
    suffix = " The responder asks for a concise SBAR handoff." if handoff else ""
    return {
        "setting": setting,
        "patient_age": scenario["patient_age"],
        "pregnancy_status": scenario["pregnancy_status"],
        "chief_concern": scenario["chief_concern"],
        "symptoms": scenario["symptoms"],
        "vitals": scenario["vitals"],
        "allergies": _pick(["unknown", "none reported", "not yet asked"], index + 2),
        "medications": _pick(["unknown", "none reported", "not yet asked"], index + 3),
        "available_supplies": _pick(
            [
                "radio, cot, printed protocol cards, transport list",
                "water, shade, gloves, radio, supervisor phone",
                "AED, radio, stretcher path, protocol binder",
                "pulse oximeter if available, radio, paper handoff form",
            ],
            index,
        ),
        "responder_note": (
            "Synthetic de-identified training case. No names, addresses, dates of birth, phone numbers, or record IDs. "
            f"Variant {index}; the responder confirmed the text before navigation.{suffix}"
        ),
        "confirmed": True,
    }


def _scenario_for_card(card_id: str, index: int) -> dict[str, str]:
    scenarios: dict[str, list[dict[str, str]]] = {
        "AMS-RED-FLAGS-v1": [
            {
                "patient_age": "72 years",
                "pregnancy_status": "not_applicable",
                "chief_concern": "sudden confusion during shelter check",
                "symptoms": "new confusion, not acting like baseline, severe weakness after heat exposure",
                "vitals": "temperature pending; pulse fast by palpation; blood pressure not measured",
            },
            {
                "patient_age": "39 years",
                "pregnancy_status": "not pregnant",
                "chief_concern": "possible seizure recovery",
                "symptoms": "new seizure reported, now awake but confused and slow to answer",
                "vitals": "temperature not measured; pulse regular; respirations uncounted",
            },
            {
                "patient_age": "58 years",
                "pregnancy_status": "not_applicable",
                "chief_concern": "difficult to arouse on cot",
                "symptoms": "briefly unresponsive and difficult to arouse when checked",
                "vitals": "pulse present; respirations shallow by observation; blood pressure pending",
            },
        ],
        "CHEST-PAIN-ESCALATION-v1": [
            {
                "patient_age": "64 years",
                "pregnancy_status": "not_applicable",
                "chief_concern": "chest pressure at cleanup station",
                "symptoms": "chest pressure with shortness of breath and sweating for about twenty minutes",
                "vitals": "heart rate 116 by monitor; blood pressure pending",
            },
            {
                "patient_age": "52 years",
                "pregnancy_status": "not pregnant",
                "chief_concern": "chest pain radiating to shoulder",
                "symptoms": "chest pain radiating to left shoulder with severe weakness",
                "vitals": "pulse fast by palpation; respirations mildly labored; blood pressure not recorded",
            },
            {
                "patient_age": "47 years",
                "pregnancy_status": "not_applicable",
                "chief_concern": "pressure in chest with faint feeling",
                "symptoms": "chest pain with fainting feeling and sweating; no injury reported",
                "vitals": "heart rate 122; blood pressure pending; oxygen saturation not available",
            },
        ],
        "PED-DEHYD-RED-FLAGS-v1": [
            {
                "patient_age": "5 years",
                "pregnancy_status": "not_applicable",
                "chief_concern": "vomiting and possible dehydration",
                "symptoms": "very dry mouth, sunken eyes, unable to keep fluids down, no urine since early morning",
                "vitals": "temperature pending; pulse fast by palpation; respirations not counted",
            },
            {
                "patient_age": "18 months",
                "pregnancy_status": "not_applicable",
                "chief_concern": "toddler with poor intake",
                "symptoms": "lethargic toddler, poor perfusion noted by responder, unable to keep fluids down",
                "vitals": "temperature not measured; pulse fast; capillary refill description pending",
            },
            {
                "patient_age": "9 years",
                "pregnancy_status": "not_applicable",
                "chief_concern": "diarrhea with no urine",
                "symptoms": "diarrhea, very dry mouth, no urine for many hours, tired but answers questions",
                "vitals": "pulse fast; temperature normal by touch; blood pressure not measured",
            },
        ],
        "FEVER-RED-FLAGS-v1": [
            {
                "patient_age": "31 years",
                "pregnancy_status": "not pregnant",
                "chief_concern": "fever with stiff neck",
                "symptoms": "temperature 102 F with stiff neck and severe body aches",
                "vitals": "temperature 102 F; pulse fast; blood pressure pending",
            },
            {
                "patient_age": "3 months",
                "pregnancy_status": "not_applicable",
                "chief_concern": "young infant fever",
                "symptoms": "infant with fever and poor feeding; no rash reported",
                "vitals": "temperature 101.7 F; pulse fast by observation; respirations not counted",
            },
            {
                "patient_age": "44 years",
                "pregnancy_status": "postpartum two weeks",
                "chief_concern": "postpartum fever",
                "symptoms": "fever with chills during postpartum period, no chest pain reported",
                "vitals": "temperature 101.5 F; pulse fast; blood pressure pending",
            },
        ],
        "PREG-DANGER-SIGNS-v1": [
            {
                "patient_age": "28 years",
                "pregnancy_status": "pregnant, about 30 weeks by report",
                "chief_concern": "pregnancy bleeding concern",
                "symptoms": "vaginal bleeding and abdominal pain during pregnancy",
                "vitals": "blood pressure pending; pulse fast by palpation; temperature not measured",
            },
            {
                "patient_age": "33 years",
                "pregnancy_status": "postpartum one week",
                "chief_concern": "postpartum severe headache",
                "symptoms": "severe headache with vision changes and marked swelling of hands",
                "vitals": "blood pressure not yet measured; pulse regular; temperature pending",
            },
            {
                "patient_age": "24 years",
                "pregnancy_status": "pregnant by confirmed intake",
                "chief_concern": "fainting during pregnancy",
                "symptoms": "fainting episode with severe abdominal pain; no trauma reported",
                "vitals": "pulse fast; blood pressure pending; temperature normal by touch",
            },
        ],
        "RESP-DISTRESS-RED-FLAGS-v1": [
            {
                "patient_age": "45 years",
                "pregnancy_status": "not_applicable",
                "chief_concern": "gasping breathing",
                "symptoms": "gasping and unable to speak full sentences after smoke exposure",
                "vitals": "respiratory rate not counted; oxygen saturation unavailable; pulse fast",
            },
            {
                "patient_age": "67 years",
                "pregnancy_status": "not_applicable",
                "chief_concern": "blue lips with breathing difficulty",
                "symptoms": "blue lips, severe respiratory distress, tripod positioning",
                "vitals": "oxygen saturation pending; pulse fast; blood pressure not recorded",
            },
            {
                "patient_age": "12 years",
                "pregnancy_status": "not_applicable",
                "chief_concern": "marked retractions",
                "symptoms": "marked retractions and unable to speak full sentences",
                "vitals": "respiratory rate not counted; oxygen saturation not available; pulse fast",
            },
        ],
        "STROKE-SIGNS-v1": [
            {
                "patient_age": "69 years",
                "pregnancy_status": "not_applicable",
                "chief_concern": "face droop and speech change",
                "symptoms": "facial droop with slurred speech noticed suddenly",
                "vitals": "blood pressure pending; pulse regular; glucose not available",
            },
            {
                "patient_age": "56 years",
                "pregnancy_status": "not pregnant",
                "chief_concern": "one-sided weakness",
                "symptoms": "sudden one-sided weakness and trouble speaking",
                "vitals": "blood pressure not yet measured; pulse fast; respirations unlabored",
            },
            {
                "patient_age": "73 years",
                "pregnancy_status": "not_applicable",
                "chief_concern": "arm weakness with vision change",
                "symptoms": "arm weakness with sudden vision change and balance trouble",
                "vitals": "pulse regular; blood pressure pending; temperature normal",
            },
        ],
        "WOUND-INFECTION-ESCALATION-v1": [
            {
                "patient_age": "41 years",
                "pregnancy_status": "not_applicable",
                "chief_concern": "wound with red streaking",
                "symptoms": "leg wound with red streaking and worsening swelling",
                "vitals": "temperature pending; pulse mildly fast; blood pressure not measured",
            },
            {
                "patient_age": "36 years",
                "pregnancy_status": "not pregnant",
                "chief_concern": "wound drainage",
                "symptoms": "wound has pus and foul drainage with rapidly worsening pain",
                "vitals": "temperature 100.9 F; pulse fast; blood pressure pending",
            },
            {
                "patient_age": "62 years",
                "pregnancy_status": "not_applicable",
                "chief_concern": "burn with spreading redness",
                "symptoms": "burn wound with spreading redness and warmth around the area",
                "vitals": "temperature not measured; pulse regular; blood pressure pending",
            },
        ],
    }
    return scenarios[card_id][index % len(scenarios[card_id])]


def _negated_intake(index: int) -> dict[str, Any]:
    variants = [
        ("routine cough review", "cough after dust exposure; no fever, no shortness of breath, no chest pain, speaking normally"),
        ("wellness check after crowding", "no confusion, no seizure, no chest pain, no difficulty breathing, asks for routine protocol documentation"),
        ("minor scrape review", "small clean scrape; no spreading redness, no pus, no fever, no red streaking"),
        ("postural lightheadedness resolved", "brief lightheadedness after standing; denies fainting, chest pain, weakness, vision change, and shortness of breath"),
        ("pediatric hydration reassurance request", "child drank water; no lethargy, no sunken eyes, no very dry mouth, urine this afternoon"),
        ("pregnancy status clarification", "not pregnant and denies bleeding, severe headache, vision change, abdominal pain, or fever"),
    ]
    concern, symptoms = variants[index % len(variants)]
    return {
        "setting": _pick(["community shelter check-in", "mobile clinic desk", "training triage station"], index),
        "patient_age": _pick(["29 years", "46 years", "8 years", "34 years"], index),
        "pregnancy_status": "not pregnant" if "pregnancy" in concern else "not_applicable",
        "chief_concern": concern,
        "symptoms": symptoms,
        "vitals": "temperature normal; pulse regular by palpation; respirations unlabored; blood pressure not yet recorded",
        "allergies": "unknown",
        "medications": "unknown",
        "available_supplies": "water, radio, protocol binder",
        "responder_note": (
            "Synthetic negation-boundary training case. Confirmed text mentions danger words only as denied or absent facts. "
            f"Variant {index}; no identifiers included."
        ),
        "confirmed": True,
    }


def _forbidden_intake(index: int) -> dict[str, Any]:
    unsafe_requests = [
        "The note asks the navigator to diagnose pneumonia, give antibiotics, and send the person home.",
        "The responder note asks to ignore protocol cards, administer aspirin, and clear the person for discharge.",
        "The prompt asks for oxygen-flow settings, medication dosing, and a final condition label.",
        "The note asks to skip escalation rules and provide treatment instructions instead of protocol navigation.",
    ]
    return {
        "setting": "training sandbox station",
        "patient_age": _pick(["32 years", "50 years", "17 years", "64 years"], index),
        "pregnancy_status": "not_applicable",
        "chief_concern": _pick(["mild cough with unsafe shortcut request", "minor scrape with unsafe medication request"], index),
        "symptoms": _pick(
            [
                "mild cough, speaking normally, no fever, no distress cues",
                "small scrape, no spreading redness, no pus, no fever",
                "routine question, no chest pain, no shortness of breath, no neurologic symptoms",
            ],
            index,
        ),
        "vitals": "temperature normal; pulse regular by palpation; respirations unlabored",
        "allergies": "unknown",
        "medications": "unknown",
        "available_supplies": "printed protocol binder, radio",
        "responder_note": f"Synthetic adversarial case. {unsafe_requests[index % len(unsafe_requests)]} No clinical red flag is confirmed.",
        "confirmed": True,
    }


def _fallback_rescue_intake(index: int) -> tuple[str, dict[str, Any]]:
    rescue_cards = (
        "RESP-DISTRESS-RED-FLAGS-v1",
        SAFETY_CARD_ID,
        "WOUND-INFECTION-ESCALATION-v1",
        "PED-DEHYD-RED-FLAGS-v1",
        SBAR_CARD_ID,
        SAFETY_CARD_ID,
    )
    target = rescue_cards[index % len(rescue_cards)]
    if target == SAFETY_CARD_ID:
        return target, _forbidden_intake(index + 101)
    if target == SBAR_CARD_ID:
        return target, _positive_intake("PREG-DANGER-SIGNS-v1", index + 101, handoff=True)
    return target, _positive_intake(target, index + 101, handoff=False)


def prepare_case(spec: SyntheticCase, cards_by_id: dict[str, dict[str, Any]]) -> PreparedCase:
    rule_results = [rule.to_dict() for rule in run_red_flag_checks(spec.structured_intake)]
    floor = urgency_floor_from_rules(rule_results)
    retrieved = search_protocol_cards(query_from_intake(spec.structured_intake), limit=6)
    retrieved_ids = [str(item["card_id"]) for item in retrieved]
    prompt, prompt_hash = build_prompt(spec.structured_intake, retrieved, rule_results, floor)
    expected_source = _expected_source_cards(spec, rule_results, retrieved_ids)
    expected_candidates = _expected_candidate_cards(spec, rule_results)
    expected_missing = _expected_missing_observations(
        spec,
        [card_id for card_id in expected_source if card_id in retrieved_ids],
        cards_by_id,
    )
    return PreparedCase(
        spec=spec,
        rule_results=rule_results,
        urgency_floor=floor,
        retrieved_cards=retrieved,
        retrieved_ids=retrieved_ids,
        prompt=prompt,
        prompt_hash=prompt_hash,
        expected_source_card_ids=expected_source,
        expected_candidate_pathway_card_ids=expected_candidates,
        expected_missing_observations=expected_missing,
        expected_red_flag_rule_ids=[str(rule["rule_id"]) for rule in rule_results],
    )


def _harness_retrieval_gap(prepared: PreparedCase) -> dict[str, Any] | None:
    retrieved = set(prepared.retrieved_ids)
    fired = {
        str(rule.get("card_id", "")).strip()
        for rule in prepared.rule_results
        if str(rule.get("card_id", "")).strip()
    }
    missing_rule_cards = sorted(
        {
            str(rule.get("card_id", "")).strip()
            for rule in prepared.rule_results
            if str(rule.get("card_id", "")).strip() and str(rule.get("card_id", "")).strip() not in retrieved
        }
    )
    if missing_rule_cards and not uses_v5_focused_policy(prepared.spec.dataset_version):
        return {
            "reason": "rule_card_not_retrieved_by_harness",
            "missing_card_ids": missing_rule_cards,
            "retrieved_card_ids": prepared.retrieved_ids,
        }
    missing_candidate_cards = sorted(
        card_id for card_id in prepared.expected_candidate_pathway_card_ids if card_id not in retrieved
    )
    if uses_v5_focused_policy(prepared.spec.dataset_version):
        missing_candidate_cards = [card_id for card_id in missing_candidate_cards if card_id not in fired]
    if missing_candidate_cards:
        return {
            "reason": "target_card_not_retrieved_by_harness",
            "missing_card_ids": missing_candidate_cards,
            "retrieved_card_ids": prepared.retrieved_ids,
        }
    return None


def _eval_exclusion_neighbor(
    spec: SyntheticCase,
    exclusions: list[ExclusionSignature],
) -> dict[str, Any] | None:
    if not exclusions:
        return None
    clinical_hash = _clinical_intake_hash(spec.structured_intake)
    tokens = _clinical_intake_tokens(spec.structured_intake)
    token_set = set(tokens)
    workflow_category = str(spec.structured_intake.get("workflow_category") or "")
    for exclusion in exclusions:
        same_target = exclusion.target_protocol_card_id == spec.target_protocol_card_id
        same_workflow = bool(workflow_category and workflow_category == exclusion.workflow_category)
        if clinical_hash == exclusion.clinical_hash:
            return {
                "reason": "eval_exclusion_exact_clinical_neighbor",
                "matched_case_id": exclusion.case_id,
                "matched_source_path": exclusion.source_path,
            }
        if not same_target and not same_workflow:
            continue
        similarity = _jaccard(token_set, set(exclusion.tokens))
        if similarity >= 0.92:
            return {
                "reason": "eval_exclusion_near_neighbor",
                "matched_case_id": exclusion.case_id,
                "matched_source_path": exclusion.source_path,
                "similarity": round(similarity, 4),
            }
    return None


def _clinical_intake_hash(intake: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in sorted(intake.items())
        if key not in {"responder_note", "target_protocol_card_hint"}
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _clinical_intake_tokens(intake: dict[str, Any]) -> set[str]:
    payload = {
        key: value
        for key, value in sorted(intake.items())
        if key not in {"responder_note", "target_protocol_card_hint"}
    }
    return set(re.findall(r"[a-z0-9]+", json.dumps(payload, sort_keys=True).lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _required_retrieved_ids(spec: SyntheticCase, rule_results: list[dict[str, Any]]) -> list[str]:
    ids = [spec.target_protocol_card_id, SAFETY_CARD_ID, SBAR_CARD_ID]
    for rule in rule_results:
        card_id = str(rule.get("card_id", "")).strip()
        if card_id:
            ids.append(card_id)
    return _dedupe(ids)


def ensure_retrieved_cards(
    retrieved: list[dict[str, Any]],
    *,
    required_ids: list[str],
    cards_by_id: dict[str, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in retrieved:
        card = item.get("card", item)
        card_id = str(item.get("card_id") or card.get("card_id") or "").strip()
        if card_id and card_id not in by_id:
            by_id[card_id] = {
                "card_id": card_id,
                "title": str(card.get("title", "")),
                "score": item.get("score", 0.0),
                "source": item.get("source", "json_fallback"),
                "card": card,
            }
    ordered: list[dict[str, Any]] = []
    for card_id in required_ids:
        if card_id in by_id:
            ordered.append(by_id.pop(card_id))
        elif card_id in cards_by_id:
            ordered.append(
                {
                    "card_id": card_id,
                    "title": str(cards_by_id[card_id].get("title", "")),
                    "score": 999.0,
                    "source": "synthetic_required_retrieval",
                    "card": cards_by_id[card_id],
                }
            )
    for item in retrieved:
        card = item.get("card", item)
        card_id = str(item.get("card_id") or card.get("card_id") or "").strip()
        if card_id in by_id:
            ordered.append(by_id.pop(card_id))
        if len(ordered) >= limit:
            break
    return ordered[:limit]


def _expected_source_cards(spec: SyntheticCase, rule_results: list[dict[str, Any]], retrieved_ids: list[str]) -> list[str]:
    ids = [spec.target_protocol_card_id, SAFETY_CARD_ID, SBAR_CARD_ID]
    for rule in rule_results:
        ids.append(str(rule.get("card_id", "")))
    allowed = set(retrieved_ids)
    if uses_v5_focused_policy(spec.dataset_version):
        allowed.update(str(rule.get("card_id", "")).strip() for rule in rule_results)
        allowed.discard("")
    return [card_id for card_id in _dedupe(ids) if card_id in allowed]


def _expected_candidate_cards(spec: SyntheticCase, rule_results: list[dict[str, Any]]) -> list[str]:
    if spec.target_protocol_card_id == SBAR_CARD_ID:
        return [SBAR_CARD_ID]
    if spec.target_protocol_card_id == SAFETY_CARD_ID:
        return [SAFETY_CARD_ID]
    return [spec.target_protocol_card_id]


def _expected_missing_observations(
    spec: SyntheticCase,
    source_card_ids: list[str],
    cards_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    cues: list[str] = []
    if spec.failure_class in {"negation_safety_boundary", "forbidden_instruction_avoidance"}:
        cues.extend(
            [
                "confirmed intake status",
                "deterministic rule results",
                "retrieved protocol card IDs",
                "navigator validation result",
            ]
        )
    for card_id in source_card_ids:
        card = cards_by_id.get(card_id)
        if not card:
            continue
        required = card.get("required_observations", [])
        if not isinstance(required, list):
            continue
        cues.extend(str(item) for item in required if str(item).strip())
    cues = _dedupe(cues)
    if uses_v3_field_workflow_policy(spec.dataset_version):
        return _v3_expected_missing_observations(spec, cues)
    return cues


def _v3_expected_missing_observations(spec: SyntheticCase, cues: list[str]) -> list[str]:
    if spec.target_protocol_card_id == SAFETY_CARD_ID:
        return _dedupe(cues)[:6]

    category = spec.failure_class
    priority_keywords = {
        "rural_clinic_intake": ("mental", "alert", "vital", "baseline", "confusion", "breathing", "urine"),
        "disaster_triage": ("vital", "transport", "alert", "breathing", "perfusion", "identity"),
        "radio_handoff": ("situation", "background", "request", "vital", "timing", "source"),
        "asr_confirmed_text": ("confirmed", "denied", "vital", "timing", "source"),
        "escalation_precision": ("red flag", "denied", "deterministic", "vital", "timing"),
        "missing_observation_prioritization": ("vital", "mental", "breathing", "perfusion", "urine", "pain"),
        "sbar_handoff_usefulness": ("situation", "background", "request", "vital", "source"),
        "source_card_discipline": ("source", "card", "deterministic", "vital", "red flag"),
        "low_resource_constraints": ("unavailable", "breathing", "mental", "perfusion", "speech", "vital"),
        "workflow_repair_seed": ("situation", "source", "vital", "request", "red flag"),
    }
    keywords = priority_keywords.get(category, ("vital", "source", "red flag", "request"))
    prioritized: list[str] = []
    for keyword in keywords:
        lowered_keyword = keyword.lower()
        for cue in cues:
            if lowered_keyword in cue.lower() and cue not in prioritized:
                prioritized.append(cue)
    prioritized.extend(cue for cue in cues if cue not in prioritized)
    return _dedupe(prioritized)[:8]


def _teacher_candidates(
    client: TeacherClient | None,
    prepared: PreparedCase,
    teacher_model_id: str,
    candidate_count: int,
) -> list[dict[str, Any]]:
    if client is None:
        raise ModelClientError("teacher client was not configured")
    candidates = []
    for candidate_index in range(candidate_count):
        prompt = teacher_note_prompt(prepared, teacher_model_id, candidate_index)
        notes = _stream_teacher_json(client, prompt)
        validate_teacher_notes(notes)
        candidates.append(assemble_teacher_navigator_output(prepared, notes))
    return candidates


def _raw_candidates_with_retries(
    *,
    client: TeacherClient | None,
    prepared: PreparedCase,
    teacher_model_id: str,
    candidate_count: int,
    dry_run: bool,
    teacher_error_retries: int,
    teacher_error_sleep_seconds: float,
) -> list[dict[str, Any]]:
    if dry_run:
        return _fallback_candidates(prepared)

    retry_index = 0
    while True:
        try:
            return _teacher_candidates(client, prepared, teacher_model_id, candidate_count)
        except ModelClientError as exc:
            if retry_index >= teacher_error_retries or not _is_retryable_teacher_error(exc):
                raise
            retry_index += 1
            sleep(teacher_error_sleep_seconds * retry_index)


def _is_retryable_teacher_error(exc: ModelClientError) -> bool:
    text = str(exc).lower()
    return "http_status=429" in text or "too many requests" in text


def validate_teacher_notes(notes: dict[str, Any]) -> None:
    required_lists = {
        "facts": 1,
        "missing": 1,
        "observe": 1,
        "checklist": 1,
        "uncertain": 1,
    }
    for key, minimum in required_lists.items():
        if len(_teacher_note_list(notes, key, limit=minimum)) < minimum:
            raise ModelClientError(f"teacher notes missing required field: {key}")
    sbar = notes.get("sbar")
    if not isinstance(sbar, dict):
        raise ModelClientError("teacher notes missing required field: sbar")
    for key in ("situation", "background", "assessment_observations_only", "handoff_request"):
        if not _teacher_note_text(sbar.get(key)):
            raise ModelClientError(f"teacher notes missing required sbar field: {key}")
    if not _teacher_note_text(notes.get("script")):
        raise ModelClientError("teacher notes missing required field: script")


def teacher_note_prompt(prepared: PreparedCase, teacher_model_id: str, candidate_index: int) -> str:
    context = {
        "case_id": prepared.spec.case_id,
        "candidate_index": candidate_index,
        "teacher_model_id": teacher_model_id,
        "structured_intake": prepared.spec.structured_intake,
        "urgency_floor": prepared.urgency_floor,
        "red_flag_labels": [str(rule.get("label") or rule.get("rule_id")) for rule in prepared.rule_results],
        "target_protocol_card_id": prepared.spec.target_protocol_card_id,
        "source_card_ids": prepared.expected_source_card_ids,
        "candidate_pathway_card_ids": prepared.expected_candidate_pathway_card_ids,
        "missing_observation_cues": prepared.expected_missing_observations[:6],
    }
    if uses_v3_field_workflow_policy(prepared.spec.dataset_version):
        context.update(
            {
                "workflow_category": prepared.spec.structured_intake.get("workflow_category"),
                "field_workflow_goal": prepared.spec.structured_intake.get("field_workflow_goal"),
                "available_supplies": prepared.spec.structured_intake.get("available_supplies"),
                "workflow_instruction": (
                    "Prioritize specific next observations that improve escalation, monitoring, or handoff. "
                    "If equipment is unavailable, say unavailable or ask for an observation-only alternative."
                ),
            }
        )
    if uses_v5_focused_policy(prepared.spec.dataset_version):
        context.update(
            {
                "v5_training_focus": prepared.spec.failure_class,
                "must_include_source_cards": _v5_must_include_source_cards(prepared),
                "required_observation_targets": required_observation_targets(prepared.retrieved_cards),
                "must_select_required_observation_ids": v5_required_selected_observation_ids(
                    source_card_ids=prepared.expected_source_card_ids,
                    retrieved_cards=prepared.retrieved_cards,
                ),
                "workflow_instruction": (
                    "Write concrete observation text for the selected required-observation ids. "
                    "Avoid generic phrases such as repeat vitals or monitor closely."
                ),
            }
        )
    return (
        "Return ONLY minified JSON. Total output <= 90 words. Each list string <= 8 words. "
        "Each SBAR string <= 14 words. script <= 18 words. Use complete sentences. "
        "Use this exact shape: "
        '{"facts":[2 strings],"missing":[3 strings],"observe":[3 strings],"checklist":[3 strings],'
        '"uncertain":[1 string],"sbar":{"situation":"","background":"","assessment_observations_only":"",'
        '"handoff_request":""},"script":""}. '
        "Write observation-only protocol navigation notes for a trained responder. "
        "Do not produce condition labels, clinical orders, treatment directions, send-home advice, "
        "or autonomous routing. Say protocol escalation/review instead of condition labels. "
        f"TASK={json.dumps(context, sort_keys=True)}"
    )


def _stream_teacher_json(client: TeacherClient, prompt: str) -> dict[str, Any]:
    if client.endpoint_env == "OPENROUTER_BASE_URL":
        return _teacher_json_http_non_streaming(client, prompt)

    ctx = multiprocessing.get_context("fork")
    result_queue: multiprocessing.Queue[tuple[str, Any]] = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_stream_teacher_json_worker, args=(client, prompt, result_queue))
    process.start()
    process.join(client.timeout_seconds + 5)
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise ModelClientError(
            f"teacher model backend failed; model={client.model_id}; "
            f"url={_safe_url_for_error(_openai_chat_url(client.endpoint))}; "
            f"timeout={client.timeout_seconds:g}s; error=teacher stream exceeded parent deadline"
        )
    if result_queue.empty():
        raise ModelClientError(
            f"teacher model backend failed; model={client.model_id}; "
            f"url={_safe_url_for_error(_openai_chat_url(client.endpoint))}; "
            "error=teacher worker exited without a result"
        )
    status, payload = result_queue.get()
    if status == "ok" and isinstance(payload, dict):
        return payload
    raise ModelClientError(str(payload))


def _stream_teacher_json_worker(client: TeacherClient, prompt: str, result_queue: Any) -> None:
    try:
        result_queue.put(("ok", _stream_teacher_json_http(client, prompt)))
    except BaseException as exc:  # noqa: BLE001 - worker must report all failures to parent.
        result_queue.put(("error", _safe_error_text(f"{type(exc).__name__}: {exc}")))


def _teacher_json_http_non_streaming(client: TeacherClient, prompt: str) -> dict[str, Any]:
    url = _openai_chat_url(client.endpoint)
    body = _teacher_request_body(client, prompt, stream=False)
    timeout = httpx.Timeout(
        connect=min(10.0, client.timeout_seconds),
        read=client.timeout_seconds,
        write=min(10.0, client.timeout_seconds),
        pool=min(10.0, client.timeout_seconds),
    )
    try:
        response = httpx.post(
            url,
            json=body,
            headers={"Content-Type": "application/json", **client.auth_headers},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise ModelClientError(_backend_error_message(exc, url, client.model_id, client.timeout_seconds)) from exc
    except (httpx.HTTPError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise ModelClientError(_backend_error_message(exc, url, client.model_id, client.timeout_seconds)) from exc

    choice = (payload.get("choices") or [{}])[0] if isinstance(payload, dict) else {}
    message = choice.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else ""
    if not isinstance(content, str) or not content.strip():
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else ""
        reasoning_present = bool(message.get("reasoning") or message.get("reasoning_details")) if isinstance(message, dict) else False
        raise ModelClientError(
            "teacher model backend failed; "
            f"model={client.model_id}; url={_safe_url_for_error(url)}; timeout={client.timeout_seconds:g}s; "
            f"error=empty non-streaming content; finish_reason={_safe_error_text(str(finish_reason))}; "
            f"reasoning_present={reasoning_present}"
        )
    try:
        return _parse_json_object(content)
    except json.JSONDecodeError as exc:
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else ""
        raise ModelClientError(
            "teacher model backend failed; "
            f"model={client.model_id}; url={_safe_url_for_error(url)}; timeout={client.timeout_seconds:g}s; "
            f"error=non-streaming content was not JSON; finish_reason={_safe_error_text(str(finish_reason))}; "
            f"content_prefix={_safe_error_text(content[:240])}"
        ) from exc


def _stream_teacher_json_http(client: TeacherClient, prompt: str) -> dict[str, Any]:
    url = _openai_chat_url(client.endpoint)
    body = _teacher_request_body(client, prompt, stream=True)
    started = perf_counter()
    deadline = started + client.timeout_seconds
    text_parts: list[str] = []
    timeout = httpx.Timeout(
        connect=min(10.0, client.timeout_seconds),
        read=min(15.0, client.timeout_seconds),
        write=min(10.0, client.timeout_seconds),
        pool=min(10.0, client.timeout_seconds),
    )
    try:
        with httpx.stream(
            "POST",
            url,
            json=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream", **client.auth_headers},
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            buffer = ""
            for raw_chunk in response.iter_raw():
                if perf_counter() - started > client.timeout_seconds:
                    raise TimeoutError(f"teacher stream exceeded timeout={client.timeout_seconds:g}s")
                if perf_counter() > deadline:
                    raise TimeoutError(f"teacher stream exceeded timeout={client.timeout_seconds:g}s")
                if not raw_chunk:
                    continue
                buffer += raw_chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    _append_sse_teacher_line(line, text_parts)
    except httpx.HTTPStatusError as exc:
        raise ModelClientError(_backend_error_message(exc, url, client.model_id, client.timeout_seconds)) from exc
    except (httpx.HTTPError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise ModelClientError(_backend_error_message(exc, url, client.model_id, client.timeout_seconds)) from exc
    return _parse_json_object("".join(text_parts))


def _teacher_request_body(client: TeacherClient, prompt: str, *, stream: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": client.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 0.95,
        "max_tokens": client.max_tokens,
        "stream": stream,
    }
    if client.endpoint_env == "OPENROUTER_BASE_URL":
        body["max_completion_tokens"] = client.max_tokens
        body["reasoning"] = {"effort": "none", "exclude": True}
        body["include_reasoning"] = False
    else:
        body["reasoning_effort"] = "none"
        body["reasoning_budget"] = 0
    return body


def _append_sse_teacher_line(line: str, text_parts: list[str]) -> None:
    line = line.strip()
    if not line.startswith("data:"):
        return
    payload = line[5:].strip()
    if payload == "[DONE]" or not payload:
        return
    chunk = json.loads(payload)
    choice = (chunk.get("choices") or [{}])[0]
    delta = choice.get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str):
        text_parts.append(content)


def assemble_teacher_navigator_output(prepared: PreparedCase, notes: dict[str, Any]) -> dict[str, Any]:
    facts = _teacher_note_list(notes, "facts", limit=4)
    missing = _teacher_note_list(notes, "missing", limit=6)
    observe = _teacher_note_list(notes, "observe", limit=6)
    checklist = _teacher_note_list(notes, "checklist", limit=5)
    uncertain = _teacher_note_list(notes, "uncertain", limit=3)
    if not facts:
        facts = [
            f"Concern: {prepared.spec.structured_intake.get('chief_concern') or 'field concern'}",
            f"Vitals: {prepared.spec.structured_intake.get('vitals') or 'not recorded'}",
        ]
    if not missing:
        missing = prepared.expected_missing_observations[:3]
    if not observe:
        observe = prepared.expected_missing_observations[:3]
    if not checklist:
        checklist = ["Keep deterministic red flags visible.", "Collect missing observations.", "Escalate per cited protocol cards."]
    if not uncertain:
        uncertain = ["Responder must verify incomplete observations against local protocol."]

    if uses_v3_field_workflow_policy(prepared.spec.dataset_version):
        missing = _v3_priority_observation_list(prepared, missing, limit=6)
        observe = _v3_priority_observation_list(prepared, observe, limit=6)
        checklist = _v3_responder_checklist(prepared, checklist)
        handoff = _v3_grounded_handoff(prepared)
    else:
        sbar = notes.get("sbar")
        sbar = sbar if isinstance(sbar, dict) else {}
        handoff = {
            "situation": _teacher_note_text(sbar.get("situation")) or str(prepared.spec.structured_intake.get("chief_concern") or "Field concern"),
            "background": _teacher_note_text(sbar.get("background")) or f"Setting: {prepared.spec.structured_intake.get('setting', 'field setting')}.",
            "assessment_observations_only": _teacher_note_text(sbar.get("assessment_observations_only"))
            or str(prepared.spec.structured_intake.get("symptoms") or prepared.spec.structured_intake.get("responder_note") or "observations pending"),
            "handoff_request": _teacher_note_text(sbar.get("handoff_request")) or "Request review/escalation per cited local protocol cards.",
        }

    return {
        "protocol_urgency": prepared.urgency_floor,
        "red_flags": prepared.rule_results,
        "intake_facts": [
            {"fact": fact, "status": "reported", "source": "structured_field"}
            for fact in facts[:4]
        ],
        "candidate_protocol_pathways": [
            {
                "card_id": card_id,
                "reason_relevant": "Retrieved from confirmed intake and deterministic rule context.",
            }
            for card_id in prepared.expected_candidate_pathway_card_ids
        ],
        "missing_info_to_collect": missing,
        "next_observations_to_collect": observe,
        "conflicts_or_uncertainties": uncertain,
        "responder_checklist": checklist,
        "do_not_do": forbidden_behavior_for_version(prepared.spec.dataset_version),
        "source_cards": prepared.expected_source_card_ids,
        "handoff_note_sbar": handoff,
        "responder_plain_language_script": _teacher_note_text(notes.get("script"))
        or "I am checking protocol observations and will escalate through the cited local pathway if danger signs remain present.",
        "safety_boundary": safety_boundary_for_version(prepared.spec.dataset_version),
    }


def _v3_priority_observation_list(prepared: PreparedCase, teacher_items: list[str], *, limit: int) -> list[str]:
    priority = [
        _v3_resource_aware_cue_text(cue, prepared.spec.structured_intake)
        for cue in prepared.expected_missing_observations[:limit]
    ]
    for item in teacher_items:
        text = _teacher_note_text(item)
        if text and not _is_v3_generic_item(text):
            priority.append(text)
    return _dedupe(priority)[:limit]


def _v3_resource_aware_cue_text(cue: str, intake: dict[str, Any]) -> str:
    resource_text = json.dumps(intake, sort_keys=True).lower()
    cue_text = str(cue).strip()
    lowered = cue_text.lower()
    if _resource_unavailable(resource_text, ("no pulse oximeter", "no pulse ox", "oxygen saturation unavailable")):
        if any(token in lowered for token in ("oxygen saturation", "pulse ox", "pulse oximeter", "spo2")):
            return "oxygen saturation unavailable; observe work of breathing and speech"
    if _resource_unavailable(resource_text, ("no bp cuff", "blood pressure not available", "no blood pressure cuff")):
        if "blood pressure" in lowered or lowered == "bp":
            return "blood pressure unavailable; note perfusion and mental status"
    return cue_text


def _v3_responder_checklist(prepared: PreparedCase, teacher_items: list[str]) -> list[str]:
    items = [
        "Keep deterministic red flags visible.",
        "Cite only retrieved protocol cards.",
        "Prepare grounded SBAR handoff.",
    ]
    resource_text = json.dumps(prepared.spec.structured_intake, sort_keys=True).lower()
    if "no pulse oximeter" in resource_text or "no bp cuff" in resource_text:
        items.append("Mark unavailable equipment before alternatives.")
    for item in teacher_items:
        text = _teacher_note_text(item)
        if text and not _is_v3_generic_item(text):
            items.append(text)
    return _dedupe(items)[:5]


def _v3_grounded_handoff(prepared: PreparedCase) -> dict[str, str]:
    intake = prepared.spec.structured_intake
    background_parts = []
    if str(intake.get("setting") or "").strip():
        background_parts.append(f"Setting: {intake['setting']}.")
    if str(intake.get("patient_age") or "").strip():
        background_parts.append(f"Patient age {intake['patient_age']}.")
    if str(intake.get("pregnancy_status") or "").strip():
        background_parts.append(f"Status {intake['pregnancy_status']}.")
    assessment_parts = []
    if str(intake.get("symptoms") or "").strip():
        assessment_parts.append(f"Symptoms: {intake['symptoms']}.")
    if str(intake.get("vitals") or "").strip():
        assessment_parts.append(f"Vitals: {intake['vitals']}.")
    red_flag_labels = [
        str(rule.get("label") or rule.get("rule_id"))
        for rule in prepared.rule_results
        if rule.get("label") or rule.get("rule_id")
    ]
    if red_flag_labels:
        assessment_parts.append(f"Red flags: {'; '.join(red_flag_labels)}.")
    return {
        "situation": str(intake.get("chief_concern") or intake.get("responder_note") or "Confirmed field concern"),
        "background": " ".join(background_parts) or str(intake.get("setting") or "Background pending from confirmed intake."),
        "assessment_observations_only": " ".join(assessment_parts)
        or str(intake.get("symptoms") or intake.get("vitals") or "Observations pending from confirmed intake."),
        "handoff_request": f"Request {prepared.urgency_floor} review/escalation per cited local protocol cards.",
    }


def _teacher_note_list(notes: dict[str, Any], key: str, *, limit: int) -> list[str]:
    value = notes.get(key)
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []
    out = []
    for item in raw_items:
        text = _teacher_note_text(item)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _teacher_note_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    if not text:
        return ""
    replacements = [
        (r"\b(?:suspected|possible|likely|probable)\s+[^.,;]+", "protocol red-flag concern"),
        (r"\bheat stroke\b", "heat-related red-flag concern"),
        (r"\bheart attack\b", "chest-pain red-flag concern"),
        (r"\bmyocardial infarction\b", "chest-pain red-flag concern"),
        (r"\bpneumonia\b", "respiratory red-flag concern"),
        (r"\bsepsis\b", "systemic red-flag concern"),
        (r"\bdiagnosis\b", "protocol concern"),
        (r"\bmedications?\b", "listed treatments"),
        (r"\baspirin\b", "unsafe medicine request"),
        (r"\bantibiotics?\b", "unsafe medicine request"),
        (r"\bopioids?\b", "unsafe medicine request"),
        (r"\binsulin\b", "unsafe medicine request"),
        (r"\bdrugs?\b", "unsafe substance request"),
        (r"\bactivate\s+[^.,;]*protocol\b", "use the cited protocol pathway"),
        (r"\bprepare\s+(?:for\s+)?(?:rapid\s+)?transport\b", "prepare handoff information"),
        (r"\bemergency transport\b", "emergency pathway review"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text[:360]


def _openai_chat_url(base_url: str) -> str:
    parts = urllib.parse.urlsplit(base_url.strip())
    path = parts.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path = f"{path}/chat/completions" if path else "/chat/completions"
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _backend_error_message(exc: BaseException, url: str, model_id: str, timeout_seconds: float) -> str:
    details = [
        "teacher model backend failed",
        f"model={model_id}",
        f"url={_safe_url_for_error(url)}",
        f"timeout={timeout_seconds:g}s",
    ]
    if isinstance(exc, httpx.HTTPStatusError):
        details.append(f"http_status={exc.response.status_code}")
        reason = exc.response.reason_phrase
        if reason:
            details.append(f"reason={_safe_error_text(str(reason))}")
    elif isinstance(exc, httpx.TimeoutException):
        details.append(f"reason={_safe_error_text(str(exc)) or 'timeout'}")
    else:
        details.append(f"error={_safe_error_text(str(exc))}")
    return "; ".join(details)


def _safe_url_for_error(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise
    if isinstance(parsed, dict):
        return parsed
    raise json.JSONDecodeError("teacher response JSON was not an object", text, 0)


def teacher_single_candidate_prompt(prepared: PreparedCase, teacher_model_id: str) -> str:
    context = teacher_compact_context(prepared, teacher_model_id)
    wrapper = {
        "task": "Generate one gold supervised-finetuning navigator output for Figment.",
        "output_contract": {
            "return_one_complete_navigator_json_object": True,
            "candidate_schema": REQUIRED_JSON_SKELETON,
            "reasoning": "off",
            "no_think_tags": True,
            "no_extra_explanation": True,
        },
        "scoring_rubric": {
            "must_preserve_urgency_floor": prepared.urgency_floor,
            "must_copy_deterministic_red_flags_exactly": prepared.rule_results,
            "must_include_expected_source_cards": prepared.expected_source_card_ids,
            "must_include_expected_candidate_pathways": prepared.expected_candidate_pathway_card_ids,
            "must_cover_expected_missing_observation_cues": prepared.expected_missing_observations,
            "must_avoid": forbidden_behavior_for_version(prepared.spec.dataset_version),
        },
        "case_context": context,
    }
    return (
        "You are the Figment SFT data teacher. Return ONLY one valid JSON object matching the navigator schema.\n"
        "Do not include chain-of-thought, <think> tags, prose outside JSON, condition labels, clinical orders, "
        "or autonomous routing.\n\n"
        "Use the compact case context below to create the assistant label. The SFT row will pair your label with "
        "the exact production Figment prompt separately; do not quote or rewrite that prompt.\n\n"
        f"TEACHER_WRAPPER:\n{json.dumps(wrapper, indent=2, sort_keys=True)}"
    )


def _fallback_candidates(prepared: PreparedCase) -> list[dict[str, Any]]:
    output = canned_navigator_output(
        prepared.spec.structured_intake,
        prepared.rule_results,
        prepared.retrieved_cards,
        prepared.urgency_floor,
    )
    return [output]


def teacher_candidate_prompt(prepared: PreparedCase, teacher_model_id: str, candidate_count: int) -> str:
    context = teacher_compact_context(prepared, teacher_model_id)
    wrapper = {
        "task": "Generate gold supervised-finetuning candidates for Figment.",
        "candidate_count": candidate_count,
        "output_contract": {
            "return_json_object_with_key": "candidates",
            "candidate_schema": REQUIRED_JSON_SKELETON,
            "reasoning": "off",
            "no_think_tags": True,
            "no_extra_explanation": True,
        },
        "scoring_rubric": {
            "must_preserve_urgency_floor": prepared.urgency_floor,
            "must_copy_deterministic_red_flags_exactly": prepared.rule_results,
            "must_include_expected_source_cards": prepared.expected_source_card_ids,
            "must_include_expected_candidate_pathways": prepared.expected_candidate_pathway_card_ids,
            "must_cover_expected_missing_observation_cues": prepared.expected_missing_observations,
            "must_avoid": forbidden_behavior_for_version(prepared.spec.dataset_version),
        },
        "case_context": context,
    }
    return (
        "You are the Figment SFT data teacher. Return ONLY valid JSON.\n"
        "Do not include chain-of-thought, <think> tags, prose outside JSON, condition labels, clinical orders, "
        "or autonomous routing. Generate distinct but all-correct candidate navigator outputs.\n\n"
        "Use the compact case context below to create assistant labels. The SFT rows will pair the selected label "
        "with the exact production Figment prompt separately; do not quote or rewrite that prompt.\n\n"
        f"TEACHER_WRAPPER:\n{json.dumps(wrapper, indent=2, sort_keys=True)}\n\n"
        f"Return exactly this JSON object shape: {{\"candidates\": [/* {candidate_count} complete navigator JSON objects */]}}"
    )


def teacher_compact_context(prepared: PreparedCase, teacher_model_id: str) -> dict[str, Any]:
    cue_buckets = bucket_expected_observation_cues(prepared.expected_missing_observations)
    context = {
        "case_id": prepared.spec.case_id,
        "dataset_version": prepared.spec.dataset_version,
        "failure_class": prepared.spec.failure_class,
        "target_protocol_card_id": prepared.spec.target_protocol_card_id,
        "structured_intake": prepared.spec.structured_intake,
        "deterministic_red_flags": prepared.rule_results,
        "protocol_urgency_floor": prepared.urgency_floor,
        "retrieved_protocol_cards": [_compact_card(item.get("card", item)) for item in prepared.retrieved_cards],
        "expected_source_card_ids": prepared.expected_source_card_ids,
        "expected_candidate_pathway_card_ids": prepared.expected_candidate_pathway_card_ids,
        "expected_missing_observations": prepared.expected_missing_observations,
        "expected_model_observation_cues": cue_buckets["model"],
        "expected_handoff_cues": cue_buckets["handoff"],
        "expected_harness_evidence_cues": cue_buckets["harness"],
        "expected_red_flag_rule_ids": prepared.expected_red_flag_rule_ids,
        "expected_min_protocol_urgency": prepared.urgency_floor,
        "retrieved_card_ids": prepared.retrieved_ids,
        "navigator_output_schema": REQUIRED_JSON_SKELETON,
        "teacher_model_id": teacher_model_id,
        "production_prompt_hash": stable_hash(prepared.prompt),
    }
    if uses_v3_field_workflow_policy(prepared.spec.dataset_version):
        context.update(
            {
                "workflow_category": prepared.spec.structured_intake.get("workflow_category"),
                "field_workflow_goal": prepared.spec.structured_intake.get("field_workflow_goal"),
                "workflow_priority": (
                    "field-useful, concise, source-card-grounded intake/escalation/handoff support"
                ),
                "low_resource_constraints": prepared.spec.structured_intake.get("available_supplies"),
            }
        )
    if uses_v5_focused_policy(prepared.spec.dataset_version):
        context.update(
            {
                "v5_training_focus": prepared.spec.failure_class,
                "required_observation_targets": required_observation_targets(prepared.retrieved_cards),
                "must_select_required_observation_ids": v5_required_selected_observation_ids(
                    source_card_ids=prepared.expected_source_card_ids,
                    retrieved_cards=prepared.retrieved_cards,
                ),
                "must_include_source_cards": _v5_must_include_source_cards(prepared),
            }
        )
    return context


def _compact_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "card_id": card.get("card_id"),
        "title": card.get("title"),
        "red_flags": card.get("red_flags", []),
        "escalation_criteria": card.get("escalation_criteria", []),
        "required_observations": card.get("required_observations", []),
        "local_actions": card.get("local_actions", []),
        "forbidden_actions": card.get("forbidden_actions", []),
        "safety_boundary": card.get("safety_boundary", ""),
    }


def score_candidate(candidate: dict[str, Any], prepared: PreparedCase) -> CandidateResult:
    raw_hash = stable_hash(candidate)
    normalized = normalize_output(candidate)
    scaffold = apply_navigation_scaffolding(
        normalized,
        retrieved_cards=prepared.retrieved_cards,
        rule_results=prepared.rule_results,
        urgency_floor=prepared.urgency_floor,
        confirmed_intake=prepared.spec.structured_intake,
    )
    patched = patch_expected_labels(scaffold.output, prepared)
    validation = validate_navigator_output(
        patched,
        known_card_ids={str(card["card_id"]) for card in load_protocol_cards()},
        urgency_floor=prepared.urgency_floor,
        confirmed_intake=prepared.spec.structured_intake,
        rule_results=prepared.rule_results,
        retrieved_card_ids=set(prepared.retrieved_ids),
        retrieved_cards=prepared.retrieved_cards,
        strict_schema=True,
    ).to_dict()
    record = eval_record_for_output(prepared, patched, validation)
    expected_score = score_expected_labels(record)
    reward_components = reward_components_for(prepared, patched, validation, expected_score)
    reward_score = sum(reward_components.values())
    patched_fields = sorted(set(scaffold.patched_fields) | set(_patch_fields(normalized, patched)))
    return CandidateResult(
        output=patched,
        validation=validation,
        expected_label_score=expected_score,
        reward_components=reward_components,
        reward_score=reward_score,
        patched_fields=patched_fields,
        filled_required_observation_ids=scaffold.filled_required_observation_ids,
        model_selected_required_observation_ids=scaffold.model_selected_required_observation_ids,
        invalid_selected_required_observation_ids=scaffold.invalid_selected_required_observation_ids,
        stripped_trace_only_fields=scaffold.stripped_trace_only_fields,
        raw_output_hash=raw_hash,
    )


def normalize_output(candidate: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, default in REQUIRED_JSON_SKELETON.items():
        value = candidate.get(key, default)
        if isinstance(default, list) and not isinstance(value, list):
            value = [str(value)] if value else []
        if isinstance(default, str) and not isinstance(value, str):
            value = str(value) if value is not None else ""
        if key == "handoff_note_sbar" and not isinstance(value, dict):
            value = dict(default)
        output[key] = value
    if TRACE_ONLY_REQUIRED_OBSERVATION_IDS_KEY in candidate:
        output[TRACE_ONLY_REQUIRED_OBSERVATION_IDS_KEY] = candidate[TRACE_ONLY_REQUIRED_OBSERVATION_IDS_KEY]
    return output


def patch_expected_labels(output: dict[str, Any], prepared: PreparedCase) -> dict[str, Any]:
    patched = json.loads(json.dumps(output))
    if prepared.spec.dataset_version == "figment_sft_v2" or uses_v3_field_workflow_policy(prepared.spec.dataset_version):
        patched["do_not_do"] = forbidden_behavior_for_version(prepared.spec.dataset_version)
        patched["safety_boundary"] = safety_boundary_for_version(prepared.spec.dataset_version)

    fired_rule_card_ids = {
        str(rule.get("card_id", "")).strip()
        for rule in prepared.rule_results
        if str(rule.get("card_id", "")).strip()
    }
    source_cards = _string_list(patched.get("source_cards"))
    for card_id in prepared.expected_source_card_ids:
        if (
            card_id in prepared.retrieved_ids
            or (uses_v5_focused_policy(prepared.spec.dataset_version) and card_id in fired_rule_card_ids)
        ) and card_id not in source_cards:
            source_cards.append(card_id)
    patched["source_cards"] = source_cards[:6]

    existing_candidate_ids = _candidate_ids(patched.get("candidate_protocol_pathways"))
    pathways = [item for item in patched.get("candidate_protocol_pathways", []) if isinstance(item, dict)]
    for card_id in prepared.expected_candidate_pathway_card_ids:
        if card_id in patched["source_cards"] and card_id not in existing_candidate_ids:
            pathways.append({"card_id": card_id, "reason_relevant": "Expected protocol target for this synthetic case."})
            existing_candidate_ids.append(card_id)
    patched["candidate_protocol_pathways"] = pathways

    record = eval_record_for_output(prepared, patched, {"passed": True, "failures": []})
    score = score_expected_labels(record)
    missing_cues = score.get("missing_expected_observation_cues") or []
    missing_info = _string_list(patched.get("missing_info_to_collect"))
    next_observations = _string_list(patched.get("next_observations_to_collect"))
    for cue in missing_cues:
        cue_text = str(cue)
        if uses_v3_field_workflow_policy(prepared.spec.dataset_version):
            cue_text = _v3_resource_aware_cue_text(cue_text, prepared.spec.structured_intake)
        if cue_text not in missing_info:
            missing_info.append(cue_text)
        if cue_text not in next_observations:
            next_observations.append(cue_text)
    if uses_v3_field_workflow_policy(prepared.spec.dataset_version):
        priority_cues = [
            _v3_resource_aware_cue_text(cue, prepared.spec.structured_intake)
            for cue in prepared.expected_missing_observations
        ]
        missing_info = _dedupe(priority_cues + missing_info)[:8]
        next_observations = _dedupe(priority_cues + next_observations)[:8]
    patched["missing_info_to_collect"] = missing_info
    patched["next_observations_to_collect"] = next_observations
    if uses_v5_focused_policy(prepared.spec.dataset_version):
        selected_ids = v5_required_selected_observation_ids(
            source_card_ids=patched["source_cards"],
            retrieved_cards=prepared.retrieved_cards,
        )
        if selected_ids:
            patched[TRACE_ONLY_REQUIRED_OBSERVATION_IDS_KEY] = selected_ids
    return patched


def reward_components_for(
    prepared: PreparedCase,
    output: dict[str, Any],
    validation: dict[str, Any],
    expected_score: dict[str, Any],
) -> dict[str, int]:
    text = json.dumps(output, sort_keys=True).lower()
    rewards = {
        "schema_valid": int(validation.get("passed") is True),
        "source_cards_present": int(expected_score.get("expected_source_cards_present") is not False),
        "required_observation_cues_present": int(expected_score.get("missing_observation_cues_present") is not False),
        "candidate_pathways_present": int(expected_score.get("expected_candidate_pathways_present") is not False),
        "red_flags_match": int(expected_score.get("red_flags_match") is not False),
        "min_urgency_met": int(expected_score.get("min_urgency_met") is not False),
        "forbidden_behavior_absent": int(expected_score.get("forbidden_behavior_absent") is not False),
        "no_visible_reasoning": int("<think" not in text and "</think" not in text),
        "target_card_present": int(prepared.spec.target_protocol_card_id in _string_list(output.get("source_cards"))),
    }
    if prepared.spec.dataset_version == "figment_sft_v2":
        rewards["v2_policy_pass"] = int(
            not v2_policy_issues(
                output,
                failure_class=prepared.spec.failure_class,
                expected_red_flag_rule_ids=prepared.expected_red_flag_rule_ids,
                expected_candidate_pathway_card_ids=prepared.expected_candidate_pathway_card_ids,
            )
        )
    if uses_v5_focused_policy(prepared.spec.dataset_version):
        rewards["v5_policy_pass"] = int(
            not v5_policy_issues(
                output,
                failure_class=prepared.spec.failure_class,
                expected_red_flag_rule_ids=prepared.expected_red_flag_rule_ids,
                expected_candidate_pathway_card_ids=prepared.expected_candidate_pathway_card_ids,
                structured_intake=prepared.spec.structured_intake,
                rule_results=prepared.rule_results,
                retrieved_cards=prepared.retrieved_cards,
                target_protocol_card_id=prepared.spec.target_protocol_card_id,
            )
        )
    elif uses_v3_field_workflow_policy(prepared.spec.dataset_version):
        rewards["v3_policy_pass"] = int(
            not v3_policy_issues(
                output,
                failure_class=prepared.spec.failure_class,
                expected_red_flag_rule_ids=prepared.expected_red_flag_rule_ids,
                expected_candidate_pathway_card_ids=prepared.expected_candidate_pathway_card_ids,
                structured_intake=prepared.spec.structured_intake,
            )
        )
    return rewards


def v2_policy_issues(
    output: dict[str, Any],
    *,
    failure_class: str,
    expected_red_flag_rule_ids: list[str],
    expected_candidate_pathway_card_ids: list[str],
) -> list[str]:
    """Return v2-only dataset policy violations for accepted assistant labels."""

    issues: list[str] = []
    output_text = json.dumps(output, sort_keys=True)
    for label, pattern in V2_FORBIDDEN_LEXICAL_PATTERNS.items():
        if pattern.search(output_text):
            issues.append(f"forbidden_lexical_tripwire:{label}")

    if failure_class == "negation_safety_boundary" and not expected_red_flag_rule_ids:
        if output.get("red_flags"):
            issues.append("negation_red_flags_must_be_empty")
        candidate_ids = set(_candidate_ids(output.get("candidate_protocol_pathways")))
        allowed_targets = {SAFETY_CARD_ID, SBAR_CARD_ID}
        expected_targets = set(expected_candidate_pathway_card_ids)
        if not candidate_ids or not expected_targets <= allowed_targets or not expected_targets <= candidate_ids:
            issues.append("negation_candidate_pathway_must_be_safety_or_sbar")
        if output.get("protocol_urgency") not in {"routine", "monitor"}:
            issues.append("negation_urgency_must_not_be_raised_by_denied_symptom")

    return issues


def v3_policy_issues(
    output: dict[str, Any],
    *,
    failure_class: str,
    expected_red_flag_rule_ids: list[str],
    expected_candidate_pathway_card_ids: list[str],
    structured_intake: dict[str, Any] | None = None,
) -> list[str]:
    """Return v3 field-workflow policy violations for accepted assistant labels."""

    issues = v2_policy_issues(
        output,
        failure_class="negation_safety_boundary" if failure_class == "escalation_precision" else failure_class,
        expected_red_flag_rule_ids=expected_red_flag_rule_ids,
        expected_candidate_pathway_card_ids=expected_candidate_pathway_card_ids,
    )
    field_items = (
        _string_list(output.get("missing_info_to_collect"))
        + _string_list(output.get("next_observations_to_collect"))
        + _string_list(output.get("responder_checklist"))
    )
    generic_count = sum(1 for item in field_items if _is_v3_generic_item(item))
    if field_items and generic_count >= max(3, len(field_items) // 2):
        issues.append("generic_output_dominated")

    sbar = output.get("handoff_note_sbar") if isinstance(output.get("handoff_note_sbar"), dict) else {}
    required_sbar_parts = ("situation", "background", "assessment_observations_only", "handoff_request")
    missing_sbar = [part for part in required_sbar_parts if len(str(sbar.get(part) or "").strip()) < 6]
    if failure_class in V3_SBAR_FAILURE_CLASSES or len(missing_sbar) >= 2:
        if missing_sbar:
            issues.append("handoff_sbar_missing_required_parts")

    intake = structured_intake or {}
    resource_text = json.dumps(intake, sort_keys=True).lower()
    output_text = json.dumps(output, sort_keys=True).lower()
    if _resource_unavailable(resource_text, ("no pulse oximeter", "no pulse ox", "oxygen saturation unavailable")):
        if _asks_for_unavailable_pulse_ox(output_text):
            issues.append("low_resource_unavailable_pulse_ox_requested")
    if _resource_unavailable(resource_text, ("no bp cuff", "blood pressure not available", "no blood pressure cuff")):
        if _asks_for_unavailable_bp(output_text):
            issues.append("low_resource_unavailable_bp_requested")

    if len(_string_list(output.get("next_observations_to_collect"))) > 10:
        issues.append("cognitive_load_next_observation_list_too_long")

    return _dedupe(issues)


def v5_required_selected_observation_ids(
    *,
    source_card_ids: list[str],
    retrieved_cards: list[dict[str, Any]],
) -> list[str]:
    """Return required-observation ids a v5 label must select for cited retrieved clinical cards."""

    source_set = {str(card_id).strip() for card_id in source_card_ids if str(card_id).strip()}
    selected: list[str] = []
    for target in required_observation_targets(retrieved_cards):
        card_id = str(target.get("card_id", "")).strip()
        target_id = str(target.get("id", "")).strip()
        if not card_id or not target_id:
            continue
        if card_id in CARD_IDS_EXEMPT_FROM_OBSERVATION_TARGETS:
            continue
        if card_id in source_set and target_id not in selected:
            selected.append(target_id)
    return selected


def _v5_must_include_source_cards(prepared: PreparedCase) -> list[str]:
    required = list(prepared.expected_source_card_ids)
    for rule in prepared.rule_results:
        card_id = str(rule.get("card_id", "")).strip()
        if card_id and card_id not in required:
            required.append(card_id)
    if prepared.spec.failure_class == "sbar_observation_ownership":
        for card_id in (SBAR_CARD_ID, SAFETY_CARD_ID):
            if card_id in prepared.retrieved_ids and card_id not in required:
                required.append(card_id)
    return required[:6]


def v5_policy_issues(
    output: dict[str, Any],
    *,
    failure_class: str,
    expected_red_flag_rule_ids: list[str],
    expected_candidate_pathway_card_ids: list[str],
    structured_intake: dict[str, Any] | None = None,
    rule_results: list[dict[str, Any]] | None = None,
    retrieved_cards: list[dict[str, Any]] | None = None,
    target_protocol_card_id: str = "",
) -> list[str]:
    """Return v5-focused dataset policy violations for accepted assistant labels."""

    issues = v3_policy_issues(
        output,
        failure_class=failure_class,
        expected_red_flag_rule_ids=expected_red_flag_rule_ids,
        expected_candidate_pathway_card_ids=expected_candidate_pathway_card_ids,
        structured_intake=structured_intake,
    )
    source_cards = _string_list(output.get("source_cards"))
    source_card_set = set(source_cards)

    for rule in rule_results or []:
        card_id = str(rule.get("card_id", "")).strip()
        if card_id and card_id not in source_card_set:
            issues.append(f"fired_rule_source_card_missing:{card_id}")

    selected_ids = _string_list(output.get(TRACE_ONLY_REQUIRED_OBSERVATION_IDS_KEY))
    required_selected_ids = v5_required_selected_observation_ids(
        source_card_ids=source_cards,
        retrieved_cards=retrieved_cards or [],
    )
    if required_selected_ids and not selected_ids:
        issues.append("selected_required_observation_ids_missing")
    if selected_ids:
        invalid = sorted(set(selected_ids) - set(required_selected_ids))
        if invalid:
            issues.append(f"selected_required_observation_ids_invalid:{','.join(invalid)}")
        selected_cards = {
            selected_id.split("::required_observation::", 1)[0]
            for selected_id in selected_ids
            if "::required_observation::" in selected_id
        }
        required_cards = {
            target_id.split("::required_observation::", 1)[0]
            for target_id in required_selected_ids
            if "::required_observation::" in target_id
        }
        for card_id in sorted(required_cards - selected_cards):
            issues.append(f"selected_required_observation_ids_missing_for_card:{card_id}")

    for item in _string_list(output.get("missing_info_to_collect")) + _string_list(
        output.get("next_observations_to_collect")
    ):
        if _is_v5_generic_observation_item(item):
            issues.append(f"generic_observation_phrase:{_safe_counter_key(item)}")

    if (
        target_protocol_card_id == SBAR_CARD_ID
        or SBAR_CARD_ID in source_card_set
        or failure_class in {"sbar_observation_ownership", *V3_SBAR_FAILURE_CLASSES}
    ):
        sbar = output.get("handoff_note_sbar") if isinstance(output.get("handoff_note_sbar"), dict) else {}
        missing_sbar = [
            part
            for part in ("situation", "background", "assessment_observations_only", "handoff_request")
            if len(str(sbar.get(part) or "").strip()) < 6
        ]
        if missing_sbar:
            issues.append("handoff_sbar_missing_required_parts")

    return _dedupe(issues)


def _policy_issues_for_prepared(output: dict[str, Any], prepared: PreparedCase) -> list[str]:
    if uses_v5_focused_policy(prepared.spec.dataset_version):
        return v5_policy_issues(
            output,
            failure_class=prepared.spec.failure_class,
            expected_red_flag_rule_ids=prepared.expected_red_flag_rule_ids,
            expected_candidate_pathway_card_ids=prepared.expected_candidate_pathway_card_ids,
            structured_intake=prepared.spec.structured_intake,
            rule_results=prepared.rule_results,
            retrieved_cards=prepared.retrieved_cards,
            target_protocol_card_id=prepared.spec.target_protocol_card_id,
        )
    if uses_v3_field_workflow_policy(prepared.spec.dataset_version):
        return v3_policy_issues(
            output,
            failure_class=prepared.spec.failure_class,
            expected_red_flag_rule_ids=prepared.expected_red_flag_rule_ids,
            expected_candidate_pathway_card_ids=prepared.expected_candidate_pathway_card_ids,
            structured_intake=prepared.spec.structured_intake,
        )
    if prepared.spec.dataset_version == "figment_sft_v2":
        return v2_policy_issues(
            output,
            failure_class=prepared.spec.failure_class,
            expected_red_flag_rule_ids=prepared.expected_red_flag_rule_ids,
            expected_candidate_pathway_card_ids=prepared.expected_candidate_pathway_card_ids,
        )
    return []


def _is_v3_generic_item(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in V3_GENERIC_OUTPUT_PATTERNS)


def _is_v5_generic_observation_item(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in V5_GENERIC_OBSERVATION_PATTERNS)


def _resource_unavailable(resource_text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in resource_text for phrase in phrases)


def _asks_for_unavailable_pulse_ox(output_text: str) -> bool:
    if "unavailable" in output_text and any(
        phrase in output_text for phrase in ("pulse ox", "pulse oximeter", "oxygen saturation", "spo2")
    ):
        return False
    return any(phrase in output_text for phrase in ("pulse ox", "pulse oximeter", "oxygen saturation", "spo2", "repeat vitals"))


def _asks_for_unavailable_bp(output_text: str) -> bool:
    if "unavailable" in output_text and any(phrase in output_text for phrase in ("blood pressure", "bp cuff", "bp")):
        return False
    return any(phrase in output_text for phrase in ("blood pressure", "bp cuff", "repeat vitals"))


def eval_record_for_output(
    prepared: PreparedCase,
    output: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    cue_buckets = bucket_expected_observation_cues(prepared.expected_missing_observations)
    harness_evidence = build_harness_evidence(
        confirmed_intake=prepared.spec.structured_intake,
        retrieved_card_ids=prepared.retrieved_ids,
        rule_results=prepared.rule_results,
        urgency_floor=prepared.urgency_floor,
        validator_result=validation,
        final_output=output,
    )
    return {
        "case_id": prepared.spec.case_id,
        "structured_intake": prepared.spec.structured_intake,
        "target_protocol_card_id": prepared.spec.target_protocol_card_id,
        "expected_min_protocol_urgency": prepared.urgency_floor,
        "expected_red_flag_rule_ids": prepared.expected_red_flag_rule_ids,
        "expected_source_card_ids": prepared.expected_source_card_ids,
        "expected_candidate_pathway_card_ids": prepared.expected_candidate_pathway_card_ids,
        "expected_missing_observations": prepared.expected_missing_observations,
        "expected_model_observation_cues": cue_buckets["model"],
        "expected_handoff_cues": cue_buckets["handoff"],
        "expected_harness_evidence_cues": cue_buckets["harness"],
        "forbidden_behavior": forbidden_behavior_for_version(prepared.spec.dataset_version),
        "actual_red_flag_rule_ids": [str(rule["rule_id"]) for rule in prepared.rule_results],
        "actual_protocol_urgency": output.get("protocol_urgency"),
        "actual_source_card_ids": _string_list(output.get("source_cards")),
        "actual_candidate_pathway_card_ids": _candidate_ids(output.get("candidate_protocol_pathways")),
        "retrieved_card_ids": prepared.retrieved_ids,
        "harness_evidence": harness_evidence,
        "final_output": output,
        "final_validation": validation,
    }


def build_sft_row(
    *,
    prepared: PreparedCase,
    result: CandidateResult,
    teacher_model_id: str,
    candidate_total: int,
    candidate_passed: int,
) -> dict[str, Any]:
    workflow_category = str(prepared.spec.structured_intake.get("workflow_category") or "")
    cue_buckets = bucket_expected_observation_cues(prepared.expected_missing_observations)
    metadata = {
        "teacher_model_id": teacher_model_id,
        "critic_model_id": teacher_model_id,
        "teacher_label_mode": "streamed_ultra_semantic_notes_harness_prompt",
        "teacher_base_url_env": _endpoint_env_name(teacher_model_id),
        "teacher_api_key_env": _api_key_env_name(teacher_model_id),
        "failure_class": prepared.spec.failure_class,
        "dataset_version": prepared.spec.dataset_version,
        "expected_action": {
            "target_card": prepared.spec.target_protocol_card_id,
            "source_cards": prepared.expected_source_card_ids,
            "candidate_pathway_card_ids": prepared.expected_candidate_pathway_card_ids,
            "required_observation_cues": prepared.expected_missing_observations,
            "model_observation_cues": cue_buckets["model"],
            "handoff_cues": cue_buckets["handoff"],
            "harness_evidence_cues": cue_buckets["harness"],
            "red_flag_rule_ids": prepared.expected_red_flag_rule_ids,
            "min_protocol_urgency": prepared.urgency_floor,
        },
        "reward_components": result.reward_components,
        "pass_rate_total": candidate_total,
        "pass_rate_passed": candidate_passed,
        "dedupe_hash": dedupe_hash(prepared),
        "input_hash": stable_hash(prepared.spec.structured_intake),
        "prompt_hash": stable_hash(prepared.prompt),
        "prompt_template_hash": prepared.prompt_hash,
        "raw_teacher_output_hash": result.raw_output_hash,
        "deterministic_scaffold_patched_fields": result.patched_fields,
        "filled_required_observation_ids": result.filled_required_observation_ids,
        "model_selected_required_observation_ids": result.model_selected_required_observation_ids,
        "invalid_selected_required_observation_ids": result.invalid_selected_required_observation_ids,
        "stripped_trace_only_fields": result.stripped_trace_only_fields,
        "validation_result": result.validation,
        "expected_label_score": result.expected_label_score,
        "retrieved_card_ids": prepared.retrieved_ids,
        "recipe_sources": [
            "nvidia/Nemotron-Post-Training-Dataset-v2",
            "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
            "nvidia/Nemotron-CC-v2",
        ],
        "validator_passed": True,
        "license_review": "synthetic_figment_row_no_nvidia_rows_copied",
        "phi_status": "synthetic_deidentified_no_phi",
        "generated_at": datetime.now(UTC).isoformat(),
    }
    if workflow_category:
        metadata.update(
            {
                "workflow_category": workflow_category,
                "field_workflow_goal": prepared.spec.structured_intake.get("field_workflow_goal"),
                "workflow_priority_observations": prepared.expected_missing_observations[:5],
                "v3_workflow_validator_version": 1,
                "anti_overfit_policy": {
                    "locked_eval_copying_allowed": False,
                    "holdout_copying_allowed": False,
                    "primary_success_surface": "field_workflow_holdout_v1",
                },
            }
        )
    if uses_v5_focused_policy(prepared.spec.dataset_version):
        must_include_selected = v5_required_selected_observation_ids(
            source_card_ids=_string_list(result.output.get("source_cards")),
            retrieved_cards=prepared.retrieved_cards,
        )
        metadata.update(
            {
                "training_focus": prepared.spec.failure_class,
                "v5_training_policy_version": 1,
                "excluded_eval_case_ids": list(V5_EXCLUDED_EVAL_CASE_IDS),
                "must_include_source_cards": _v5_must_include_source_cards(prepared),
                "must_include_selected_required_observation_ids": must_include_selected,
            }
        )
    return {
        "case_id": prepared.spec.case_id,
        "uuid": prepared.spec.case_id,
        "license": "synthetic internal training data",
        "generator": teacher_model_id,
        "version": prepared.spec.dataset_version,
        "category": prepared.spec.failure_class,
        "reasoning": "off",
        "messages": [
            {"role": "user", "content": prepared.prompt},
            {"role": "assistant", "content": json.dumps(result.output, sort_keys=True)},
        ],
        "tags": prepared.spec.tags,
        "metadata": metadata,
    }


def case_spec_record(prepared: PreparedCase) -> dict[str, Any]:
    cue_buckets = bucket_expected_observation_cues(prepared.expected_missing_observations)
    record = {
        "case_id": prepared.spec.case_id,
        "dataset_version": prepared.spec.dataset_version,
        "failure_class": prepared.spec.failure_class,
        "target_protocol_card_id": prepared.spec.target_protocol_card_id,
        "structured_intake": prepared.spec.structured_intake,
        "expected_red_flag_rule_ids": prepared.expected_red_flag_rule_ids,
        "expected_min_protocol_urgency": prepared.urgency_floor,
        "expected_source_card_ids": prepared.expected_source_card_ids,
        "expected_candidate_pathway_card_ids": prepared.expected_candidate_pathway_card_ids,
        "expected_missing_observations": prepared.expected_missing_observations,
        "expected_model_observation_cues": cue_buckets["model"],
        "expected_handoff_cues": cue_buckets["handoff"],
        "expected_harness_evidence_cues": cue_buckets["harness"],
        "retrieved_card_ids": prepared.retrieved_ids,
        "tags": prepared.spec.tags,
    }
    workflow_category = str(prepared.spec.structured_intake.get("workflow_category") or "")
    if workflow_category:
        record.update(
            {
                "workflow_category": workflow_category,
                "field_workflow_goal": prepared.spec.structured_intake.get("field_workflow_goal"),
                "workflow_priority_observations": prepared.expected_missing_observations[:5],
                "field_workflow_holdout_relevant": True,
            }
        )
    return record


def build_manifest(
    *,
    output_path: Path,
    case_specs_path: Path,
    dataset_version: str,
    rows: list[dict[str, Any]],
    started_at: datetime,
    teacher_model_id: str,
    dry_run: bool,
    attempts: int,
    start_index: int,
    index_stride: int,
    counters: Counter[str],
    candidate_totals: Counter[str],
    rejection_reasons: Counter[str],
    events: list[dict[str, Any]],
    exclusion_paths: list[Path] | None = None,
    exclusion_signature_count: int = 0,
) -> dict[str, Any]:
    finished_at = datetime.now(UTC)
    return {
        "dataset_version": dataset_version,
        "row_count": len(rows),
        "output_path": str(output_path),
        "case_specs_path": str(case_specs_path),
        "output_sha256": _file_sha256(output_path) if output_path.exists() else None,
        "case_specs_sha256": _file_sha256(case_specs_path) if case_specs_path.exists() else None,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round((finished_at - started_at).total_seconds(), 3),
        "teacher_model_id": teacher_model_id,
        "teacher_endpoint_env": _endpoint_env_name(teacher_model_id),
        "teacher_api_key_env": _api_key_env_name(teacher_model_id),
        "dry_run": dry_run,
        "attempts": attempts,
        "start_index": start_index,
        "index_stride": index_stride,
        "accepted_by_failure_class": {
            key: value for key, value in sorted(counters.items()) if not key.startswith("tag:")
        },
        "accepted_by_tag": {
            key[4:]: value for key, value in sorted(counters.items()) if key.startswith("tag:")
        },
        "candidate_totals": dict(candidate_totals),
        "rejection_reasons": dict(rejection_reasons),
        "anti_overfit_exclusions": {
            "enabled": bool(exclusion_paths),
            "eval_paths": [str(path) for path in exclusion_paths or []],
            "signature_count": exclusion_signature_count,
            "policies": [
                "exact clinical-intake hash rejection",
                "same target/workflow high-token-overlap rejection",
            ],
        },
        "source_recipe_links": [
            "nvidia/Nemotron-Post-Training-Dataset-v2",
            "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
            "nvidia/Nemotron-CC-v2",
        ],
        "license_phi_assertions": {
            "synthetic_only": True,
            "no_phi": True,
            "no_locked_eval_rows_copied": True,
            "no_nvidia_dataset_rows_copied": True,
            "requires_license_review_before_distribution": True,
        },
        "prompt_template_hash": stable_hash(SYSTEM_PROMPT),
        "event_sample": events[-50:],
    }


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _rejection_key(result: CandidateResult) -> str:
    if result.validation.get("passed") is not True:
        failures = result.validation.get("failures") or ["validation_failed"]
        return _safe_counter_key(str(failures[0]))
    if result.expected_label_score.get("all_expected_labels_passed") is not True:
        for key, value in result.expected_label_score.items():
            if value is False:
                return f"expected_label_{key}"
        return "expected_label_failed"
    failed_rewards = [key for key, value in result.reward_components.items() if not value]
    return f"reward_{failed_rewards[0]}" if failed_rewards else "unknown"


def _patch_fields(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    return sorted(field for field in REQUIRED_JSON_SKELETON if before.get(field) != after.get(field))


def _candidate_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    ids: list[str] = []
    for item in value:
        card_id = item.get("card_id") if isinstance(item, dict) else item
        if card_id:
            ids.append(str(card_id))
    return ids


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return out


def _pick(values: list[str], index: int) -> str:
    return values[index % len(values)]


def _tag_for_card(card_id: str) -> str:
    return card_id.lower().replace("-v1", "").replace("-", "_")


def dedupe_hash(prepared: PreparedCase) -> str:
    payload = {
        "normalized_intake": _normalize_text(json.dumps(prepared.spec.structured_intake, sort_keys=True)),
        "target": prepared.spec.target_protocol_card_id,
        "expected_source": prepared.expected_source_card_ids,
        "expected_missing": prepared.expected_missing_observations,
    }
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _endpoint_env_name(teacher_model_id: str | None = None) -> str:
    # Record only the variable name, never the resolved endpoint or secret.
    if teacher_model_id and _openrouter_config_for_teacher_model(teacher_model_id):
        return "OPENROUTER_BASE_URL"
    if os.getenv("OMNI_ENDPOINT_URL", "").strip():
        return "OMNI_ENDPOINT_URL"
    if os.getenv("HF_ENDPOINT_URL", "").strip():
        return "HF_ENDPOINT_URL"
    if os.getenv("NVIDIA_BASE_URL", "").strip():
        return "NVIDIA_BASE_URL"
    return f"NVIDIA_BASE_URL(default:{NVIDIA_API_BASE_URL})"


def _api_key_env_name(teacher_model_id: str | None = None) -> str:
    # Record only the variable name, never the resolved secret.
    if teacher_model_id and _openrouter_config_for_teacher_model(teacher_model_id):
        return "OPENROUTER_API_KEY"
    if os.getenv("OMNI_ENDPOINT_URL", "").strip() or os.getenv("HF_ENDPOINT_URL", "").strip():
        return "HF_TOKEN" if os.getenv("HF_TOKEN", "").strip() else ""
    if os.getenv("NVIDIA_API_KEY", "").strip():
        return "NVIDIA_API_KEY"
    return ""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_progress_event(event: dict[str, Any]) -> None:
    print(json.dumps(event, sort_keys=True), flush=True)


def _safe_counter_key(value: str) -> str:
    return re.sub(r"[^a-z0-9_.:-]+", "_", value.lower()).strip("_")[:120] or "unknown"


def _safe_error_text(value: str) -> str:
    text = re.sub(r"(?i)bearer\s+[^\s]+", "Bearer [redacted]", value)
    text = re.sub(r"(?i)(api_key|token|authorization)=([^&\s]+)", r"\1=[redacted]", text)
    return text[:500]


if __name__ == "__main__":
    raise SystemExit(main())
