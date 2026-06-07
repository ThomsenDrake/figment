#!/usr/bin/env python3
"""No-network-by-default smoke proof for Figment model routing."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figment.config import FigmentConfig, load_config  # noqa: E402
from figment.model_client import MODEL_TIMEOUT_ENV  # noqa: E402
from figment.navigator import run_navigation  # noqa: E402


SMOKE_NETWORK_FLAG = "FIGMENT_SMOKE_ALLOW_NETWORK"
SMOKE_TIMEOUT_ENV = "FIGMENT_SMOKE_TIMEOUT_SECONDS"
SMOKE_TRACE_PATH_ENV = "FIGMENT_SMOKE_TRACE_PATH"
DEFAULT_SMOKE_TIMEOUT_SECONDS = "8"
REAL_LLAMA_CPP_EVAL_COMMAND = (
    "FIGMENT_MODE=local MODEL_STACK=local_4b_parakeet MODEL_BACKEND=llama_cpp "
    "LOCAL_MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 "
    "LLAMA_BASE_URL=http://127.0.0.1:8001/v1 PYTHON_DOTENV_DISABLED=true "
    "python3 scripts/run_eval.py --backend llama_cpp --model-stack local_4b_parakeet "
    "--cases data/eval/initial_handwritten_cases.jsonl "
    "--cases data/eval/adversarial_strict_cases.jsonl "
    "--cases data/eval/comprehensive_hosted_cases.jsonl "
    "--output traces/local_llama_cpp_eval_$(date -u +%Y%m%dT%H%M%SZ).jsonl"
)

BOUNDED_FIELDS = (
    "candidate_protocol_pathways",
    "missing_info_to_collect",
    "next_observations_to_collect",
    "conflicts_or_uncertainties",
    "responder_checklist",
    "handoff_note_sbar",
)

SMOKE_INTAKE = {
    "confirmed": True,
    "setting": "desert worksite",
    "patient_age": "adult",
    "pregnancy_status": "not pregnant",
    "chief_concern": "fatigue and dizziness after working outside",
    "symptoms": "thirst, dizziness, no chest pain, no trouble breathing",
    "vitals": "BP 118/76, HR 92, RR 16, alert and speaking clearly",
    "allergies": "unknown",
    "medications": "none reported",
    "available_supplies": "oral rehydration solution, shade, phone",
    "responder_note": "Synthetic smoke case for bounded navigator fields; no deterministic red flags observed.",
}

OFF_GRID_NOTE = (
    "A no-cloud/off-grid proof depends on the recorded route being self-hosted with no runtime cloud APIs. "
    "This can be the smaller local stack or self-hosted Omni on adequate local hardware; the smoke output records "
    "the configured model_id and route instead of assuming only 4B can qualify."
)


def run_smoke() -> dict[str, Any]:
    config = load_config()
    network_enabled = _truthy(os.getenv(SMOKE_NETWORK_FLAG))
    route = _route_summary(config)

    if config.model_backend == "llama_cpp" and not network_enabled:
        return {
            "status": "skipped",
            "skip_reason": "network_disabled_for_configured_route",
            "network_enabled": False,
            "network_attempted": False,
            "route": route,
            "model_id": config.active_model_id,
            "local_llm_evidence": _local_llm_evidence(
                config,
                status="skipped",
                network_attempted=False,
                validation_passed=None,
                fallback_tier=None,
            ),
            "fallback_tier": None,
            "fallback_reason": None,
            "latency_ms": 0.0,
            "validation": {"passed": None, "failures": []},
            "bounded_field_source": None,
            "bounded_fields_present": {},
            "off_grid_note": OFF_GRID_NOTE,
            "next_step": f"Set {SMOKE_NETWORK_FLAG}=true to call LLAMA_BASE_URL for a local OpenAI-compatible proof.",
        }

    if config.model_backend not in {"canned", "llama_cpp"}:
        return {
            "status": "skipped",
            "skip_reason": "unsupported_backend_for_no_cloud_smoke",
            "network_enabled": network_enabled,
            "network_attempted": False,
            "route": route,
            "model_id": config.active_model_id,
            "local_llm_evidence": _local_llm_evidence(
                config,
                status="skipped",
                network_attempted=False,
                validation_passed=None,
                fallback_tier=None,
            ),
            "fallback_tier": None,
            "fallback_reason": None,
            "latency_ms": 0.0,
            "validation": {"passed": None, "failures": []},
            "bounded_field_source": None,
            "bounded_fields_present": {},
            "off_grid_note": OFF_GRID_NOTE,
            "next_step": "Use MODEL_BACKEND=canned for no-network smoke or MODEL_BACKEND=llama_cpp with LLAMA_BASE_URL for local proof.",
        }

    trace_path = os.getenv(SMOKE_TRACE_PATH_ENV, "").strip() or None
    started = time.perf_counter()
    with _smoke_timeout_override():
        output, trace = run_navigation(
            SMOKE_INTAKE,
            [],
            config=config,
            trace_path=trace_path,
        )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    trace_payload = trace.to_dict()
    model_route = trace_payload.get("model_route") or {}
    validation = trace_payload.get("validator_result") or {}
    fallback_tier = model_route.get("fallback_tier")
    expected_tier = "configured" if config.model_backend == "llama_cpp" else "canned"
    validation_passed = validation.get("passed") is True
    status = "passed" if fallback_tier == expected_tier and validation_passed else "failed"
    bounded_field_source = "configured_model" if fallback_tier == "configured" else "canned_fallback"

    return {
        "status": status,
        "network_enabled": network_enabled,
        "network_attempted": config.model_backend == "llama_cpp",
        "route": route | {key: model_route.get(key) for key in ("fallback_tier", "fallback_reason")},
        "model_id": model_route.get("model_id") or config.active_model_id,
        "local_llm_evidence": _local_llm_evidence(
            config,
            status=status,
            network_attempted=config.model_backend == "llama_cpp",
            validation_passed=validation_passed,
            fallback_tier=fallback_tier,
        ),
        "fallback_tier": fallback_tier,
        "fallback_reason": model_route.get("fallback_reason"),
        "latency_ms": latency_ms,
        "validation": {
            "passed": validation.get("passed"),
            "failures": validation.get("failures", []),
        },
        "bounded_field_source": bounded_field_source,
        "bounded_fields_present": _bounded_fields_present(output),
        "trace_path": str(Path(trace_path)) if trace_path else None,
        "off_grid_note": OFF_GRID_NOTE,
    }


def _route_summary(config: FigmentConfig) -> dict[str, Any]:
    if config.model_backend == "llama_cpp":
        return {
            "model_stack": config.model_stack,
            "model_backend": config.model_backend,
            "model_id": config.active_model_id,
            "route_kind": "local_openai_compatible",
            "base_url": config.llama_base_url,
        }
    if config.model_backend == "canned":
        return {
            "model_stack": config.model_stack,
            "model_backend": config.model_backend,
            "model_id": config.active_model_id,
            "route_kind": "canned",
            "base_url": None,
        }
    return {
        "model_stack": config.model_stack,
        "model_backend": config.model_backend,
        "model_id": config.active_model_id,
        "route_kind": "unsupported_for_no_cloud_smoke",
        "base_url": None,
    }


def _local_llm_evidence(
    config: FigmentConfig,
    *,
    status: str,
    network_attempted: bool,
    validation_passed: bool | None,
    fallback_tier: str | None,
) -> dict[str, Any]:
    is_local_llama = config.model_backend == "llama_cpp"
    route_smoke_passed = (
        is_local_llama
        and network_attempted
        and status == "passed"
        and fallback_tier == "configured"
        and validation_passed is True
    )
    if not is_local_llama:
        proof_status = "not_local_llama_route"
    elif not network_attempted:
        proof_status = "skipped_no_network_call"
    elif route_smoke_passed:
        proof_status = "one_case_route_smoke_passed"
    else:
        proof_status = "one_case_route_smoke_failed"
    return {
        "proof_status": proof_status,
        "evidence_scope": "one_case_route_smoke" if is_local_llama else "not_local_llama",
        "configured_model_id": config.active_model_id,
        "configured_base_url": config.llama_base_url if is_local_llama else None,
        "network_attempted": network_attempted,
        "fallback_tier": fallback_tier,
        "validation_passed": validation_passed,
        "counts_as_no_cloud_route_proof": route_smoke_passed,
        "counts_as_50_case_local_llm_competence": False,
        "real_eval_command": REAL_LLAMA_CPP_EVAL_COMMAND,
    }


def _bounded_fields_present(output: dict[str, Any]) -> dict[str, bool]:
    return {field: bool(output.get(field)) for field in BOUNDED_FIELDS}


@contextmanager
def _smoke_timeout_override() -> Any:
    previous = os.environ.get(MODEL_TIMEOUT_ENV)
    if previous is None:
        os.environ[MODEL_TIMEOUT_ENV] = os.getenv(SMOKE_TIMEOUT_ENV, DEFAULT_SMOKE_TIMEOUT_SECONDS).strip() or DEFAULT_SMOKE_TIMEOUT_SECONDS
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(MODEL_TIMEOUT_ENV, None)


def _truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"1", "true", "yes", "y", "on"})


def main() -> int:
    try:
        result = run_smoke()
    except Exception as exc:
        result = {
            "status": "error",
            "error": str(exc),
            "network_enabled": _truthy(os.getenv(SMOKE_NETWORK_FLAG)),
            "network_attempted": False,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "passed":
        return 0
    if result["status"] == "skipped":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
