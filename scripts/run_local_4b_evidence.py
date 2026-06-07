#!/usr/bin/env python3
"""Capture evidence for the full-weight local 4B route."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import sys
from time import gmtime, strftime
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figment.config import FigmentConfig, NVIDIA_NEMOTRON_3_NANO_4B_BF16_MODEL_ID  # noqa: E402
from figment.trace import stable_hash  # noqa: E402
from scripts.run_eval import run_eval  # noqa: E402
from scripts.smoke_model_route import run_smoke  # noqa: E402


DEFAULT_CASE_PATHS = (
    Path("data/eval/initial_handwritten_cases.jsonl"),
    Path("data/eval/adversarial_strict_cases.jsonl"),
    Path("data/eval/comprehensive_hosted_cases.jsonl"),
)
DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_TIMEOUT_SECONDS = 45.0
FULL_WEIGHT_MODEL_REPO = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
FULL_WEIGHT_REVISION = "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"
FULL_WEIGHT_SHA256 = "55d4e2519456c4a9bddf596b0748d630e3b2ce6ff6f4c2b7ed3e07e2b00dad42"
FULL_WEIGHT_BYTES = 7_947_142_640


def run_evidence(
    *,
    base_url: str,
    model_id: str,
    output_dir: Path,
    case_paths: list[Path],
    limit: int | None,
    timeout_seconds: float,
    smoke_only: bool = False,
    force_eval: bool = False,
) -> dict[str, Any]:
    normalized_base_url = _normalize_base_url(base_url)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint_metadata = _probe_models_endpoint(normalized_base_url, timeout_seconds)
    _write_json(output_dir / "endpoint_metadata.json", endpoint_metadata)

    summary: dict[str, Any] = {
        "status": "endpoint_unavailable" if endpoint_metadata["status"] != "passed" else "endpoint_available",
        "output_dir": str(output_dir),
        "base_url": normalized_base_url,
        "model_id": model_id,
        "full_weight_artifact": {
            "repo": FULL_WEIGHT_MODEL_REPO,
            "revision": FULL_WEIGHT_REVISION,
            "model_safetensors_bytes": FULL_WEIGHT_BYTES,
            "model_safetensors_sha256": FULL_WEIGHT_SHA256,
        },
        "endpoint_metadata_path": str(output_dir / "endpoint_metadata.json"),
        "route_smoke_path": None,
        "eval_records_path": None,
        "eval_summary_path": None,
        "counts_as_no_cloud_route_proof": False,
        "counts_as_50_case_local_llm_competence": False,
    }
    if endpoint_metadata["status"] != "passed":
        _write_json(output_dir / "summary.json", summary)
        return summary

    with _local_4b_env(normalized_base_url, model_id, timeout_seconds):
        smoke = run_smoke()
    smoke_path = output_dir / "route_smoke.json"
    _write_json(smoke_path, smoke)
    summary["route_smoke_path"] = str(smoke_path)
    summary["route_smoke_status"] = smoke.get("status")
    summary["counts_as_no_cloud_route_proof"] = bool(
        (smoke.get("local_llm_evidence") or {}).get("counts_as_no_cloud_route_proof")
    )

    if smoke_only:
        summary["status"] = "smoke_passed" if smoke.get("status") == "passed" else "smoke_failed"
        _write_json(output_dir / "summary.json", summary)
        return summary

    if smoke.get("status") != "passed" and not force_eval:
        summary["status"] = "smoke_failed_eval_skipped"
        summary["eval_skip_reason"] = "route smoke did not prove configured-model validation"
        _write_json(output_dir / "summary.json", summary)
        return summary

    eval_records_path = output_dir / "local_4b_eval.jsonl"
    config = FigmentConfig(
        figment_mode="local",
        model_stack="local_4b_parakeet",
        model_backend="llama_cpp",
        audio_backend="none",
        local_model_id=model_id,
        llama_base_url=normalized_base_url,
    ).validated()
    eval_summary = run_eval(
        case_paths=case_paths,
        output_path=eval_records_path,
        config=config,
        limit=limit,
    )
    eval_summary_path = output_dir / "eval_summary.json"
    _write_json(eval_summary_path, eval_summary)
    eval_manifest = _build_eval_evidence_manifest(
        eval_records_path=eval_records_path,
        eval_summary=eval_summary,
        endpoint_metadata=endpoint_metadata,
        route_smoke=smoke,
        config=config,
    )
    eval_manifest_path = output_dir / "eval_evidence_manifest.json"
    _write_json(eval_manifest_path, eval_manifest)

    local_evidence = eval_summary.get("local_llm_evidence") or {}
    summary.update(
        {
            "status": "eval_completed",
            "eval_records_path": str(eval_records_path),
            "eval_summary_path": str(eval_summary_path),
            "eval_evidence_manifest_path": str(eval_manifest_path),
            "total_cases": eval_summary.get("total_cases"),
            "competence_successes": eval_summary.get("competence_successes"),
            "fallback_uses": eval_summary.get("fallback_uses"),
            "final_validation_successes": eval_summary.get("final_validation_successes"),
            "latency_ms": eval_manifest.get("latency_ms"),
            "trace_hash_count": eval_manifest.get("trace_hash_count"),
            "missing_trace_hash_case_ids": eval_manifest.get("missing_trace_hash_case_ids"),
            "counts_as_50_case_local_llm_competence": bool(
                local_evidence.get("counts_as_50_case_local_llm_competence")
            ),
        }
    )
    _write_json(output_dir / "summary.json", summary)
    return summary


def _probe_models_endpoint(base_url: str, timeout_seconds: float) -> dict[str, Any]:
    models_url = _models_url(base_url)
    request = urllib.request.Request(models_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "models_url": models_url,
            "error": str(exc),
        }
    return {
        "status": "passed",
        "models_url": models_url,
        "payload": payload,
    }


def _normalize_base_url(base_url: str) -> str:
    stripped = base_url.strip().rstrip("/")
    if not stripped:
        return DEFAULT_BASE_URL
    parts = urllib.parse.urlsplit(stripped)
    path = parts.path.rstrip("/")
    for suffix in ("/chat/completions", "/models"):
        if path.endswith(suffix):
            path = path[: -len(suffix)] or "/"
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path.rstrip("/"), "", ""))


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _build_eval_evidence_manifest(
    *,
    eval_records_path: Path,
    eval_summary: dict[str, Any],
    endpoint_metadata: dict[str, Any],
    route_smoke: dict[str, Any],
    config: FigmentConfig,
) -> dict[str, Any]:
    records = _read_eval_records(eval_records_path)
    endpoint_payload = endpoint_metadata.get("payload") if isinstance(endpoint_metadata.get("payload"), dict) else {}
    smoke_evidence = route_smoke.get("local_llm_evidence") if isinstance(route_smoke.get("local_llm_evidence"), dict) else {}
    local_evidence = (
        eval_summary.get("local_llm_evidence")
        if isinstance(eval_summary.get("local_llm_evidence"), dict)
        else {}
    )
    latency_values = [
        float(record["latency_ms"])
        for record in records
        if isinstance(record.get("latency_ms"), int | float)
    ]
    trace_entries = [
        {"case_id": record.get("case_id"), "trace_hash": record.get("trace_hash")}
        for record in records
        if record.get("trace_hash")
    ]
    missing_trace_hash_case_ids = [
        str(record.get("case_id"))
        for record in records
        if not record.get("trace_hash")
    ]
    return {
        "evidence_version": 1,
        "manifest_hash_inputs": {
            "eval_records_sha256": _file_sha256(eval_records_path),
            "endpoint_payload_hash": stable_hash(endpoint_payload),
            "route_smoke_hash": stable_hash(route_smoke),
        },
        "model_server_metadata": {
            "base_url": config.llama_base_url,
            "models_url": endpoint_metadata.get("models_url"),
            "models_status": endpoint_metadata.get("status"),
            "advertised_model_ids": _advertised_model_ids(endpoint_payload),
            "models_payload_hash": stable_hash(endpoint_payload),
        },
        "configured_route": {
            "figment_mode": config.figment_mode,
            "model_stack": config.model_stack,
            "model_backend": config.model_backend,
            "audio_backend": config.audio_backend,
            "active_model_id": config.active_model_id,
            "local_model_id": config.local_model_id,
            "llama_base_url": config.llama_base_url,
            "full_weight_artifact": {
                "repo": FULL_WEIGHT_MODEL_REPO,
                "revision": FULL_WEIGHT_REVISION,
                "model_safetensors_bytes": FULL_WEIGHT_BYTES,
                "model_safetensors_sha256": FULL_WEIGHT_SHA256,
            },
        },
        "no_cloud_evidence": {
            "model_backend_is_local_openai_compatible": config.model_backend == "llama_cpp",
            "hosted_backend_disabled_for_eval": config.model_backend != "hosted_omni",
            "endpoint_models_probe_passed": endpoint_metadata.get("status") == "passed",
            "route_smoke_counts_as_no_cloud_route_proof": bool(
                smoke_evidence.get("counts_as_no_cloud_route_proof")
            ),
            "counts_as_50_case_local_llm_eval": bool(
                local_evidence.get("counts_as_50_case_local_llm_eval")
            ),
            "counts_as_50_case_local_llm_competence": bool(
                local_evidence.get("counts_as_50_case_local_llm_competence")
            ),
            "base_url_host": _base_url_host(config.llama_base_url),
            "base_url_host_class": _base_url_host_class(config.llama_base_url),
            "note": (
                "This manifest proves Figment used MODEL_BACKEND=llama_cpp against the configured "
                "OpenAI-compatible endpoint. For LAN or other local hosts, keep runtime evidence beside "
                "this bundle if judges need independent network-local attestation."
            ),
        },
        "score_summary": {
            "total_cases": eval_summary.get("total_cases", len(records)),
            "raw_configured_model_successes": eval_summary.get("raw_configured_model_successes", 0),
            "repair_successes": eval_summary.get("repair_successes", 0),
            "competence_successes": eval_summary.get("competence_successes", 0),
            "fallback_uses": eval_summary.get("fallback_uses", 0),
            "canned_fallback_uses": eval_summary.get("canned_fallback_uses", 0),
            "final_validation_successes": eval_summary.get("final_validation_successes", 0),
            "expected_label_successes": eval_summary.get("expected_label_successes"),
            "expected_label_failures": eval_summary.get("expected_label_failures"),
        },
        "case_ids": {
            "raw_success": _case_ids(records, "raw_configured_model_success"),
            "repair_success": _case_ids(records, "repair_success"),
            "full_fallback": _case_ids(records, "canned_fallback_used"),
            "competence_success": _case_ids(records, "competence_success"),
        },
        "field_provenance": {
            "records_with_field_provenance": eval_summary.get("records_with_field_provenance", 0),
            "field_provenance_fields": eval_summary.get("field_provenance_fields", 0),
            "field_provenance_counts": eval_summary.get("field_provenance_counts", {}),
            "model_retained_field_count": eval_summary.get("model_retained_field_count", 0),
            "visible_field_provenance_count": eval_summary.get("visible_field_provenance_count", 0),
            "model_visible_field_count": eval_summary.get("model_visible_field_count", 0),
            "deterministic_patch_count": eval_summary.get("deterministic_patch_count", 0),
            "model_field_pass_rate": eval_summary.get("model_field_pass_rate", 0),
            "model_visible_fields_retained": eval_summary.get("model_visible_fields_retained", 0),
        },
        "latency_ms": _latency_summary(latency_values),
        "trace_hash_count": len(trace_entries),
        "trace_hashes": trace_entries,
        "missing_trace_hash_case_ids": missing_trace_hash_case_ids,
    }


def _read_eval_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _advertised_model_ids(endpoint_payload: dict[str, Any]) -> list[str]:
    data = endpoint_payload.get("data")
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    return ids


def _base_url_host(base_url: str) -> str | None:
    return urllib.parse.urlsplit(base_url).hostname


def _base_url_host_class(base_url: str) -> str:
    host = _base_url_host(base_url)
    if not host:
        return "unknown"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        lowered = host.lower()
        if lowered == "localhost" or lowered.endswith(".localhost"):
            return "loopback"
        if lowered.endswith(".local"):
            return "mdns_local"
        return "dns_name_unclassified"
    if address.is_loopback:
        return "loopback"
    if address.is_private:
        return "private_lan"
    if address.is_link_local:
        return "link_local"
    return "public_or_unclassified_ip"


def _case_ids(records: list[dict[str, Any]], flag: str) -> list[str]:
    return [str(record.get("case_id")) for record in records if record.get(flag)]


def _latency_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "case_count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    sorted_values = sorted(values)
    return {
        "case_count": len(values),
        "min": round(sorted_values[0], 3),
        "mean": round(sum(sorted_values) / len(sorted_values), 3),
        "p50": round(_percentile(sorted_values, 0.50), 3),
        "p95": round(_percentile(sorted_values, 0.95), 3),
        "max": round(sorted_values[-1], 3),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = percentile * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


@contextmanager
def _local_4b_env(base_url: str, model_id: str, timeout_seconds: float) -> Any:
    overrides = {
        "PYTHON_DOTENV_DISABLED": "true",
        "FIGMENT_MODE": "local",
        "MODEL_STACK": "local_4b_parakeet",
        "MODEL_BACKEND": "llama_cpp",
        "AUDIO_BACKEND": "none",
        "LOCAL_MODEL_ID": model_id,
        "LLAMA_BASE_URL": base_url,
        "FIGMENT_SMOKE_ALLOW_NETWORK": "true",
        "FIGMENT_MODEL_TIMEOUT_SECONDS": str(timeout_seconds),
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")


def _default_output_dir() -> Path:
    stamp = strftime("%Y%m%dT%H%M%SZ", gmtime())
    return Path("traces") / f"local_4b_evidence_{stamp}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("LLAMA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model-id", default=NVIDIA_NEMOTRON_3_NANO_4B_BF16_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cases", action="append", default=None, help="JSONL eval case path. Repeatable.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir or _default_output_dir()
    case_paths = [Path(path) for path in args.cases] if args.cases else list(DEFAULT_CASE_PATHS)
    summary = run_evidence(
        base_url=args.base_url,
        model_id=args.model_id,
        output_dir=output_dir,
        case_paths=case_paths,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
        smoke_only=args.smoke_only,
        force_eval=args.force_eval,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] in {"eval_completed", "smoke_passed"}:
        return 0
    if summary["status"] == "endpoint_unavailable":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
