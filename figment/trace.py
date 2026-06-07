"""Trace helpers for visible, non-secret Figment execution logs."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any


RAW_AUDIO_KEY_MARKERS = ("raw_audio", "audio_data", "blob", "base64", "data_url", "dataurl")
FILENAME_KEYS = {"audio_filename", "uploaded_filename", "upload_filename", "filename", "file_name"}
AUDIO_FILENAME_PATTERN = re.compile(r"\b[\w.-]+\.(?:wav|mp3|m4a|flac|ogg|aac|webm)\b", re.IGNORECASE)
FINAL_ROUTE_LABELS = {
    "live_model_generated": "Live model generated",
    "model_repaired": "Model repaired",
    "model_with_deterministic_patches": "Model with deterministic patches",
    "validation_fallback": "Validation fallback",
    "canned_backend": "Canned backend",
    "unknown": "Unknown",
}
DETERMINISTIC_FALLBACK_PROVENANCE = "deterministic_fallback"
MODEL_PROVENANCE_VALUES = {"model_raw", "model_repaired"}


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def scrub_audio_metadata(audio: dict[str, Any] | None) -> dict[str, Any] | None:
    if audio is None:
        return None
    scrubbed = _scrub_audio_value(audio)
    if isinstance(scrubbed, dict):
        scrubbed["raw_audio_stored"] = False
        return scrubbed
    return None


def _scrub_audio_value(value: Any) -> Any:
    if isinstance(value, dict):
        scrubbed: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized != "raw_audio_stored" and any(marker in normalized for marker in RAW_AUDIO_KEY_MARKERS):
                continue
            if normalized in FILENAME_KEYS:
                scrubbed[key] = None
                continue
            scrubbed[key] = _scrub_audio_value(item)
        return scrubbed
    if isinstance(value, list):
        return [_scrub_audio_value(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    if isinstance(value, str):
        lowered = value.lower()
        if lowered.startswith("data:") or ";base64," in lowered or AUDIO_FILENAME_PATTERN.search(value):
            return "[redacted audio metadata]"
    return value


def normalize_trace_payload(trace: dict[str, Any]) -> dict[str, Any]:
    payload = dict(trace or {})
    payload["audio"] = scrub_audio_metadata(payload.get("audio"))
    payload["raw_audio_stored"] = False
    route = payload.get("model_route") if isinstance(payload.get("model_route"), dict) else {}
    validator = payload.get("validator_result") if isinstance(payload.get("validator_result"), dict) else {}
    events = payload.get("events") if isinstance(payload.get("events"), list) else []
    field_provenance = payload.get("field_provenance") if isinstance(payload.get("field_provenance"), dict) else {}
    payload["field_provenance_summary"] = summarize_field_provenance(field_provenance)
    payload["model_route"] = derive_model_route(route, validator, events, field_provenance=field_provenance)
    return payload


def derive_model_route(
    model_route: dict[str, Any] | None,
    validator_result: dict[str, Any] | None = None,
    events: list[Any] | None = None,
    *,
    field_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    route = dict(model_route or {})
    validator = validator_result or {}
    event_text = " | ".join(str(event).lower() for event in (events or []))
    raw_route = str(route.get("raw_route") or route.get("model_backend") or "unknown")
    fallback_reason = route.get("fallback_reason")
    fallback_tier = route.get("fallback_tier")
    final_route = _derive_final_route(
        raw_route,
        fallback_reason=fallback_reason,
        fallback_tier=fallback_tier,
        event_text=event_text,
        field_level_fallback_used=bool(route.get("field_level_fallback_used"))
        or _has_deterministic_patches(field_provenance),
    )
    route["raw_route"] = raw_route
    route["final_route"] = final_route
    route["runtime_contribution"] = final_route
    route["runtime_label"] = runtime_route_label(final_route)
    route["validation_status"] = _validation_status(validator)
    return route


def runtime_route_label(route_or_name: dict[str, Any] | str | None) -> str:
    if isinstance(route_or_name, dict):
        route_name = str(route_or_name.get("final_route") or route_or_name.get("runtime_contribution") or "unknown")
    elif route_or_name:
        route_name = str(route_or_name)
    else:
        route_name = "unknown"
    return FINAL_ROUTE_LABELS.get(route_name, route_name.replace("_", " ").title())


def _derive_final_route(
    raw_route: str,
    *,
    fallback_reason: Any,
    fallback_tier: Any,
    event_text: str,
    field_level_fallback_used: bool = False,
) -> str:
    if raw_route == "canned":
        return "canned_backend"
    if fallback_reason == "navigator_validation_failure":
        return "validation_fallback"
    if fallback_reason:
        return "canned_backend"
    if field_level_fallback_used or "deterministic patches" in event_text or "field-level deterministic" in event_text:
        return "model_with_deterministic_patches"
    if "repaired" in event_text:
        return "model_repaired"
    if fallback_tier == "canned":
        return "canned_backend"
    if raw_route in {"hosted_omni", "llama_cpp"}:
        return "live_model_generated"
    return raw_route or "unknown"


def _validation_status(validator: dict[str, Any]) -> str:
    if validator.get("passed") is True:
        return "passed"
    if validator.get("passed") is False:
        return "failed"
    return "unknown"


def summarize_field_provenance(provenance: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        provenance = {}
    counts: dict[str, int] = {}
    for value in provenance.values():
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return {
        "counts": dict(sorted(counts.items())),
        "total_fields": sum(counts.values()),
        "deterministic_patch_count": counts.get(DETERMINISTIC_FALLBACK_PROVENANCE, 0),
        "model_retained_count": sum(counts.get(value, 0) for value in MODEL_PROVENANCE_VALUES),
    }


def _has_deterministic_patches(provenance: dict[str, Any] | None) -> bool:
    if not isinstance(provenance, dict):
        return False
    return any(str(value) == DETERMINISTIC_FALLBACK_PROVENANCE for value in provenance.values())


@dataclass
class FigmentTrace:
    input_captured: dict[str, Any]
    red_flags: list[dict[str, Any]]
    retrieved_card_ids: list[str]
    prompt_template_hash: str
    model_route: dict[str, Any]
    navigator_output: dict[str, Any]
    validator_result: dict[str, Any]
    field_provenance: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] | None = None
    raw_audio_stored: bool = False
    events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return normalize_trace_payload(payload)


def write_trace(trace: FigmentTrace | dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = trace.to_dict() if isinstance(trace, FigmentTrace) else trace
    if isinstance(payload, dict):
        payload = normalize_trace_payload(payload)
    target.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
    return target
