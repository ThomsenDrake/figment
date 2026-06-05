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


@dataclass
class FigmentTrace:
    input_captured: dict[str, Any]
    red_flags: list[dict[str, Any]]
    retrieved_card_ids: list[str]
    prompt_template_hash: str
    model_route: dict[str, Any]
    navigator_output: dict[str, Any]
    validator_result: dict[str, Any]
    audio: dict[str, Any] | None = None
    raw_audio_stored: bool = False
    events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["audio"] = scrub_audio_metadata(payload.get("audio"))
        payload["raw_audio_stored"] = False
        return payload


def write_trace(trace: FigmentTrace | dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = trace.to_dict() if isinstance(trace, FigmentTrace) else trace
    if isinstance(payload, dict):
        payload["audio"] = scrub_audio_metadata(payload.get("audio"))
        payload["raw_audio_stored"] = False
    target.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
    return target
