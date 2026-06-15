"""Audio intake draft helpers.

The scaffold intentionally avoids loading ASR/model dependencies. Audio is a
drafting path only; confirmed typed intake remains the source of truth.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

from .config import (
    FigmentConfig,
    NVIDIA_NEMOTRON_3_NANO_4B_BF16_MODEL_ID,
    OMNI_MODEL_ID,
    load_config,
)
from .schemas import AudioDraft, AudioFieldSuggestion


LOCAL_PARAKEET_AUDIO_BACKENDS = {"parakeet_nemo", "local_4b_parakeet"}
LOCAL_PARAKEET_RUNTIME = "local_4b_parakeet"
LOCAL_PARAKEET_PATH = "parakeet_asr_plus_text_nemotron"
DEMO_AUDIO_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "demo_audio" / "manifest.json"
ALLOWED_AUDIO_SUGGESTION_FIELDS = {
    "setting",
    "patient_age",
    "pregnancy_status",
    "chief_concern",
    "symptoms",
    "vitals",
    "allergies",
    "medications",
    "available_supplies",
    "responder_note",
}
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
NON_INFORMATIVE_DRAFT_VALUES = {
    "",
    "n/a",
    "na",
    "not answerable",
    "not available",
    "not mentioned",
    "not provided",
    "not specified",
    "unable to determine",
    "unclear",
}

RED_FLAG_HINTS = (
    "chest pain",
    "trouble breathing",
    "shortness of breath",
    "bleeding",
    "lethargic",
    "confusion",
    "slurred speech",
    "no urine",
    "severe headache",
    "visual spots",
    "high blood pressure",
    "spreading redness",
)

CANNED_TRANSCRIPTS = {
    "pediatric_dehydration": (
        "Seven year old at a shelter clinic after flood cleanup. Child cannot keep fluids down, is lethargic, "
        "has a very dry mouth, and has no urine since morning. Temperature and blood pressure are missing. "
        "Supplies include oral rehydration solution, radio, and transport team."
    ),
    "wound_infection": (
        "Forty three year old at a mobile clinic with a leg cut from debris three days ago. The wound is getting "
        "worse with spreading redness, swelling, and foul drainage. Temperature is unknown. Supplies are clean "
        "dressings and radio."
    ),
    "pregnancy_danger_sign": (
        "Twenty nine year old pregnant patient at a rural clinic with vaginal bleeding, severe headache, and "
        "dizziness. Blood pressure is not available. She reports a prenatal vitamin. Phone and transport contact "
        "are available."
    ),
}


def draft_audio_intake(
    transcript: str = "",
    config: FigmentConfig | None = None,
    *,
    case_id: str = "pediatric_dehydration",
    provider_payload: dict[str, Any] | None = None,
    audio_file_received: bool = False,
) -> dict[str, Any]:
    config = (config or load_config()).validated()
    if _is_local_parakeet_backend(config) and not _local_asr_allowed(config):
        raise ValueError("Local Parakeet path requires ALLOW_LOCAL_ASR=true")

    audio_enabled = config.enable_audio_intake and config.audio_backend != "none"
    if not audio_enabled:
        draft = AudioDraft(
            audio_intake_path="typed_only",
            audio_model_id=None,
            field_fill_model_id=None,
            audio_runtime="none",
            transcript="",
            suggested_fields=[],
            missing_or_unclear_fields=[],
            provisional_red_flag_mentions=[],
            confirmed_intake_required=False,
            confirmation_status="confirmed",
            raw_audio_stored=False,
        ).to_dict()
        draft["draft_source"] = "typed_intake_only"
        draft["transcript_source"] = "typed_intake_form"
        draft["audio_source"] = "none"
        return draft

    transcript = _clean_text(transcript)
    provider_payload_used = False
    typed_transcript_used = False
    canned_demo_used = False
    transcript_source = "none"
    audio_source = "none"
    if _has_provider_payload(provider_payload):
        provider_payload_used = True
        transcript = _clean_text(provider_payload.get("transcript", transcript))
        if _is_local_parakeet_backend(config):
            transcript_source = "local_parakeet_asr_provider"
            audio_source = "local_parakeet_asr_payload"
        elif config.audio_backend == "omni_native":
            transcript_source = "hosted_omni_provider"
            audio_source = "hosted_omni_audio_payload"
        else:
            transcript_source = "provider_payload"
            audio_source = "provider_payload"
        suggestions = _provider_suggestions(provider_payload)
        if not suggestions and transcript:
            suggestions = _suggest_fields(transcript)
        if not transcript and not suggestions:
            return _unprocessed_audio_draft(
                "Audio provider payload did not include a transcript or valid field suggestions.",
                missing_field="transcript_or_valid_provider_suggestions",
            )
        missing = _string_list(provider_payload.get("missing_or_unclear_fields", []))
        if not missing:
            missing = _missing_fields(suggestions, transcript)
        mentions = _merged_strings(
            _string_list(provider_payload.get("provisional_red_flag_mentions", [])),
            _red_flag_mentions(transcript),
        )
    elif transcript:
        typed_transcript_used = True
        transcript_source = "typed_transcript"
        audio_source = "typed_text_only"
        suggestions = _suggest_fields(transcript)
        missing = _missing_fields(suggestions, transcript)
        mentions = _red_flag_mentions(transcript)
    elif config.audio_backend == "canned" and not audio_file_received:
        canned_demo_used = True
        transcript = _canned_transcript(case_id)
        transcript_source = "synthetic_demo_manifest"
        audio_source = "committed_synthetic_demo_asset"
        suggestions = _suggest_fields(transcript)
        missing = _missing_fields(suggestions, transcript)
        mentions = _red_flag_mentions(transcript)
    else:
        return _unprocessed_audio_draft("Audio received but needs transcript/model support before field drafting.")

    audio_model_id = None
    field_fill_model_id = None
    runtime = "typed_transcript_heuristic" if typed_transcript_used else config.audio_backend
    path = "typed_transcript_heuristic" if typed_transcript_used else "omni_native"
    draft_source = "typed_transcript_heuristic" if typed_transcript_used else "provider_payload"
    if canned_demo_used:
        path = "canned_audio_demo"
        runtime = "canned"
        draft_source = "canned_audio_demo"
    elif provider_payload_used and audio_enabled and config.audio_backend == "omni_native":
        audio_model_id = OMNI_MODEL_ID
        draft_source = "omni_audio_provider"
    elif provider_payload_used and audio_enabled and _is_local_parakeet_backend(config):
        audio_model_id = config.audio_model_id
        field_fill_model_id = _local_field_fill_model_id(config)
        runtime = LOCAL_PARAKEET_RUNTIME
        path = LOCAL_PARAKEET_PATH
        draft_source = "parakeet_asr_provider"

    draft = AudioDraft(
        audio_intake_path=path,
        audio_model_id=audio_model_id,
        field_fill_model_id=field_fill_model_id,
        audio_runtime=runtime,
        transcript=transcript,
        suggested_fields=suggestions,
        missing_or_unclear_fields=missing,
        provisional_red_flag_mentions=mentions,
        confirmed_intake_required=True,
        confirmation_status="unconfirmed",
        raw_audio_stored=False,
    ).to_dict()
    draft["draft_source"] = draft_source
    draft["transcript_source"] = transcript_source
    draft["audio_source"] = audio_source
    return draft


def _unprocessed_audio_draft(
    processing_status: str,
    *,
    missing_field: str = "transcript_or_provider_payload",
) -> dict[str, Any]:
    draft = AudioDraft(
        audio_intake_path="audio_received_needs_transcript_or_model",
        audio_model_id=None,
        field_fill_model_id=None,
        audio_runtime="unprocessed_audio",
        transcript="",
        suggested_fields=[],
        missing_or_unclear_fields=[missing_field],
        provisional_red_flag_mentions=[],
        confirmed_intake_required=False,
        confirmation_status="confirmed",
        raw_audio_stored=False,
    ).to_dict()
    draft["processing_status"] = processing_status
    draft["draft_source"] = "unprocessed_audio"
    draft["transcript_source"] = "none"
    draft["audio_source"] = "unprocessed_audio"
    return draft


def _canned_transcript(case_id: str) -> str:
    manifest_transcripts = _demo_manifest_source_scripts()
    return manifest_transcripts.get(case_id) or CANNED_TRANSCRIPTS.get(case_id, CANNED_TRANSCRIPTS["pediatric_dehydration"])


@lru_cache(maxsize=1)
def _demo_manifest_source_scripts() -> dict[str, str]:
    try:
        payload = json.loads(DEMO_AUDIO_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    transcripts: dict[str, str] = {}
    for item in payload.get("cases", []):
        if not isinstance(item, dict):
            continue
        slug = _clean_text(item.get("slug", ""))
        source_script = _clean_text(item.get("source_script", "")) or _clean_text(item.get("voxtral_transcript", ""))
        if slug and source_script:
            transcripts[slug] = source_script
    return transcripts


def confirm_audio_draft(
    intake: dict[str, Any],
    audio_draft: dict[str, Any],
    *,
    accept: bool = True,
    edits: dict[str, str] | None = None,
    reject_fields: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(intake)
    confirmed = dict(audio_draft)
    edits = edits or {}
    reject_fields = reject_fields or set()
    suggestions = []
    for suggestion in confirmed.get("suggested_fields", []):
        item = dict(suggestion)
        field = item.get("field")
        if field in reject_fields:
            item["status"] = "rejected"
            item["needs_confirmation"] = False
        elif field in edits:
            item["draft_value"] = edits[field]
            if field:
                updated[field] = edits[field]
            item["status"] = "edited"
            item["needs_confirmation"] = False
        elif accept:
            field = item.get("field")
            value = item.get("draft_value", "")
            if field and value and not updated.get(field):
                updated[field] = value
            item["status"] = "accepted"
            item["needs_confirmation"] = False
        else:
            item["status"] = "rejected"
            item["needs_confirmation"] = False
        suggestions.append(item)
    confirmed["suggested_fields"] = suggestions
    confirmed["confirmation_status"] = "confirmed"
    confirmed["confirmed_intake_required"] = True
    confirmed["raw_audio_stored"] = False
    return updated, confirmed


def _provider_suggestions(payload: dict[str, Any]) -> list[AudioFieldSuggestion]:
    suggestions = []
    for item in payload.get("suggested_fields", []):
        if not isinstance(item, dict) or not item.get("field"):
            continue
        field = _clean_text(item.get("field", ""))
        if field not in ALLOWED_AUDIO_SUGGESTION_FIELDS:
            continue
        draft_value = _clean_text(item.get("draft_value", ""))
        if not _is_informative_draft_value(draft_value):
            continue
        suggestions.append(
            AudioFieldSuggestion(
                field=field,
                draft_value=draft_value,
                source_snippet=_clean_text(item.get("source_snippet", "")),
                source_timecode=_clean_text(item.get("source_timecode", "")),
            )
        )
    return suggestions


def _has_provider_payload(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if _clean_text(payload.get("transcript", "")):
        return True
    suggestions = payload.get("suggested_fields", [])
    if isinstance(suggestions, list):
        for item in suggestions:
            if isinstance(item, dict) and _clean_text(item.get("field", "")):
                return True
    return bool(
        _string_list(payload.get("missing_or_unclear_fields", []))
        or _string_list(payload.get("provisional_red_flag_mentions", []))
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_clean_text(item) for item in value if _clean_text(item)]


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _is_informative_draft_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in NON_INFORMATIVE_DRAFT_VALUES:
        return False
    return not normalized.startswith("provisional intake draft")


def _red_flag_mentions(transcript: str) -> list[str]:
    lower = transcript.lower()
    return [hint for hint in RED_FLAG_HINTS if hint in lower]


def _merged_strings(*values: list[str]) -> list[str]:
    merged = []
    seen = set()
    for value_list in values:
        for value in value_list:
            normalized = value.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(value)
    return merged


def _is_local_parakeet_backend(config: FigmentConfig) -> bool:
    return str(getattr(config, "audio_backend", "")) in LOCAL_PARAKEET_AUDIO_BACKENDS


def _local_asr_allowed(config: FigmentConfig) -> bool:
    return bool(getattr(config, "allow_local_asr", False) or getattr(config, "allow_stretch_stack", False))


def _local_field_fill_model_id(config: FigmentConfig) -> str:
    local_model_id = _clean_text(getattr(config, "local_model_id", ""))
    if local_model_id and "omni" not in local_model_id.lower():
        return local_model_id
    active_model_id = _clean_text(getattr(config, "active_model_id", ""))
    if active_model_id and "4b" in active_model_id.lower():
        return active_model_id
    return NVIDIA_NEMOTRON_3_NANO_4B_BF16_MODEL_ID


def _suggest_fields(transcript: str) -> list[AudioFieldSuggestion]:
    text = transcript.strip()
    if not text:
        return []
    suggestions = [AudioFieldSuggestion(field="responder_note", draft_value=text, source_snippet=text[:160])]
    lower = text.lower()
    if "chest pain" in lower:
        suggestions.append(AudioFieldSuggestion(field="chief_concern", draft_value="chest pain", source_snippet="chest pain"))
    elif "wound" in lower:
        suggestions.append(AudioFieldSuggestion(field="chief_concern", draft_value="wound concern after disaster injury", source_snippet=_snippet(text, "wound")))
    elif "preg" in lower or "bleeding" in lower:
        suggestions.append(AudioFieldSuggestion(field="chief_concern", draft_value="pregnancy danger sign concern", source_snippet=_snippet(text, "bleeding")))
        suggestions.append(AudioFieldSuggestion(field="pregnancy_status", draft_value="pregnant", source_snippet=_snippet(text, "preg")))
    elif "watery stool" in lower or "no urine" in lower:
        age = _age_from_transcript(text) or "child"
        concern = "possible dehydration concern after repeated watery stool" if "watery stool" in lower else "possible dehydration concern"
        concern_term = "stool" if "watery stool" in lower else "no urine"
        suggestions.append(AudioFieldSuggestion(field="patient_age", draft_value=age, source_snippet=_snippet(text, "year")))
        suggestions.append(AudioFieldSuggestion(field="chief_concern", draft_value=concern, source_snippet=_snippet(text, concern_term)))
        suggestions.append(AudioFieldSuggestion(field="symptoms", draft_value="very tired, dry mouth, no urine since morning", source_snippet=_snippet(text, "no urine")))
    if "trouble breathing" in lower or "shortness of breath" in lower:
        suggestions.append(AudioFieldSuggestion(field="symptoms", draft_value="trouble breathing", source_snippet=_snippet(text, "breath")))
    if "visual spots" in lower or "severe headache" in lower:
        suggestions.append(AudioFieldSuggestion(field="symptoms", draft_value="severe headache and visual spots", source_snippet=_snippet(text, "headache")))
    if "blood pressure" in lower:
        suggestions.append(AudioFieldSuggestion(field="vitals", draft_value="blood pressure high on first check; repeat value pending", source_snippet=_snippet(text, "blood pressure")))
    if "pulse is fast" in lower:
        suggestions.append(AudioFieldSuggestion(field="vitals", draft_value="pulse described as fast; full vitals not yet taken", source_snippet=_snippet(text, "pulse")))
    return suggestions


def _missing_fields(suggestions: list[AudioFieldSuggestion], transcript: str) -> list[str]:
    present = {suggestion.field for suggestion in suggestions}
    missing = []
    for field in ("patient_age", "vitals", "allergies", "medications", "available_supplies"):
        if field not in present:
            missing.append(field)
    if "not taken" in transcript.lower() and "vitals" not in missing:
        missing.append("vitals")
    return missing


def _snippet(text: str, term: str) -> str:
    match = re.search(term, text, flags=re.IGNORECASE)
    if not match:
        return text[:120]
    start = max(0, match.start() - 40)
    end = min(len(text), match.end() + 40)
    return text[start:end]


def _age_from_transcript(text: str) -> str:
    match = re.search(r"\b(\d{1,3})\s*[- ]?year[- ]old\b", text, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)} years"
    match = re.search(r"\b([a-z]+(?:[- ][a-z]+)?)\s*[- ]?year[- ]old\b", text, flags=re.IGNORECASE)
    if not match:
        return ""
    value = _number_words_value(match.group(1).lower())
    return f"{value} years" if value else ""


def _number_words_value(value: str) -> int:
    parts = value.replace("-", " ").split()
    total = 0
    for part in parts:
        if part not in NUMBER_WORDS:
            return 0
        total += NUMBER_WORDS[part]
    return total
