"""Audio intake draft helpers.

The scaffold intentionally avoids loading ASR/model dependencies. Audio is a
drafting path only; confirmed typed intake remains the source of truth.
"""

from __future__ import annotations

import re
from typing import Any

from .config import FigmentConfig, OMNI_MODEL_ID, STRETCH_AUDIO_MODEL_ID, STRETCH_TEXT_MODEL_ID, load_config
from .schemas import AudioDraft, AudioFieldSuggestion


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
        "Four-year-old at a disaster shelter with repeated watery stool, very tired, dry mouth, "
        "and no urine since this morning. I think pulse is fast but I have not taken a full set of vitals."
    ),
    "wound_infection": (
        "Adult with a leg cut from debris three days ago. The area is more painful with spreading redness "
        "and warmth. Need to check temperature and drainage."
    ),
    "pregnancy_danger_sign": (
        "Pregnant patient, about thirty-two weeks, reports severe headache and visual spots. "
        "Blood pressure was high on the first check. Needs immediate protocol review."
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
    if config.audio_backend == "parakeet_nemo" and not config.allow_stretch_stack:
        raise ValueError("Parakeet stretch path requires ALLOW_STRETCH_STACK=true")

    audio_enabled = config.enable_audio_intake and config.audio_backend != "none"
    if not audio_enabled:
        return AudioDraft(
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

    transcript = transcript.strip()
    if provider_payload is not None:
        transcript = str(provider_payload.get("transcript", transcript))
        suggestions = _provider_suggestions(provider_payload)
        missing = list(provider_payload.get("missing_or_unclear_fields", []))
        mentions = list(provider_payload.get("provisional_red_flag_mentions", []))
    elif transcript:
        suggestions = _suggest_fields(transcript)
        missing = _missing_fields(suggestions, transcript)
        mentions = [hint for hint in RED_FLAG_HINTS if hint in transcript.lower()]
    elif config.audio_backend == "canned" and not audio_file_received:
        transcript = CANNED_TRANSCRIPTS.get(case_id, CANNED_TRANSCRIPTS["pediatric_dehydration"])
        suggestions = _suggest_fields(transcript)
        missing = _missing_fields(suggestions, transcript)
        mentions = [hint for hint in RED_FLAG_HINTS if hint in transcript.lower()]
    else:
        draft = AudioDraft(
            audio_intake_path="audio_received_needs_transcript_or_model",
            audio_model_id=None,
            field_fill_model_id=None,
            audio_runtime="unprocessed_audio",
            transcript="",
            suggested_fields=[],
            missing_or_unclear_fields=["transcript_or_provider_payload"],
            provisional_red_flag_mentions=[],
            confirmed_intake_required=False,
            confirmation_status="confirmed",
            raw_audio_stored=False,
        ).to_dict()
        draft["processing_status"] = "Audio received but needs transcript/model support before field drafting."
        return draft

    audio_model_id = None
    field_fill_model_id = None
    runtime = config.audio_backend
    path = "canned_audio_demo" if config.audio_backend == "canned" else "omni_native"
    if audio_enabled and config.audio_backend == "omni_native":
        audio_model_id = OMNI_MODEL_ID
    elif audio_enabled and config.audio_backend == "parakeet_nemo":
        audio_model_id = STRETCH_AUDIO_MODEL_ID
        field_fill_model_id = STRETCH_TEXT_MODEL_ID
        path = "parakeet_rnnt_plus_text_nemotron"

    return AudioDraft(
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
        suggestions.append(
            AudioFieldSuggestion(
                field=str(item["field"]),
                draft_value=str(item.get("draft_value", "")),
                source_snippet=str(item.get("source_snippet", "")),
                source_timecode=str(item.get("source_timecode", "")),
            )
        )
    return suggestions


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
        suggestions.append(AudioFieldSuggestion(field="patient_age", draft_value="4 years", source_snippet=_snippet(text, "four")))
        suggestions.append(AudioFieldSuggestion(field="chief_concern", draft_value="possible dehydration concern after repeated watery stool", source_snippet=_snippet(text, "stool")))
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
