"""Deterministic safety validators."""

from __future__ import annotations

import re
from typing import Any

from .schemas import ValidationResult, urgency_at_least


FORBIDDEN_PATTERNS = [
    re.compile(r"\bdiagnos(?:e|is|ed)\b", re.IGNORECASE),
    re.compile(r"\bprescrib(?:e|ing|ed)\b", re.IGNORECASE),
    re.compile(r"\bstart (?:an )?(?:antibiotic|opioid|insulin)\b", re.IGNORECASE),
    re.compile(r"\bgive \d+(?:\.\d+)?\s*(?:mg|mcg|ml)\b", re.IGNORECASE),
]


def _text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_text_values(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_text_values(item))
        return out
    return []


def validate_audio_ready(audio_draft: dict[str, Any] | None) -> ValidationResult:
    result = ValidationResult(passed=True)
    if not audio_draft:
        return result
    if audio_draft.get("raw_audio_stored") is True:
        result.add("raw audio must not be stored in traces")
    if audio_draft.get("confirmed_intake_required", True) and audio_draft.get("confirmation_status") != "confirmed":
        result.add("audio-derived intake must be confirmed before navigation")
    for suggestion in audio_draft.get("suggested_fields", []):
        if suggestion.get("status") == "audio_draft":
            result.add(f"audio suggestion for {suggestion.get('field', '<unknown>')} is still provisional")
    return result


def validate_confirmed_intake(intake: dict[str, Any]) -> ValidationResult:
    result = ValidationResult(passed=True)
    if not intake.get("confirmed"):
        result.add("intake must be confirmed before rules or navigation run")
    if not intake.get("chief_concern") and not intake.get("responder_note"):
        result.add("chief_concern or responder_note is required")
    return result


def urgency_floor_from_rules(rule_results: list[dict[str, Any]]) -> str:
    order = {"routine": 0, "monitor": 1, "urgent": 2, "emergency": 3}
    floor = "routine"
    for rule in rule_results:
        urgency = str(rule.get("urgency", "routine"))
        if order.get(urgency, -1) > order[floor]:
            floor = urgency
    return floor


def validate_navigator_output(
    output: dict[str, Any],
    known_card_ids: set[str],
    urgency_floor: str = "routine",
) -> ValidationResult:
    result = ValidationResult(passed=True)
    if not isinstance(output, dict):
        return ValidationResult(False, ["navigator output is not an object"])

    urgency = output.get("protocol_urgency")
    if urgency not in {"routine", "monitor", "urgent", "emergency"}:
        result.add("protocol_urgency is missing or invalid")
    elif not urgency_at_least(urgency, urgency_floor):
        result.add(f"protocol_urgency {urgency} is below deterministic floor {urgency_floor}")

    known_cards = set(known_card_ids)
    source_cards = output.get("source_cards")
    if not isinstance(source_cards, list) or not source_cards:
        result.add("source_cards must be a non-empty list")
        source_cards = []
    unknown = sorted(set(source_cards) - known_cards)
    if unknown:
        result.add(f"unknown source_cards: {', '.join(unknown)}")

    for pathway in output.get("candidate_protocol_pathways", []):
        card_id = pathway.get("card_id") if isinstance(pathway, dict) else None
        if not card_id:
            result.add("candidate_protocol_pathway is missing card_id")
        elif card_id not in source_cards:
            result.add(f"candidate pathway {card_id} is not cited in source_cards")

    all_text = "\n".join(_text_values(output))
    for pattern in FORBIDDEN_PATTERNS:
        match = pattern.search(all_text)
        if match:
            result.add(f"forbidden clinical language: {match.group(0)}")
    return result
