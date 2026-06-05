"""Grounded SBAR rendering for validated Figment navigator output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SBAR_FIELDS = (
    ("situation", "S"),
    ("background", "B"),
    ("assessment_observations_only", "A"),
    ("handoff_request", "R"),
)


def _validation_passed(validation_result: Any) -> bool:
    if validation_result is None:
        return True
    if hasattr(validation_result, "passed"):
        return bool(validation_result.passed)
    if isinstance(validation_result, Mapping):
        return bool(validation_result.get("passed"))
    return False


def sbar_sections(navigator_output: Mapping[str, Any], validation_result: Any = None) -> dict[str, str]:
    """Return SBAR sections from a navigator output that has passed validation."""
    if not _validation_passed(validation_result):
        raise ValueError("cannot render SBAR from navigator output that failed validation")
    raw_sbar = navigator_output.get("handoff_note_sbar")
    if not isinstance(raw_sbar, Mapping):
        raise ValueError("navigator output is missing handoff_note_sbar")

    sections: dict[str, str] = {}
    missing: list[str] = []
    for field, _label in SBAR_FIELDS:
        value = str(raw_sbar.get(field, "")).strip()
        if not value:
            missing.append(field)
        sections[field] = value
    if missing:
        raise ValueError(f"handoff_note_sbar is missing: {', '.join(missing)}")
    return sections


def render_sbar(navigator_output: Mapping[str, Any], validation_result: Any = None) -> str:
    """Render a factual SBAR note without adding facts beyond navigator output."""
    sections = sbar_sections(navigator_output, validation_result)
    lines = [
        f"S: {sections['situation']}",
        f"B: {sections['background']}",
        f"A (observations only): {sections['assessment_observations_only']}",
        f"R: {sections['handoff_request']}",
    ]
    source_cards = navigator_output.get("source_cards") or []
    if source_cards:
        lines.append("Source cards: " + ", ".join(str(card_id) for card_id in source_cards))
    safety_boundary = str(navigator_output.get("safety_boundary", "")).strip()
    if safety_boundary:
        lines.append(f"Safety boundary: {safety_boundary}")
    return "\n".join(lines)


render_sbar_note = render_sbar
