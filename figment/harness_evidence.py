"""Deterministic evidence surfaces for Figment navigator outputs and traces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def build_harness_evidence(
    *,
    confirmed_intake: Mapping[str, Any] | None,
    retrieved_card_ids: Iterable[Any],
    rule_results: Iterable[Mapping[str, Any]],
    urgency_floor: str,
    validator_result: Mapping[str, Any],
    final_output: Mapping[str, Any] | None = None,
    model_route: Mapping[str, Any] | None = None,
    audio: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build app-owned, non-secret evidence badges for a navigator result."""

    route = model_route or {}
    output = final_output or {}
    validation_status = _validation_status(validator_result)
    return {
        "confirmed_intake": bool(confirmed_intake and confirmed_intake.get("confirmed") is True),
        "retrieved_card_ids": _unique_strings(retrieved_card_ids),
        "deterministic_rule_ids": _unique_strings(rule.get("rule_id") for rule in rule_results),
        "deterministic_rule_card_ids": _unique_strings(rule.get("card_id") for rule in rule_results),
        "urgency_floor": str(urgency_floor),
        "validator_status": validation_status,
        "audio_correction_status": _audio_correction_status(audio),
        "source_card_ids": _unique_strings(output.get("source_cards", [])),
        "fallback_tier": _optional_string(route.get("fallback_tier")),
        "fallback_reason": _optional_string(route.get("fallback_reason")),
        "final_route": _optional_string(route.get("final_route") or route.get("runtime_contribution"))
        or "unknown",
        "field_level_fallback_used": bool(route.get("field_level_fallback_used")),
        "repair_attempt_count": _int_or_zero(route.get("repair_attempt_count")),
        "repair_scopes": _unique_strings(route.get("repair_scopes", [])),
    }


def _validation_status(validator_result: Mapping[str, Any]) -> str:
    if validator_result.get("passed") is True:
        return "passed"
    if validator_result.get("passed") is False:
        return "failed"
    return "unknown"


def _audio_correction_status(audio: Mapping[str, Any] | None) -> str:
    if not isinstance(audio, Mapping) or not audio:
        return "not_applicable"
    correction_keys = (
        "manual_corrections",
        "manual_correction",
        "corrected_fields",
        "corrections_applied",
        "human_corrected_fields",
    )
    if any(audio.get(key) for key in correction_keys):
        return "corrected"
    if audio.get("transcript") or audio.get("fields") or audio.get("structured_intake_patch"):
        return "no_manual_correction_recorded"
    return "not_applicable"


def _unique_strings(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return out


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["build_harness_evidence"]
