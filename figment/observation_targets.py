"""Deterministic scaffolding helpers for required observation targets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any


CARD_IDS_EXEMPT_FROM_OBSERVATION_TARGETS = {"SAFETY-BOUNDARIES-v1", "REFERRAL-SBAR-v1"}
TRACE_ONLY_REQUIRED_OBSERVATION_IDS_KEY = "selected_required_observation_ids"
URGENCY_ORDER = {"routine": 0, "monitor": 1, "urgent": 2, "emergency": 3}
TARGET_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "or",
    "the",
    "to",
    "of",
    "for",
    "if",
    "by",
    "with",
}


@dataclass(frozen=True)
class NavigationScaffoldResult:
    """Navigator output after deterministic scaffolding plus changed fields."""

    output: dict[str, Any]
    patched_fields: set[str]
    filled_required_observation_ids: list[str]
    model_selected_required_observation_ids: list[str] = field(default_factory=list)
    invalid_selected_required_observation_ids: list[str] = field(default_factory=list)
    stripped_trace_only_fields: list[str] = field(default_factory=list)


def required_observation_targets(retrieved_cards: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return stable target ids for required observations on retrieved cards."""

    targets: list[dict[str, Any]] = []
    for item in retrieved_cards:
        card = _card_payload(item)
        card_id = str(card.get("card_id", "")).strip()
        if not card_id:
            continue
        required_observations = card.get("required_observations")
        if not isinstance(required_observations, list):
            continue
        title = str(card.get("title", "")).strip()
        for index, observation in enumerate(required_observations, start=1):
            display_text = str(observation).strip()
            if not display_text:
                continue
            targets.append(
                {
                    "id": f"{card_id}::required_observation::{index}",
                    "card_id": card_id,
                    "title": title,
                    "display_text": display_text,
                    "cue_tokens": _target_tokens(display_text),
                }
            )
    return targets


def build_case_fact_ledger(intake: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Split confirmed intake into present, absent/denied, and unclear facts."""

    ledger = {"present": [], "absent_or_denied": [], "unclear": []}
    if intake.get("confirmed") is not True:
        ledger["unclear"].append(
            {
                "field": "confirmed",
                "value": "intake is not confirmed",
            }
        )
        return ledger

    for field, value in sorted(intake.items()):
        if field == "confirmed" or value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in {"unknown", "pending", "not recorded", "not available", "unclear"}:
            ledger["unclear"].append({"field": field, "value": text})
            continue
        ledger["present"].append({"field": field, "value": text})
        for phrase in _negated_phrases(text):
            ledger["absent_or_denied"].append({"field": field, "value": phrase})
    return ledger


def apply_navigation_scaffolding(
    output: Mapping[str, Any],
    *,
    retrieved_cards: Iterable[Mapping[str, Any]],
    rule_results: list[dict[str, Any]],
    urgency_floor: str,
    confirmed_intake: Mapping[str, Any] | None = None,
) -> NavigationScaffoldResult:
    """Patch deterministic control fields and required-observation omissions."""

    patched = deepcopy(dict(output))
    patched_fields: set[str] = set()
    retrieved_card_list = list(retrieved_cards)
    retrieved_ids = _retrieved_card_ids(retrieved_card_list)
    fired_card_ids = _fired_rule_card_ids(rule_results)
    targets = required_observation_targets(retrieved_card_list)
    (
        model_selected_required_observation_ids,
        invalid_selected_required_observation_ids,
        stripped_trace_only_fields,
    ) = _pop_trace_only_required_observation_ids(patched, targets)

    current_urgency = str(patched.get("protocol_urgency", "")).strip()
    if not _urgency_at_least(current_urgency, urgency_floor):
        patched["protocol_urgency"] = urgency_floor
        patched_fields.add("protocol_urgency")

    expected_red_flags = list(rule_results)
    if patched.get("red_flags") != expected_red_flags:
        patched["red_flags"] = expected_red_flags
        patched_fields.add("red_flags")

    source_cards = _scaffold_source_cards(patched.get("source_cards"), retrieved_ids, fired_card_ids)
    if patched.get("source_cards") != source_cards:
        patched["source_cards"] = source_cards
        patched_fields.add("source_cards")

    pathways = _scaffold_candidate_pathways(patched.get("candidate_protocol_pathways"), source_cards)
    if patched.get("candidate_protocol_pathways") != pathways:
        patched["candidate_protocol_pathways"] = pathways
        patched_fields.add("candidate_protocol_pathways")

    filled_ids = _fill_required_observation_targets(
        patched,
        targets,
        selected_required_observation_ids=model_selected_required_observation_ids,
    )
    if filled_ids:
        patched_fields.update({"missing_info_to_collect", "next_observations_to_collect"})

    if confirmed_intake is not None and _scaffold_handoff_note_sbar(
        patched,
        confirmed_intake=confirmed_intake,
        rule_results=rule_results,
        urgency_floor=urgency_floor,
        source_card_ids=source_cards,
    ):
        patched_fields.add("handoff_note_sbar")

    return NavigationScaffoldResult(
        output=patched,
        patched_fields=patched_fields,
        filled_required_observation_ids=filled_ids,
        model_selected_required_observation_ids=model_selected_required_observation_ids,
        invalid_selected_required_observation_ids=invalid_selected_required_observation_ids,
        stripped_trace_only_fields=stripped_trace_only_fields,
    )


def build_handoff_note_sbar_template(
    intake: Mapping[str, Any],
    rule_results: list[dict[str, Any]],
    urgency_floor: str,
    *,
    source_card_ids: Iterable[Any] = (),
) -> dict[str, str]:
    """Build a deterministic, grounded SBAR draft from confirmed harness facts."""

    situation = _first_text(
        intake.get("chief_concern"),
        intake.get("responder_note"),
        "Confirmed field concern",
    )
    background_parts = []
    if _has_value(intake.get("setting")):
        background_parts.append(f"Setting: {intake['setting']}.")
    if _has_value(intake.get("patient_age")):
        background_parts.append(f"Age: {intake['patient_age']}.")
    if _has_value(intake.get("pregnancy_status")):
        background_parts.append(f"Pregnancy status: {intake['pregnancy_status']}.")

    assessment_parts = []
    if _has_value(intake.get("symptoms")):
        assessment_parts.append(f"Symptoms: {intake['symptoms']}.")
    if _has_value(intake.get("vitals")):
        assessment_parts.append(f"Vitals: {intake['vitals']}.")
    red_flag_labels = [
        str(rule.get("label") or rule.get("rule_id"))
        for rule in rule_results
        if rule.get("label") or rule.get("rule_id")
    ]
    if red_flag_labels:
        assessment_parts.append(f"Red flags: {'; '.join(red_flag_labels)}.")

    source_suffix = _source_card_suffix(source_card_ids)
    return {
        "situation": str(situation),
        "background": " ".join(background_parts) or "Background details pending from confirmed intake.",
        "assessment_observations_only": " ".join(assessment_parts)
        or "Assessment observations pending from confirmed intake.",
        "handoff_request": f"Request {urgency_floor} review/escalation per cited local protocol cards{source_suffix}.",
    }


def targets_for_failure_cards(
    targets: Iterable[Mapping[str, Any]],
    failures: Iterable[str],
) -> list[dict[str, Any]]:
    """Filter required-observation targets to card ids named in validation failures."""

    failure_text = "\n".join(str(failure) for failure in failures)
    card_ids = set(re.findall(r"\b[A-Z][A-Z0-9-]+-v\d+\b", failure_text))
    selected = []
    for target in targets:
        card_id = str(target.get("card_id", "")).strip()
        if card_id and (not card_ids or card_id in card_ids):
            selected.append(dict(target))
    return selected


def _scaffold_handoff_note_sbar(
    output: dict[str, Any],
    *,
    confirmed_intake: Mapping[str, Any],
    rule_results: list[dict[str, Any]],
    urgency_floor: str,
    source_card_ids: Iterable[Any],
) -> bool:
    template = build_handoff_note_sbar_template(
        confirmed_intake,
        rule_results,
        urgency_floor,
        source_card_ids=source_card_ids,
    )
    handoff = output.get("handoff_note_sbar")
    if not isinstance(handoff, Mapping):
        output["handoff_note_sbar"] = template
        return True

    patched_handoff = dict(handoff)
    changed = False
    for field, template_value in template.items():
        value = str(patched_handoff.get(field) or "").strip()
        if not value or _handoff_slot_needs_scaffold(field, value, rule_results):
            patched_handoff[field] = template_value
            changed = True
    if changed:
        output["handoff_note_sbar"] = patched_handoff
    return changed


def _handoff_slot_needs_scaffold(field: str, value: str, rule_results: list[dict[str, Any]]) -> bool:
    normalized = _normalize_text(value)
    if field == "assessment_observations_only" and _unsafe_assessment_language(normalized):
        return True
    if field == "assessment_observations_only" and rule_results:
        rule_markers = [
            _normalize_text(str(rule.get(key, "")))
            for rule in rule_results
            for key in ("rule_id", "label")
            if rule.get(key)
        ]
        return "red flag" not in normalized and "rule" not in normalized and not any(
            marker and marker in normalized for marker in rule_markers
        )
    return False


def _unsafe_assessment_language(normalized_text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:diagnos(?:e|is|ed)|prescrib(?:e|ed|ing)|administer|discharge|treat(?:ment)?|dose|dosing)\b",
            normalized_text,
        )
    )


def _fill_required_observation_targets(
    output: dict[str, Any],
    targets: list[dict[str, Any]],
    *,
    selected_required_observation_ids: Iterable[str] = (),
) -> list[str]:
    source_cards = {str(card_id) for card_id in output.get("source_cards", []) if str(card_id)}
    actionable_targets = [
        target
        for target in targets
        if target["card_id"] in source_cards
        and target["card_id"] not in CARD_IDS_EXEMPT_FROM_OBSERVATION_TARGETS
    ]
    if not actionable_targets:
        return []

    missing_info = _string_list(output.get("missing_info_to_collect"))
    next_observations = _string_list(output.get("next_observations_to_collect"))
    combined_text = "\n".join(missing_info + next_observations)
    combined_tokens = set(_target_tokens(combined_text))

    missing_targets = [
        target
        for target in actionable_targets
        if not _target_present(target, combined_text, combined_tokens)
    ]
    if not missing_targets:
        return []

    for target in missing_targets:
        display_text = str(target["display_text"])
        missing_info.append(display_text)
        next_observations.append(display_text)
    output["missing_info_to_collect"] = missing_info
    output["next_observations_to_collect"] = next_observations
    return [str(target["id"]) for target in missing_targets]


def _target_present(target: Mapping[str, Any], text: str, tokens: set[str]) -> bool:
    target_tokens = set(str(token) for token in target.get("cue_tokens", []))
    if target_tokens and target_tokens <= tokens:
        return True
    normalized_target = _normalize_text(str(target.get("display_text", "")))
    return bool(normalized_target and normalized_target in _normalize_text(text))


def _pop_trace_only_required_observation_ids(
    output: dict[str, Any],
    targets: Iterable[Mapping[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    if TRACE_ONLY_REQUIRED_OBSERVATION_IDS_KEY not in output:
        return [], [], []

    raw_value = output.pop(TRACE_ONLY_REQUIRED_OBSERVATION_IDS_KEY)
    target_ids = {str(target.get("id", "")).strip() for target in targets}
    target_ids.discard("")
    selected: list[str] = []
    invalid: list[str] = []
    for target_id in _string_list(raw_value):
        normalized = str(target_id).strip()
        if not normalized:
            continue
        if normalized in target_ids:
            _append_unique(selected, normalized)
        else:
            _append_unique(invalid, normalized)
    return selected, invalid, [TRACE_ONLY_REQUIRED_OBSERVATION_IDS_KEY]


def _target_tokens(value: Any) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    return [token for token in tokens if token and token not in TARGET_TOKEN_STOPWORDS]


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _card_payload(item: Mapping[str, Any]) -> Mapping[str, Any]:
    card = item.get("card", item)
    return card if isinstance(card, Mapping) else {}


def _retrieved_card_ids(retrieved_cards: Iterable[Mapping[str, Any]]) -> list[str]:
    ids: list[str] = []
    for item in retrieved_cards:
        card = _card_payload(item)
        card_id = str(item.get("card_id") or card.get("card_id") or "").strip()
        if card_id and card_id not in ids:
            ids.append(card_id)
    return ids


def _fired_rule_card_ids(rule_results: Iterable[Mapping[str, Any]]) -> list[str]:
    ids: list[str] = []
    for rule in rule_results:
        card_id = str(rule.get("card_id", "")).strip()
        if card_id and card_id not in ids:
            ids.append(card_id)
    return ids


def _scaffold_source_cards(value: Any, retrieved_ids: list[str], fired_card_ids: list[str]) -> list[str]:
    raw_source_cards = [str(card_id) for card_id in value if str(card_id)] if isinstance(value, list) else []
    allowed = set(retrieved_ids) | set(fired_card_ids)
    source_cards: list[str] = []
    for card_id in fired_card_ids:
        if card_id not in source_cards:
            source_cards.append(card_id)
    for card_id in raw_source_cards:
        if card_id in allowed and card_id not in source_cards:
            source_cards.append(card_id)
    if not source_cards:
        source_cards.extend(retrieved_ids[:3])
    return source_cards[:6]


def _scaffold_candidate_pathways(value: Any, source_cards: list[str]) -> list[dict[str, str]]:
    source_set = set(source_cards)
    pathways: list[dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, Mapping):
                continue
            card_id = str(item.get("card_id", "")).strip()
            if card_id not in source_set:
                continue
            pathways.append(
                {
                    "card_id": card_id,
                    "reason_relevant": str(item.get("reason_relevant") or "Retrieved from confirmed intake."),
                }
            )
    existing_pathway_ids = {item["card_id"] for item in pathways}
    for card_id in source_cards:
        if card_id in existing_pathway_ids:
            continue
        pathways.append(
            {
                "card_id": card_id,
                "reason_relevant": "Retrieved from confirmed intake and deterministic rule context.",
            }
        )
        existing_pathway_ids.add(card_id)
        if len(pathways) >= 3:
            break
    if not pathways:
        pathways = [
            {
                "card_id": card_id,
                "reason_relevant": "Retrieved from confirmed intake and deterministic protocol context.",
            }
            for card_id in source_cards[:3]
        ]
    return pathways


def _urgency_at_least(actual: str, expected_minimum: str) -> bool:
    if actual not in URGENCY_ORDER or expected_minimum not in URGENCY_ORDER:
        return False
    return URGENCY_ORDER[actual] >= URGENCY_ORDER[expected_minimum]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _negated_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for match in re.finditer(r"\b(no|denies|denied|without)\s+([^,.;]+)", text, re.IGNORECASE):
        marker = match.group(1).lower()
        phrase = " ".join(match.group(2).split()).strip()
        if not phrase:
            continue
        phrases.append(f"{marker} {phrase}")
    return phrases


def _source_card_suffix(source_card_ids: Iterable[Any]) -> str:
    ids = [str(card_id).strip() for card_id in source_card_ids if str(card_id).strip()]
    return f" ({', '.join(ids[:4])})" if ids else ""


def _first_text(*values: Any) -> str:
    for value in values:
        if _has_value(value):
            return str(value).strip()
    return ""


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


__all__ = [
    "NavigationScaffoldResult",
    "TRACE_ONLY_REQUIRED_OBSERVATION_IDS_KEY",
    "apply_navigation_scaffolding",
    "build_case_fact_ledger",
    "build_handoff_note_sbar_template",
    "required_observation_targets",
    "targets_for_failure_cards",
]
