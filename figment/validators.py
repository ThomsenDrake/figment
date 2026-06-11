"""Deterministic safety validators."""

from __future__ import annotations

import re
from typing import Any

from .schemas import ValidationResult, urgency_at_least


FORBIDDEN_PATTERNS = [
    re.compile(r"\bdiagnos(?:e|is|ed)\b", re.IGNORECASE),
    re.compile(r"\bprescrib(?:e|ing|ed)\b", re.IGNORECASE),
    re.compile(r"\b(?:administer|inject|dispense)\s+(?:an?\s+)?(?:aspirin|antibiotic|opioid|insulin|medication|drug|dose)\b", re.IGNORECASE),
    re.compile(r"\bstart (?:an )?(?:antibiotic|opioid|insulin)\b", re.IGNORECASE),
    re.compile(r"\bgive \d+(?:\.\d+)?\s*(?:mg|mcg|ml)\b", re.IGNORECASE),
    re.compile(r"\b(?:discharge(?: home)?|send (?:them|the patient|the person|him|her)?\s*home)\b", re.IGNORECASE),
    re.compile(r"\b(?:no|without) escalation (?:needed|required)\b", re.IGNORECASE),
    re.compile(r"\b(?:clear|clears|cleared|rule out|ruled out) (?:the )?(?:red flag|emergency|urgent concern)\b", re.IGNORECASE),
    re.compile(r"\b(?:override|downgrade|ignore|skip) (?:the )?(?:red flag|protocol|escalation)\b", re.IGNORECASE),
]

NAVIGATOR_REQUIRED_KEYS = {
    "protocol_urgency",
    "red_flags",
    "intake_facts",
    "candidate_protocol_pathways",
    "missing_info_to_collect",
    "next_observations_to_collect",
    "conflicts_or_uncertainties",
    "responder_checklist",
    "do_not_do",
    "source_cards",
    "handoff_note_sbar",
    "responder_plain_language_script",
    "safety_boundary",
}

NAVIGATOR_LIST_FIELDS = {
    "red_flags",
    "intake_facts",
    "candidate_protocol_pathways",
    "missing_info_to_collect",
    "next_observations_to_collect",
    "conflicts_or_uncertainties",
    "responder_checklist",
    "do_not_do",
    "source_cards",
}

NAVIGATOR_TEXT_FIELDS = {"responder_plain_language_script", "safety_boundary"}
CARD_IDS_EXEMPT_FROM_OBSERVATION_GROUNDING = {"SAFETY-BOUNDARIES-v1", "REFERRAL-SBAR-v1"}
HIGH_RISK_GROUNDING_TOKENS = {
    "antibiotic",
    "aspirin",
    "bleeding",
    "blue",
    "cyanosis",
    "dose",
    "dosing",
    "fainting",
    "fever",
    "fracture",
    "headache",
    "insulin",
    "opioid",
    "oxygen",
    "pregnancy",
    "pregnant",
    "pressure",
    "rash",
    "seizure",
    "shock",
    "skull",
    "stroke",
    "unconscious",
    "unresponsive",
    "vision",
}


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
    *,
    confirmed_intake: dict[str, Any] | None = None,
    rule_results: list[dict[str, Any]] | None = None,
    retrieved_card_ids: set[str] | None = None,
    retrieved_cards: list[dict[str, Any]] | None = None,
    strict_schema: bool = False,
) -> ValidationResult:
    result = ValidationResult(passed=True)
    if not isinstance(output, dict):
        return ValidationResult(False, ["navigator output is not an object"])
    if strict_schema:
        _validate_schema_shape(result, output)

    urgency = output.get("protocol_urgency")
    if urgency not in {"routine", "monitor", "urgent", "emergency"}:
        result.add("protocol_urgency is missing or invalid")
    elif not urgency_at_least(urgency, urgency_floor):
        result.add(f"protocol_urgency {urgency} is below deterministic floor {urgency_floor}")

    known_cards = set(known_card_ids)
    retrieved_cards_by_id = _retrieved_cards_by_id(retrieved_cards or [])
    if retrieved_card_ids is None and retrieved_cards_by_id:
        retrieved_card_ids = set(retrieved_cards_by_id)
    fired_rule_card_ids = {
        str(rule.get("card_id", "")).strip()
        for rule in rule_results or []
        if str(rule.get("card_id", "")).strip()
    }
    allowed_cards = set(retrieved_card_ids) if retrieved_card_ids is not None else known_cards
    allowed_cards.update(fired_rule_card_ids & known_cards)
    source_cards = output.get("source_cards")
    if not isinstance(source_cards, list) or not source_cards:
        result.add("source_cards must be a non-empty list")
        source_cards = []
    normalized_source_cards = []
    for card_id in source_cards:
        if not isinstance(card_id, str) or not card_id:
            result.add("source_cards entries must be non-empty strings")
            continue
        normalized_source_cards.append(card_id)
    source_card_set = set(normalized_source_cards)
    unknown = sorted(source_card_set - known_cards)
    if unknown:
        result.add(f"unknown source_cards: {', '.join(unknown)}")
    disallowed = sorted(source_card_set - allowed_cards)
    if disallowed:
        result.add(f"source_cards not in allowed/retrieved card IDs: {', '.join(disallowed)}")

    for rule in rule_results or []:
        rule_card_id = str(rule.get("card_id", "")).strip()
        if rule_card_id and rule_card_id not in source_card_set:
            result.add(f"fired rule card {rule_card_id} is not cited in source_cards")

    pathways = output.get("candidate_protocol_pathways", [])
    if pathways is None:
        pathways = []
    if not isinstance(pathways, list):
        result.add("candidate_protocol_pathways must be a list")
        pathways = []
    for pathway in pathways:
        card_id = pathway.get("card_id") if isinstance(pathway, dict) else None
        if not card_id:
            result.add("candidate_protocol_pathway is missing card_id")
        elif card_id not in source_card_set:
            result.add(f"candidate pathway {card_id} is not cited in source_cards")
        elif card_id not in allowed_cards:
            result.add(f"candidate pathway {card_id} is not in allowed/retrieved card IDs")

    if strict_schema and retrieved_cards_by_id:
        _validate_missing_observations_against_cards(result, output, source_card_set, retrieved_cards_by_id)

    handoff = output.get("handoff_note_sbar")
    required_handoff_fields = {
        "situation",
        "background",
        "assessment_observations_only",
        "handoff_request",
    }
    if not isinstance(handoff, dict):
        result.add("handoff_note_sbar must be an object")
    else:
        missing_handoff = sorted(field for field in required_handoff_fields if not handoff.get(field))
        if missing_handoff:
            result.add(f"handoff_note_sbar is missing: {', '.join(missing_handoff)}")
        elif confirmed_intake is not None:
            _validate_handoff_grounding(result, handoff, confirmed_intake, rule_results or [])

    all_text = "\n".join(_text_values(output))
    for pattern in FORBIDDEN_PATTERNS:
        for match in pattern.finditer(all_text):
            if _is_negated_safety_phrase(all_text, match.start()):
                continue
            result.add(f"forbidden clinical language: {match.group(0)}")
            break
    return result


def _validate_schema_shape(result: ValidationResult, output: dict[str, Any]) -> None:
    missing = sorted(NAVIGATOR_REQUIRED_KEYS - set(output))
    if missing:
        result.add(f"missing required schema keys: {', '.join(missing)}")
    for field in sorted(NAVIGATOR_LIST_FIELDS & set(output)):
        if not isinstance(output.get(field), list):
            result.add(f"{field} must be a list")
    for field in sorted(NAVIGATOR_TEXT_FIELDS & set(output)):
        if not isinstance(output.get(field), str):
            result.add(f"{field} must be a string")


def _retrieved_cards_by_id(retrieved_cards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in retrieved_cards:
        if not isinstance(item, dict):
            continue
        card = item.get("card", item)
        if not isinstance(card, dict):
            continue
        card_id = str(card.get("card_id", "")).strip()
        if card_id:
            by_id[card_id] = card
    return by_id


def _validate_missing_observations_against_cards(
    result: ValidationResult,
    output: dict[str, Any],
    source_card_ids: set[str],
    retrieved_cards_by_id: dict[str, dict[str, Any]],
) -> None:
    observation_tokens = _grounding_tokens(
        output.get("missing_info_to_collect", []),
        output.get("next_observations_to_collect", []),
    )
    for card_id, card in retrieved_cards_by_id.items():
        if card_id not in source_card_ids or card_id in CARD_IDS_EXEMPT_FROM_OBSERVATION_GROUNDING:
            continue
        required_observations = card.get("required_observations", [])
        if not isinstance(required_observations, list) or not required_observations:
            continue
        required_sets = [_grounding_tokens(observation) for observation in required_observations]
        if not any(tokens & observation_tokens for tokens in required_sets):
            result.add(f"missing_info_to_collect does not reference required observations for {card_id}")


def _is_negated_safety_phrase(text: str, match_start: int) -> bool:
    sentence_start = max(text.rfind(boundary, 0, match_start) for boundary in ".!?\n;")
    prefix = text[sentence_start + 1 : match_start].lower()
    tail = prefix[-80:]
    if re.search(r"\b(?:do not|don't|never|must not|cannot|does not|not)\s+$", tail):
        return True
    if re.search(r"\bnot\s+an?\s+$", tail):
        return True
    negated_clause = re.search(r"\b(?:do not|don't|never|must not|cannot|does not)\s+([^.!?\n;]*)$", prefix)
    if negated_clause and _clause_is_negated_safety_instruction(negated_clause.group(1)):
        return True

    list_marker = re.search(r"\b(?:do not|don't|never|must not|cannot|does not)\s+([^.!?\n;]*)$", prefix)
    if not list_marker:
        return False
    allowed_list_text = re.sub(
        r"\b(?:diagnos(?:e|is|ed)|prescrib(?:e|ing|ed)|administer|inject|dispense|dose|dosing|"
        r"override|downgrade|ignore|skip|discharge|send|clear|rule|ruled|out|treat(?:ment)?|"
        r"condition|label|drug|medication|order|start|antibiotic|antibiotics|opioid|insulin|"
        r"aspirin|oxygen|escalation|protocol|red|flag|home|or|and)\b|[,/()\s-]+",
        "",
        list_marker.group(1),
    )
    return not allowed_list_text


def _clause_is_negated_safety_instruction(clause_prefix: str) -> bool:
    if re.search(
        r"\b(?:diagnos(?:e|is|ed)|prescrib(?:e|ing|ed)|administer|inject|dispense|start|give|"
        r"dose|dosing|override|downgrade|ignore|skip|discharge|send home)\b",
        clause_prefix,
    ):
        return True
    return bool(
        re.search(
            r"\b(?:provide|offer|make|assign|state|use|treat as)\b[^.!?\n;]{0,80}"
            r"\b(?:diagnos(?:e|is|ed)|prescription|medication order|condition label)\b",
            clause_prefix,
        )
    )


GROUNDING_STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "that",
    "this",
    "only",
    "protocol",
    "local",
    "cited",
    "review",
    "request",
    "escalate",
    "escalation",
    "observed",
    "reported",
    "vitals",
    "setting",
    "field",
    "clinic",
    "case",
    "synthetic",
}


def _validate_handoff_grounding(
    result: ValidationResult,
    handoff: dict[str, Any],
    confirmed_intake: dict[str, Any],
    rule_results: list[dict[str, Any]],
) -> None:
    rule_text = " ".join(
        str(rule.get(field, ""))
        for rule in rule_results
        for field in ("rule_id", "label", "evidence", "card_id")
    )
    grounding_sources = {
        "situation": [
            confirmed_intake.get("chief_concern", ""),
            confirmed_intake.get("symptoms", ""),
            confirmed_intake.get("responder_note", ""),
        ],
        "background": [
            confirmed_intake.get("setting", ""),
            confirmed_intake.get("patient_age", ""),
            confirmed_intake.get("pregnancy_status", ""),
            rule_text,
        ],
        "assessment_observations_only": [
            confirmed_intake.get("symptoms", ""),
            confirmed_intake.get("vitals", ""),
            confirmed_intake.get("responder_note", ""),
            rule_text,
        ],
    }
    for field, sources in grounding_sources.items():
        field_tokens = _grounding_tokens(handoff.get(field, ""))
        if not field_tokens:
            continue
        source_tokens = _grounding_tokens(*sources)
        if not field_tokens & source_tokens:
            result.add(f"handoff_note_sbar {field} is not grounded in confirmed intake or rules")
            continue
        unsupported_high_risk = sorted(_high_risk_tokens(handoff.get(field, "")) - _high_risk_tokens(*sources))
        if unsupported_high_risk:
            result.add(
                f"handoff_note_sbar {field} has unsupported high-risk handoff facts: "
                f"{', '.join(unsupported_high_risk[:8])}"
            )


def _grounding_tokens(*values: Any) -> set[str]:
    text = " ".join(_text_values(list(values))).lower()
    return {
        _normalize_grounding_token(token)
        for token in re.findall(r"[a-z0-9]+", text)
        if len(token) >= 3 and token not in GROUNDING_STOPWORDS
    }


def _normalize_grounding_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _high_risk_tokens(*values: Any) -> set[str]:
    tokens = _grounding_tokens(*values)
    return {token for token in tokens if token in HIGH_RISK_GROUNDING_TOKENS or re.fullmatch(r"\d{2,3}", token)}
