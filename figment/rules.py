"""Deterministic red-flag checks over confirmed Figment intake."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from .schemas import RuleResult, Urgency, highest_urgency


TEXT_FIELDS = (
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
)


def _text_from_intake(intake: Mapping[str, Any]) -> str:
    values: list[str] = []
    for field in TEXT_FIELDS:
        value = intake.get(field)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            value = " ".join(str(item) for item in value)
        values.append(str(value))
    return "\n".join(values).lower()


def _find(patterns: tuple[str, ...], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def _is_pediatric(intake: Mapping[str, Any], text: str) -> bool:
    age = str(intake.get("patient_age", "")).strip().lower()
    month_match = re.search(r"\b(\d{1,3})\s*(?:mo|mos|month|months)\b", age)
    if month_match:
        return int(month_match.group(1)) < 216
    week_day_match = re.search(r"\b(\d{1,3})\s*(?:d|day|days|w|wk|wks|week|weeks)\b", age)
    if week_day_match:
        return True
    year_match = re.search(r"\b(\d{1,3})\s*(?:y|yr|yrs|year|years)\b", age)
    if year_match and int(year_match.group(1)) < 18:
        return True
    bare_age_match = re.fullmatch(r"\s*(\d{1,2})\s*", age)
    if bare_age_match and int(bare_age_match.group(1)) < 18:
        return True
    return bool(re.search(r"\b(child|infant|toddler|baby|pediatric)\b", text))


def _is_pregnant_or_postpartum(intake: Mapping[str, Any], text: str) -> bool:
    status = str(intake.get("pregnancy_status", "")).strip().lower().replace("_", " ")
    false_statuses = {
        "",
        "n/a",
        "na",
        "no",
        "none",
        "not applicable",
        "not pregnant",
        "nonpregnant",
        "non pregnant",
        "unknown",
    }
    if status in false_statuses or re.search(r"\b(?:not|no|denies)\s+(?:currently\s+)?pregnan\w*\b", status):
        return False
    if re.search(r"\b(?:pregnant|pregnancy|postpartum)\b", status):
        return True
    if _has_negated_pregnancy(text):
        return False
    return bool(re.search(r"\b(?:pregnant|pregnancy|postpartum)\b", text))


def _has_negated_pregnancy(text: str) -> bool:
    return bool(
        re.search(r"\b(?:not|no|denies|is not|isn't)\s+(?:currently\s+)?pregnan\w*\b", text)
        or re.search(r"\bnon[- ]pregnan\w*\b", text)
    )


def _has_positive_fever(text: str) -> bool:
    measured_fever = (
        r"\b(?:temperature|temp)\s*(?:of|is|:|now|currently|measured|at)?\s*"
        r"(?:10[0-9](?:\.\d+)?\s*f|3[89](?:\.\d+)?\s*c|40(?:\.\d+)?\s*c)\b"
    )
    if re.search(measured_fever, text):
        return True
    for match in re.finditer(r"\b(?:fever|febrile)\b", text):
        fever_context = text[max(0, match.start() - 40) : match.end() + 20]
        if not re.search(r"\b(?:no|not|denies|without)\s+(?:current(?:ly)?\s+|reported\s+)?(?:fever|febrile)\b", fever_context):
            return True
    return False


def _rule(rule_id: str, label: str, urgency: Urgency, evidence: str, card_id: str) -> RuleResult:
    return RuleResult(rule_id=rule_id, label=label, urgency=urgency, evidence=evidence, card_id=card_id)


def _altered_mental_status(_: Mapping[str, Any], text: str) -> RuleResult | None:
    evidence = _find(
        (
            r"\baltered mental status\b",
            r"\bnew confusion\b",
            r"\bconfus(?:ed|ion)\b",
            r"\bunresponsive\b",
            r"\bdifficult to (?:wake|arouse)\b",
            r"\bnew seizure\b",
        ),
        text,
    )
    if not evidence:
        return None
    return _rule("AMS-001", "Altered mental status", "emergency", evidence, "AMS-RED-FLAGS-v1")


def _respiratory_distress(_: Mapping[str, Any], text: str) -> RuleResult | None:
    evidence = _find(
        (
            r"\bsevere respiratory distress\b",
            r"\bgasping\b",
            r"\bunable to speak(?: full sentences)?\b",
            r"\bblue lips\b",
            r"\bcyanosis\b",
            r"\bmarked retractions\b",
            r"\btripod position(?:ing)?\b",
        ),
        text,
    )
    if not evidence:
        return None
    return _rule("RESP-001", "Severe respiratory distress", "emergency", evidence, "RESP-DISTRESS-RED-FLAGS-v1")


def _pregnancy_danger(intake: Mapping[str, Any], text: str) -> RuleResult | None:
    if not _is_pregnant_or_postpartum(intake, text):
        return None
    evidence = _find(
        (
            r"\bvaginal bleeding\b",
            r"\bbleeding\b",
            r"\bseizure\b",
            r"\bfainting\b",
            r"\bsevere headache\b",
            r"\bvision changes?\b",
            r"\bsevere abdominal pain\b",
        ),
        text,
    )
    if not evidence and _has_positive_fever(text):
        evidence = "fever"
    if not evidence:
        return None
    return _rule("PREG-001", "Pregnancy danger sign", "emergency", evidence, "PREG-DANGER-SIGNS-v1")


def _chest_pain(_: Mapping[str, Any], text: str) -> RuleResult | None:
    evidence = _find(
        (
            r"\bchest pain\b",
            r"\bchest pressure\b",
            r"\bpain radiat(?:es|ing) to (?:arm|jaw|back|shoulder)\b",
            r"\bchest pain with (?:shortness of breath|sweating|fainting|severe weakness)\b",
        ),
        text,
    )
    if not evidence:
        return None
    return _rule("red_flag_chest_pain", "Chest pain escalation cue", "emergency", evidence, "CHEST-PAIN-ESCALATION-v1")


def _stroke_signs(_: Mapping[str, Any], text: str) -> RuleResult | None:
    evidence = _find(
        (
            r"\bfacial droop\b",
            r"\bface droop\b",
            r"\bone[- ]sided weakness\b",
            r"\barm weakness\b",
            r"\bslurred speech\b",
            r"\btrouble speaking\b",
            r"\bsudden vision change\b",
            r"\bsudden balance trouble\b",
        ),
        text,
    )
    if not evidence:
        return None
    return _rule("STROKE-001", "Stroke sign", "emergency", evidence, "STROKE-SIGNS-v1")


def _pediatric_dehydration(intake: Mapping[str, Any], text: str) -> RuleResult | None:
    if not _is_pediatric(intake, text):
        return None
    evidence = _find(
        (
            r"\bletharg(?:ic|y)\b",
            r"\bdifficult to arouse\b",
            r"\bno urine\b",
            r"\bsunken eyes\b",
            r"\bvery dry mouth\b",
            r"\bpoor perfusion\b",
            r"\bunable to keep fluids down\b",
        ),
        text,
    )
    if not evidence:
        return None
    urgency: Urgency = "emergency" if re.search(r"\bletharg(?:ic|y)|difficult to arouse\b", evidence) else "urgent"
    return _rule("PED-DEHYD-001", "Pediatric dehydration red flag", urgency, evidence, "PED-DEHYD-RED-FLAGS-v1")


def _fever_escalation(intake: Mapping[str, Any], text: str) -> RuleResult | None:
    if not _has_positive_fever(text):
        return None
    danger = _find(
        (
            r"\bstiff neck\b",
            r"\bconfus(?:ed|ion)\b",
            r"\baltered mental status\b",
            r"\bnon[- ]blanching rash\b",
            r"\brapidly spreading rash\b",
            r"\bsevere dehydration\b",
        ),
        text,
    )
    if not danger and (_is_pregnant_or_postpartum(intake, text) or re.search(r"\binfant\b", text)):
        danger = "pregnancy/infant fever context"
    if not danger:
        return None
    return _rule("FEVER-001", "Fever escalation cue", "urgent", danger, "FEVER-RED-FLAGS-v1")


def _wound_infection(_: Mapping[str, Any], text: str) -> RuleResult | None:
    if not re.search(r"\bwound\b|cut|laceration|burn", text):
        return None
    evidence = _find(
        (
            r"\bspreading redness\b",
            r"\bred streak(?:ing|s)?\b",
            r"\bpus\b",
            r"\bfoul drainage\b",
            r"\brapidly worsening pain\b",
            r"\bworsening swelling\b",
        ),
        text,
    )
    if not evidence and _has_positive_fever(text):
        evidence = "fever"
    if not evidence:
        return None
    return _rule("WOUND-001", "Wound infection escalation cue", "urgent", evidence, "WOUND-INFECTION-ESCALATION-v1")


CHECKS = (
    _altered_mental_status,
    _respiratory_distress,
    _pregnancy_danger,
    _chest_pain,
    _stroke_signs,
    _pediatric_dehydration,
    _fever_escalation,
    _wound_infection,
)


def run_red_flag_checks(intake: Mapping[str, Any]) -> list[RuleResult]:
    """Return fired deterministic rules. Requires confirmed intake."""
    if not intake.get("confirmed"):
        raise ValueError("red-flag rules require confirmed intake")
    text = _text_from_intake(intake)
    fired: list[RuleResult] = []
    seen: set[str] = set()
    for check in CHECKS:
        result = check(intake, text)
        if result and result.rule_id not in seen:
            fired.append(result)
            seen.add(result.rule_id)
    return fired


def evaluate_rules(intake: Mapping[str, Any], urgency_floor: Urgency = "routine") -> dict[str, Any]:
    """Evaluate red flags and return the locked protocol_urgency floor."""
    fired = run_red_flag_checks(intake)
    protocol_urgency = highest_urgency([urgency_floor, *[rule.urgency for rule in fired]])
    return {
        "confirmed": True,
        "protocol_urgency": protocol_urgency,
        "urgency_floor": protocol_urgency,
        "red_flags": [rule.to_dict() for rule in fired],
    }
