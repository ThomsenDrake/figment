"""Prompt assembly for the Figment protocol navigator."""

from __future__ import annotations

import json
from typing import Any

from .trace import stable_hash


SYSTEM_PROMPT = """You are Figment, an offline protocol navigator for a trained responder.
You are NOT a clinician. Do not diagnose and do not prescribe.
Use ONLY the protocol cards provided below.

Rules:
- Extract relevant facts from messy notes and mark them as reported, missing, unclear, or conflicting.
- Treat audio draft text only as confirmed intake if the medic accepted or edited it; never treat unconfirmed audio drafts as facts.
- Select candidate protocol pathways only from retrieved cards and cite every card you rely on in source_cards.
- Stay inside the retrieved cards.
- Use allowed_facts_inventory as the complete fact boundary. Do not introduce handoff facts that are absent from confirmed intake, deterministic rules, or retrieved cards.
- Fill every key shown in REQUIRED_JSON_SKELETON, including every handoff_note_sbar subkey.
- Use required_observations_inventory when choosing missing_info_to_collect and next_observations_to_collect.
- Do not discharge, clear for discharge, or send anyone home. Do not provide autonomous routing; route only to local protocol, supervisor, clinician, or emergency pathway when the cited cards support it.
- Do not give a drug dose unless a cited card explicitly contains it.
- If critical info is missing, list it in missing_info_to_collect and prioritize the next 3 to 5 observations to collect.
- Convert card guidance into a case-specific responder checklist.
- Never lower deterministic red-flag urgency.
- Denied or absent symptoms are absence facts, not red_flags. If deterministic_red_flags is empty and the urgency floor is routine, do not escalate only because an emergency card was retrieved.
- If no relevant card was retrieved, direct the responder to local protocol, supervisor, clinician, or emergency pathway; do not improvise.
- Refuse out-of-scope or unsafe requests via safety_boundary.
- Return ONLY JSON matching the required navigator schema. No chain-of-thought."""


OUTPUT_SCHEMA = {
    "protocol_urgency": "routine | monitor | urgent | emergency",
    "red_flags": [],
    "intake_facts": [{"fact": "", "status": "reported | missing | unclear | conflicting", "source": "structured_field | responder_note | protocol_card"}],
    "candidate_protocol_pathways": [{"card_id": "", "reason_relevant": ""}],
    "missing_info_to_collect": [],
    "next_observations_to_collect": [],
    "conflicts_or_uncertainties": [],
    "responder_checklist": [],
    "do_not_do": [],
    "source_cards": [],
    "handoff_note_sbar": {
        "situation": "",
        "background": "",
        "assessment_observations_only": "",
        "handoff_request": "",
    },
    "responder_plain_language_script": "",
    "safety_boundary": "",
}


REQUIRED_JSON_SKELETON = {
    "protocol_urgency": "routine",
    "red_flags": [],
    "intake_facts": [{"fact": "", "status": "reported", "source": "structured_field"}],
    "candidate_protocol_pathways": [{"card_id": "", "reason_relevant": ""}],
    "missing_info_to_collect": [],
    "next_observations_to_collect": [],
    "conflicts_or_uncertainties": [],
    "responder_checklist": [],
    "do_not_do": [],
    "source_cards": [],
    "handoff_note_sbar": {
        "situation": "",
        "background": "",
        "assessment_observations_only": "",
        "handoff_request": "",
    },
    "responder_plain_language_script": "",
    "safety_boundary": "",
}

ROUTINE_OR_NEGATED_CASE_GUIDANCE = [
    "Do not convert denied or absent symptoms into red_flags; keep them as absence facts only when they appear in confirmed intake.",
    "If deterministic_red_flags is empty and protocol_urgency_floor is routine, keep protocol_urgency routine unless confirmed allowed facts match retrieved card escalation criteria.",
    "Nearby emergency card language is not escalation by itself; do not copy nearby emergency card language into red_flags, SBAR, or checklist unless an allowed fact supports it.",
]


def build_prompt(
    intake: dict[str, Any],
    retrieved_cards: list[dict[str, Any]],
    rule_results: list[dict[str, Any]],
    urgency_floor: str,
    audio_draft: dict[str, Any] | None = None,
) -> tuple[str, str]:
    card_payload = [item.get("card", item) for item in retrieved_cards]
    context = {
        "structured_intake": intake,
        "deterministic_red_flags": rule_results,
        "protocol_urgency_floor": urgency_floor,
        "retrieved_protocol_cards": card_payload,
        "allowed_facts_inventory": _allowed_facts_inventory(intake, card_payload, rule_results, urgency_floor),
        "required_observations_inventory": _required_observations_inventory(card_payload),
        "routine_or_negated_case_guidance": ROUTINE_OR_NEGATED_CASE_GUIDANCE,
        "audio_draft_policy": {
            "audio_is_pre_navigation_only": True,
            "unconfirmed_audio_is_not_a_fact": True,
            "audio_draft": _safe_audio_draft_context(audio_draft),
        },
        "navigator_output_schema": OUTPUT_SCHEMA,
        "required_json_skeleton": REQUIRED_JSON_SKELETON,
    }
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"REQUIRED_JSON_SKELETON:\n{json.dumps(REQUIRED_JSON_SKELETON, indent=2, sort_keys=True)}\n\n"
        f"CONTEXT:\n{json.dumps(context, indent=2, sort_keys=True)}"
    )
    return prompt, stable_hash(SYSTEM_PROMPT)


def _allowed_facts_inventory(
    intake: dict[str, Any],
    card_payload: list[dict[str, Any]],
    rule_results: list[dict[str, Any]],
    urgency_floor: str,
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = [
        {
            "source": "confirmed_intake",
            "field": "confirmed",
            "value": intake.get("confirmed") is True,
        },
        {
            "source": "deterministic_rule",
            "field": "protocol_urgency_floor",
            "value": urgency_floor,
        },
    ]
    if intake.get("confirmed") is True:
        for field, value in sorted(intake.items()):
            if field == "confirmed" or not _has_value(value):
                continue
            inventory.append({"source": "confirmed_intake", "field": field, "value": value})
    else:
        inventory.append(
            {
                "source": "confirmed_intake",
                "field": "unconfirmed_intake_policy",
                "value": "Do not use unconfirmed intake fields as navigator facts.",
            }
        )

    if not rule_results:
        inventory.append({"source": "deterministic_rule", "field": "red_flags", "value": []})
    for rule in rule_results:
        inventory.append(
            {
                "source": "deterministic_rule",
                "field": "red_flag",
                "rule_id": rule.get("rule_id"),
                "label": rule.get("label"),
                "urgency": rule.get("urgency"),
                "evidence": rule.get("evidence"),
                "card_id": rule.get("card_id"),
            }
        )

    for card in card_payload:
        card_id = str(card.get("card_id", "")).strip()
        if not card_id:
            continue
        for field in (
            "title",
            "applies_to",
            "red_flags",
            "escalation_criteria",
            "local_actions",
            "forbidden_actions",
            "safety_boundary",
        ):
            value = card.get(field)
            if _has_value(value):
                inventory.append(
                    {
                        "source": "retrieved_protocol_card",
                        "card_id": card_id,
                        "field": field,
                        "value": value,
                    }
                )
    return inventory


def _required_observations_inventory(card_payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for card in card_payload:
        required = card.get("required_observations")
        if not isinstance(required, list) or not required:
            continue
        card_id = str(card.get("card_id", "")).strip()
        if not card_id:
            continue
        inventory.append(
            {
                "card_id": card_id,
                "title": str(card.get("title", "")).strip(),
                "required_observations": [str(item) for item in required if _has_value(item)],
            }
        )
    return inventory


def _safe_audio_draft_context(audio_draft: dict[str, Any] | None) -> dict[str, Any] | None:
    if not audio_draft:
        return None
    accepted_or_edited_fields = []
    for suggestion in audio_draft.get("suggested_fields", []):
        if not isinstance(suggestion, dict) or suggestion.get("status") not in {"accepted", "edited"}:
            continue
        accepted_or_edited_fields.append(
            {
                "field": suggestion.get("field"),
                "status": suggestion.get("status"),
            }
        )
    return {
        "audio_intake_path": audio_draft.get("audio_intake_path"),
        "audio_runtime": audio_draft.get("audio_runtime"),
        "confirmation_status": audio_draft.get("confirmation_status"),
        "confirmed_intake_required": audio_draft.get("confirmed_intake_required"),
        "accepted_or_edited_fields": accepted_or_edited_fields,
        "raw_audio_stored": False,
    }


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True
