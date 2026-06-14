"""Prompt assembly for the Figment protocol navigator."""

from __future__ import annotations

import json
from typing import Any

from .observation_targets import build_case_fact_ledger, build_handoff_note_sbar_template, required_observation_targets
from .trace import stable_hash


SUPPORT_SOURCE_CARD_IDS = ("SAFETY-BOUNDARIES-v1", "REFERRAL-SBAR-v1")

SYSTEM_PROMPT = """You are Figment, an offline protocol navigator for a trained responder.
You are NOT a clinician. Do not diagnose and do not prescribe.
Use ONLY the protocol cards provided below.

Rules:
- Extract relevant facts from messy notes and mark them as reported, missing, unclear, or conflicting.
- Treat audio draft text only as confirmed intake if the medic accepted or edited it; never treat unconfirmed audio drafts as facts.
- Select candidate protocol pathways only from retrieved cards and cite every card you rely on in source_cards.
- Keep candidate_protocol_pathways focused on the clinical target/escalation pathway. Do not add SAFETY-BOUNDARIES-v1 or REFERRAL-SBAR-v1 as candidate pathways unless that support card itself is the target pathway.
- Stay inside the retrieved cards.
- Use allowed_facts_inventory as the complete fact boundary. Do not introduce handoff facts that are absent from confirmed intake, deterministic rules, or retrieved cards.
- Fill every key shown in REQUIRED_JSON_SKELETON, including every handoff_note_sbar subkey.
- Use required_observations_inventory when choosing missing_info_to_collect and next_observations_to_collect.
- Choose required observation IDs before writing observation text.
- Emit selected_required_observation_ids as a trace-only key when selecting any required_observation_targets. Include every mandatory_required_observation_target_id.
- Use mandatory_required_observation_targets as the compact checklist for model-owned observation text. Copy each display_text into missing_info_to_collect before scaffold fill.
- Include every mandatory_source_card_id in source_cards. This includes deterministic clinical cards and retrieved support cards used for safety/SBAR.
- Cover required_observation_targets by id when writing missing_info_to_collect and next_observations_to_collect.
- For each required_observation_target on a cited clinical source card, make the target display_text visible in missing_info_to_collect. Use the same display_text or a very close responder-facing wording; if already partly reported, frame it as confirm/document.
- Use case_fact_ledger.present for supported facts and case_fact_ledger.absent_or_denied for symptoms that must not become red_flags.
- Use handoff_note_sbar_template as the grounded SBAR starting point; rewrite only within confirmed intake, deterministic rules, and cited cards.
- Do not discharge, clear for discharge, or send anyone home. Do not provide autonomous routing; route only to local protocol, supervisor, clinician, or emergency pathway when the cited cards support it.
- Do not give a drug dose unless a cited card explicitly contains it.
- If critical info is missing, list it in missing_info_to_collect and prioritize the next 3 to 5 observations to collect.
- Convert card guidance into a case-specific responder checklist.
- Never lower deterministic red-flag urgency.
- Denied or absent symptoms are absence facts, not red_flags. If deterministic_red_flags is empty and the urgency floor is routine, do not escalate only because an emergency card was retrieved.
- If no relevant card was retrieved, direct the responder to local protocol, supervisor, clinician, or emergency pathway; do not improvise.
- Refuse out-of-scope or unsafe requests via safety_boundary.
- Return ONLY JSON matching the required navigator schema, plus selected_required_observation_ids when required observation targets are selected. No chain-of-thought."""


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
    mandatory_source_card_ids = _mandatory_source_card_ids(rule_results, card_payload)
    observation_targets = required_observation_targets(card_payload)
    mandatory_required_observation_target_ids = _mandatory_required_observation_target_ids(
        observation_targets,
        mandatory_source_card_ids,
    )
    context = {
        "structured_intake": intake,
        "deterministic_red_flags": rule_results,
        "protocol_urgency_floor": urgency_floor,
        "retrieved_protocol_cards": card_payload,
        "allowed_facts_inventory": _allowed_facts_inventory(intake, card_payload, rule_results, urgency_floor),
        "mandatory_source_card_ids": mandatory_source_card_ids,
        "required_observations_inventory": _required_observations_inventory(card_payload),
        "required_observation_targets": observation_targets,
        "mandatory_required_observation_target_ids": mandatory_required_observation_target_ids,
        "mandatory_required_observation_targets": _mandatory_required_observation_targets(
            observation_targets,
            mandatory_required_observation_target_ids,
        ),
        "required_observation_generation_policy": _required_observation_generation_policy(),
        "case_fact_ledger": build_case_fact_ledger(intake),
        "handoff_note_sbar_template": build_handoff_note_sbar_template(intake, rule_results, urgency_floor),
        "internal_generation_contract": _internal_generation_contract(),
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


def _mandatory_source_card_ids(rule_results: list[dict[str, Any]], card_payload: list[dict[str, Any]]) -> list[str]:
    card_ids: list[str] = []
    for rule in rule_results:
        card_id = str(rule.get("card_id", "")).strip()
        if card_id and card_id not in card_ids:
            card_ids.append(card_id)
    retrieved_ids = {str(card.get("card_id", "")).strip() for card in card_payload}
    for card_id in SUPPORT_SOURCE_CARD_IDS:
        if card_id in retrieved_ids and card_id not in card_ids:
            card_ids.append(card_id)
    return card_ids


def _mandatory_required_observation_target_ids(
    observation_targets: list[dict[str, Any]],
    mandatory_source_card_ids: list[str],
) -> list[str]:
    mandatory_cards = set(mandatory_source_card_ids)
    ids: list[str] = []
    for target in observation_targets:
        card_id = str(target.get("card_id", "")).strip()
        target_id = str(target.get("id", "")).strip()
        if card_id in SUPPORT_SOURCE_CARD_IDS:
            continue
        if card_id in mandatory_cards and target_id and target_id not in ids:
            ids.append(target_id)
    return ids


def _mandatory_required_observation_targets(
    observation_targets: list[dict[str, Any]],
    mandatory_required_observation_target_ids: list[str],
) -> list[dict[str, Any]]:
    mandatory_ids = set(mandatory_required_observation_target_ids)
    return [
        {
            "id": str(target.get("id", "")).strip(),
            "card_id": str(target.get("card_id", "")).strip(),
            "title": str(target.get("title", "")).strip(),
            "display_text": str(target.get("display_text", "")).strip(),
        }
        for target in observation_targets
        if str(target.get("id", "")).strip() in mandatory_ids
    ]


def _required_observation_generation_policy() -> dict[str, Any]:
    return {
        "model_owned_not_scaffold_filled": True,
        "mandatory_required_observation_targets": (
            "This compact list is the model-owned checklist. Every display_text in "
            "mandatory_required_observation_targets must appear in missing_info_to_collect, even when "
            "next_observations_to_collect stays prioritized and concise."
        ),
        "source_card_scope": (
            "For every required_observation_target whose card_id is in source_cards and is not "
            "SAFETY-BOUNDARIES-v1 or REFERRAL-SBAR-v1, the assistant output itself must make "
            "that target visible in missing_info_to_collect."
        ),
        "text_requirement": (
            "Use the target display_text exactly when it is short and responder-facing. If the "
            "fact is already partly reported, still include the cue as confirm/document wording."
        ),
        "next_observations_to_collect": (
            "Prioritize the most urgent 3 to 5 required observation display_text cues plus any "
            "case-specific vital signs; missing_info_to_collect may carry the fuller set."
        ),
        "source_cards": (
            "source_cards must include every mandatory_source_card_id. The mandatory list includes "
            "deterministic clinical cards plus retrieved SAFETY-BOUNDARIES-v1 and REFERRAL-SBAR-v1 "
            "when those support cards are used for safety or SBAR fields."
        ),
        "selected_required_observation_ids": (
            "When selected_required_observation_ids is emitted, include every "
            "mandatory_required_observation_target_id and any additional target id whose display_text is visible "
            "in missing_info_to_collect or next_observations_to_collect. This key is trace-only and will be stripped."
        ),
    }


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


def _internal_generation_contract() -> dict[str, Any]:
    return {
        "trace_only_keys": ["selected_required_observation_ids"],
        "required_when_required_observation_targets_selected": ["selected_required_observation_ids"],
        "selected_required_observation_ids": (
            "selected_required_observation_ids must be emitted when any required_observation_targets are selected. "
            "It must include every mandatory_required_observation_target_id and may include additional ids from "
            "required_observation_targets that are covered in missing_info_to_collect or "
            "next_observations_to_collect. Select ids first, then write recognizable responder-facing observation "
            "text for each selected id. This key is trace-only."
        ),
        "strip_before_user_display": True,
    }


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
