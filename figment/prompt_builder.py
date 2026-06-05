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
- Do not give a drug dose unless a cited card explicitly contains it.
- If critical info is missing, list it in missing_info_to_collect and prioritize the next 3 to 5 observations to collect.
- Convert card guidance into a case-specific responder checklist.
- Never lower deterministic red-flag urgency.
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
        "audio_draft_policy": {
            "audio_is_pre_navigation_only": True,
            "unconfirmed_audio_is_not_a_fact": True,
            "audio_draft": audio_draft,
        },
        "navigator_output_schema": OUTPUT_SCHEMA,
    }
    prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{json.dumps(context, indent=2, sort_keys=True)}"
    return prompt, stable_hash(SYSTEM_PROMPT)
