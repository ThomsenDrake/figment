"""Prompt contract tests for load-bearing navigator model calls."""

from __future__ import annotations

import json
from typing import Any

from figment.prompt_builder import build_prompt


REQUIRED_TOP_LEVEL_KEYS = {
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


def _context_from_prompt(prompt: str) -> dict[str, Any]:
    return json.loads(prompt.split("\n\nCONTEXT:\n", maxsplit=1)[1])


def _confirmed_chest_pain_intake() -> dict[str, Any]:
    return {
        "setting": "mobile clinic",
        "patient_age": "52",
        "pregnancy_status": "not_applicable",
        "chief_concern": "Chest pain",
        "symptoms": "Crushing chest pain and shortness of breath",
        "vitals": "HR 118; blood pressure pending",
        "allergies": "unknown",
        "medications": "unknown",
        "available_supplies": "AED, radio",
        "responder_note": "Adult reports chest pain after cleanup work.",
        "confirmed": True,
    }


def _emergency_chest_pain_rules() -> list[dict[str, Any]]:
    return [
        {
            "rule_id": "red_flag_chest_pain",
            "label": "Chest pain escalation cue",
            "urgency": "emergency",
            "evidence": "chest pain",
            "card_id": "CHEST-PAIN-ESCALATION-v1",
        }
    ]


def _retrieved_cards() -> list[dict[str, Any]]:
    return [
        {
            "card_id": "CHEST-PAIN-ESCALATION-v1",
            "score": 1.0,
            "source": "test",
            "card": {
                "card_id": "CHEST-PAIN-ESCALATION-v1",
                "title": "Chest pain escalation",
                "required_observations": [
                    "chest pain description",
                    "onset and duration",
                    "available vital signs",
                ],
                "red_flags": ["chest pain or pressure"],
                "escalation_criteria": [
                    "Chest pain with shortness of breath requires emergency escalation.",
                ],
                "local_actions": ["Document onset, duration, and vital signs."],
                "forbidden_actions": ["Do not diagnose the cause of chest pain."],
            },
        },
        {
            "card_id": "REFERRAL-SBAR-v1",
            "score": 0.5,
            "source": "test",
            "card": {
                "card_id": "REFERRAL-SBAR-v1",
                "title": "Referral and SBAR format",
                "required_observations": [
                    "situation or reason for handoff",
                    "objective observations only",
                    "source protocol card IDs",
                ],
                "local_actions": ["Use Situation, Background, Assessment, and Request."],
                "forbidden_actions": ["Do not add a diagnosis as the assessment."],
            },
        },
    ]


def test_prompt_includes_literal_required_json_skeleton_with_all_schema_keys() -> None:
    prompt, _ = build_prompt(
        _confirmed_chest_pain_intake(),
        _retrieved_cards(),
        _emergency_chest_pain_rules(),
        "emergency",
    )

    assert "REQUIRED_JSON_SKELETON:" in prompt
    assert "Do not discharge" in prompt
    assert "autonomous routing" in prompt
    context = _context_from_prompt(prompt)

    skeleton = context["required_json_skeleton"]
    assert set(skeleton) == REQUIRED_TOP_LEVEL_KEYS
    assert set(skeleton["handoff_note_sbar"]) == {
        "situation",
        "background",
        "assessment_observations_only",
        "handoff_request",
    }
    assert isinstance(skeleton["source_cards"], list)
    assert isinstance(skeleton["missing_info_to_collect"], list)


def test_prompt_context_lists_allowed_facts_and_required_observations() -> None:
    prompt, _ = build_prompt(
        _confirmed_chest_pain_intake(),
        _retrieved_cards(),
        _emergency_chest_pain_rules(),
        "emergency",
        audio_draft={
            "confirmation_status": "unconfirmed",
            "transcript": "Audio-only chest pain phrase must not become an allowed fact.",
        },
    )
    context = _context_from_prompt(prompt)

    allowed_facts_text = json.dumps(context["allowed_facts_inventory"], sort_keys=True)
    assert "confirmed_intake" in allowed_facts_text
    assert "Crushing chest pain and shortness of breath" in allowed_facts_text
    assert "deterministic_rule" in allowed_facts_text
    assert "red_flag_chest_pain" in allowed_facts_text
    assert "retrieved_protocol_card" in allowed_facts_text
    assert "Chest pain escalation" in allowed_facts_text
    assert "Audio-only chest pain phrase" not in allowed_facts_text
    assert "Audio-only chest pain phrase" not in prompt

    observations = context["required_observations_inventory"]
    chest = next(item for item in observations if item["card_id"] == "CHEST-PAIN-ESCALATION-v1")
    assert chest["required_observations"] == [
        "chest pain description",
        "onset and duration",
        "available vital signs",
    ]


def test_prompt_context_guides_routine_and_negated_cases() -> None:
    prompt, _ = build_prompt(
        {
            **_confirmed_chest_pain_intake(),
            "chief_concern": "routine cough check",
            "symptoms": "mild cough, no chest pain, no shortness of breath, speaking normally",
        },
        _retrieved_cards(),
        [],
        "routine",
    )
    context = _context_from_prompt(prompt)

    guidance_text = " ".join(context["routine_or_negated_case_guidance"])
    assert "Do not convert denied or absent symptoms into red_flags" in guidance_text
    assert "keep protocol_urgency routine" in guidance_text
    assert "nearby emergency card language" in guidance_text
