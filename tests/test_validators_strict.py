from typing import Any

from figment.validators import validate_navigator_output


CHEST_CARD = {
    "card_id": "CHEST-PAIN-ESCALATION-v1",
    "title": "Chest pain escalation",
    "required_observations": [
        "chest pain description",
        "onset and duration",
        "shortness of breath report",
        "sweating or fainting report",
        "radiation to arm, jaw, back, or shoulder",
        "available vital signs",
    ],
}


def _confirmed_chest_pain_intake() -> dict[str, Any]:
    return {
        "setting": "mobile clinic",
        "patient_age": "57 years",
        "pregnancy_status": "not_applicable",
        "chief_concern": "chest pressure",
        "symptoms": "chest pain with shortness of breath and sweating",
        "vitals": "heart rate 118; blood pressure pending",
        "allergies": "unknown",
        "medications": "unknown",
        "available_supplies": "AED, radio, transport path",
        "responder_note": "Synthetic case. Adult reports chest pressure after cleanup work.",
        "confirmed": True,
    }


def _chest_rule() -> dict[str, str]:
    return {
        "rule_id": "red_flag_chest_pain",
        "label": "Chest pain escalation cue",
        "urgency": "emergency",
        "evidence": "chest pain with shortness of breath",
        "card_id": "CHEST-PAIN-ESCALATION-v1",
    }


def _navigator_output(**overrides: Any) -> dict[str, Any]:
    output: dict[str, Any] = {
        "protocol_urgency": "emergency",
        "red_flags": [_chest_rule()],
        "intake_facts": [
            {
                "fact": "Chest pain with shortness of breath reported.",
                "status": "reported",
                "source": "structured_field",
            }
        ],
        "candidate_protocol_pathways": [
            {
                "card_id": "CHEST-PAIN-ESCALATION-v1",
                "reason_relevant": "Chest pain with shortness of breath was reported.",
            }
        ],
        "missing_info_to_collect": [
            "chest pain description",
            "onset and duration",
            "shortness of breath report",
            "sweating or fainting report",
            "radiation to arm, jaw, back, or shoulder",
            "available vital signs",
        ],
        "next_observations_to_collect": [
            "chest pain description",
            "onset and duration",
            "shortness of breath report",
            "available vital signs",
        ],
        "conflicts_or_uncertainties": [],
        "responder_checklist": ["Escalate per cited local protocol."],
        "do_not_do": ["Do not diagnose or prescribe."],
        "source_cards": ["CHEST-PAIN-ESCALATION-v1"],
        "handoff_note_sbar": {
            "situation": "Chest pressure with shortness of breath.",
            "background": "Mobile clinic synthetic case after cleanup work.",
            "assessment_observations_only": "Chest pain and sweating reported. Heart rate 118; blood pressure pending.",
            "handoff_request": "Request escalation per cited local protocol.",
        },
        "responder_plain_language_script": "I am going to keep checking observations and follow the local escalation path.",
        "safety_boundary": "This output does not diagnose or prescribe and does not replace local protocol.",
    }
    output.update(overrides)
    return output


def test_strict_validator_requires_full_schema_and_fired_rule_card_citation() -> None:
    output = _navigator_output(source_cards=["SAFETY-BOUNDARIES-v1"])
    del output["red_flags"]
    del output["intake_facts"]

    result = validate_navigator_output(
        output,
        {
            "CHEST-PAIN-ESCALATION-v1",
            "SAFETY-BOUNDARIES-v1",
        },
        urgency_floor="emergency",
        confirmed_intake=_confirmed_chest_pain_intake(),
        rule_results=[_chest_rule()],
        strict_schema=True,
    )

    assert not result.passed
    assert any("missing required schema keys" in failure for failure in result.failures)
    assert any("fired rule card CHEST-PAIN-ESCALATION-v1 is not cited" in failure for failure in result.failures)


def test_strict_validator_rejects_known_but_unretrieved_card_citation() -> None:
    output = _navigator_output(
        source_cards=["CHEST-PAIN-ESCALATION-v1", "WOUND-INFECTION-ESCALATION-v1"],
        candidate_protocol_pathways=[
            {
                "card_id": "WOUND-INFECTION-ESCALATION-v1",
                "reason_relevant": "The model reached for a known card that was not retrieved.",
            }
        ],
    )

    result = validate_navigator_output(
        output,
        {
            "CHEST-PAIN-ESCALATION-v1",
            "SAFETY-BOUNDARIES-v1",
            "WOUND-INFECTION-ESCALATION-v1",
        },
        urgency_floor="emergency",
        confirmed_intake=_confirmed_chest_pain_intake(),
        rule_results=[_chest_rule()],
        retrieved_card_ids={"CHEST-PAIN-ESCALATION-v1", "SAFETY-BOUNDARIES-v1"},
        strict_schema=True,
    )

    assert not result.passed
    assert any("not in allowed/retrieved card IDs" in failure for failure in result.failures)


def test_strict_validator_allows_known_fired_rule_card_even_when_retrieval_missed_it() -> None:
    output = _navigator_output(
        source_cards=["CHEST-PAIN-ESCALATION-v1"],
        candidate_protocol_pathways=[
            {
                "card_id": "CHEST-PAIN-ESCALATION-v1",
                "reason_relevant": "Chest pain red flag fired deterministically.",
            }
        ],
    )

    result = validate_navigator_output(
        output,
        {
            "CHEST-PAIN-ESCALATION-v1",
            "SAFETY-BOUNDARIES-v1",
            "WOUND-INFECTION-ESCALATION-v1",
        },
        urgency_floor="emergency",
        confirmed_intake=_confirmed_chest_pain_intake(),
        rule_results=[_chest_rule()],
        retrieved_card_ids={"SAFETY-BOUNDARIES-v1"},
        strict_schema=True,
    )

    assert result.passed


def test_strict_validator_rejects_generic_missing_info_and_hallucinated_sbar_facts() -> None:
    output = _navigator_output(
        missing_info_to_collect=["ask anything else that seems relevant"],
        next_observations_to_collect=["keep monitoring"],
        handoff_note_sbar={
            "situation": "Chest pressure with shortness of breath.",
            "background": "Mobile clinic synthetic case after cleanup work.",
            "assessment_observations_only": "Chest pain reported with skull fracture and blood pressure 220/140 observed.",
            "handoff_request": "Request escalation per cited local protocol.",
        },
    )

    result = validate_navigator_output(
        output,
        {"CHEST-PAIN-ESCALATION-v1", "SAFETY-BOUNDARIES-v1"},
        urgency_floor="emergency",
        confirmed_intake=_confirmed_chest_pain_intake(),
        rule_results=[_chest_rule()],
        retrieved_cards=[{"card": CHEST_CARD}],
        strict_schema=True,
    )

    assert not result.passed
    assert any("missing_info_to_collect does not reference required observations" in failure for failure in result.failures)
    assert any("unsupported high-risk handoff facts" in failure for failure in result.failures)


def test_validator_rejects_broader_unsafe_action_phrases() -> None:
    output = _navigator_output(
        responder_checklist=["Administer aspirin now and discharge home if symptoms improve."],
    )

    result = validate_navigator_output(
        output,
        {"CHEST-PAIN-ESCALATION-v1"},
        urgency_floor="emergency",
    )

    assert not result.passed
    assert any("forbidden clinical language" in failure for failure in result.failures)
