import importlib

import pytest

from figment.config import FigmentConfig, OMNI_MODEL_ID
from figment.validators import validate_audio_ready, validate_navigator_output


def test_config_defaults_to_omni_and_gates_split_stack() -> None:
    config = FigmentConfig().validated()

    assert config.model_stack == "omni_native"
    assert config.active_model_id == OMNI_MODEL_ID

    with pytest.raises(ValueError, match="ALLOW_STRETCH_STACK"):
        FigmentConfig(model_stack="base_nano_parakeet").validated()

    with pytest.raises(ValueError, match="parakeet_nemo requires ALLOW_STRETCH_STACK"):
        FigmentConfig(audio_backend="parakeet_nemo").validated()


def test_red_flag_rules_run_only_on_confirmed_intake() -> None:
    app = importlib.import_module("app")
    unconfirmed = app.collect_intake(
        setting="shelter clinic",
        patient_age="52",
        pregnancy_status="not_applicable",
        chief_concern="Chest pain and shortness of breath",
        symptoms="crushing chest pain",
        vitals="HR 118",
        allergies="unknown",
        medications="unknown",
        available_supplies="oxygen, AED",
        responder_note="Adult reports chest pain after cleanup work.",
    )

    assert app.evaluate_red_flags(unconfirmed) == []

    confirmed = app.confirm_intake(unconfirmed)
    rules = app.evaluate_red_flags(confirmed)

    assert {rule["rule_id"] for rule in rules} >= {"red_flag_chest_pain"}
    assert app.urgency_floor_from_rules(rules) == "emergency"


def test_validators_reject_downgraded_or_unsafe_navigator_output() -> None:
    output = {
        "protocol_urgency": "routine",
        "candidate_protocol_pathways": [
            {"card_id": "CHEST-PAIN-v1", "reason_relevant": "Chest pain was reported."}
        ],
        "missing_info_to_collect": [],
        "next_observations_to_collect": [],
        "conflicts_or_uncertainties": [],
        "responder_checklist": ["Diagnose the cause and give 5 mg medication."],
        "do_not_do": [],
        "source_cards": ["CHEST-PAIN-v1"],
        "handoff_note_sbar": {
            "situation": "Chest pain",
            "background": "Synthetic case",
            "assessment_observations_only": "Pain reported",
            "handoff_request": "Escalate per protocol",
        },
        "responder_plain_language_script": "",
        "safety_boundary": "",
    }

    result = validate_navigator_output(output, {"CHEST-PAIN-v1"}, urgency_floor="emergency")

    assert not result.passed
    assert any("below deterministic floor" in item for item in result.failures)
    assert any("forbidden clinical language" in item for item in result.failures)


def test_validator_allows_negated_safety_boundary_language() -> None:
    output = {
        "protocol_urgency": "emergency",
        "candidate_protocol_pathways": [
            {"card_id": "CHEST-PAIN-v1", "reason_relevant": "Chest pain was reported."}
        ],
        "missing_info_to_collect": [],
        "next_observations_to_collect": [],
        "conflicts_or_uncertainties": [],
        "responder_checklist": ["Escalate per cited local protocol."],
        "do_not_do": ["Do not diagnose the cause."],
        "source_cards": ["CHEST-PAIN-v1"],
        "handoff_note_sbar": {
            "situation": "Chest pain",
            "background": "Synthetic case",
            "assessment_observations_only": "Pain reported",
            "handoff_request": "Escalate per protocol",
        },
        "responder_plain_language_script": "",
        "safety_boundary": "This is not a diagnosis or prescription and cannot diagnose, prescribe, dose, or override local protocol.",
    }

    result = validate_navigator_output(output, {"CHEST-PAIN-v1"}, urgency_floor="emergency")

    assert result.passed


def test_validator_allows_does_not_diagnose_or_prescribe_boundary_language() -> None:
    output = {
        "protocol_urgency": "routine",
        "candidate_protocol_pathways": [
            {"card_id": "SAFETY-BOUNDARIES-v1", "reason_relevant": "Safety boundary review."}
        ],
        "missing_info_to_collect": [],
        "next_observations_to_collect": [],
        "conflicts_or_uncertainties": [],
        "responder_checklist": ["Escalate per cited local protocol."],
        "do_not_do": [],
        "source_cards": ["SAFETY-BOUNDARIES-v1"],
        "handoff_note_sbar": {
            "situation": "Mild symptoms",
            "background": "Synthetic case",
            "assessment_observations_only": "Temperature normal.",
            "handoff_request": "Escalate per protocol",
        },
        "responder_plain_language_script": "",
        "safety_boundary": "This output does not diagnose or prescribe and does not replace local protocol.",
    }

    result = validate_navigator_output(output, {"SAFETY-BOUNDARIES-v1"})

    assert result.passed


def test_validator_allows_negated_do_not_start_or_prescribe_list() -> None:
    output = {
        "protocol_urgency": "urgent",
        "candidate_protocol_pathways": [
            {"card_id": "WOUND-INFECTION-ESCALATION-v1", "reason_relevant": "Wound cue."}
        ],
        "missing_info_to_collect": [],
        "next_observations_to_collect": [],
        "conflicts_or_uncertainties": [],
        "responder_checklist": ["Escalate per cited local protocol."],
        "do_not_do": [
            "Do not start antibiotics, prescribe medication, or provide dosing instructions unless local protocol explicitly provides them."
        ],
        "source_cards": ["WOUND-INFECTION-ESCALATION-v1"],
        "handoff_note_sbar": {
            "situation": "Wound concern",
            "background": "Synthetic case",
            "assessment_observations_only": "Spreading redness reported.",
            "handoff_request": "Escalate per protocol",
        },
        "responder_plain_language_script": "",
        "safety_boundary": "Prototype protocol navigation only.",
    }

    result = validate_navigator_output(output, {"WOUND-INFECTION-ESCALATION-v1"}, urgency_floor="urgent")

    assert result.passed


def test_validator_rejects_incomplete_handoff_schema() -> None:
    output = {
        "protocol_urgency": "urgent",
        "candidate_protocol_pathways": [
            {"card_id": "CHEST-PAIN-v1", "reason_relevant": "Chest pain was reported."}
        ],
        "missing_info_to_collect": [],
        "next_observations_to_collect": [],
        "conflicts_or_uncertainties": [],
        "responder_checklist": ["Escalate per cited local protocol."],
        "do_not_do": ["Do not diagnose the cause."],
        "source_cards": ["CHEST-PAIN-v1"],
        "handoff_note_sbar": {
            "situation": "Chest pain",
            "background": "Synthetic case",
            "assessment_observations_only": "Pain reported",
        },
        "responder_plain_language_script": "",
        "safety_boundary": "Prototype protocol navigation only.",
    }

    result = validate_navigator_output(output, {"CHEST-PAIN-v1"}, urgency_floor="urgent")

    assert not result.passed
    assert any("handoff_note_sbar is missing" in item for item in result.failures)


def test_validator_rejects_forbidden_language_after_unrelated_not() -> None:
    output = {
        "protocol_urgency": "routine",
        "candidate_protocol_pathways": [
            {"card_id": "SAFETY-BOUNDARIES-v1", "reason_relevant": "Safety boundary review."}
        ],
        "missing_info_to_collect": [],
        "next_observations_to_collect": [],
        "conflicts_or_uncertainties": [],
        "responder_checklist": ["Escalate per cited local protocol."],
        "do_not_do": [],
        "source_cards": ["SAFETY-BOUNDARIES-v1"],
        "handoff_note_sbar": {
            "situation": "Mild symptoms",
            "background": "Synthetic case",
            "assessment_observations_only": "Patient not stable, prescribe opioid now.",
            "handoff_request": "Escalate per protocol",
        },
        "responder_plain_language_script": "",
        "safety_boundary": "Prototype protocol navigation only.",
    }

    result = validate_navigator_output(output, {"SAFETY-BOUNDARIES-v1"})

    assert not result.passed
    assert any("forbidden clinical language: prescribe" in item for item in result.failures)


def test_validator_rejects_ungrounded_sbar_when_intake_context_is_available() -> None:
    output = {
        "protocol_urgency": "routine",
        "candidate_protocol_pathways": [
            {"card_id": "SAFETY-BOUNDARIES-v1", "reason_relevant": "Safety boundary review."}
        ],
        "missing_info_to_collect": [],
        "next_observations_to_collect": [],
        "conflicts_or_uncertainties": [],
        "responder_checklist": ["Escalate per cited local protocol."],
        "do_not_do": ["Do not diagnose."],
        "source_cards": ["SAFETY-BOUNDARIES-v1"],
        "handoff_note_sbar": {
            "situation": "Head injury with loss of consciousness",
            "background": "Fall from ladder at home",
            "assessment_observations_only": "Blood pressure 220/140 and skull fracture observed.",
            "handoff_request": "Escalate per protocol",
        },
        "responder_plain_language_script": "",
        "safety_boundary": "Prototype protocol navigation only.",
    }
    intake = {
        "setting": "shelter clinic",
        "patient_age": "40",
        "pregnancy_status": "not_applicable",
        "chief_concern": "mild cough",
        "symptoms": "speaking normally, no distress cues",
        "vitals": "temperature normal",
        "allergies": "unknown",
        "medications": "unknown",
        "available_supplies": "radio",
        "responder_note": "No injury reported.",
        "confirmed": True,
    }

    result = validate_navigator_output(
        output,
        {"SAFETY-BOUNDARIES-v1"},
        confirmed_intake=intake,
        rule_results=[],
    )

    assert not result.passed
    assert any("handoff_note_sbar" in item and "not grounded" in item for item in result.failures)


def test_audio_drafts_are_non_authoritative_until_confirmed() -> None:
    app = importlib.import_module("app")
    disabled_config = FigmentConfig(enable_audio_intake=False, audio_backend="none").validated()

    disabled_draft = app.draft_audio_intake(
        transcript="Patient says chest pain, but audio is disabled.",
        config=disabled_config,
    )

    assert disabled_draft["audio_runtime"] == "none"
    assert disabled_draft["confirmed_intake_required"] is False
    assert disabled_draft["provisional_red_flag_mentions"] == []
    assert validate_audio_ready(disabled_draft).passed

    enabled_config = FigmentConfig(
        model_backend="canned",
        enable_audio_intake=True,
        audio_backend="omni_native",
    ).validated()
    audio_draft = app.draft_audio_intake(
        transcript="Adult with chest pain and trouble breathing.",
        config=enabled_config,
    )

    assert audio_draft["confirmation_status"] == "unconfirmed"
    assert "chest pain" in " ".join(audio_draft["provisional_red_flag_mentions"]).lower()
    assert not validate_audio_ready(audio_draft).passed

    intake = app.collect_intake(
        setting="mobile clinic",
        patient_age="adult",
        pregnancy_status="not_applicable",
        chief_concern="",
        symptoms="",
        vitals="",
        allergies="",
        medications="",
        available_supplies="basic kit",
        responder_note="",
    )

    with pytest.raises(ValueError, match="audio-derived intake must be confirmed"):
        app.confirm_intake(intake, audio_draft=audio_draft)

    updated_intake, confirmed_audio = app.confirm_audio_draft(
        intake,
        audio_draft,
        accept=True,
    )
    confirmed_intake = app.confirm_intake(updated_intake, audio_draft=confirmed_audio)

    assert confirmed_audio["confirmation_status"] == "confirmed"
    assert confirmed_intake["confirmed"] is True
    assert app.evaluate_red_flags(confirmed_intake)


def test_app_import_and_blocks_smoke() -> None:
    app = importlib.import_module("app")
    config = FigmentConfig(model_backend="canned", audio_backend="none").validated()

    demo = app.build_app(config=config)

    assert hasattr(demo, "queue")
    assert app.TAB_TITLES == [
        "Intake",
        "Risk Check",
        "Protocol Guidance",
        "Navigator Output + Handoff",
        "Trace",
    ]
