import json
from pathlib import Path
from typing import Any

import figment.navigator as navigator
from figment.config import FigmentConfig


class UnsafeDowngradingModelClient:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def generate_json(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {
            "protocol_urgency": "routine",
            "candidate_protocol_pathways": [
                {
                    "card_id": "CHEST-PAIN-ESCALATION-v1",
                    "reason_relevant": "Chest pain was reported.",
                }
            ],
            "missing_info_to_collect": [],
            "next_observations_to_collect": [],
            "conflicts_or_uncertainties": [],
            "responder_checklist": ["Diagnose the cause and give 5 mg medication."],
            "do_not_do": [],
            "source_cards": ["CHEST-PAIN-ESCALATION-v1"],
            "handoff_note_sbar": {
                "situation": "Chest pain",
                "background": "Synthetic case",
                "assessment_observations_only": "Pain reported",
                "handoff_request": "Escalate per protocol",
            },
            "responder_plain_language_script": "",
            "safety_boundary": "",
        }


class FailingTransportModelClient:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def generate_json(self, *_: Any, **__: Any) -> dict[str, Any]:
        raise navigator.ModelClientError("transport failed")


class RepairingModelClient:
    calls = 0

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def generate_json(self, *_: Any, **__: Any) -> dict[str, Any]:
        type(self).calls += 1
        if type(self).calls == 1:
            return {
                "protocol_urgency": "routine",
                "red_flags": _emergency_chest_pain_rules(),
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
                        "reason_relevant": "Chest pain was reported.",
                    }
                ],
                "missing_info_to_collect": ["repeat vital signs"],
                "next_observations_to_collect": [],
                "conflicts_or_uncertainties": [],
                "responder_checklist": ["Escalate per cited local protocol."],
                "do_not_do": ["Do not diagnose."],
                "source_cards": ["CHEST-PAIN-ESCALATION-v1"],
                "handoff_note_sbar": {
                    "situation": "Chest pain",
                    "background": "Synthetic case",
                    "assessment_observations_only": "Pain reported",
                },
                "responder_plain_language_script": "",
                "safety_boundary": "This output does not diagnose or prescribe and does not replace local protocol.",
            }
        return {
            "protocol_urgency": "emergency",
            "red_flags": _emergency_chest_pain_rules(),
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
            "missing_info_to_collect": ["repeat vital signs"],
            "next_observations_to_collect": ["work of breathing", "level of alertness"],
            "conflicts_or_uncertainties": [],
            "responder_checklist": ["Escalate per cited local protocol."],
            "do_not_do": ["Do not diagnose."],
            "source_cards": ["CHEST-PAIN-ESCALATION-v1"],
            "handoff_note_sbar": {
                "situation": "Chest pain",
                "background": "Adult reports chest pain after cleanup work.",
                "assessment_observations_only": "Crushing chest pain and shortness of breath reported.",
                "handoff_request": "Escalate per protocol",
            },
            "responder_plain_language_script": "",
            "safety_boundary": "This output does not diagnose or prescribe and does not replace local protocol.",
        }


class PartiallyInvalidModelClient:
    calls = 0

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def generate_json(self, *_: Any, **__: Any) -> dict[str, Any]:
        type(self).calls += 1
        if type(self).calls > 1:
            raise navigator.ModelClientError("focused repair unavailable")
        return {
            "protocol_urgency": "emergency",
            "red_flags": _emergency_chest_pain_rules(),
            "intake_facts": [
                {
                    "fact": "Model retained chest pain fact.",
                    "status": "reported",
                    "source": "structured_field",
                }
            ],
            "candidate_protocol_pathways": [
                {
                    "card_id": "CHEST-PAIN-ESCALATION-v1",
                    "reason_relevant": "Model retained cited chest pain pathway.",
                }
            ],
            "missing_info_to_collect": ["Model retained onset and duration question."],
            "next_observations_to_collect": ["Model retained work of breathing check."],
            "conflicts_or_uncertainties": ["Model retained uncertainty note."],
            "responder_checklist": ["Model retained checklist item."],
            "do_not_do": ["Do not diagnose."],
            "source_cards": ["CHEST-PAIN-ESCALATION-v1"],
            "handoff_note_sbar": {
                "situation": "Chest pain",
                "background": "Adult reports chest pain after cleanup work.",
                "assessment_observations_only": "Crushing chest pain and shortness of breath reported.",
            },
            "responder_plain_language_script": "Model retained plain language script.",
            "safety_boundary": "Prototype protocol navigation only; do not diagnose or prescribe.",
        }


class SparseSchemaModelClient:
    calls = 0

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def generate_json(self, *_: Any, **__: Any) -> dict[str, Any]:
        type(self).calls += 1
        if type(self).calls > 1:
            raise navigator.ModelClientError("focused repair unavailable")
        return {
            "protocol_urgency": "emergency",
            "candidate_protocol_pathways": [
                {
                    "card_id": "CHEST-PAIN-ESCALATION-v1",
                    "reason_relevant": "Chest pain with shortness of breath was reported.",
                }
            ],
            "missing_info_to_collect": ["repeat vital signs"],
            "next_observations_to_collect": ["work of breathing"],
            "conflicts_or_uncertainties": [],
            "responder_checklist": ["Model retained checklist item."],
            "do_not_do": ["Do not diagnose."],
            "source_cards": ["CHEST-PAIN-ESCALATION-v1"],
            "handoff_note_sbar": {
                "situation": "Chest pain",
                "background": "Adult reports chest pain after cleanup work.",
                "assessment_observations_only": "Crushing chest pain and shortness of breath reported.",
                "handoff_request": "Escalate per protocol",
            },
            "responder_plain_language_script": "Model retained plain language script.",
            "safety_boundary": "Prototype protocol navigation only; do not diagnose or prescribe.",
        }


class UnretrievedUngroundedModelClient:
    calls = 0

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def generate_json(self, *_: Any, **__: Any) -> dict[str, Any]:
        type(self).calls += 1
        if type(self).calls > 1:
            raise navigator.ModelClientError("focused repair unavailable")
        return {
            "protocol_urgency": "emergency",
            "red_flags": _emergency_chest_pain_rules(),
            "intake_facts": [
                {
                    "fact": "Chest pain with shortness of breath reported.",
                    "status": "reported",
                    "source": "structured_field",
                }
            ],
            "candidate_protocol_pathways": [
                {
                    "card_id": "WOUND-INFECTION-ESCALATION-v1",
                    "reason_relevant": "The model reached for a known but unretrieved card.",
                }
            ],
            "missing_info_to_collect": ["ask anything else that seems relevant"],
            "next_observations_to_collect": ["keep monitoring"],
            "conflicts_or_uncertainties": [],
            "responder_checklist": ["Model retained checklist item."],
            "do_not_do": ["Do not diagnose."],
            "source_cards": ["CHEST-PAIN-ESCALATION-v1", "WOUND-INFECTION-ESCALATION-v1"],
            "handoff_note_sbar": {
                "situation": "Chest pain",
                "background": "Adult reports chest pain after cleanup work.",
                "assessment_observations_only": "Crushing chest pain and shortness of breath reported.",
                "handoff_request": "Escalate per protocol",
            },
            "responder_plain_language_script": "Model retained plain language script.",
            "safety_boundary": "Prototype protocol navigation only; do not diagnose or prescribe.",
        }


class MultiFailureRepairModelClient:
    calls = 0

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def generate_json(self, *_: Any, **__: Any) -> dict[str, Any]:
        type(self).calls += 1
        if type(self).calls > 1:
            raise navigator.ModelClientError("focused repair unavailable")
        return {
            "protocol_urgency": "routine",
            "red_flags": _emergency_chest_pain_rules(),
            "intake_facts": [
                {
                    "fact": "Chest pain with shortness of breath reported.",
                    "status": "reported",
                    "source": "structured_field",
                }
            ],
            "candidate_protocol_pathways": [
                {
                    "card_id": "WOUND-INFECTION-ESCALATION-v1",
                    "reason_relevant": "The model reached for a known but unretrieved card.",
                }
            ],
            "missing_info_to_collect": ["ask anything else that seems relevant"],
            "next_observations_to_collect": ["keep monitoring"],
            "conflicts_or_uncertainties": [],
            "responder_checklist": ["Prescribe opioid now."],
            "do_not_do": [],
            "source_cards": ["WOUND-INFECTION-ESCALATION-v1"],
            "handoff_note_sbar": {
                "situation": "Skull fracture with chest pain",
                "background": "Unrelated unsupported background.",
                "assessment_observations_only": "Blood pressure 220/140 observed.",
            },
            "responder_plain_language_script": ["Model returned the wrong schema type."],
            "safety_boundary": "Prototype protocol navigation only.",
        }


class ObservationThinModelClient:
    calls = 0

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def generate_json(self, *_: Any, **__: Any) -> dict[str, Any]:
        type(self).calls += 1
        return {
            "protocol_urgency": "emergency",
            "red_flags": _emergency_chest_pain_rules(),
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
            "missing_info_to_collect": ["available vital signs"],
            "next_observations_to_collect": [],
            "conflicts_or_uncertainties": [],
            "responder_checklist": ["Escalate per cited local protocol."],
            "do_not_do": ["Do not diagnose."],
            "source_cards": ["CHEST-PAIN-ESCALATION-v1"],
            "handoff_note_sbar": {
                "situation": "Chest pain with shortness of breath.",
                "background": "Mobile clinic adult case.",
                "assessment_observations_only": "Crushing chest pain and shortness of breath reported. HR 118.",
                "handoff_request": "Escalate per protocol.",
            },
            "responder_plain_language_script": "I am going to keep checking observations and follow the local escalation path.",
            "safety_boundary": "This output does not diagnose or prescribe and does not replace local protocol.",
        }


class SelectedObservationIdsModelClient:
    calls = 0

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def generate_json(self, *_: Any, **__: Any) -> dict[str, Any]:
        type(self).calls += 1
        return {
            "protocol_urgency": "emergency",
            "red_flags": _emergency_chest_pain_rules(),
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
                "Ask the patient to describe the pain in plain words.",
                "Ask when the pain started and whether it changed.",
                "available vital signs",
            ],
            "next_observations_to_collect": [],
            "conflicts_or_uncertainties": [],
            "responder_checklist": ["Escalate per cited local protocol."],
            "do_not_do": ["Do not diagnose."],
            "source_cards": ["CHEST-PAIN-ESCALATION-v1"],
            "handoff_note_sbar": {
                "situation": "Chest pain with shortness of breath.",
                "background": "Mobile clinic adult case.",
                "assessment_observations_only": "Crushing chest pain and shortness of breath reported. HR 118.",
                "handoff_request": "Escalate per protocol.",
            },
            "responder_plain_language_script": "I am going to keep checking observations and follow the local escalation path.",
            "safety_boundary": "This output does not diagnose or prescribe and does not replace local protocol.",
            "selected_required_observation_ids": [
                "CHEST-PAIN-ESCALATION-v1::required_observation::1",
                "CHEST-PAIN-ESCALATION-v1::required_observation::2",
                "NOT-A-REAL-TARGET",
            ],
        }


def _confirmed_chest_pain_intake() -> dict[str, Any]:
    return {
        "setting": "mobile clinic",
        "patient_age": "52",
        "pregnancy_status": "not_applicable",
        "chief_concern": "Chest pain",
        "symptoms": "Crushing chest pain and shortness of breath",
        "vitals": "HR 118",
        "allergies": "unknown",
        "medications": "unknown",
        "available_supplies": "oxygen, AED",
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


def _retrieved_chest_pain_cards() -> list[dict[str, Any]]:
    return [
        {
            "card_id": "CHEST-PAIN-ESCALATION-v1",
            "title": "Chest pain escalation",
            "score": 1.0,
            "source": "test",
            "card": {
                "card_id": "CHEST-PAIN-ESCALATION-v1",
                "title": "Chest pain escalation",
                "required_observations": [
                    "chest pain description",
                    "onset and duration",
                    "shortness of breath report",
                    "available vital signs",
                ],
            },
        }
    ]


def _confirmed_audio_draft_with_raw_metadata() -> dict[str, Any]:
    return {
        "audio_intake_path": "/tmp/uploads/field-case.wav",
        "audio_model_id": "test-audio-model",
        "field_fill_model_id": "test-fill-model",
        "audio_runtime": "omni_native",
        "transcript": "Adult reports chest pain after cleanup work.",
        "suggested_fields": [
            {
                "field": "chief_concern",
                "draft_value": "Chest pain",
                "status": "accepted",
                "source_snippet": "Adult reports chest pain.",
            }
        ],
        "confirmed_intake_required": True,
        "confirmation_status": "confirmed",
        "raw_audio_stored": False,
        "raw_audio_bytes": "RIFF raw bytes",
        "audio_data": "UklGRmZha2U=",
        "blob": {
            "name": "field-case.wav",
            "payload": "data:audio/wav;base64,UklGRmZha2U=",
        },
        "metadata": {
            "uploaded_filename": "field-case.wav",
            "filename": "field-case.wav",
        },
    }


def test_run_navigation_returns_safe_fallback_for_invalid_model_output(monkeypatch) -> None:
    monkeypatch.setattr(navigator, "ModelClient", UnsafeDowngradingModelClient)

    output, trace = navigator.run_navigation(
        _confirmed_chest_pain_intake(),
        _emergency_chest_pain_rules(),
        config=FigmentConfig(model_backend="hosted_omni", nvidia_api_key="test-nvidia-key"),
        retrieved_cards=_retrieved_chest_pain_cards(),
    )

    output_text = json.dumps(output).lower()
    assert output["protocol_urgency"] == "emergency"
    assert "diagnose" not in output_text
    assert "give 5 mg" not in output_text
    assert trace.navigator_output == output
    assert trace.validator_result["passed"] is True
    assert trace.model_route["fallback_tier"] == "canned"
    assert any("fallback applied" in event for event in trace.events)


def test_run_navigation_labels_transport_fallback_in_trace(monkeypatch) -> None:
    monkeypatch.setattr(navigator, "ModelClient", FailingTransportModelClient)

    output, trace = navigator.run_navigation(
        _confirmed_chest_pain_intake(),
        _emergency_chest_pain_rules(),
        config=FigmentConfig(model_backend="hosted_omni", nvidia_api_key="test-nvidia-key"),
        retrieved_cards=_retrieved_chest_pain_cards(),
    )

    assert output["protocol_urgency"] == "emergency"
    assert trace.model_route["model_backend"] == "hosted_omni"
    assert trace.model_route["fallback_tier"] == "canned"
    assert any("model backend failed" in event for event in trace.events)


def test_run_navigation_retries_hosted_output_repair_before_fallback(monkeypatch) -> None:
    RepairingModelClient.calls = 0
    monkeypatch.setattr(navigator, "ModelClient", RepairingModelClient)

    output, trace = navigator.run_navigation(
        _confirmed_chest_pain_intake(),
        _emergency_chest_pain_rules(),
        config=FigmentConfig(model_backend="hosted_omni", nvidia_api_key="test-nvidia-key"),
        retrieved_cards=_retrieved_chest_pain_cards(),
    )

    assert RepairingModelClient.calls == 2
    assert output["protocol_urgency"] == "emergency"
    assert trace.validator_result["passed"] is True
    assert trace.model_route["fallback_tier"] == "configured"
    assert trace.model_route["fallback_reason"] is None
    assert any("repaired by hosted retry" in event for event in trace.events)


def test_run_navigation_retains_valid_model_fields_with_field_provenance(monkeypatch) -> None:
    PartiallyInvalidModelClient.calls = 0
    monkeypatch.setattr(navigator, "ModelClient", PartiallyInvalidModelClient)

    output, trace = navigator.run_navigation(
        _confirmed_chest_pain_intake(),
        _emergency_chest_pain_rules(),
        config=FigmentConfig(model_backend="hosted_omni", nvidia_api_key="test-nvidia-key"),
        retrieved_cards=_retrieved_chest_pain_cards(),
    )

    assert output["responder_checklist"] == ["Model retained checklist item."]
    assert output["handoff_note_sbar"]["handoff_request"]
    assert trace.validator_result["passed"] is True
    assert trace.model_route["fallback_tier"] == "configured"
    assert trace.model_route["field_level_fallback_used"] is True
    assert trace.field_provenance["responder_checklist"] == "model_raw"
    assert trace.field_provenance["handoff_note_sbar"] == "deterministic_fallback"
    assert trace.to_dict()["field_provenance"]["responder_checklist"] == "model_raw"
    assert any("field-level" in event for event in trace.events)


def test_run_navigation_scrubs_audio_trace_payload(tmp_path: Path) -> None:
    trace_path = tmp_path / "navigator-trace.json"

    _, trace = navigator.run_navigation(
        _confirmed_chest_pain_intake(),
        _emergency_chest_pain_rules(),
        audio_draft=_confirmed_audio_draft_with_raw_metadata(),
        config=FigmentConfig(model_backend="canned"),
        retrieved_cards=_retrieved_chest_pain_cards(),
        trace_path=str(trace_path),
    )

    serialized_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    for payload in (trace.to_dict(), serialized_trace):
        trace_text = json.dumps(payload).lower()
        assert payload["raw_audio_stored"] is False
        assert payload["audio"]["raw_audio_stored"] is False
        assert "raw_audio_bytes" not in trace_text
        assert "audio_data" not in trace_text
        assert "blob" not in trace_text
        assert "base64" not in trace_text
        assert "data:audio" not in trace_text
        assert "field-case.wav" not in trace_text


def test_run_navigation_keeps_safe_audio_route_labels_in_trace() -> None:
    audio_draft = {
        "audio_intake_path": "omni_native",
        "audio_runtime": "omni_native",
        "suggested_fields": [],
        "confirmed_intake_required": True,
        "confirmation_status": "confirmed",
        "raw_audio_stored": False,
    }

    _, trace = navigator.run_navigation(
        _confirmed_chest_pain_intake(),
        _emergency_chest_pain_rules(),
        audio_draft=audio_draft,
        config=FigmentConfig(model_backend="canned"),
        retrieved_cards=_retrieved_chest_pain_cards(),
    )

    assert trace.to_dict()["audio"]["audio_intake_path"] == "omni_native"


def test_run_navigation_strict_schema_prevents_sparse_output_from_model_raw_label(monkeypatch) -> None:
    SparseSchemaModelClient.calls = 0
    monkeypatch.setattr(navigator, "ModelClient", SparseSchemaModelClient)

    output, trace = navigator.run_navigation(
        _confirmed_chest_pain_intake(),
        _emergency_chest_pain_rules(),
        config=FigmentConfig(model_backend="hosted_omni", nvidia_api_key="test-nvidia-key"),
        retrieved_cards=_retrieved_chest_pain_cards(),
    )
    payload = trace.to_dict()

    assert output["red_flags"] == _emergency_chest_pain_rules()
    assert trace.validator_result["passed"] is True
    assert trace.model_route["strict_validation"] is True
    assert trace.field_provenance["red_flags"] == "deterministic_fallback"
    assert trace.field_provenance["responder_checklist"] == "model_raw"
    assert payload["model_route"]["final_route"] == "model_with_deterministic_patches"
    assert payload["field_provenance_summary"]["counts"]["deterministic_fallback"] >= 1


def test_run_navigation_enforces_retrieved_cards_and_observation_grounding(monkeypatch) -> None:
    UnretrievedUngroundedModelClient.calls = 0
    monkeypatch.setattr(navigator, "ModelClient", UnretrievedUngroundedModelClient)

    output, trace = navigator.run_navigation(
        _confirmed_chest_pain_intake(),
        _emergency_chest_pain_rules(),
        config=FigmentConfig(model_backend="hosted_omni", nvidia_api_key="test-nvidia-key"),
        retrieved_cards=_retrieved_chest_pain_cards(),
    )

    assert "WOUND-INFECTION-ESCALATION-v1" not in output["source_cards"]
    assert output["source_cards"] == ["CHEST-PAIN-ESCALATION-v1"]
    assert trace.validator_result["passed"] is True
    assert trace.field_provenance["source_cards"] == "deterministic_fallback"
    assert trace.field_provenance["missing_info_to_collect"] == "deterministic_fallback"
    assert trace.to_dict()["model_route"]["final_route"] == "model_with_deterministic_patches"


def test_run_navigation_caps_focused_repair_attempts_and_traces_metrics(monkeypatch) -> None:
    MultiFailureRepairModelClient.calls = 0
    monkeypatch.setattr(navigator, "ModelClient", MultiFailureRepairModelClient)

    _output, trace = navigator.run_navigation(
        _confirmed_chest_pain_intake(),
        _emergency_chest_pain_rules(),
        config=FigmentConfig(model_backend="hosted_omni", nvidia_api_key="test-nvidia-key"),
        retrieved_cards=_retrieved_chest_pain_cards(),
    )

    assert MultiFailureRepairModelClient.calls == 3
    assert trace.model_route["repair_attempt_count"] == 2
    assert trace.model_route["repair_attempt_cap"] == 2
    assert trace.model_route["repair_capped"] is True
    assert trace.model_route["repair_latency_ms"] >= 0


def test_run_navigation_fills_required_observation_targets_without_counting_as_model_raw(monkeypatch) -> None:
    ObservationThinModelClient.calls = 0
    monkeypatch.setattr(navigator, "ModelClient", ObservationThinModelClient)

    output, trace = navigator.run_navigation(
        _confirmed_chest_pain_intake(),
        _emergency_chest_pain_rules(),
        config=FigmentConfig(model_backend="hosted_omni", nvidia_api_key="test-nvidia-key"),
        retrieved_cards=_retrieved_chest_pain_cards(),
    )

    observation_text = json.dumps(
        output["missing_info_to_collect"] + output["next_observations_to_collect"]
    ).lower()
    assert ObservationThinModelClient.calls == 1
    assert "chest pain description" in observation_text
    assert "onset and duration" in observation_text
    assert "shortness of breath report" in observation_text
    assert "available vital signs" in observation_text
    assert trace.validator_result["passed"] is True
    assert trace.model_route["field_level_fallback_used"] is True
    assert trace.field_provenance["missing_info_to_collect"] == "deterministic_fallback"
    assert trace.field_provenance["next_observations_to_collect"] == "deterministic_fallback"
    assert trace.model_route["filled_required_observation_ids"] == [
        "CHEST-PAIN-ESCALATION-v1::required_observation::1",
        "CHEST-PAIN-ESCALATION-v1::required_observation::2",
        "CHEST-PAIN-ESCALATION-v1::required_observation::3",
    ]
    assert any("required-observation targets filled" in event for event in trace.events)


def test_run_navigation_strips_and_traces_selected_required_observation_ids(monkeypatch) -> None:
    SelectedObservationIdsModelClient.calls = 0
    monkeypatch.setattr(navigator, "ModelClient", SelectedObservationIdsModelClient)

    output, trace = navigator.run_navigation(
        _confirmed_chest_pain_intake(),
        _emergency_chest_pain_rules(),
        config=FigmentConfig(model_backend="hosted_omni", nvidia_api_key="test-nvidia-key"),
        retrieved_cards=_retrieved_chest_pain_cards(),
    )

    observation_text = json.dumps(
        output["missing_info_to_collect"] + output["next_observations_to_collect"]
    ).lower()
    assert SelectedObservationIdsModelClient.calls == 1
    assert "selected_required_observation_ids" not in output
    assert "shortness of breath report" in observation_text
    assert "chest pain description" not in observation_text
    assert "onset and duration" not in observation_text
    assert trace.validator_result["passed"] is True
    assert trace.model_route["model_selected_required_observation_ids"] == [
        "CHEST-PAIN-ESCALATION-v1::required_observation::1",
        "CHEST-PAIN-ESCALATION-v1::required_observation::2",
    ]
    assert trace.model_route["invalid_selected_required_observation_ids"] == ["NOT-A-REAL-TARGET"]
    assert trace.model_route["stripped_trace_only_fields"] == ["selected_required_observation_ids"]
    assert trace.model_route["filled_required_observation_ids"] == [
        "CHEST-PAIN-ESCALATION-v1::required_observation::3"
    ]
    assert trace.field_provenance["missing_info_to_collect"] == "deterministic_fallback"
    assert any("trace-only required-observation target ids stripped" in event for event in trace.events)
