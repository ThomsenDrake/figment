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
        config=FigmentConfig(model_backend="hosted_omni"),
        retrieved_cards=_retrieved_chest_pain_cards(),
    )

    output_text = json.dumps(output).lower()
    assert output["protocol_urgency"] == "emergency"
    assert "diagnose" not in output_text
    assert "give 5 mg" not in output_text
    assert trace.navigator_output == output
    assert trace.validator_result["passed"] is True


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
