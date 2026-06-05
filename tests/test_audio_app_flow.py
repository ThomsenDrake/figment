import importlib

from figment.config import FigmentConfig


def test_confirm_intake_does_not_silently_bulk_accept_audio_suggestions() -> None:
    app = importlib.import_module("app")
    audio_draft = {
        "audio_intake_path": "omni_native",
        "audio_runtime": "omni_native",
        "transcript": "Adult with chest pain.",
        "suggested_fields": [
            {
                "field": "chief_concern",
                "draft_value": "chest pain",
                "source_snippet": "chest pain",
                "status": "audio_draft",
                "needs_confirmation": True,
            }
        ],
        "confirmed_intake_required": True,
        "confirmation_status": "unconfirmed",
        "raw_audio_stored": False,
    }

    confirmed, _, confirmed_audio = app._confirm_ui_intake(
        "mobile clinic",
        "52",
        "not_applicable",
        "",
        "",
        "",
        "unknown",
        "unknown",
        "basic kit",
        "Responder typed note remains source of truth.",
        audio_draft,
    )

    assert confirmed["chief_concern"] == ""
    assert confirmed["responder_note"] == "Responder typed note remains source of truth."
    assert confirmed_audio["suggested_fields"][0]["status"] == "rejected"
    assert confirmed_audio["suggested_fields"][0]["needs_confirmation"] is False


def test_uploaded_audio_without_transcript_or_payload_is_not_fabricated_as_omni() -> None:
    app = importlib.import_module("app")
    config = FigmentConfig(
        model_backend="canned",
        enable_audio_intake=True,
        audio_backend="omni_native",
    ).validated()

    draft = app.draft_audio_intake(
        transcript="",
        config=config,
        audio_file="/tmp/field-note.wav",
    )

    assert draft["audio_file_received"] is True
    assert draft["audio_filename"] == "field-note.wav"
    assert draft["transcript"] == ""
    assert draft["suggested_fields"] == []
    assert draft["audio_intake_path"] == "audio_received_needs_transcript_or_model"
    assert draft["audio_runtime"] == "unprocessed_audio"
    assert "needs transcript/model support" in draft["processing_status"].lower()


def test_canned_audio_backend_is_labeled_as_canned_demo_runtime() -> None:
    app = importlib.import_module("app")
    config = FigmentConfig(
        model_backend="canned",
        enable_audio_intake=True,
        audio_backend="canned",
    ).validated()

    draft = app.draft_audio_intake(transcript="", config=config)

    assert draft["audio_runtime"] == "canned"
    assert draft["audio_intake_path"] == "canned_audio_demo"
    assert draft["audio_model_id"] is None
    assert draft["transcript"]
    assert draft["suggested_fields"]


def test_navigator_ui_uses_supplied_runtime_config(monkeypatch) -> None:
    app = importlib.import_module("app")
    config = FigmentConfig(
        model_backend="hosted_omni",
        audio_backend="omni_native",
        enable_audio_intake=True,
        omni_endpoint_url="http://omni.example.test/v1",
    ).validated()
    intake = {
        "confirmed": True,
        "setting": "mobile clinic",
        "chief_concern": "typed chest pain concern",
        "responder_note": "typed source of truth",
    }
    seen = {}

    def fake_run_case(intake_arg, config_arg=None, audio_draft=None):
        seen["config"] = config_arg
        return {
            "navigator_output": {"protocol_urgency": "routine"},
            "sbar": "handoff",
            "trace": {"model_route": {"model_backend": config_arg.model_backend}},
        }

    monkeypatch.setattr(app, "run_case", fake_run_case)

    output, sbar, trace, state = app._navigate_ui(intake, None, config=config)

    assert seen["config"] is config
    assert output["protocol_urgency"] == "routine"
    assert sbar == "handoff"
    assert trace == state
