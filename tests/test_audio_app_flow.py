import importlib

from figment.config import FigmentConfig, OMNI_MODEL_ID


def test_hosted_omni_provider_payload_becomes_unconfirmed_audio_draft() -> None:
    app = importlib.import_module("app")
    config = FigmentConfig(
        model_backend="hosted_omni",
        nvidia_api_key="test-nvidia-key",
        enable_audio_intake=True,
        audio_backend="omni_native",
    ).validated()
    provider_payload = {
        "transcript": "Adult reports severe headache and visual spots after flood cleanup.",
        "suggested_fields": [
            {
                "field": "symptoms",
                "draft_value": "severe headache and visual spots",
                "source_snippet": "severe headache and visual spots",
                "source_timecode": "00:01-00:05",
            }
        ],
        "missing_or_unclear_fields": ["blood_pressure", "pregnancy_status"],
        "provisional_red_flag_mentions": ["severe headache", "visual spots"],
    }

    draft = app.draft_audio_intake(config=config, provider_payload=provider_payload)

    assert draft["audio_intake_path"] == "omni_native"
    assert draft["audio_runtime"] == "omni_native"
    assert draft["audio_model_id"] == OMNI_MODEL_ID
    assert draft["field_fill_model_id"] is None
    assert draft["transcript"] == provider_payload["transcript"]
    assert draft["missing_or_unclear_fields"] == ["blood_pressure", "pregnancy_status"]
    assert draft["provisional_red_flag_mentions"] == ["severe headache", "visual spots"]
    assert draft["confirmed_intake_required"] is True
    assert draft["confirmation_status"] == "unconfirmed"
    assert draft["raw_audio_stored"] is False
    assert draft["suggested_fields"] == [
        {
            "field": "symptoms",
            "draft_value": "severe headache and visual spots",
            "source_snippet": "severe headache and visual spots",
            "source_timecode": "00:01-00:05",
            "status": "audio_draft",
            "needs_confirmation": True,
        }
    ]


def test_hosted_omni_provider_payload_with_invalid_field_names_falls_back_to_transcript_suggestions() -> None:
    app = importlib.import_module("app")
    config = FigmentConfig(
        model_backend="hosted_omni",
        nvidia_api_key="test-nvidia-key",
        enable_audio_intake=True,
        audio_backend="omni_native",
    ).validated()
    provider_payload = {
        "transcript": (
            "Seven-year-old at a shelter clinic after flood cleanup. Child cannot keep fluids down, "
            "is lethargic, has a very dry mouth, and has no urine since morning."
        ),
        "suggested_fields": [
            {
                "field": "chief_concern|symptoms|vitals|patient_age|pregnancy_status|allergies|medications|available_supplies|responder_note",
                "draft_value": "lethargic, dry mouth, no urine",
                "source_snippet": "lethargic, dry mouth, no urine",
            }
        ],
        "provisional_red_flag_mentions": ["lethargic", "no urine"],
    }

    draft = app.draft_audio_intake(config=config, provider_payload=provider_payload)
    fields = {suggestion["field"] for suggestion in draft["suggested_fields"]}

    assert "chief_concern|symptoms|vitals|patient_age|pregnancy_status|allergies|medications|available_supplies|responder_note" not in fields
    assert {"responder_note", "patient_age", "chief_concern", "symptoms"}.issubset(fields)
    assert next(item for item in draft["suggested_fields"] if item["field"] == "patient_age")["draft_value"] == "7 years"


def test_hosted_omni_provider_payload_without_transcript_or_valid_suggestions_fails_closed() -> None:
    app = importlib.import_module("app")
    config = FigmentConfig(
        model_backend="hosted_omni",
        nvidia_api_key="test-nvidia-key",
        enable_audio_intake=True,
        audio_backend="omni_native",
    ).validated()
    provider_payload = {
        "suggested_fields": [
            {
                "field": "chief_concern|symptoms",
                "draft_value": "chest pain",
                "source_snippet": "chest pain",
            }
        ]
    }

    draft = app.draft_audio_intake(config=config, provider_payload=provider_payload)

    assert draft["audio_intake_path"] == "audio_received_needs_transcript_or_model"
    assert draft["audio_runtime"] == "unprocessed_audio"
    assert draft["transcript"] == ""
    assert draft["suggested_fields"] == []
    assert draft["confirmed_intake_required"] is False
    assert draft["confirmation_status"] == "confirmed"
    assert draft["missing_or_unclear_fields"] == ["transcript_or_valid_provider_suggestions"]
    assert "valid field suggestions" in draft["processing_status"].lower()


def test_hosted_omni_provider_payload_drops_non_informative_field_values_and_preserves_red_flag_mentions() -> None:
    app = importlib.import_module("app")
    config = FigmentConfig(
        model_backend="hosted_omni",
        nvidia_api_key="test-nvidia-key",
        enable_audio_intake=True,
        audio_backend="omni_native",
    ).validated()
    provider_payload = {
        "transcript": (
            "Seven-year-old at a shelter clinic after flood cleanup. Child cannot keep fluids down, "
            "is lethargic, has a very dry mouth, and has no urine since morning."
        ),
        "suggested_fields": [
            {
                "field": "symptoms",
                "draft_value": "cannot keep fluids down, lethargic, very dry mouth, no urine since morning",
                "source_snippet": "cannot keep fluids down, lethargic",
            },
            {
                "field": "allergies",
                "draft_value": "Not answerable",
                "source_snippet": "",
            },
            {
                "field": "medications",
                "draft_value": "not provided",
                "source_snippet": "",
            },
            {
                "field": "responder_note",
                "draft_value": "Provisional intake draft for trained responder",
                "source_snippet": "",
            },
        ],
        "missing_or_unclear_fields": [],
        "provisional_red_flag_mentions": [],
    }

    draft = app.draft_audio_intake(config=config, provider_payload=provider_payload)

    assert {item["field"] for item in draft["suggested_fields"]} == {"symptoms"}
    assert set(draft["missing_or_unclear_fields"]) >= {"allergies", "medications"}
    assert set(draft["provisional_red_flag_mentions"]) >= {"lethargic", "no urine"}


def test_manual_edit_wins_over_hosted_omni_audio_draft_on_confirmation() -> None:
    app = importlib.import_module("app")
    config = FigmentConfig(
        model_backend="hosted_omni",
        nvidia_api_key="test-nvidia-key",
        enable_audio_intake=True,
        audio_backend="omni_native",
    ).validated()
    audio_draft = app.draft_audio_intake(
        config=config,
        provider_payload={
            "transcript": "Audio says chest pain, but the medic corrects it.",
            "suggested_fields": [
                {
                    "field": "chief_concern",
                    "draft_value": "chest pain",
                    "source_snippet": "chest pain",
                }
            ],
        },
    )

    confirmed, _, confirmed_audio = app._confirm_ui_intake(
        "mobile clinic",
        "52",
        "not_applicable",
        "manual correction: wound concern",
        "",
        "",
        "unknown",
        "unknown",
        "basic kit",
        "Typed note remains source of truth.",
        audio_draft,
    )

    assert confirmed["chief_concern"] == "manual correction: wound concern"
    assert confirmed_audio["suggested_fields"][0]["draft_value"] == "manual correction: wound concern"
    assert confirmed_audio["suggested_fields"][0]["status"] == "edited"
    assert confirmed_audio["suggested_fields"][0]["needs_confirmation"] is False


def test_uploaded_audio_uses_hosted_omni_provider_when_enabled(tmp_path, monkeypatch) -> None:
    app = importlib.import_module("app")
    audio_path = tmp_path / "field-note.wav"
    audio_path.write_bytes(b"fake wav bytes")
    config = FigmentConfig(
        model_backend="hosted_omni",
        enable_audio_intake=True,
        audio_backend="omni_native",
        nvidia_api_key="test-nvidia-key",
    ).validated()
    seen = {}

    class FakeModelClient:
        def __init__(self, config_arg):
            seen["config"] = config_arg

        def generate_audio_draft(self, audio_file):
            seen["audio_file"] = audio_file
            return {
                "transcript": "Adult reports trouble breathing.",
                "suggested_fields": [
                    {
                        "field": "symptoms",
                        "draft_value": "trouble breathing",
                        "source_snippet": "trouble breathing",
                    }
                ],
                "missing_or_unclear_fields": ["vitals"],
                "provisional_red_flag_mentions": ["trouble breathing"],
            }

    monkeypatch.setattr(app, "ModelClient", FakeModelClient)

    draft = app.draft_audio_intake(audio_file=str(audio_path), config=config)

    assert seen["config"] is config
    assert seen["audio_file"] == str(audio_path)
    assert draft["audio_intake_path"] == "omni_native"
    assert draft["audio_runtime"] == "omni_native"
    assert draft["audio_model_id"] == OMNI_MODEL_ID
    assert draft["transcript"] == "Adult reports trouble breathing."
    assert draft["suggested_fields"][0]["field"] == "symptoms"
    assert draft["confirmation_status"] == "unconfirmed"
    assert draft["audio_filename"] == "field-note.wav"
    assert draft["raw_audio_stored"] is False
    assert "not written to figment traces" in draft["audio_retention_note"].lower()


def test_hosted_omni_audio_failure_fails_closed(tmp_path, monkeypatch) -> None:
    app = importlib.import_module("app")
    audio_path = tmp_path / "field-note.wav"
    audio_path.write_bytes(b"fake wav bytes")
    config = FigmentConfig(
        model_backend="hosted_omni",
        enable_audio_intake=True,
        audio_backend="omni_native",
        nvidia_api_key="test-nvidia-key",
    ).validated()

    class FailingModelClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate_audio_draft(self, _audio_file):
            raise app.ModelClientError("endpoint unavailable")

    monkeypatch.setattr(app, "ModelClient", FailingModelClient)

    draft = app.draft_audio_intake(audio_file=str(audio_path), config=config)

    assert draft["audio_intake_path"] == "audio_received_needs_transcript_or_model"
    assert draft["audio_runtime"] == "unprocessed_audio"
    assert draft["suggested_fields"] == []
    assert "hosted omni audio draft failed" in draft["processing_status"].lower()
    assert draft["audio_filename"] == "field-note.wav"
    assert draft["raw_audio_stored"] is False
    assert "not written to figment traces" in draft["audio_retention_note"].lower()


def test_local_parakeet_path_uses_local_4b_runtime_label_when_gated() -> None:
    app = importlib.import_module("app")
    config = FigmentConfig(
        model_stack="local_4b_parakeet",
        model_backend="llama_cpp",
        enable_audio_intake=True,
        audio_backend="parakeet_nemo",
        allow_local_asr=True,
    ).validated()

    draft = app.draft_audio_intake(
        transcript="Local Parakeet transcript says the patient has trouble breathing.",
        config=config,
    )

    assert draft["audio_intake_path"] == "parakeet_rnnt_plus_text_nemotron"
    assert draft["audio_runtime"] == "local_4b_parakeet"
    assert draft["audio_model_id"] == "nvidia/parakeet-rnnt-1.1b"
    assert draft["field_fill_model_id"] == "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    assert draft["confirmation_status"] == "unconfirmed"


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


def test_rejected_audio_transcript_is_not_passed_to_navigator_prompt(monkeypatch) -> None:
    app = importlib.import_module("app")
    import figment.navigator as navigator

    audio_draft = {
        "audio_intake_path": "omni_native",
        "audio_runtime": "omni_native",
        "transcript": "Adult says chest pain from audio only.",
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
        "Typed note remains the source of truth.",
        audio_draft,
    )
    captured: dict[str, str] = {}

    class CapturePromptModelClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def generate_json(self, prompt, context):
            captured["prompt"] = prompt
            return navigator.canned_navigator_output(
                context["intake"],
                context["rule_results"],
                context["retrieved_cards"],
                context["urgency_floor"],
            )

    monkeypatch.setattr(navigator, "ModelClient", CapturePromptModelClient)

    navigator.run_navigation(
        confirmed,
        [],
        audio_draft=confirmed_audio,
        config=FigmentConfig(model_backend="hosted_omni", nvidia_api_key="unused"),
        retrieved_cards=[
            {
                "card_id": "SAFETY-BOUNDARIES-v1",
                "card": {"card_id": "SAFETY-BOUNDARIES-v1", "title": "Safety boundaries"},
            }
        ],
    )

    assert "Adult says chest pain from audio only." not in captured["prompt"]
    assert "chest pain" not in captured["prompt"].lower()


def test_apply_audio_draft_prefills_empty_fields_without_confirming() -> None:
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

    *fields, audio_json, audio_state = app._apply_audio_draft_ui(
        "mobile clinic",
        "52",
        "not_applicable",
        "",
        "",
        "",
        "unknown",
        "unknown",
        "basic kit",
        "",
        audio_draft,
    )

    assert fields[3] == "chest pain"
    assert audio_json["confirmation_status"] == "unconfirmed"
    assert audio_state is audio_json


def test_loading_demo_case_resets_audio_and_downstream_state() -> None:
    app = importlib.import_module("app")

    values = app._load_demo_case_and_reset("Rural clinic: pregnancy danger sign")

    assert values[:10] == app._load_demo_case("Rural clinic: pregnancy danger sign")
    assert values[10] is None
    assert values[11] == ""
    assert values[12] is None
    assert values[13] is None
    assert values[14] == {"red_flags": [], "protocol_urgency": "routine"}
    assert "PROTOCOL_URGENCY" in values[15]
    assert values[16] == []
    assert values[17] == ""
    assert "Selected Protocol Card" in values[18]
    assert values[19] == {}
    assert values[20] == ""
    assert "Protocol Urgency" in values[21]
    assert values[22] == {}
    assert values[23] is None
    assert "Run the navigator" in values[24]
    assert values[25] == {}
    assert values[26] is None
    assert values[27] == {}


def test_source_and_audio_changes_clear_stale_pipeline_outputs() -> None:
    app = importlib.import_module("app")

    assert app._clear_source_outputs() == [
        None,
        {"red_flags": [], "protocol_urgency": "routine"},
        app._risk_summary_html({"red_flags": [], "protocol_urgency": "routine"}),
        [],
        "",
        app._protocol_results_html([]),
        {},
        "",
        app._navigator_summary_html({}),
        {},
        None,
        app._trace_audit_html({}),
        {},
        {},
    ]
    assert app._clear_audio_outputs() == [
        None,
        None,
        {"red_flags": [], "protocol_urgency": "routine"},
        app._risk_summary_html({"red_flags": [], "protocol_urgency": "routine"}),
        [],
        "",
        app._protocol_results_html([]),
        {},
        "",
        app._navigator_summary_html({}),
        {},
        None,
        app._trace_audit_html({}),
        {},
        None,
        {},
    ]


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
        provider_payload={},
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
