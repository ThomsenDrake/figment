import importlib
from pathlib import Path

import pytest

from figment import config as config_module
from figment.config import FIGMENT_CANNED_MODEL_ID, FigmentConfig, NVIDIA_OMNI_API_MODEL_ID
from figment.validators import validate_audio_ready, validate_navigator_output


CONFIG_ENV_KEYS = (
    "FIGMENT_MODE",
    "MODEL_STACK",
    "MODEL_BACKEND",
    "AUDIO_BACKEND",
    "ENABLE_AUDIO_INTAKE",
    "ALLOW_LOCAL_ASR",
    "ALLOW_SELF_HOSTED_OMNI",
    "ALLOW_STRETCH_STACK",
    "HF_MODEL_ID",
    "NVIDIA_MODEL_ID",
    "NVIDIA_BASE_URL",
    "NVIDIA_API_KEY",
    "LOCAL_MODEL_ID",
    "HF_ENDPOINT_URL",
    "OMNI_ENDPOINT_URL",
    "HF_TOKEN",
    "LLAMA_BASE_URL",
    "FIGMENT_TRACE_DIR",
)


def _clear_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_config_defaults_to_omni_with_canned_fallback_without_hosted_secret(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_config_env(monkeypatch)

    config = FigmentConfig.from_env()

    assert config.model_stack == "omni_native"
    assert config.model_backend == "canned"
    assert config.active_model_id == FIGMENT_CANNED_MODEL_ID


def test_config_uses_hosted_omni_api_model_id_when_secret_is_present(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")

    config = FigmentConfig.from_env()

    assert config.model_stack == "omni_native"
    assert config.model_backend == "hosted_omni"
    assert config.nvidia_model_id == NVIDIA_OMNI_API_MODEL_ID
    assert config.active_model_id == NVIDIA_OMNI_API_MODEL_ID


def test_hf_token_without_endpoint_does_not_select_hosted_omni(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")

    config = FigmentConfig.from_env()

    assert config.model_backend == "canned"
    assert config.active_model_id == FIGMENT_CANNED_MODEL_ID


def test_hf_endpoint_selects_hosted_omni_with_hf_token(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv("HF_ENDPOINT_URL", "https://hf.example.test/v1")

    config = FigmentConfig.from_env()

    assert config.model_backend == "hosted_omni"
    assert config.hf_endpoint_url == "https://hf.example.test/v1"


def test_explicit_hosted_omni_nvidia_endpoint_requires_key() -> None:
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        FigmentConfig(model_backend="hosted_omni", nvidia_api_key="").validated()


def test_demo_audio_examples_cover_committed_audio_assets() -> None:
    app = importlib.import_module("app")

    examples = app._demo_audio_examples()

    assert len(examples) == 3
    assert {Path(example[0]).name for example in examples} == {
        "case_1_dictated_intake.wav",
        "case_2_dictated_intake.wav",
        "case_3_dictated_intake.wav",
    }
    assert all(Path(example[0]).exists() for example in examples)
    assert all(example[1] == "" for example in examples)


def test_config_gates_local_4b_parakeet_stack_with_local_asr(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_4b_model_id = getattr(config_module, "NVIDIA_NEMOTRON_3_NANO_4B_BF16_MODEL_ID", None)
    parakeet_model_id = getattr(config_module, "PARAKEET_ASR_MODEL_ID", None)

    assert local_4b_model_id == "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    assert parakeet_model_id == "nvidia/parakeet-rnnt-1.1b"

    config = FigmentConfig(model_stack="local_4b_parakeet", model_backend="llama_cpp").validated()

    assert config.local_model_id == local_4b_model_id
    assert config.active_model_id == local_4b_model_id

    with pytest.raises(ValueError, match="parakeet_nemo requires ALLOW_LOCAL_ASR"):
        FigmentConfig(audio_backend="parakeet_nemo").validated()

    with pytest.raises(ValueError, match="MODEL_STACK=local_4b_parakeet"):
        FigmentConfig(
            model_stack="omni_native",
            audio_backend="parakeet_nemo",
            allow_local_asr=True,
        ).validated()

    monkeypatch.chdir(tmp_path)
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("FIGMENT_MODE", "local")
    monkeypatch.setenv("MODEL_BACKEND", "llama_cpp")
    monkeypatch.setenv("MODEL_STACK", "local_4b_parakeet")
    monkeypatch.setenv("AUDIO_BACKEND", "parakeet_nemo")
    monkeypatch.setenv("ALLOW_LOCAL_ASR", "true")

    local_config = FigmentConfig.from_env()

    assert local_config.allow_local_asr is True
    assert local_config.audio_model_id == parakeet_model_id
    assert local_config.active_model_id == local_4b_model_id


def test_legacy_split_stack_contract_is_rejected(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="base_nano_parakeet.*local_4b_parakeet"):
        FigmentConfig(model_stack="base_nano_parakeet").validated()

    monkeypatch.chdir(tmp_path)
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("ALLOW_STRETCH_STACK", "true")

    with pytest.raises(ValueError, match="ALLOW_STRETCH_STACK.*ALLOW_LOCAL_ASR"):
        FigmentConfig.from_env()


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


def test_validator_allows_negated_diagnosis_safety_instruction_with_condition_context() -> None:
    output = {
        "protocol_urgency": "emergency",
        "candidate_protocol_pathways": [
            {"card_id": "PED-DEHYD-RED-FLAGS-v1", "reason_relevant": "Pediatric danger signs were reported."}
        ],
        "missing_info_to_collect": ["repeat vitals"],
        "next_observations_to_collect": ["urine output"],
        "conflicts_or_uncertainties": [],
        "responder_checklist": ["Escalate per cited local protocol."],
        "do_not_do": ["Do not diagnose dehydration severity as a final clinical diagnosis."],
        "source_cards": ["PED-DEHYD-RED-FLAGS-v1"],
        "handoff_note_sbar": {
            "situation": "Pediatric dehydration concern",
            "background": "Synthetic case",
            "assessment_observations_only": "No urine since morning.",
            "handoff_request": "Escalate per protocol",
        },
        "responder_plain_language_script": "",
        "safety_boundary": "Prototype protocol navigation only.",
    }

    result = validate_navigator_output(output, {"PED-DEHYD-RED-FLAGS-v1"}, urgency_floor="emergency")

    assert result.passed


def test_validator_rejects_prescribing_after_unrelated_negated_instruction() -> None:
    output = {
        "protocol_urgency": "routine",
        "candidate_protocol_pathways": [
            {"card_id": "SAFETY-BOUNDARIES-v1", "reason_relevant": "Safety boundary review."}
        ],
        "missing_info_to_collect": [],
        "next_observations_to_collect": [],
        "conflicts_or_uncertainties": [],
        "responder_checklist": ["Do not wait, prescribe opioid now."],
        "do_not_do": [],
        "source_cards": ["SAFETY-BOUNDARIES-v1"],
        "handoff_note_sbar": {
            "situation": "Mild symptoms",
            "background": "Synthetic case",
            "assessment_observations_only": "Temperature normal.",
            "handoff_request": "Escalate per protocol",
        },
        "responder_plain_language_script": "",
        "safety_boundary": "Prototype protocol navigation only.",
    }

    result = validate_navigator_output(output, {"SAFETY-BOUNDARIES-v1"})

    assert not result.passed
    assert any("forbidden clinical language: prescribe" in item for item in result.failures)


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


def test_run_case_uses_environment_config_when_not_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    app = importlib.import_module("app")
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    monkeypatch.setenv("MODEL_BACKEND", "hosted_omni")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")
    seen = {}

    class FakeTrace:
        validator_result = {"passed": True, "failures": []}

        def to_dict(self) -> dict[str, object]:
            return {"model_route": {"model_backend": seen["config"].model_backend}}

    def fake_run_navigation(intake, rules, *, audio_draft=None, config=None, **_kwargs):
        seen["config"] = config
        return {"protocol_urgency": "routine", "source_cards": []}, FakeTrace()

    monkeypatch.setattr(app, "run_navigation", fake_run_navigation)
    monkeypatch.setattr(app, "render_sbar", lambda *_args, **_kwargs: "handoff")
    intake = app.collect_intake(
        setting="mobile clinic",
        patient_age="52",
        pregnancy_status="not_applicable",
        chief_concern="wound concern",
        symptoms="spreading redness",
        vitals="unknown",
        allergies="unknown",
        medications="unknown",
        available_supplies="dressings",
        responder_note="Cut from debris.",
    )

    result = app.run_case(intake)

    assert seen["config"].model_backend == "hosted_omni"
    assert result["trace"]["model_route"]["model_backend"] == "hosted_omni"


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
