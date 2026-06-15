import importlib
from typing import Any

import pytest

from figment.config import FigmentConfig
from figment.trace import FigmentTrace


CONFIG_ENV_KEYS = (
    "FIGMENT_MODE",
    "MODEL_STACK",
    "MODEL_BACKEND",
    "AUDIO_BACKEND",
    "ENABLE_AUDIO_INTAKE",
    "ALLOW_LOCAL_ASR",
    "ALLOW_SELF_HOSTED_OMNI",
    "ALLOW_STRETCH_STACK",
    "ZEROGPU_MODEL_REPO",
    "ZEROGPU_MODEL_SUBFOLDER",
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
    "PYTHON_DOTENV_DISABLED",
)


def _clear_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")


def test_env_config_rejects_cross_mode_backend_drift(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("FIGMENT_MODE", "hosted")
    monkeypatch.setenv("MODEL_BACKEND", "llama_cpp")

    with pytest.raises(ValueError, match="FIGMENT_MODE=hosted.*MODEL_BACKEND=llama_cpp"):
        FigmentConfig.from_env()

    _clear_config_env(monkeypatch)
    monkeypatch.setenv("FIGMENT_MODE", "local")
    monkeypatch.setenv("MODEL_BACKEND", "hosted_omni")
    monkeypatch.setenv("MODEL_STACK", "omni_native")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")

    with pytest.raises(ValueError, match="FIGMENT_MODE=local.*hosted_omni"):
        FigmentConfig.from_env()


def test_env_config_allows_explicit_local_self_hosted_omni(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("FIGMENT_MODE", "local")
    monkeypatch.setenv("MODEL_STACK", "omni_native")
    monkeypatch.setenv("MODEL_BACKEND", "llama_cpp")
    monkeypatch.setenv("LOCAL_MODEL_ID", "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")
    monkeypatch.setenv("ALLOW_SELF_HOSTED_OMNI", "true")

    config = FigmentConfig.from_env()

    assert config.allow_self_hosted_omni is True
    assert config.figment_mode == "local"
    assert config.model_stack == "omni_native"
    assert config.model_backend == "llama_cpp"
    assert config.active_model_id == "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"


def test_env_config_zerogpu_backend_defaults_to_v14p(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("FIGMENT_MODE", "hosted")
    monkeypatch.setenv("MODEL_BACKEND", "hf_zerogpu")

    config = FigmentConfig.from_env()

    assert config.model_backend == "hf_zerogpu"
    assert config.model_stack == "local_4b_parakeet"
    assert config.local_model_id == "figment-sft-v14p-lora-merged-bf16"
    assert config.zerogpu_model_repo == "build-small-hackathon/figment-finetuned-model-archive"
    assert config.zerogpu_model_subfolder == "figment_sft_v14p/figment-sft-v14p-lora-merged-bf16"
    assert config.active_model_id == "figment-sft-v14p-lora-merged-bf16"


def test_env_config_parakeet_asr_can_use_transformers_model_on_zerogpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("FIGMENT_MODE", "hosted")
    monkeypatch.setenv("MODEL_BACKEND", "hf_zerogpu")
    monkeypatch.setenv("AUDIO_BACKEND", "parakeet_nemo")
    monkeypatch.setenv("ENABLE_AUDIO_INTAKE", "true")
    monkeypatch.setenv("ALLOW_LOCAL_ASR", "true")
    monkeypatch.setenv("PARAKEET_ASR_MODEL_ID", "nvidia/parakeet-ctc-1.1b")
    monkeypatch.setenv("PARAKEET_ASR_RUNTIME", "transformers")

    config = FigmentConfig.from_env()

    assert config.model_backend == "hf_zerogpu"
    assert config.audio_backend == "parakeet_nemo"
    assert config.enable_audio_intake is True
    assert config.allow_local_asr is True
    assert config.parakeet_asr_model_id == "nvidia/parakeet-ctc-1.1b"
    assert config.parakeet_asr_runtime == "transformers"
    assert config.audio_model_id == "nvidia/parakeet-ctc-1.1b"


@pytest.mark.parametrize(
    ("model_route", "events", "expected_final_route"),
    [
        ({"model_backend": "hosted_omni", "fallback_reason": None}, [], "live_model_generated"),
        ({"model_backend": "hf_zerogpu", "fallback_reason": None}, [], "live_model_generated"),
        ({"model_backend": "hosted_omni", "fallback_reason": None}, ["navigator output repaired by hosted retry"], "model_repaired"),
        (
            {"model_backend": "hosted_omni", "fallback_reason": None, "field_level_fallback_used": True},
            ["navigator output retained with field-level deterministic patches"],
            "model_with_deterministic_patches",
        ),
        ({"model_backend": "hosted_omni", "fallback_reason": "navigator_validation_failure"}, [], "validation_fallback"),
        ({"model_backend": "canned", "fallback_reason": None}, [], "canned_backend"),
    ],
)
def test_trace_json_derives_final_route_from_actual_trace_state(
    model_route: dict[str, Any],
    events: list[str],
    expected_final_route: str,
) -> None:
    trace = FigmentTrace(
        input_captured={},
        red_flags=[],
        retrieved_card_ids=[],
        prompt_template_hash="prompt",
        model_route=model_route,
        navigator_output={},
        validator_result={"passed": True, "failures": []},
        events=events,
    ).to_dict()

    assert trace["model_route"]["raw_route"] == model_route["model_backend"]
    assert trace["model_route"]["final_route"] == expected_final_route
    assert trace["model_route"]["validation_status"] == "passed"


def test_trace_and_navigator_ui_show_final_route_not_just_configured_backend() -> None:
    app = importlib.import_module("app")
    trace = {
        "events": ["navigator output generated", "navigator output failed validation; deterministic fallback applied"],
        "retrieved_card_ids": ["CHEST-PAIN-ESCALATION-v1"],
        "model_route": {
            "model_backend": "hosted_omni",
            "model_id": "test-hosted-model",
            "fallback_tier": "canned",
            "fallback_reason": "navigator_validation_failure",
        },
        "validator_result": {"passed": True, "failures": []},
        "raw_audio_stored": False,
    }
    output = {
        "protocol_urgency": "emergency",
        "missing_info_to_collect": [],
        "responder_checklist": [],
        "do_not_do": [],
        "source_cards": ["CHEST-PAIN-ESCALATION-v1"],
        "handoff_note_sbar": {},
    }

    trace_html = app._trace_audit_html(trace)
    navigator_html = app._navigator_summary_html(output, trace)

    assert "Validation fallback" in trace_html
    assert "Validation fallback" in navigator_html
    assert "navigator_validation_failure" in trace_html


def test_trace_and_navigator_ui_show_field_provenance_counts_for_hybrid_output() -> None:
    app = importlib.import_module("app")
    trace = {
        "events": ["navigator output retained with field-level deterministic patches"],
        "retrieved_card_ids": ["CHEST-PAIN-ESCALATION-v1"],
        "model_route": {
            "model_backend": "hosted_omni",
            "model_id": "test-hosted-model",
            "fallback_tier": "configured",
            "fallback_reason": None,
            "field_level_fallback_used": True,
            "repair_attempt_count": 2,
            "repair_latency_ms": 12.5,
        },
        "validator_result": {"passed": True, "failures": []},
        "field_provenance": {
            "protocol_urgency": "model_raw",
            "source_cards": "deterministic_fallback",
            "handoff_note_sbar": "deterministic_fallback",
        },
        "raw_audio_stored": False,
    }
    output = {
        "protocol_urgency": "emergency",
        "missing_info_to_collect": [],
        "responder_checklist": [],
        "do_not_do": [],
        "source_cards": ["CHEST-PAIN-ESCALATION-v1"],
        "handoff_note_sbar": {},
    }

    trace_html = app._trace_audit_html(trace)
    navigator_html = app._navigator_summary_html(output, trace)

    assert "Model with deterministic patches" in trace_html
    assert "Model with deterministic patches" in navigator_html
    assert "Field provenance" in trace_html
    assert "deterministic_fallback=2" in trace_html
    assert "model_raw=1" in navigator_html
    assert "Repair calls" in trace_html
    assert "Configured backend" in navigator_html


def test_harness_evidence_is_visible_outside_model_authored_text() -> None:
    app = importlib.import_module("app")
    evidence = {
        "confirmed_intake": True,
        "retrieved_card_ids": ["CHEST-PAIN-ESCALATION-v1", "REFERRAL-SBAR-v1"],
        "deterministic_rule_ids": ["CHEST-001"],
        "urgency_floor": "emergency",
        "validator_status": "passed",
        "audio_correction_status": "not_applicable",
        "source_card_ids": ["CHEST-PAIN-ESCALATION-v1"],
        "final_route": "live_model_generated",
    }
    trace = {
        "events": ["validation complete"],
        "model_route": {"model_backend": "llama_cpp", "fallback_tier": "configured"},
        "validator_result": {"passed": True, "failures": []},
        "navigator_output": {"harness_evidence": evidence},
        "raw_audio_stored": False,
    }
    output = {
        "protocol_urgency": "emergency",
        "missing_info_to_collect": ["repeat vitals"],
        "responder_checklist": [],
        "do_not_do": [],
        "source_cards": ["CHEST-PAIN-ESCALATION-v1"],
        "handoff_note_sbar": {},
        "harness_evidence": evidence,
    }

    navigator_html = app._navigator_summary_html(output, trace)
    trace_html = app._trace_audit_html(trace)

    assert "Harness Evidence" in navigator_html
    assert "Harness Evidence" in trace_html
    assert "Retrieved cards: 2" in navigator_html
    assert "Validation: passed" in navigator_html
    assert "Harness evidence: visible" in trace_html


def test_typed_transcript_audio_draft_is_not_labeled_as_real_omni_audio() -> None:
    app = importlib.import_module("app")
    config = FigmentConfig(
        model_backend="canned",
        enable_audio_intake=True,
        audio_backend="omni_native",
    ).validated()

    draft = app.draft_audio_intake(
        transcript="Adult says chest pain and trouble breathing.",
        config=config,
    )

    assert draft["audio_intake_path"] == "typed_transcript_heuristic"
    assert draft["audio_runtime"] == "typed_transcript_heuristic"
    assert draft["audio_model_id"] is None
    assert draft["draft_source"] == "typed_transcript_heuristic"
    assert draft["transcript_source"] == "typed_transcript"


def test_retrieval_source_is_exposed_in_ui_and_app_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    app = importlib.import_module("app")
    retrieved_cards = [
        {
            "card_id": "SAFETY-BOUNDARIES-v1",
            "title": "Safety boundaries",
            "score": 1.0,
            "source": "json_fallback",
            "card": {"card_id": "SAFETY-BOUNDARIES-v1", "title": "Safety boundaries"},
        }
    ]

    class FakeTrace:
        validator_result = {"passed": True, "failures": []}

        def to_dict(self) -> dict[str, Any]:
            return {
                "model_route": {"model_backend": "canned"},
                "validator_result": {"passed": True, "failures": []},
            }

    def fake_run_navigation(*_args: Any, retrieved_cards: list[dict[str, Any]] | None = None, **_kwargs: Any):
        assert retrieved_cards == retrieved_cards_fixture
        return {"protocol_urgency": "routine", "source_cards": ["SAFETY-BOUNDARIES-v1"]}, FakeTrace()

    retrieved_cards_fixture = retrieved_cards
    monkeypatch.setattr(app, "search_protocol_cards", lambda *_args, **_kwargs: retrieved_cards_fixture)
    monkeypatch.setattr(app, "run_navigation", fake_run_navigation)
    monkeypatch.setattr(app, "render_sbar", lambda *_args, **_kwargs: "handoff")

    panel = app.protocol_evidence_panel(retrieved_cards)
    results_html = app._protocol_results_html(retrieved_cards)
    result = app.run_case(
        app.collect_intake(
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
        ),
        config=FigmentConfig(model_backend="canned"),
    )

    assert "json_fallback" in panel
    assert "json_fallback" in results_html
    assert result["trace"]["retrieval"]["sources"] == ["json_fallback"]
    assert result["trace"]["retrieval"]["primary_source"] == "json_fallback"
