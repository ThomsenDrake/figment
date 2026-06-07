import json
from types import SimpleNamespace
from typing import Any

import pytest

from figment.config import FigmentConfig
from figment.model_client import ModelClient, ModelClientError
from scripts import smoke_model_route


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _clear_route_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "FIGMENT_MODE",
        "MODEL_STACK",
        "MODEL_BACKEND",
        "LOCAL_MODEL_ID",
        "LLAMA_BASE_URL",
        "NVIDIA_API_KEY",
        "HF_TOKEN",
        "HF_ENDPOINT_URL",
        "OMNI_ENDPOINT_URL",
        "FIGMENT_SMOKE_ALLOW_NETWORK",
        "FIGMENT_SMOKE_TIMEOUT_SECONDS",
        "FIGMENT_MODEL_TIMEOUT_SECONDS",
        "ALLOW_SELF_HOSTED_OMNI",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")


def test_canned_smoke_reports_canned_route_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_route_env(monkeypatch)
    monkeypatch.setenv("MODEL_BACKEND", "canned")

    result = smoke_model_route.run_smoke()

    assert result["status"] == "passed"
    assert result["network_enabled"] is False
    assert result["network_attempted"] is False
    assert result["route"]["model_backend"] == "canned"
    assert result["fallback_tier"] == "canned"
    assert result["bounded_field_source"] == "canned_fallback"
    assert result["validation"]["passed"] is True
    assert result["model_id"]
    assert result["latency_ms"] >= 0


def test_local_llama_smoke_requires_network_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_route_env(monkeypatch)
    monkeypatch.setenv("FIGMENT_MODE", "local")
    monkeypatch.setenv("MODEL_BACKEND", "llama_cpp")
    monkeypatch.setenv("LLAMA_BASE_URL", "http://127.0.0.1:65534/v1")
    monkeypatch.setenv("LOCAL_MODEL_ID", "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")

    def fail_if_called(*_: Any, **__: Any) -> None:
        raise AssertionError("local smoke must not call the model route without FIGMENT_SMOKE_ALLOW_NETWORK")

    monkeypatch.setattr(smoke_model_route, "run_navigation", fail_if_called)

    result = smoke_model_route.run_smoke()

    assert result["status"] == "skipped"
    assert result["skip_reason"] == "network_disabled_for_configured_route"
    assert result["network_enabled"] is False
    assert result["network_attempted"] is False
    assert result["route"]["route_kind"] == "local_openai_compatible"
    assert result["model_id"] == "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    assert result["local_llm_evidence"]["proof_status"] == "skipped_no_network_call"
    assert result["local_llm_evidence"]["counts_as_no_cloud_route_proof"] is False
    assert result["local_llm_evidence"]["counts_as_50_case_local_llm_competence"] is False
    assert "MODEL_BACKEND=llama_cpp" in result["local_llm_evidence"]["real_eval_command"]
    assert "comprehensive_hosted_cases.jsonl" in result["local_llm_evidence"]["real_eval_command"]
    assert "self-hosted Omni" in result["off_grid_note"]


def test_gated_local_llama_smoke_reports_configured_route_for_any_local_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_route_env(monkeypatch)
    monkeypatch.setenv("FIGMENT_MODE", "local")
    monkeypatch.setenv("MODEL_BACKEND", "llama_cpp")
    monkeypatch.setenv("FIGMENT_SMOKE_ALLOW_NETWORK", "true")
    monkeypatch.setenv("LLAMA_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("LOCAL_MODEL_ID", "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")

    captured: dict[str, Any] = {}

    def fake_run_navigation(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], Any]:
        captured["config"] = kwargs["config"]
        output = {
            "candidate_protocol_pathways": [{"card_id": "SAFETY-BOUNDARIES-v1"}],
            "missing_info_to_collect": ["repeat vitals"],
            "next_observations_to_collect": ["mental status"],
            "conflicts_or_uncertainties": ["synthetic smoke case"],
            "responder_checklist": ["keep red flags visible"],
            "handoff_note_sbar": {"situation": "fatigue"},
        }
        trace = SimpleNamespace(
            model_route={
                "model_stack": "local_4b_parakeet",
                "model_backend": "llama_cpp",
                "model_id": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
                "fallback_tier": "configured",
                "fallback_reason": None,
            },
            validator_result={"passed": True, "failures": []},
            navigator_output=output,
            to_dict=lambda: {
                "model_route": trace.model_route,
                "validator_result": trace.validator_result,
                "navigator_output": output,
            },
        )
        return output, trace

    monkeypatch.setattr(smoke_model_route, "run_navigation", fake_run_navigation)

    result = smoke_model_route.run_smoke()

    assert isinstance(captured["config"], FigmentConfig)
    assert captured["config"].local_model_id == "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    assert result["status"] == "passed"
    assert result["network_enabled"] is True
    assert result["network_attempted"] is True
    assert result["route"]["base_url"] == "http://127.0.0.1:8001/v1"
    assert result["fallback_tier"] == "configured"
    assert result["bounded_field_source"] == "configured_model"
    assert result["model_id"] == "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    assert all(result["bounded_fields_present"].values())
    assert result["local_llm_evidence"]["proof_status"] == "one_case_route_smoke_passed"
    assert result["local_llm_evidence"]["counts_as_no_cloud_route_proof"] is True
    assert result["local_llm_evidence"]["counts_as_50_case_local_llm_competence"] is False


def test_model_client_error_mentions_sanitized_route_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        raise OSError("endpoint unavailable")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ModelClient(
        FigmentConfig(
            model_backend="llama_cpp",
            llama_base_url="http://127.0.0.1:8001/v1?token=secret",
            local_model_id="local-smoke-model",
        )
    )

    with pytest.raises(ModelClientError) as excinfo:
        client.generate_json("Return JSON.", {})

    message = str(excinfo.value)
    assert "local-smoke-model" in message
    assert "http://127.0.0.1:8001/v1/chat/completions" in message
    assert "secret" not in message


def test_model_client_timeout_can_be_shortened_for_smoke_without_changing_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["timeout"] = timeout
        return _FakeResponse({"choices": [{"message": {"content": '{"source_cards": ["SAFETY-BOUNDARIES-v1"]}'}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("FIGMENT_MODEL_TIMEOUT_SECONDS", "3.5")

    client = ModelClient(
        FigmentConfig(
            model_backend="llama_cpp",
            llama_base_url="http://127.0.0.1:8001/v1",
            local_model_id="local-smoke-model",
        )
    )
    client.generate_json("Return JSON.", {})

    assert captured["timeout"] == 3.5
