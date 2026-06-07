import json
from pathlib import Path
from typing import Any

from figment import config as config_module
from figment.config import FigmentConfig
import pytest

from figment.model_client import ModelClient, ModelClientError


EXPECTED_NVIDIA_OMNI_API_MODEL_ID = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _fake_chat_response(content: dict[str, Any]) -> _FakeResponse:
    return _fake_raw_chat_response(json.dumps(content))


def _fake_raw_chat_response(content: Any) -> _FakeResponse:
    return _FakeResponse({"choices": [{"message": {"content": content}}]})


def test_hosted_omni_posts_to_nvidia_api_with_nvidia_key(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _fake_chat_response({"protocol_urgency": "routine", "source_cards": []})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ModelClient(
        FigmentConfig(model_backend="hosted_omni", nvidia_api_key="test-nvidia-key"),
        timeout_seconds=12.0,
    )
    result = client.generate_json("Return JSON.", {})

    assert result["protocol_urgency"] == "routine"
    assert captured["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-nvidia-key"
    assert captured["timeout"] == 12.0
    assert captured["body"]["model"] == EXPECTED_NVIDIA_OMNI_API_MODEL_ID
    assert captured["body"]["messages"] == [{"role": "user", "content": "Return JSON."}]
    assert captured["body"]["temperature"] == 0.0
    assert captured["body"]["max_tokens"] >= 8192
    assert captured["body"]["stream"] is False
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_hosted_omni_requires_nvidia_key_for_nvidia_endpoint(monkeypatch: Any) -> None:
    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        raise AssertionError("NVIDIA requests without a key must not reach the network")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        FigmentConfig(model_backend="hosted_omni", nvidia_api_key="").validated()


def test_hosted_omni_audio_draft_posts_audio_data_url(tmp_path: Path, monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    audio_path = tmp_path / "field-note.wav"
    audio_path.write_bytes(b"fake wav bytes")

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _fake_chat_response(
            {
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
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ModelClient(FigmentConfig(model_backend="hosted_omni", nvidia_api_key="test-nvidia-key"))
    result = client.generate_audio_draft(audio_path)

    message = captured["body"]["messages"][0]
    audio_part = message["content"][1]
    assert captured["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-nvidia-key"
    assert captured["body"]["model"] == EXPECTED_NVIDIA_OMNI_API_MODEL_ID
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert message["content"][0]["type"] == "text"
    assert audio_part["type"] == "audio_url"
    assert audio_part["audio_url"]["url"].startswith("data:audio/")
    assert ";base64," in audio_part["audio_url"]["url"]
    assert result["transcript"] == "Adult reports trouble breathing."
    assert result["suggested_fields"][0]["field"] == "symptoms"


def test_local_llama_cpp_uses_local_base_url_without_hosted_auth(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _fake_chat_response({"protocol_urgency": "monitor", "source_cards": []})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ModelClient(
        FigmentConfig(
            model_backend="llama_cpp",
            llama_base_url="http://127.0.0.1:8001/v1",
            local_model_id="local-test-model",
            nvidia_api_key="hosted-key-not-for-local",
        )
    )
    result = client.generate_json("Local JSON please.", {})

    assert result["protocol_urgency"] == "monitor"
    assert captured["url"] == "http://127.0.0.1:8001/v1/chat/completions"
    assert captured["authorization"] is None
    assert captured["body"]["model"] == "local-test-model"


def test_custom_hf_endpoint_prefers_hf_token_over_nvidia_key(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _fake_chat_response({"protocol_urgency": "routine", "source_cards": []})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ModelClient(
        FigmentConfig(
            model_backend="hosted_omni",
            hf_endpoint_url="https://hf.example.test/v1",
            hf_token="hf-test-token",
            nvidia_api_key="nvidia-key-not-for-hf",
        )
    )
    client.generate_json("Return JSON.", {})

    assert captured["url"] == "https://hf.example.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer hf-test-token"
    assert "chat_template_kwargs" not in captured["body"]


def test_malformed_chat_response_raises_model_client_error(monkeypatch: Any) -> None:
    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        return _FakeResponse({"choices": []})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ModelClient(
        FigmentConfig(
            model_backend="llama_cpp",
            llama_base_url="http://127.0.0.1:8001/v1",
        )
    )

    with pytest.raises(ModelClientError, match="model response"):
        client.generate_json("Return JSON.", {})


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"protocol_urgency": "routine", "source_cards": []}', "routine"),
        ('Assistant note:\n{"protocol_urgency": "monitor", "source_cards": []}\nDone.', "monitor"),
        ([{"type": "text", "text": '{"protocol_urgency": "urgent", "source_cards": []}'}], "urgent"),
        (
            [
                {"type": "refusal", "refusal": ""},
                {
                    "type": "output_text",
                    "text": '```json\n{"protocol_urgency": "emergency", "source_cards": []}\n```',
                },
            ],
            "emergency",
        ),
    ],
)
def test_openai_compatible_response_content_variants_are_extracted(
    monkeypatch: Any,
    content: Any,
    expected: str,
) -> None:
    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        return _fake_raw_chat_response(content)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ModelClient(
        FigmentConfig(
            model_backend="llama_cpp",
            llama_base_url="http://127.0.0.1:8001/v1",
        )
    )

    result = client.generate_json("Return JSON.", {})

    assert result["protocol_urgency"] == expected


@pytest.mark.parametrize("content", ["not json", "[1, 2, 3]"])
def test_invalid_or_non_object_json_raises_model_client_error(monkeypatch: Any, content: str) -> None:
    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        return _fake_raw_chat_response(content)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ModelClient(
        FigmentConfig(
            model_backend="llama_cpp",
            llama_base_url="http://127.0.0.1:8001/v1",
        )
    )

    with pytest.raises(ModelClientError, match="JSON"):
        client.generate_json("Return JSON.", {})


def test_transport_failures_raise_model_client_error_for_fallback(monkeypatch: Any) -> None:
    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        raise OSError("endpoint unavailable")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ModelClient(
        FigmentConfig(
            model_backend="llama_cpp",
            llama_base_url="http://127.0.0.1:8001/v1",
        )
    )

    with pytest.raises(ModelClientError, match="model backend failed"):
        client.generate_json("Return JSON.", {})


def test_config_loads_nvidia_api_key_from_dotenv(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_BACKEND", raising=False)
    (tmp_path / ".env").write_text(
        "NVIDIA_API_KEY=dotenv-nvidia-key\nMODEL_BACKEND=hosted_omni\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        config_module,
        "_load_dotenv",
        lambda: config_module._load_simple_dotenv(tmp_path / ".env"),
    )

    config = FigmentConfig.from_env()

    assert config.model_backend == "hosted_omni"
    assert config.nvidia_api_key == "dotenv-nvidia-key"
