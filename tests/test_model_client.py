import json
from pathlib import Path
from typing import Any

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
    return _FakeResponse({"choices": [{"message": {"content": json.dumps(content)}}]})


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
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}


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
            nvidia_api_key="hosted-key-not-for-local",
        )
    )
    result = client.generate_json("Local JSON please.", {})

    assert result["protocol_urgency"] == "monitor"
    assert captured["url"] == "http://127.0.0.1:8001/v1/chat/completions"
    assert captured["authorization"] is None
    assert captured["body"]["model"] == EXPECTED_NVIDIA_OMNI_API_MODEL_ID


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


def test_config_loads_nvidia_api_key_from_dotenv(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_BACKEND", raising=False)
    (tmp_path / ".env").write_text(
        "NVIDIA_API_KEY=dotenv-nvidia-key\nMODEL_BACKEND=hosted_omni\n",
        encoding="utf-8",
    )

    config = FigmentConfig.from_env()

    assert config.model_backend == "hosted_omni"
    assert config.nvidia_api_key == "dotenv-nvidia-key"
