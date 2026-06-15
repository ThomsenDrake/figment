import sys
from types import ModuleType
from types import SimpleNamespace
from typing import Any

from figment.zerogpu_runtime import _ZeroGpuRuntime


def test_zerogpu_runtime_uses_native_transformers_with_mamba_kernels_disabled(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}
    fake_config = SimpleNamespace(use_mamba_kernels=True, use_cache=False)

    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = "bf16"

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            calls["config"] = {"args": args, "kwargs": kwargs}
            return fake_config

    class FakeTokenizer:
        pad_token = "<pad>"
        eos_token = "</s>"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> FakeTokenizer:
            calls["tokenizer"] = {"args": args, "kwargs": kwargs}
            return FakeTokenizer()

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(use_cache=False)

        def to(self, device: str) -> None:
            calls["device"] = device

        def eval(self) -> None:
            calls["eval"] = True

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> FakeModel:
            calls["model"] = {"args": args, "kwargs": kwargs}
            return FakeModel()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoConfig = FakeAutoConfig
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    _ZeroGpuRuntime(model_repo="repo", model_subfolder="subfolder", model_id="model")

    assert calls["tokenizer"]["kwargs"]["trust_remote_code"] is False
    assert calls["config"]["kwargs"]["trust_remote_code"] is False
    assert fake_config.use_mamba_kernels is False
    assert calls["model"]["kwargs"]["config"] is fake_config
    assert calls["model"]["kwargs"]["trust_remote_code"] is False
    assert calls["model"]["kwargs"]["torch_dtype"] == "bf16"
    assert calls["device"] == "cuda"
    assert calls["eval"] is True


def test_zerogpu_runtime_generate_json_accepts_batch_encoding_tokenizer_output() -> None:
    calls: dict[str, Any] = {}

    class FakeTensor:
        def __init__(self, shape: tuple[int, int]) -> None:
            self.shape = shape

        def to(self, device: str) -> "FakeTensor":
            calls.setdefault("moved_to", []).append(device)
            return self

    class FakeGeneratedIds:
        def __getitem__(self, key: Any) -> list[int]:
            calls["generated_slice"] = key
            return [4, 5, 6]

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, *_: Any, **__: Any) -> dict[str, FakeTensor]:
            return {
                "input_ids": FakeTensor((1, 3)),
                "attention_mask": FakeTensor((1, 3)),
            }

        def decode(self, generated_ids: list[int], *, skip_special_tokens: bool) -> str:
            calls["decoded"] = {"ids": generated_ids, "skip_special_tokens": skip_special_tokens}
            return '{"protocol_urgency": "monitor", "source_cards": []}'

    class FakeTorch:
        @staticmethod
        def ones_like(tensor: FakeTensor) -> FakeTensor:
            calls["ones_like"] = tensor
            return FakeTensor(tensor.shape)

        @staticmethod
        def inference_mode() -> Any:
            class Context:
                def __enter__(self) -> None:
                    return None

                def __exit__(self, *_: Any) -> None:
                    return None

            return Context()

    class FakeModel:
        def parameters(self) -> Any:
            return iter([SimpleNamespace(device="cuda")])

        def generate(self, **kwargs: Any) -> FakeGeneratedIds:
            calls["generate"] = kwargs
            return FakeGeneratedIds()

    class FakeLock:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_: Any) -> None:
            return None

    runtime = object.__new__(_ZeroGpuRuntime)
    runtime.model_id = "model"
    runtime.max_context_tokens = 32
    runtime.max_generation_tokens = 8
    runtime.lock = FakeLock()
    runtime.torch = FakeTorch()
    runtime.tokenizer = FakeTokenizer()
    runtime.model = FakeModel()

    result = runtime.generate_json("Return JSON.")

    assert result == {"protocol_urgency": "monitor", "source_cards": []}
    assert calls["moved_to"] == ["cuda", "cuda"]
    assert calls["generate"]["input_ids"].shape == (1, 3)
    assert calls["generate"]["attention_mask"].shape == (1, 3)
    assert calls["generated_slice"] == (0, slice(3, None, None))
    assert calls["decoded"] == {"ids": [4, 5, 6], "skip_special_tokens": True}
