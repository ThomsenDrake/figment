"""ZeroGPU Transformers runtime for Figment's published v14p model."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
import threading
from typing import Any


try:  # pragma: no cover - exercised in Hugging Face Spaces.
    import spaces
except ImportError:  # pragma: no cover - local tests use the no-op decorator.

    class _SpacesCompat:
        @staticmethod
        def GPU(*_: Any, **__: Any) -> Any:  # noqa: N802 - mirrors the spaces API.
            def decorator(function: Any) -> Any:
                return function

            return decorator

    spaces = _SpacesCompat()


_RUNTIMES: dict[tuple[str, str], "_ZeroGpuRuntime"] = {}
_RUNTIME_LOCK = threading.Lock()


@spaces.GPU(
    duration=int(os.getenv("ZEROGPU_DURATION_SECONDS", "60")),
    size=os.getenv("ZEROGPU_SIZE", "large"),
)
def generate_zero_gpu_json(
    *,
    prompt: str,
    model_repo: str,
    model_subfolder: str,
    model_id: str,
) -> dict[str, Any]:
    runtime = _runtime_for(model_repo, model_subfolder, model_id)
    return runtime.generate_json(prompt)


def _runtime_for(model_repo: str, model_subfolder: str, model_id: str) -> "_ZeroGpuRuntime":
    key = (model_repo, model_subfolder)
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(key)
        if runtime is None:
            runtime = _ZeroGpuRuntime(model_repo=model_repo, model_subfolder=model_subfolder, model_id=model_id)
            _RUNTIMES[key] = runtime
        return runtime


class _ZeroGpuRuntime:
    def __init__(self, *, model_repo: str, model_subfolder: str, model_id: str) -> None:
        import torch
        from transformers import AutoConfig
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.max_context_tokens = int(os.getenv("ZEROGPU_MAX_CONTEXT_TOKENS", "1536"))
        self.max_generation_tokens = int(os.getenv("ZEROGPU_MAX_GENERATION_TOKENS", "512"))
        self.lock = threading.Lock()
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_repo,
            subfolder=model_subfolder,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_config = AutoConfig.from_pretrained(
            model_repo,
            subfolder=model_subfolder,
            trust_remote_code=False,
        )
        model_config.use_mamba_kernels = False
        self.model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            subfolder=model_subfolder,
            config=model_config,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
        )
        self.model.to("cuda")
        self.model.eval()
        self.model.config.use_cache = True

    def generate_json(self, prompt: str) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        with self.lock:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            if isinstance(encoded, Mapping):
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
            else:
                input_ids = encoded
                attention_mask = None
            input_ids = input_ids.to(device)
            if attention_mask is None:
                attention_mask = self.torch.ones_like(input_ids)
            else:
                attention_mask = attention_mask.to(device)
            max_input_tokens = max(1, self.max_context_tokens - self.max_generation_tokens - 8)
            if int(input_ids.shape[-1]) > max_input_tokens:
                input_ids = input_ids[:, -max_input_tokens:]
                attention_mask = attention_mask[:, -max_input_tokens:]
            input_len = int(input_ids.shape[-1])
            available_tokens = max(1, self.max_context_tokens - input_len - 8)
            max_new_tokens = max(1, min(self.max_generation_tokens, available_tokens))
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated_ids = output_ids[0, input_len:]
            content = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return _parse_json_object(content)


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise
    if isinstance(parsed, dict):
        return parsed
    raise json.JSONDecodeError("model response JSON was not an object", text, 0)


def _preload_default_runtime_from_env() -> None:
    if os.getenv("MODEL_BACKEND") != "hf_zerogpu":
        return
    if os.getenv("FIGMENT_SKIP_ZEROGPU_PRELOAD", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    model_repo = os.getenv("ZEROGPU_MODEL_REPO", "build-small-hackathon/figment-finetuned-model-archive").strip()
    model_subfolder = os.getenv(
        "ZEROGPU_MODEL_SUBFOLDER",
        "figment_sft_v14p/figment-sft-v14p-lora-merged-bf16",
    ).strip()
    model_id = os.getenv("LOCAL_MODEL_ID", "figment-sft-v14p-lora-merged-bf16").strip()
    if model_repo and model_subfolder and model_id:
        _runtime_for(model_repo, model_subfolder, model_id)


_preload_default_runtime_from_env()
