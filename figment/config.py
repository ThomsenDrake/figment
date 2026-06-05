"""Configuration and model identity contracts for Figment."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


OMNI_MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
OMNI_FP8_MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
OMNI_NVFP4_MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
OMNI_GGUF_MODEL_ID = "ggml-org/NVIDIA-Nemotron-3-Nano-Omni"
STRETCH_TEXT_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
STRETCH_BASE_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
STRETCH_AUDIO_MODEL_ID = "nvidia/parakeet-rnnt-1.1b"
STRETCH_GGUF_MODEL_ID = "bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF"

MODEL_STACKS = {"omni_native", "base_nano_parakeet"}
MODEL_BACKENDS = {"hosted_omni", "llama_cpp", "hosted_text_nemotron", "canned"}
AUDIO_BACKENDS = {"omni_native", "parakeet_nemo", "canned", "none"}
FIGMENT_MODES = {"hosted", "local", "canned"}


def _bool_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class FigmentConfig:
    figment_mode: str = "hosted"
    model_stack: str = "omni_native"
    model_backend: str = "canned"
    audio_backend: str = "none"
    enable_audio_intake: bool = False
    allow_stretch_stack: bool = False
    hf_model_id: str = OMNI_MODEL_ID
    hf_token: str = ""
    omni_endpoint_url: str = ""
    hf_endpoint_url: str = ""
    llama_base_url: str = "http://127.0.0.1:8001/v1"
    trace_dir: Path = Path("traces")

    @classmethod
    def from_env(cls) -> "FigmentConfig":
        mode = os.getenv("FIGMENT_MODE", "hosted").strip() or "hosted"
        stack = os.getenv("MODEL_STACK", "omni_native").strip() or "omni_native"
        backend = os.getenv("MODEL_BACKEND", os.getenv("FIGMENT_MODE", "canned")).strip()
        if backend in FIGMENT_MODES:
            backend = {"hosted": "hosted_omni", "local": "llama_cpp", "canned": "canned"}[backend]
        audio = os.getenv("AUDIO_BACKEND", "none").strip() or "none"
        enable_audio = _bool_env(os.getenv("ENABLE_AUDIO_INTAKE"), False)
        allow_stretch = _bool_env(os.getenv("ALLOW_STRETCH_STACK"), False)
        return cls(
            figment_mode=mode,
            model_stack=stack,
            model_backend=backend,
            audio_backend=audio,
            enable_audio_intake=enable_audio,
            allow_stretch_stack=allow_stretch,
            hf_model_id=os.getenv("HF_MODEL_ID", OMNI_MODEL_ID).strip() or OMNI_MODEL_ID,
            hf_token=os.getenv("HF_TOKEN", "").strip(),
            omni_endpoint_url=os.getenv("OMNI_ENDPOINT_URL", "").strip(),
            hf_endpoint_url=os.getenv("HF_ENDPOINT_URL", "").strip(),
            llama_base_url=os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8001/v1").strip(),
            trace_dir=Path(os.getenv("FIGMENT_TRACE_DIR", "traces").strip() or "traces"),
        ).validated()

    def validated(self) -> "FigmentConfig":
        errors = []
        if self.figment_mode not in FIGMENT_MODES:
            errors.append(f"FIGMENT_MODE must be one of {sorted(FIGMENT_MODES)}")
        if self.model_stack not in MODEL_STACKS:
            errors.append(f"MODEL_STACK must be one of {sorted(MODEL_STACKS)}")
        if self.model_backend not in MODEL_BACKENDS:
            errors.append(f"MODEL_BACKEND must be one of {sorted(MODEL_BACKENDS)}")
        if self.audio_backend not in AUDIO_BACKENDS:
            errors.append(f"AUDIO_BACKEND must be one of {sorted(AUDIO_BACKENDS)}")
        if self.model_stack == "base_nano_parakeet" and not self.allow_stretch_stack:
            errors.append("MODEL_STACK=base_nano_parakeet requires ALLOW_STRETCH_STACK=true")
        if self.audio_backend == "parakeet_nemo" and not self.allow_stretch_stack:
            errors.append("AUDIO_BACKEND=parakeet_nemo requires ALLOW_STRETCH_STACK=true")
        if self.audio_backend == "parakeet_nemo" and self.model_stack != "base_nano_parakeet":
            errors.append("AUDIO_BACKEND=parakeet_nemo requires MODEL_STACK=base_nano_parakeet")
        if self.model_backend == "hosted_text_nemotron" and self.model_stack != "base_nano_parakeet":
            errors.append("hosted_text_nemotron backend is stretch-only")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    @property
    def active_model_id(self) -> str:
        if self.model_stack == "base_nano_parakeet":
            return STRETCH_TEXT_MODEL_ID
        return self.hf_model_id or OMNI_MODEL_ID

    @property
    def audio_model_id(self) -> str | None:
        if self.audio_backend == "omni_native":
            return OMNI_MODEL_ID
        if self.audio_backend == "parakeet_nemo":
            return STRETCH_AUDIO_MODEL_ID
        return None


def load_config() -> FigmentConfig:
    return FigmentConfig.from_env()

