"""Configuration and model identity contracts for Figment."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


OMNI_MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
OMNI_FP8_MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
OMNI_NVFP4_MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
NVIDIA_NEMOTRON_3_NANO_4B_BF16_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
PARAKEET_ASR_MODEL_ID = "nvidia/parakeet-rnnt-1.1b"
PARAKEET_TRANSFORMERS_CTC_MODEL_ID = "nvidia/parakeet-ctc-1.1b"
NVIDIA_OMNI_API_MODEL_ID = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"
FIGMENT_CANNED_MODEL_ID = "figment-canned-deterministic"
FIGMENT_V14P_MODEL_REPO = "build-small-hackathon/figment-finetuned-model-archive"
FIGMENT_V14P_MODEL_SUBFOLDER = "figment_sft_v14p/figment-sft-v14p-lora-merged-bf16"
FIGMENT_V14P_MODEL_ID = "figment-sft-v14p-lora-merged-bf16"

MODEL_STACKS = {"omni_native", "local_4b_parakeet"}
MODEL_BACKENDS = {"hosted_omni", "llama_cpp", "hf_zerogpu", "canned"}
AUDIO_BACKENDS = {"omni_native", "parakeet_nemo", "canned", "none"}
FIGMENT_MODES = {"hosted", "local", "canned"}


def _bool_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _default_backend_for_env(
    mode: str,
    *,
    nvidia_api_key: str,
    hf_token: str,
    omni_endpoint_url: str,
    hf_endpoint_url: str,
) -> str:
    if mode == "local":
        return "llama_cpp"
    if mode == "canned":
        return "canned"
    if nvidia_api_key or omni_endpoint_url or hf_endpoint_url:
        return "hosted_omni"
    return "canned"


def _mode_consistency_errors(
    mode: str,
    stack: str,
    backend: str,
    audio_backend: str,
    *,
    allow_self_hosted_omni: bool = False,
    strict_hosted_local_backend: bool = False,
) -> list[str]:
    errors: list[str] = []
    if strict_hosted_local_backend and mode == "hosted" and backend == "llama_cpp":
        errors.append("FIGMENT_MODE=hosted cannot use MODEL_BACKEND=llama_cpp; use FIGMENT_MODE=local for llama_cpp")
    if mode == "local":
        self_hosted_omni_path = allow_self_hosted_omni and stack == "omni_native" and backend in {"hosted_omni", "llama_cpp"}
        if backend == "hosted_omni" and not self_hosted_omni_path:
            errors.append("FIGMENT_MODE=local with MODEL_BACKEND=hosted_omni requires ALLOW_SELF_HOSTED_OMNI=true")
        if stack == "omni_native" and not self_hosted_omni_path:
            errors.append("FIGMENT_MODE=local cannot use MODEL_STACK=omni_native without a proven self-hosted Omni path")
        if audio_backend == "omni_native" and not self_hosted_omni_path:
            errors.append("FIGMENT_MODE=local cannot use AUDIO_BACKEND=omni_native without ALLOW_SELF_HOSTED_OMNI=true")
    return errors


@dataclass(frozen=True)
class FigmentConfig:
    figment_mode: str = "hosted"
    model_stack: str = "omni_native"
    model_backend: str = "canned"
    audio_backend: str = "none"
    enable_audio_intake: bool = False
    allow_local_asr: bool = False
    allow_self_hosted_omni: bool = False
    hf_model_id: str = OMNI_MODEL_ID
    hf_token: str = ""
    nvidia_api_key: str = ""
    nvidia_base_url: str = NVIDIA_API_BASE_URL
    nvidia_model_id: str = NVIDIA_OMNI_API_MODEL_ID
    local_model_id: str = NVIDIA_NEMOTRON_3_NANO_4B_BF16_MODEL_ID
    omni_endpoint_url: str = ""
    hf_endpoint_url: str = ""
    llama_base_url: str = "http://127.0.0.1:8001/v1"
    zerogpu_model_repo: str = FIGMENT_V14P_MODEL_REPO
    zerogpu_model_subfolder: str = FIGMENT_V14P_MODEL_SUBFOLDER
    parakeet_asr_model_id: str = PARAKEET_ASR_MODEL_ID
    parakeet_asr_runtime: str = "auto"
    trace_dir: Path = Path("traces")

    @classmethod
    def from_env(cls) -> "FigmentConfig":
        _load_dotenv()
        mode = os.getenv("FIGMENT_MODE", "hosted").strip() or "hosted"
        nvidia_api_key = os.getenv("NVIDIA_API_KEY", "").strip()
        hf_token = os.getenv("HF_TOKEN", "").strip()
        omni_endpoint_url = os.getenv("OMNI_ENDPOINT_URL", "").strip()
        hf_endpoint_url = os.getenv("HF_ENDPOINT_URL", "").strip()
        backend = os.getenv("MODEL_BACKEND", "").strip()
        if not backend:
            backend = _default_backend_for_env(
                mode,
                nvidia_api_key=nvidia_api_key,
                hf_token=hf_token,
                omni_endpoint_url=omni_endpoint_url,
                hf_endpoint_url=hf_endpoint_url,
            )
        if backend in FIGMENT_MODES:
            backend = {"hosted": "hosted_omni", "local": "llama_cpp", "canned": "canned"}[backend]
        stack = os.getenv("MODEL_STACK", "").strip()
        if not stack:
            stack = "local_4b_parakeet" if backend in {"llama_cpp", "hf_zerogpu"} else "omni_native"
        audio = os.getenv("AUDIO_BACKEND", "none").strip() or "none"
        allow_self_hosted_omni = _bool_env(os.getenv("ALLOW_SELF_HOSTED_OMNI"), False)
        mode_errors = _mode_consistency_errors(
            mode,
            stack,
            backend,
            audio,
            allow_self_hosted_omni=allow_self_hosted_omni,
            strict_hosted_local_backend=True,
        )
        if mode_errors:
            raise ValueError("; ".join(mode_errors))
        enable_audio = _bool_env(os.getenv("ENABLE_AUDIO_INTAKE"), False)
        if _bool_env(os.getenv("ALLOW_STRETCH_STACK"), False):
            raise ValueError("ALLOW_STRETCH_STACK is retired; use ALLOW_LOCAL_ASR=true with MODEL_STACK=local_4b_parakeet")
        allow_local_asr = _bool_env(os.getenv("ALLOW_LOCAL_ASR"), False)
        return cls(
            figment_mode=mode,
            model_stack=stack,
            model_backend=backend,
            audio_backend=audio,
            enable_audio_intake=enable_audio,
            allow_local_asr=allow_local_asr,
            allow_self_hosted_omni=allow_self_hosted_omni,
            hf_model_id=os.getenv("HF_MODEL_ID", OMNI_MODEL_ID).strip() or OMNI_MODEL_ID,
            hf_token=os.getenv("HF_TOKEN", "").strip(),
            nvidia_api_key=nvidia_api_key,
            nvidia_base_url=os.getenv("NVIDIA_BASE_URL", NVIDIA_API_BASE_URL).strip() or NVIDIA_API_BASE_URL,
            nvidia_model_id=os.getenv("NVIDIA_MODEL_ID", NVIDIA_OMNI_API_MODEL_ID).strip() or NVIDIA_OMNI_API_MODEL_ID,
            local_model_id=os.getenv(
                "LOCAL_MODEL_ID",
                FIGMENT_V14P_MODEL_ID if backend == "hf_zerogpu" else NVIDIA_NEMOTRON_3_NANO_4B_BF16_MODEL_ID,
            ).strip()
            or (FIGMENT_V14P_MODEL_ID if backend == "hf_zerogpu" else NVIDIA_NEMOTRON_3_NANO_4B_BF16_MODEL_ID),
            omni_endpoint_url=omni_endpoint_url,
            hf_endpoint_url=hf_endpoint_url,
            llama_base_url=os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8001/v1").strip(),
            zerogpu_model_repo=os.getenv("ZEROGPU_MODEL_REPO", FIGMENT_V14P_MODEL_REPO).strip()
            or FIGMENT_V14P_MODEL_REPO,
            zerogpu_model_subfolder=os.getenv("ZEROGPU_MODEL_SUBFOLDER", FIGMENT_V14P_MODEL_SUBFOLDER).strip()
            or FIGMENT_V14P_MODEL_SUBFOLDER,
            parakeet_asr_model_id=os.getenv("PARAKEET_ASR_MODEL_ID", PARAKEET_ASR_MODEL_ID).strip()
            or PARAKEET_ASR_MODEL_ID,
            parakeet_asr_runtime=os.getenv("PARAKEET_ASR_RUNTIME", "auto").strip().lower() or "auto",
            trace_dir=Path(os.getenv("FIGMENT_TRACE_DIR", "traces").strip() or "traces"),
        ).validated()

    def validated(self) -> "FigmentConfig":
        errors = []
        if self.figment_mode not in FIGMENT_MODES:
            errors.append(f"FIGMENT_MODE must be one of {sorted(FIGMENT_MODES)}")
        if self.model_stack == "base_nano_parakeet":
            errors.append("MODEL_STACK=base_nano_parakeet is retired; use MODEL_STACK=local_4b_parakeet")
        elif self.model_stack not in MODEL_STACKS:
            errors.append(f"MODEL_STACK must be one of {sorted(MODEL_STACKS)}")
        if self.model_backend == "hosted_text_nemotron":
            errors.append("MODEL_BACKEND=hosted_text_nemotron is retired; use MODEL_BACKEND=llama_cpp for local_4b_parakeet")
        elif self.model_backend not in MODEL_BACKENDS:
            errors.append(f"MODEL_BACKEND must be one of {sorted(MODEL_BACKENDS)}")
        if self.model_backend == "hf_zerogpu" and self.model_stack != "local_4b_parakeet":
            errors.append("MODEL_BACKEND=hf_zerogpu requires MODEL_STACK=local_4b_parakeet")
        if self.audio_backend not in AUDIO_BACKENDS:
            errors.append(f"AUDIO_BACKEND must be one of {sorted(AUDIO_BACKENDS)}")
        if self.audio_backend == "parakeet_nemo" and not self.allow_local_asr:
            errors.append("AUDIO_BACKEND=parakeet_nemo requires ALLOW_LOCAL_ASR=true")
        if self.audio_backend == "parakeet_nemo" and self.model_stack != "local_4b_parakeet":
            errors.append("AUDIO_BACKEND=parakeet_nemo requires MODEL_STACK=local_4b_parakeet")
        if self.parakeet_asr_runtime not in {"auto", "nemo", "transformers"}:
            errors.append("PARAKEET_ASR_RUNTIME must be one of ['auto', 'nemo', 'transformers']")
        errors.extend(
            _mode_consistency_errors(
                self.figment_mode,
                self.model_stack,
                self.model_backend,
                self.audio_backend,
                allow_self_hosted_omni=self.allow_self_hosted_omni,
            )
        )
        if self.model_backend == "hosted_omni":
            hosted_endpoint = self.omni_endpoint_url or self.hf_endpoint_url or self.nvidia_base_url
            if not hosted_endpoint:
                errors.append("hosted_omni requires NVIDIA_BASE_URL, OMNI_ENDPOINT_URL, or HF_ENDPOINT_URL")
            elif _is_nvidia_endpoint(hosted_endpoint) and not self.nvidia_api_key:
                errors.append("hosted_omni with NVIDIA endpoint requires NVIDIA_API_KEY")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    @property
    def active_model_id(self) -> str:
        if self.model_backend == "hosted_omni":
            return self.nvidia_model_id
        if self.model_backend in {"llama_cpp", "hf_zerogpu"} or self.model_stack == "local_4b_parakeet":
            return self.local_model_id
        return FIGMENT_CANNED_MODEL_ID

    @property
    def audio_model_id(self) -> str | None:
        if self.audio_backend == "omni_native":
            return OMNI_MODEL_ID
        if self.audio_backend == "parakeet_nemo":
            return self.parakeet_asr_model_id
        return None

    @property
    def allow_stretch_stack(self) -> bool:
        return self.allow_local_asr


def load_config() -> FigmentConfig:
    return FigmentConfig.from_env()


def _load_dotenv() -> None:
    if _bool_env(os.getenv("PYTHON_DOTENV_DISABLED"), False):
        return
    dotenv_path = Path.cwd() / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_simple_dotenv(dotenv_path)
        return
    load_dotenv(dotenv_path=dotenv_path, override=False)


def _load_simple_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("'\"")


def _is_nvidia_endpoint(url: str) -> bool:
    return "integrate.api.nvidia.com" in url.lower()
