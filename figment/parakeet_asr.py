"""Parakeet ASR adapters for Figment audio draft intake."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import threading
from typing import Any
from contextlib import suppress
import wave

from .config import FigmentConfig, PARAKEET_ASR_MODEL_ID


PARAKEET_TRANSFORMERS_CTC_MODEL_ID = "nvidia/parakeet-ctc-1.1b"
LOCAL_ASR_MAX_BYTES = int(os.getenv("PARAKEET_ASR_MAX_BYTES", str(25 * 1024 * 1024)))
LOCAL_ASR_MAX_SECONDS = float(os.getenv("PARAKEET_ASR_MAX_SECONDS", "120"))


try:  # pragma: no cover - exercised on Hugging Face Spaces.
    import spaces
except ImportError:  # pragma: no cover - local tests run without spaces.

    class _SpacesCompat:
        @staticmethod
        def GPU(**_kwargs: Any):
            def decorator(func):
                return func

            return decorator

    spaces = _SpacesCompat()


class ParakeetAsrError(RuntimeError):
    """Raised when Parakeet cannot produce a usable transcript."""


@spaces.GPU(
    duration=int(os.getenv("PARAKEET_ASR_DURATION_SECONDS", os.getenv("ZEROGPU_DURATION_SECONDS", "60"))),
    size=os.getenv("PARAKEET_ASR_SIZE", os.getenv("ZEROGPU_SIZE", "large")),
)
def transcribe_audio_with_parakeet(audio_file: str, *, config: FigmentConfig) -> dict[str, Any]:
    """Return a Figment provider payload from a local or ZeroGPU Parakeet ASR run."""

    validate_parakeet_audio_file(audio_file)
    model_id = getattr(config, "parakeet_asr_model_id", PARAKEET_ASR_MODEL_ID) or PARAKEET_ASR_MODEL_ID
    runtime_name = getattr(config, "parakeet_asr_runtime", "auto") or "auto"
    runtime = _runtime_for(runtime_name=runtime_name, model_id=model_id)
    transcript = _clean_transcript(runtime.transcribe(audio_file))
    if not transcript:
        raise ParakeetAsrError("Parakeet ASR returned an empty transcript")
    return {
        "transcript": transcript,
        "suggested_fields": [],
        "missing_or_unclear_fields": [],
        "provisional_red_flag_mentions": [],
        "provider_metadata": {
            "asr_model_id": model_id,
            "asr_runtime": runtime.runtime_name,
            "raw_audio_stored": False,
        },
    }


def validate_parakeet_audio_file(audio_file: str) -> None:
    path = Path(audio_file)
    if not path.exists():
        raise ParakeetAsrError(f"audio file does not exist: {audio_file}")
    size = path.stat().st_size
    if size > LOCAL_ASR_MAX_BYTES:
        raise ParakeetAsrError(
            f"audio file exceeds Parakeet ASR size limit: {size} bytes > {LOCAL_ASR_MAX_BYTES} bytes"
        )
    duration = _wav_duration_seconds(path)
    if duration is not None and duration > LOCAL_ASR_MAX_SECONDS:
        raise ParakeetAsrError(
            f"audio file exceeds Parakeet ASR duration limit: {duration:.1f}s > {LOCAL_ASR_MAX_SECONDS:.1f}s"
        )


def parakeet_audio_limits_text() -> str:
    mb = LOCAL_ASR_MAX_BYTES / (1024 * 1024)
    return f"{mb:.0f} MB, {LOCAL_ASR_MAX_SECONDS:.0f} seconds"


def _wav_duration_seconds(path: Path) -> float | None:
    with suppress(wave.Error, OSError, EOFError):
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            if not rate:
                return None
            return wav.getnframes() / rate
    return None


def _clean_transcript(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(value or "").strip()


@lru_cache(maxsize=8)
def _runtime_for(*, runtime_name: str, model_id: str):
    normalized = runtime_name.strip().lower()
    if normalized not in {"auto", "nemo", "transformers"}:
        raise ParakeetAsrError("PARAKEET_ASR_RUNTIME must be one of auto, nemo, transformers")
    if normalized == "nemo":
        return _NemoParakeetRuntime(model_id)
    if normalized == "transformers":
        return _TransformersParakeetRuntime(model_id)
    if model_id == PARAKEET_ASR_MODEL_ID:
        try:
            return _NemoParakeetRuntime(model_id)
        except ParakeetAsrError as exc:
            raise ParakeetAsrError(
                f"{model_id} is a NeMo .nemo checkpoint; install NeMo or set "
                f"PARAKEET_ASR_MODEL_ID={PARAKEET_TRANSFORMERS_CTC_MODEL_ID} with "
                "PARAKEET_ASR_RUNTIME=transformers"
            ) from exc
    return _TransformersParakeetRuntime(model_id)


class _NemoParakeetRuntime:
    runtime_name = "nemo"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.lock = threading.Lock()
        try:
            import nemo.collections.asr as nemo_asr
        except Exception as exc:  # pragma: no cover - NeMo is optional in local CI.
            raise ParakeetAsrError(f"NeMo ASR is unavailable: {exc}") from exc
        try:
            if Path(model_id).exists():
                self.model = nemo_asr.models.ASRModel.restore_from(restore_path=model_id)
            else:
                self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
            with suppress(Exception):
                self.model.eval()
        except Exception as exc:  # pragma: no cover - heavyweight model path.
            raise ParakeetAsrError(f"failed to load NeMo Parakeet model {model_id}: {exc}") from exc

    def transcribe(self, audio_file: str) -> str:
        with self.lock:
            try:
                result = self.model.transcribe([audio_file])
            except Exception as exc:  # pragma: no cover - heavyweight model path.
                raise ParakeetAsrError(f"NeMo Parakeet transcription failed: {exc}") from exc
        return _transcription_text(result)


class _TransformersParakeetRuntime:
    runtime_name = "transformers"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.lock = threading.Lock()
        try:
            import torch
            from transformers import pipeline
        except Exception as exc:
            raise ParakeetAsrError(f"Transformers ASR dependencies are unavailable: {exc}") from exc
        device = 0 if torch.cuda.is_available() else -1
        kwargs: dict[str, Any] = {"model": model_id, "device": device}
        try:
            self.pipeline = pipeline("automatic-speech-recognition", **kwargs)
        except Exception as exc:
            raise ParakeetAsrError(f"failed to load Transformers Parakeet model {model_id}: {exc}") from exc

    def transcribe(self, audio_file: str) -> str:
        with self.lock:
            try:
                result = self.pipeline(audio_file)
            except Exception as exc:
                raise ParakeetAsrError(f"Transformers Parakeet transcription failed: {exc}") from exc
        return _transcription_text(result)


def _transcription_text(result: Any) -> str:
    if isinstance(result, list):
        if not result:
            return ""
        return _transcription_text(result[0])
    if isinstance(result, dict):
        return _clean_transcript(result.get("text") or result.get("transcript"))
    text = getattr(result, "text", None)
    if text is not None:
        return _clean_transcript(text)
    return _clean_transcript(result)
