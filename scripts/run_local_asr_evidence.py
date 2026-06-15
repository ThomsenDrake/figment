#!/usr/bin/env python3
"""Capture evidence for the local Parakeet ASR draft-intake path."""

from __future__ import annotations

import argparse
from contextlib import suppress
import hashlib
import json
from pathlib import Path
import sys
from time import gmtime, strftime
from typing import Any
import wave

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figment.audio_intake import draft_audio_intake  # noqa: E402
from figment.config import (  # noqa: E402
    FigmentConfig,
    NVIDIA_NEMOTRON_3_NANO_4B_BF16_MODEL_ID,
    PARAKEET_ASR_MODEL_ID,
)
from figment.parakeet_asr import transcribe_audio_with_parakeet  # noqa: E402


PARAKEET_REPO = "nvidia/parakeet-rnnt-1.1b"
PARAKEET_REVISION = "a07b19e98a26c1873a3f2622c446a4a1ca6316cb"
PARAKEET_NEMO_BYTES = 4_283_105_280
PARAKEET_NEMO_SHA256 = "535896f014953d945b287ac533560e20da8103c6781b152de4645528e2b60738"
PARAKEET_SNAPSHOT_PATH = Path(
    "/Users/drake.thomsen/.cache/huggingface/hub/"
    "models--nvidia--parakeet-rnnt-1.1b/snapshots/a07b19e98a26c1873a3f2622c446a4a1ca6316cb"
)
PARAKEET_NEMO_PATH = PARAKEET_SNAPSHOT_PATH / "parakeet-rnnt-1.1b.nemo"


def run_evidence(
    *,
    output_dir: Path,
    provider_payload_path: Path | None = None,
    audio_path: Path | None = None,
    provider_note: str = "",
    transcribe_audio: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = _artifact_summary()
    _write_json(output_dir / "artifact_metadata.json", artifact)

    audio_metadata = _audio_metadata(audio_path) if audio_path else None
    if audio_metadata:
        _write_json(output_dir / "audio_metadata.json", audio_metadata)

    summary: dict[str, Any] = {
        "status": "artifact_missing" if not artifact["present"] else "artifact_present_provider_payload_required",
        "output_dir": str(output_dir),
        "artifact_metadata_path": str(output_dir / "artifact_metadata.json"),
        "audio_metadata_path": str(output_dir / "audio_metadata.json") if audio_metadata else None,
        "asr_evidence_manifest_path": str(output_dir / "asr_evidence_manifest.json"),
        "provider_payload_path": str(provider_payload_path) if provider_payload_path else None,
        "draft_path": None,
        "provider_note": provider_note,
        "raw_audio_stored": False,
        "counts_as_local_asr_artifact": bool(artifact["present"]),
        "counts_as_local_asr_proof": False,
    }
    config = FigmentConfig(
        figment_mode="local",
        model_stack="local_4b_parakeet",
        model_backend="llama_cpp",
        audio_backend="parakeet_nemo",
        enable_audio_intake=True,
        allow_local_asr=True,
        local_model_id=NVIDIA_NEMOTRON_3_NANO_4B_BF16_MODEL_ID,
    ).validated()
    if artifact["present"] and provider_payload_path is None and audio_path is not None and transcribe_audio:
        provider_payload = transcribe_audio_with_parakeet(str(audio_path), config=config)
        provider_payload_path = output_dir / "provider_payload.json"
        _write_json(provider_payload_path, provider_payload)
        summary["provider_payload_path"] = str(provider_payload_path)

    if not artifact["present"] or provider_payload_path is None:
        _write_json(
            output_dir / "asr_evidence_manifest.json",
            _asr_evidence_manifest(
                artifact=artifact,
                audio_metadata=audio_metadata,
                provider_payload_metadata=None,
                draft=None,
                checks=None,
                provider_note=provider_note,
                config=config,
            ),
        )
        _write_json(output_dir / "summary.json", summary)
        return summary

    provider_payload = _read_json(provider_payload_path)
    provider_payload_metadata = _provider_payload_metadata(provider_payload_path)
    _write_json(output_dir / "provider_payload_metadata.json", provider_payload_metadata)

    draft = draft_audio_intake(config=config, provider_payload=provider_payload)
    draft_path = output_dir / "audio_draft.json"
    _write_json(draft_path, draft)
    checks = _draft_checks(draft)
    _write_json(output_dir / "draft_checks.json", checks)
    _write_json(
        output_dir / "asr_evidence_manifest.json",
        _asr_evidence_manifest(
            artifact=artifact,
            audio_metadata=audio_metadata,
            provider_payload_metadata=provider_payload_metadata,
            draft=draft,
            checks=checks,
            provider_note=provider_note,
            config=config,
        ),
    )

    summary.update(
        {
            "status": "local_asr_proof_passed" if checks["counts_as_local_asr_proof"] else "local_asr_proof_failed",
            "draft_path": str(draft_path),
            "draft_checks_path": str(output_dir / "draft_checks.json"),
            "provider_payload_metadata_path": str(output_dir / "provider_payload_metadata.json"),
            "counts_as_local_asr_proof": checks["counts_as_local_asr_proof"],
        }
    )
    _write_json(output_dir / "summary.json", summary)
    return summary


def _artifact_summary() -> dict[str, Any]:
    present = PARAKEET_NEMO_PATH.exists()
    size = PARAKEET_NEMO_PATH.stat().st_size if present else None
    sha256 = _sha256(PARAKEET_NEMO_PATH) if present else None
    return {
        "repo": PARAKEET_REPO,
        "revision": PARAKEET_REVISION,
        "snapshot_path": str(PARAKEET_SNAPSHOT_PATH),
        "nemo_path": str(PARAKEET_NEMO_PATH),
        "expected_nemo_bytes": PARAKEET_NEMO_BYTES,
        "expected_nemo_sha256": PARAKEET_NEMO_SHA256,
        "present": present,
        "nemo_bytes": size,
        "nemo_sha256": sha256,
        "matches_expected": present and size == PARAKEET_NEMO_BYTES and sha256 == PARAKEET_NEMO_SHA256,
    }


def _audio_metadata(audio_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(audio_path),
        "exists": audio_path.exists(),
        "raw_audio_copied_to_evidence": False,
    }
    if not audio_path.exists():
        return metadata
    metadata.update(
        {
            "bytes": audio_path.stat().st_size,
            "sha256": _sha256(audio_path),
        }
    )
    with suppress(wave.Error, OSError):
        with wave.open(str(audio_path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            metadata.update(
                {
                    "channels": wav.getnchannels(),
                    "sample_rate_hz": rate,
                    "duration_seconds": round(frames / rate, 3) if rate else None,
                }
            )
    return metadata


def _provider_payload_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _draft_checks(draft: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "has_transcript": bool(str(draft.get("transcript", "")).strip()),
        "has_suggested_fields": bool(draft.get("suggested_fields")),
        "audio_intake_path_is_parakeet": draft.get("audio_intake_path") == "parakeet_asr_plus_text_nemotron",
        "audio_runtime_is_local_4b": draft.get("audio_runtime") == "local_4b_parakeet",
        "audio_model_is_parakeet": draft.get("audio_model_id") == PARAKEET_ASR_MODEL_ID,
        "field_fill_model_is_4b": draft.get("field_fill_model_id") == NVIDIA_NEMOTRON_3_NANO_4B_BF16_MODEL_ID,
        "transcript_source_is_local_parakeet": draft.get("transcript_source") == "local_parakeet_asr_provider",
        "audio_source_is_local_parakeet": draft.get("audio_source") == "local_parakeet_asr_payload",
        "requires_confirmation": draft.get("confirmed_intake_required") is True,
        "is_unconfirmed": draft.get("confirmation_status") == "unconfirmed",
        "raw_audio_stored_false": draft.get("raw_audio_stored") is False,
    }
    proof = all(checks.values())
    return {"checks": checks, "counts_as_local_asr_proof": proof}


def _asr_evidence_manifest(
    *,
    artifact: dict[str, Any],
    audio_metadata: dict[str, Any] | None,
    provider_payload_metadata: dict[str, Any] | None,
    draft: dict[str, Any] | None,
    checks: dict[str, Any] | None,
    provider_note: str,
    config: FigmentConfig | None = None,
) -> dict[str, Any]:
    return {
        "evidence_version": 1,
        "artifact": artifact,
        "provider_payload": provider_payload_metadata,
        "provider_note": provider_note,
        "configured_route": _configured_route_summary(config),
        "raw_audio_handling": {
            "audio_metadata_present": audio_metadata is not None,
            "audio_metadata_sha256": audio_metadata.get("sha256") if audio_metadata else None,
            "raw_audio_copied_to_evidence": False,
            "raw_audio_stored": bool(draft.get("raw_audio_stored")) if draft else False,
        },
        "draft_summary": _draft_summary(draft),
        "draft_checks": checks,
        "proof_flags": {
            "counts_as_local_asr_artifact": bool(artifact.get("present")),
            "counts_as_local_asr_proof": bool(checks and checks.get("counts_as_local_asr_proof")),
        },
    }


def _configured_route_summary(config: FigmentConfig | None) -> dict[str, Any]:
    if config is None:
        return {
            "figment_mode": "local",
            "model_stack": "local_4b_parakeet",
            "model_backend": "llama_cpp",
            "audio_backend": "parakeet_nemo",
            "text_model_id": NVIDIA_NEMOTRON_3_NANO_4B_BF16_MODEL_ID,
            "audio_model_id": PARAKEET_ASR_MODEL_ID,
        }
    return {
        "figment_mode": config.figment_mode,
        "model_stack": config.model_stack,
        "model_backend": config.model_backend,
        "audio_backend": config.audio_backend,
        "text_model_id": config.local_model_id,
        "audio_model_id": config.audio_model_id,
    }


def _draft_summary(draft: dict[str, Any] | None) -> dict[str, Any] | None:
    if draft is None:
        return None
    transcript = str(draft.get("transcript") or "")
    suggested_fields = draft.get("suggested_fields")
    missing_or_unclear_fields = draft.get("missing_or_unclear_fields")
    provisional_red_flags = draft.get("provisional_red_flag_mentions")
    return {
        "audio_intake_path": draft.get("audio_intake_path"),
        "audio_runtime": draft.get("audio_runtime"),
        "transcript_source": draft.get("transcript_source"),
        "audio_source": draft.get("audio_source"),
        "confirmation_status": draft.get("confirmation_status"),
        "transcript_hash": hashlib.sha256(transcript.encode("utf-8")).hexdigest() if transcript else None,
        "transcript_char_count": len(transcript),
        "suggested_field_count": len(suggested_fields) if isinstance(suggested_fields, list) else 0,
        "missing_or_unclear_field_count": (
            len(missing_or_unclear_fields) if isinstance(missing_or_unclear_fields, list) else 0
        ),
        "provisional_red_flag_mention_count": (
            len(provisional_red_flags) if isinstance(provisional_red_flags, list) else 0
        ),
        "raw_audio_stored": draft.get("raw_audio_stored"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("provider payload must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")


def _default_output_dir() -> Path:
    stamp = strftime("%Y%m%dT%H%M%SZ", gmtime())
    return Path("traces") / f"local_asr_parakeet_evidence_{stamp}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-payload", type=Path, default=None)
    parser.add_argument("--audio", type=Path, default=None, help="Optional source audio path; metadata only, not copied.")
    parser.add_argument("--transcribe-audio", action="store_true", help="Run Parakeet ASR on --audio to create provider_payload.json.")
    parser.add_argument("--provider-note", default="")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = run_evidence(
        output_dir=args.output_dir or _default_output_dir(),
        provider_payload_path=args.provider_payload,
        audio_path=args.audio,
        provider_note=args.provider_note,
        transcribe_audio=args.transcribe_audio,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["counts_as_local_asr_proof"]:
        return 0
    if summary["counts_as_local_asr_artifact"]:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
