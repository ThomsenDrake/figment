#!/usr/bin/env python3
"""Generate Figment demo audio with Mistral Voxtral TTS and transcribe it back."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "demo_audio"
TTS_MODEL = "voxtral-mini-tts-2603"
TRANSCRIPTION_MODEL = "voxtral-mini-latest"

DEMO_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "case_1",
        "slug": "pediatric_dehydration",
        "label": "Disaster clinic: pediatric dehydration",
        "audio_filename": "case_1_dictated_intake.wav",
        "voice_id": "148251ff-f7d0-4d88-b723-f7733925448b",
        "voice_name": "Abi",
        "script": (
            "Seven year old at a shelter clinic after flood cleanup. Child cannot keep fluids down, "
            "is lethargic, has a very dry mouth, and has no urine since morning. Temperature "
            "and blood pressure are missing. Supplies include oral rehydration solution, radio, "
            "and transport team."
        ),
    },
    {
        "case_id": "case_2",
        "slug": "wound_infection",
        "label": "Disaster injury: wound infection",
        "audio_filename": "case_2_dictated_intake.wav",
        "voice_id": "c69964a6-ab8b-4f8a-9465-ec0925096ec8",
        "voice_name": "Paul - Neutral",
        "script": (
            "Forty three year old at a mobile clinic with a leg cut from debris three days ago. "
            "The wound is getting worse with spreading redness, swelling, and foul drainage. "
            "Temperature is unknown. Supplies are clean dressings and radio."
        ),
    },
    {
        "case_id": "case_3",
        "slug": "pregnancy_danger_sign",
        "label": "Rural clinic: pregnancy danger sign",
        "audio_filename": "case_3_dictated_intake.wav",
        "voice_id": "5a271406-039d-46fe-835b-fbbb00eaf08d",
        "voice_name": "Marie - Neutral",
        "script": (
            "Twenty nine year old pregnant patient at a rural clinic with vaginal bleeding, severe "
            "headache, and dizziness. Blood pressure is not available. She reports a prenatal "
            "vitamin. Phone and transport contact are available."
        ),
    },
)


def _request_json(client: httpx.Client, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    response = client.request(method, url, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"{method} {url} failed with {response.status_code}: {response.text[:800]}")
    return response.json()


def _audio_bytes(audio_data: str) -> bytes:
    if "," in audio_data and audio_data.lstrip().startswith("data:"):
        audio_data = audio_data.split(",", 1)[1]
    return base64.b64decode(audio_data)


def generate_speech(client: httpx.Client, headers: dict[str, str], case: dict[str, Any], output_path: Path) -> None:
    payload = {
        "model": TTS_MODEL,
        "input": case["script"],
        "voice_id": case["voice_id"],
        "response_format": "wav",
    }
    data = _request_json(client, "POST", "https://api.mistral.ai/v1/audio/speech", headers=headers, json=payload)
    output_path.write_bytes(_audio_bytes(str(data["audio_data"])))


def transcribe_audio(
    client: httpx.Client,
    headers: dict[str, str],
    audio_path: Path,
    transcript_path: Path,
) -> dict[str, Any]:
    with audio_path.open("rb") as audio_file:
        data = {
            "model": TRANSCRIPTION_MODEL,
            "language": "en",
            "timestamp_granularities": "segment",
        }
        files = {"file": (audio_path.name, audio_file, "audio/wav")}
        response = client.post(
            "https://api.mistral.ai/v1/audio/transcriptions",
            headers=headers,
            data=data,
            files=files,
        )
    if response.status_code >= 400:
        raise RuntimeError(f"transcription failed with {response.status_code}: {response.text[:800]}")
    payload = response.json()
    transcript_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_manifest(output_dir: Path, generated_cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tts_model": TTS_MODEL,
        "transcription_model": TRANSCRIPTION_MODEL,
        "cases": generated_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and transcribe Figment demo dictated-intake audio.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("MISTRAL_API_KEY is required")

    output_dir = args.output_dir
    transcript_dir = output_dir / "transcripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": f"Bearer {api_key}"}
    generated_cases: list[dict[str, Any]] = []
    with httpx.Client(timeout=120) as client:
        for case in DEMO_CASES:
            audio_path = output_dir / case["audio_filename"]
            transcript_path = transcript_dir / f"{case['case_id']}_{case['slug']}.voxtral.json"
            generate_speech(client, headers, case, audio_path)
            transcript_payload = transcribe_audio(client, headers, audio_path, transcript_path)
            generated_cases.append(
                {
                    "case_id": case["case_id"],
                    "slug": case["slug"],
                    "label": case["label"],
                    "voice_id": case["voice_id"],
                    "voice_name": case["voice_name"],
                    "source_script": case["script"],
                    "audio_path": str(audio_path.relative_to(PROJECT_ROOT)),
                    "transcript_path": str(transcript_path.relative_to(PROJECT_ROOT)),
                    "voxtral_transcript": str(transcript_payload.get("text", "")).strip(),
                    "language": transcript_payload.get("language"),
                    "usage": transcript_payload.get("usage", {}),
                }
            )
            print(f"generated {audio_path.relative_to(PROJECT_ROOT)}")

    manifest = build_manifest(output_dir, generated_cases)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
