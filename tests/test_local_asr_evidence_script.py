import json
from pathlib import Path
import wave

from scripts import run_local_asr_evidence


def _provider_payload(path: Path) -> Path:
    payload = {
        "transcript": "Local Parakeet transcript says the patient has trouble breathing.",
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
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_asr_evidence_without_provider_payload_records_artifact_only(tmp_path: Path, monkeypatch) -> None:
    artifact_path = tmp_path / "parakeet-rnnt-1.1b.nemo"
    artifact_path.write_bytes(b"fake parakeet")
    monkeypatch.setattr(run_local_asr_evidence, "PARAKEET_NEMO_PATH", artifact_path)
    monkeypatch.setattr(run_local_asr_evidence, "PARAKEET_NEMO_BYTES", len(b"fake parakeet"))
    monkeypatch.setattr(
        run_local_asr_evidence,
        "PARAKEET_NEMO_SHA256",
        run_local_asr_evidence._sha256(artifact_path),
    )

    summary = run_local_asr_evidence.run_evidence(output_dir=tmp_path / "evidence")

    assert summary["status"] == "artifact_present_provider_payload_required"
    assert summary["counts_as_local_asr_artifact"] is True
    assert summary["counts_as_local_asr_proof"] is False
    assert (tmp_path / "evidence" / "artifact_metadata.json").exists()
    assert summary["asr_evidence_manifest_path"] == str(tmp_path / "evidence" / "asr_evidence_manifest.json")
    manifest = json.loads((tmp_path / "evidence" / "asr_evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["proof_flags"]["counts_as_local_asr_artifact"] is True
    assert manifest["proof_flags"]["counts_as_local_asr_proof"] is False
    assert manifest["provider_payload"] is None


def test_asr_evidence_provider_payload_can_pass_gated_draft_checks(tmp_path: Path, monkeypatch) -> None:
    artifact_path = tmp_path / "parakeet-rnnt-1.1b.nemo"
    artifact_path.write_bytes(b"fake parakeet")
    monkeypatch.setattr(run_local_asr_evidence, "PARAKEET_NEMO_PATH", artifact_path)
    monkeypatch.setattr(run_local_asr_evidence, "PARAKEET_NEMO_BYTES", len(b"fake parakeet"))
    monkeypatch.setattr(
        run_local_asr_evidence,
        "PARAKEET_NEMO_SHA256",
        run_local_asr_evidence._sha256(artifact_path),
    )
    payload_path = _provider_payload(tmp_path / "provider.json")

    summary = run_local_asr_evidence.run_evidence(
        output_dir=tmp_path / "evidence",
        provider_payload_path=payload_path,
        provider_note="local adapter smoke",
    )

    assert summary["status"] == "local_asr_proof_passed"
    assert summary["counts_as_local_asr_artifact"] is True
    assert summary["counts_as_local_asr_proof"] is True
    draft = json.loads((tmp_path / "evidence" / "audio_draft.json").read_text(encoding="utf-8"))
    assert draft["transcript_source"] == "local_parakeet_asr_provider"
    assert draft["raw_audio_stored"] is False
    manifest = json.loads((tmp_path / "evidence" / "asr_evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["evidence_version"] == 1
    assert manifest["provider_payload"]["sha256"] == run_local_asr_evidence._sha256(payload_path)
    assert manifest["draft_summary"]["transcript_hash"]
    assert manifest["draft_summary"]["suggested_field_count"] == 1
    assert manifest["draft_checks"]["counts_as_local_asr_proof"] is True
    assert manifest["configured_route"]["audio_backend"] == "parakeet_nemo"
    assert manifest["configured_route"]["model_stack"] == "local_4b_parakeet"
    assert manifest["raw_audio_handling"]["raw_audio_copied_to_evidence"] is False
    assert manifest["raw_audio_handling"]["raw_audio_stored"] is False


def test_asr_evidence_audio_metadata_hashes_without_copying_audio(tmp_path: Path, monkeypatch) -> None:
    artifact_path = tmp_path / "parakeet-rnnt-1.1b.nemo"
    artifact_path.write_bytes(b"fake parakeet")
    monkeypatch.setattr(run_local_asr_evidence, "PARAKEET_NEMO_PATH", artifact_path)
    monkeypatch.setattr(run_local_asr_evidence, "PARAKEET_NEMO_BYTES", len(b"fake parakeet"))
    monkeypatch.setattr(
        run_local_asr_evidence,
        "PARAKEET_NEMO_SHA256",
        run_local_asr_evidence._sha256(artifact_path),
    )
    audio_path = tmp_path / "source.wav"
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\0\0" * 8000)

    summary = run_local_asr_evidence.run_evidence(output_dir=tmp_path / "evidence", audio_path=audio_path)

    assert summary["counts_as_local_asr_proof"] is False
    audio_metadata = json.loads((tmp_path / "evidence" / "audio_metadata.json").read_text(encoding="utf-8"))
    assert audio_metadata["duration_seconds"] == 1.0
    assert audio_metadata["raw_audio_copied_to_evidence"] is False
    manifest = json.loads((tmp_path / "evidence" / "asr_evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_audio_handling"]["audio_metadata_sha256"] == audio_metadata["sha256"]
    assert manifest["raw_audio_handling"]["raw_audio_copied_to_evidence"] is False
