import json
from pathlib import Path
from typing import Any

from scripts import run_local_4b_evidence


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_normalize_base_url_accepts_models_or_chat_url() -> None:
    assert (
        run_local_4b_evidence._normalize_base_url("http://local-runtime.local:8001/v1/models")
        == "http://local-runtime.local:8001/v1"
    )
    assert (
        run_local_4b_evidence._normalize_base_url("http://local-runtime.local:8001/v1/chat/completions")
        == "http://local-runtime.local:8001/v1"
    )


def test_endpoint_failure_writes_summary_without_running_smoke(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_urlopen(*_: Any, **__: Any) -> _FakeResponse:
        raise OSError("no route")

    def fail_smoke() -> dict[str, Any]:
        raise AssertionError("smoke should not run if /v1/models is unavailable")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(run_local_4b_evidence, "run_smoke", fail_smoke)

    summary = run_local_4b_evidence.run_evidence(
        base_url="http://local-runtime.local:8001/v1",
        model_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        output_dir=tmp_path,
        case_paths=[],
        limit=None,
        timeout_seconds=0.1,
    )

    assert summary["status"] == "endpoint_unavailable"
    assert summary["counts_as_no_cloud_route_proof"] is False
    assert summary["counts_as_50_case_local_llm_competence"] is False
    assert (tmp_path / "endpoint_metadata.json").exists()
    assert (tmp_path / "summary.json").exists()


def test_smoke_failure_skips_eval_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _FakeResponse({"data": [{"id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"}]}),
    )
    monkeypatch.setattr(
        run_local_4b_evidence,
        "run_smoke",
        lambda: {
            "status": "failed",
            "local_llm_evidence": {
                "counts_as_no_cloud_route_proof": False,
                "counts_as_50_case_local_llm_competence": False,
            },
        },
    )

    def fail_eval(*_: Any, **__: Any) -> dict[str, Any]:
        raise AssertionError("eval should be skipped when smoke fails")

    monkeypatch.setattr(run_local_4b_evidence, "run_eval", fail_eval)

    summary = run_local_4b_evidence.run_evidence(
        base_url="http://local-runtime.local:8001/v1",
        model_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        output_dir=tmp_path,
        case_paths=[],
        limit=None,
        timeout_seconds=1.0,
    )

    assert summary["status"] == "smoke_failed_eval_skipped"
    assert summary["eval_skip_reason"] == "route smoke did not prove configured-model validation"
    assert (tmp_path / "route_smoke.json").exists()


def test_smoke_only_pass_records_route_proof(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _FakeResponse({"data": [{"id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"}]}),
    )
    monkeypatch.setattr(
        run_local_4b_evidence,
        "run_smoke",
        lambda: {
            "status": "passed",
            "local_llm_evidence": {
                "counts_as_no_cloud_route_proof": True,
                "counts_as_50_case_local_llm_competence": False,
            },
        },
    )

    summary = run_local_4b_evidence.run_evidence(
        base_url="http://local-runtime.local:8001/v1/models",
        model_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        output_dir=tmp_path,
        case_paths=[],
        limit=None,
        timeout_seconds=1.0,
        smoke_only=True,
    )

    assert summary["status"] == "smoke_passed"
    assert summary["base_url"] == "http://local-runtime.local:8001/v1"
    assert summary["counts_as_no_cloud_route_proof"] is True
    assert summary["counts_as_50_case_local_llm_competence"] is False


def test_completed_eval_writes_evidence_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _FakeResponse({"data": [{"id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"}]}),
    )
    monkeypatch.setattr(
        run_local_4b_evidence,
        "run_smoke",
        lambda: {
            "status": "passed",
            "local_llm_evidence": {
                "counts_as_no_cloud_route_proof": True,
                "counts_as_50_case_local_llm_competence": False,
            },
        },
    )

    def fake_run_eval(*, output_path: Path, **_: Any) -> dict[str, Any]:
        records = [
            {
                "case_id": "case-1",
                "raw_configured_model_success": True,
                "repair_success": False,
                "canned_fallback_used": False,
                "competence_success": True,
                "latency_ms": 10.0,
                "trace_hash": "trace-a",
            },
            {
                "case_id": "case-2",
                "raw_configured_model_success": False,
                "repair_success": False,
                "canned_fallback_used": True,
                "competence_success": False,
                "latency_ms": 30.0,
                "trace_hash": "trace-b",
            },
        ]
        output_path.write_text(
            "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records),
            encoding="utf-8",
        )
        return {
            "total_cases": 2,
            "raw_configured_model_successes": 1,
            "repair_successes": 0,
            "competence_successes": 1,
            "fallback_uses": 1,
            "canned_fallback_uses": 1,
            "final_validation_successes": 2,
            "records_with_field_provenance": 2,
            "field_provenance_fields": 26,
            "field_provenance_counts": {"model_raw": 13, "deterministic_fallback": 13},
            "model_retained_field_count": 13,
            "visible_field_provenance_count": 26,
            "model_visible_field_count": 13,
            "deterministic_patch_count": 13,
            "model_field_pass_rate": 0.5,
            "model_visible_fields_retained": 0.5,
            "local_llm_evidence": {
                "counts_as_50_case_local_llm_eval": False,
                "counts_as_50_case_local_llm_competence": False,
            },
        }

    monkeypatch.setattr(run_local_4b_evidence, "run_eval", fake_run_eval)

    summary = run_local_4b_evidence.run_evidence(
        base_url="http://192.168.1.7:8001/v1",
        model_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        output_dir=tmp_path,
        case_paths=[],
        limit=None,
        timeout_seconds=1.0,
    )

    manifest_path = Path(summary["eval_evidence_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["status"] == "eval_completed"
    assert summary["trace_hash_count"] == 2
    assert summary["latency_ms"]["mean"] == 20.0
    assert manifest["model_server_metadata"]["advertised_model_ids"] == [
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    ]
    assert manifest["no_cloud_evidence"]["base_url_host_class"] == "private_lan"
    assert manifest["score_summary"]["raw_configured_model_successes"] == 1
    assert manifest["score_summary"]["canned_fallback_uses"] == 1
    assert manifest["case_ids"]["raw_success"] == ["case-1"]
    assert manifest["case_ids"]["full_fallback"] == ["case-2"]
    assert manifest["field_provenance"]["model_field_pass_rate"] == 0.5
    assert manifest["trace_hashes"] == [
        {"case_id": "case-1", "trace_hash": "trace-a"},
        {"case_id": "case-2", "trace_hash": "trace-b"},
    ]
