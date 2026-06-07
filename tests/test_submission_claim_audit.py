from pathlib import Path

from scripts import audit_submission_claims


def test_current_submission_claims_stay_evidence_gated() -> None:
    report = audit_submission_claims.audit_claims()

    assert report["status"] == "passed"
    assert report["violations"] == []
    assert report["gate_status"]["off_grid"] is False
    assert report["gate_status"]["local_4b"] is False
    assert report["gate_status"]["local_asr"] is False
    assert report["gate_status"]["backyard_user_use"] is False


def test_audit_flags_unproven_off_grid_claim() -> None:
    violations = audit_submission_claims.scan_text(
        "Figment has achieved Off the Grid and the no-cloud route is proven.",
        gate_status={gate.key: False for gate in audit_submission_claims.CLAIM_GATES},
    )

    assert {violation["gate"] for violation in violations} == {"off_grid"}
    assert "recorded no-cloud trace" in violations[0]["required_evidence"]


def test_audit_allows_proof_needed_language() -> None:
    text = "\n".join(
        [
            "Off the Grid is targeted / proof-needed until a recorded no-cloud run exists.",
            "Parakeet remains not demo-visible as local ASR proof.",
            "Do not say the target user used or tested Figment until factual notes exist.",
        ]
    )

    assert audit_submission_claims.scan_text(text) == []


def test_audit_flags_unproven_user_use_and_local_asr_claims() -> None:
    text = "\n".join(
        [
            "The responder tested Figment and approved it for field documentation.",
            "Parakeet ASR is demo-visible and proven for local audio intake.",
        ]
    )
    violations = audit_submission_claims.scan_text(
        text,
        relative_path=Path("submission.md"),
        gate_status={gate.key: False for gate in audit_submission_claims.CLAIM_GATES},
    )

    assert [violation["gate"] for violation in violations] == ["backyard_user_use", "local_asr"]
    assert {violation["file"] for violation in violations} == {"submission.md"}
