#!/usr/bin/env python3
"""Report Figment evidence gates without upgrading unsupported claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import audit_submission_claims  # noqa: E402


REPO_ROOT = PROJECT_ROOT


def build_report(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    claim_audit = audit_submission_claims.audit_claims(repo_root)
    gate_status = claim_audit["gate_status"]
    gates = {
        "public_space_no_secret": _public_space_gate(repo_root),
        "hosted_omni_eval": _hosted_eval_gate(repo_root),
        "local_4b_50_case_eval": _local_4b_eval_gate(repo_root),
        "no_cloud_route": _no_cloud_route_gate(repo_root),
        "llama_champion_route": _llama_champion_gate(repo_root),
        "local_asr_provider_proof": _local_asr_gate(repo_root),
        "trained_responder_user_test": _simple_gate(
            passed=gate_status.get("backyard_user_use", False),
            label="Trained-responder user test",
            required_evidence="Completed user-test notes from a real trained responder.",
            evidence_paths=_existing_paths(repo_root, [Path("docs/user_test_notes.md")]),
            next_action="Fill docs/user_test_notes.md from a real trained-responder session.",
        ),
        "demo_video": _simple_gate(
            passed=gate_status.get("demo_video", False),
            label="Demo video",
            required_evidence="Final demo video link.",
            evidence_paths=_existing_paths(repo_root, [Path("docs/submission_checklist.md")]),
            next_action="Add the final demo video link after recording a route-supported demo.",
        ),
        "social_post": _simple_gate(
            passed=gate_status.get("social_post", False),
            label="Social post",
            required_evidence="Final social post link with achieved-versus-targeted wording.",
            evidence_paths=_existing_paths(repo_root, [Path("docs/submission_checklist.md")]),
            next_action="Add the final social post link after proof-sensitive copy is ready.",
        ),
        "well_tuned_adapter": _simple_gate(
            passed=gate_status.get("well_tuned", False),
            label="Well-Tuned adapter",
            required_evidence="Published tuned model or adapter used by the app and measured.",
            evidence_paths=_existing_paths(repo_root, [Path("docs/model_parameter_evidence_ledger.md")]),
            next_action="Leave Well-Tuned as stretch until a published measured adapter exists.",
        ),
        "claim_audit": _simple_gate(
            passed=claim_audit["status"] == "passed",
            label="Submission claim audit",
            required_evidence="No premature achieved/proven/used/tested claims in submission-facing copy.",
            evidence_paths=_existing_paths(repo_root, audit_submission_claims.AUDITED_FILES),
            next_action="Run make audit-claims and fix any overclaiming lines.",
            extra={"violation_count": len(claim_audit["violations"])},
        ),
    }
    missing_gate_keys = [key for key, gate in gates.items() if not gate["passed"]]
    return {
        "status": "complete" if not missing_gate_keys else "incomplete",
        "ready_for_badge_claims": not missing_gate_keys,
        "repo_root": str(repo_root),
        "gates": gates,
        "missing_gate_keys": missing_gate_keys,
    }


def _public_space_gate(repo_root: Path) -> dict[str, Any]:
    checklist = _read_text(repo_root / "docs/submission_checklist.md")
    passed = "Public Hugging Face Space | Runnable" in checklist and "Space cold boot with app files present | Verified" in checklist
    return _simple_gate(
        passed=passed,
        label="Public Hugging Face Space",
        required_evidence="Public Space URL plus cold-boot evidence with app files present.",
        evidence_paths=_existing_paths(repo_root, [Path("docs/submission_checklist.md")]),
        next_action="Re-verify public Space cold boot and record the current Space commit.",
    )


def _hosted_eval_gate(repo_root: Path) -> dict[str, Any]:
    traces = sorted(repo_root.glob("traces/hosted_omni_eval*.jsonl"))
    scorecard_path = repo_root / "docs/hosted_omni_eval_results.md"
    scorecard = _read_text(scorecard_path)
    scorecard_has_current_metrics = all(
        marker in scorecard
        for marker in (
            "31/50",
            "8/50",
            "480/650",
            "170/650",
            "50/50",
        )
    )
    return _simple_gate(
        passed=bool(traces) or scorecard_has_current_metrics,
        label="Hosted Omni eval",
        required_evidence="Hosted Omni eval JSONL trace or committed scorecard.",
        evidence_paths=[str(path) for path in traces]
        + _existing_paths(repo_root, [Path("docs/hosted_omni_eval_results.md")]),
        next_action="Run or refresh the hosted Omni eval and update docs/hosted_omni_eval_results.md.",
    )


def _local_4b_eval_gate(repo_root: Path) -> dict[str, Any]:
    summaries = _local_4b_summaries(repo_root)
    passing = [
        (path, summary)
        for path, summary in summaries
        if summary.get("counts_as_50_case_local_llm_competence") is True
        and int(summary.get("total_cases") or 0) >= 50
    ]
    evidence_paths = _local_4b_evidence_paths([path for path, _summary in passing] or [path for path, _summary in summaries])
    return _simple_gate(
        passed=bool(passing),
        label="Local 4B 50-case eval",
        required_evidence="50-case local OpenAI-compatible eval with configured-model competence.",
        evidence_paths=evidence_paths,
        next_action="Run scripts/run_local_4b_evidence.py against the local full-weight endpoint.",
    )


def _no_cloud_route_gate(repo_root: Path) -> dict[str, Any]:
    summaries = _local_4b_summaries(repo_root)
    passing = [
        (path, summary)
        for path, summary in summaries
        if summary.get("counts_as_no_cloud_route_proof") is True
    ]
    return _simple_gate(
        passed=bool(passing),
        label="No-cloud/off-grid route",
        required_evidence="Recorded no-cloud route proof from a local or self-hosted endpoint.",
        evidence_paths=_local_4b_evidence_paths([path for path, _summary in passing] or [path for path, _summary in summaries]),
        next_action="Capture a no-cloud local route smoke or eval bundle.",
    )


def _llama_champion_gate(repo_root: Path) -> dict[str, Any]:
    summaries = _local_4b_summaries(repo_root)
    passing = [
        (path, summary)
        for path, summary in summaries
        if summary.get("counts_as_50_case_local_llm_competence") is True
        and int(summary.get("total_cases") or 0) >= 50
    ]
    return _simple_gate(
        passed=bool(passing),
        label="Llama Champion route",
        required_evidence="Eligible local llama.cpp/OpenAI-compatible route with trace or eval evidence.",
        evidence_paths=_local_4b_evidence_paths([path for path, _summary in passing] or [path for path, _summary in summaries]),
        next_action="Record a qualifying local model route before claiming Llama Champion.",
    )


def _local_asr_gate(repo_root: Path) -> dict[str, Any]:
    summaries = _local_asr_summaries(repo_root)
    passing = [
        (path, summary)
        for path, summary in summaries
        if summary.get("counts_as_local_asr_proof") is True
    ]
    evidence_paths = _local_asr_evidence_paths([path for path, _summary in passing] or [path for path, _summary in summaries])
    return _simple_gate(
        passed=bool(passing),
        label="Local Parakeet ASR provider proof",
        required_evidence="Real local ASR provider payload with counts_as_local_asr_proof=true.",
        evidence_paths=evidence_paths,
        next_action="Run scripts/run_local_asr_evidence.py with a real local Parakeet provider payload.",
    )


def _simple_gate(
    *,
    passed: bool,
    label: str,
    required_evidence: str,
    evidence_paths: list[str],
    next_action: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate = {
        "passed": bool(passed),
        "label": label,
        "required_evidence": required_evidence,
        "evidence_paths": evidence_paths,
        "next_action": "" if passed else next_action,
    }
    if extra:
        gate.update(extra)
    return gate


def _local_4b_summaries(repo_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [
        (path, _read_json(path))
        for path in sorted(repo_root.glob("traces/local_4b_evidence_*/summary.json"))
    ]


def _local_asr_summaries(repo_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [
        (path, _read_json(path))
        for path in sorted(repo_root.glob("traces/local_asr_parakeet_evidence_*/summary.json"))
    ]


def _local_4b_evidence_paths(summary_paths: list[Path]) -> list[str]:
    paths: list[str] = []
    for summary_path in summary_paths:
        paths.append(str(summary_path))
        manifest_path = summary_path.parent / "eval_evidence_manifest.json"
        if manifest_path.exists():
            paths.append(str(manifest_path))
    return paths


def _local_asr_evidence_paths(summary_paths: list[Path]) -> list[str]:
    paths: list[str] = []
    for summary_path in summary_paths:
        paths.append(str(summary_path))
        manifest_path = summary_path.parent / "asr_evidence_manifest.json"
        if manifest_path.exists():
            paths.append(str(manifest_path))
    return paths


def _existing_paths(repo_root: Path, relative_paths: tuple[Path, ...] | list[Path]) -> list[str]:
    return [str(repo_root / path) for path in relative_paths if (repo_root / path).exists()]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Figment Evidence Gate Status",
        "",
        f"- Status: `{report['status']}`",
        f"- Ready for badge claims: `{str(report['ready_for_badge_claims']).lower()}`",
        "",
        "| Gate | Passed | Next action |",
        "| ---- | ------ | ----------- |",
    ]
    for key, gate in report["gates"].items():
        next_action = gate["next_action"] or "Evidence recorded."
        lines.append(f"| `{key}` | `{str(gate['passed']).lower()}` | {next_action} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(args.repo_root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.markdown:
        print(_markdown_report(report))
    else:
        print(f"evidence gate status: {report['status']}")
        for key in report["missing_gate_keys"]:
            gate = report["gates"][key]
            print(f"- {key}: {gate['next_action']}")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
