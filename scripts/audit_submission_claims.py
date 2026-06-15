#!/usr/bin/env python3
"""Audit Figment submission copy for evidence-gated claim drift."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

AUDITED_FILES = (
    Path("README.md"),
    Path("docs/submission_checklist.md"),
    Path("docs/safety_statement.md"),
    Path("docs/local_llama_eval_evidence.md"),
    Path("docs/local_parakeet_asr_evidence.md"),
    Path("docs/user_test_notes.md"),
)

SAFE_CONTEXT_RE = re.compile(
    r"\b("
    r"proof[- ]needed|not yet proven|not proven|unproven|pending|targeted|stretch|tentative|"
    r"not proof|is not proof|not ready|not demo[- ]visible|artifact availability is not proof|"
    r"until|before|after|once|only if|only after|if .* proven|requires?|needed|"
    r"do not|must not|cannot|does not|artifact presence alone|template only|no completed|"
    r"no outcome recorded|claim only|claiming"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClaimGate:
    key: str
    label: str
    evidence_summary: str
    patterns: tuple[re.Pattern[str], ...]


CLAIM_GATES = (
    ClaimGate(
        key="off_grid",
        label="Off the Grid / no-cloud",
        evidence_summary="recorded no-cloud trace or completed local evidence bundle",
        patterns=(
            re.compile(r"\bOff the Grid\b.*\b(achieved|proven|validated|ready|complete|eligible)\b", re.IGNORECASE),
            re.compile(r"\boff[- ]grid\b.*\b(achieved|proven|validated|ready|complete)\b", re.IGNORECASE),
            re.compile(r"\bno[- ]cloud\b.*\b(achieved|proven|validated|ready|complete)\b", re.IGNORECASE),
        ),
    ),
    ClaimGate(
        key="llama_champion",
        label="Llama Champion",
        evidence_summary="eligible llama.cpp/local route trace or eval evidence",
        patterns=(
            re.compile(r"\bLlama Champion\b.*\b(achieved|proven|validated|ready|complete|eligible)\b", re.IGNORECASE),
            re.compile(r"\bllama\.cpp\b.*\b(achieved|proven|validated|ready|complete|eligible)\b", re.IGNORECASE),
        ),
    ),
    ClaimGate(
        key="well_tuned",
        label="Well-Tuned",
        evidence_summary="published tuned model or adapter used by the app and measured",
        patterns=(
            re.compile(r"\bWell[- ]Tuned\b.*\b(achieved|proven|validated|ready|complete|eligible)\b", re.IGNORECASE),
            re.compile(r"\b(fine[- ]tuned|adapter|LoRA)\b.*\b(published|used by the app|measured improvement|achieved)\b", re.IGNORECASE),
        ),
    ),
    ClaimGate(
        key="backyard_user_use",
        label="Backyard AI user-use",
        evidence_summary="completed trained-responder user-test notes",
        patterns=(
            re.compile(r"\b(responder|participant|target user|volunteer)\b.*\b(used|tested|validated|approved|endorsed)\b", re.IGNORECASE),
            re.compile(r"\b(used|tested|validated|approved|endorsed)\b.*\b(Figment|prototype|app)\b", re.IGNORECASE),
        ),
    ),
    ClaimGate(
        key="local_asr",
        label="Local Parakeet ASR",
        evidence_summary="local ASR provider payload with counts_as_local_asr_proof=true",
        patterns=(
            re.compile(r"\bParakeet\b.*\b(proven|validated|ready|demo[- ]visible|enabled|passes|green)\b", re.IGNORECASE),
            re.compile(r"\blocal ASR\b.*\b(proven|validated|ready|demo[- ]visible|enabled|passes|green)\b", re.IGNORECASE),
        ),
    ),
    ClaimGate(
        key="local_4b",
        label="Local 4B model competence",
        evidence_summary="50-case local eval with configured-model competence",
        patterns=(
            re.compile(r"\blocal 4B\b.*\b(proven|validated|ready|competence|passed|green|achieved)\b", re.IGNORECASE),
            re.compile(r"\blocal endpoint\b.*\b(proven|validated|ready|competence|passed|green|achieved)\b", re.IGNORECASE),
        ),
    ),
    ClaimGate(
        key="demo_video",
        label="Demo video",
        evidence_summary="final demo video link",
        patterns=(
            re.compile(r"\bdemo video\b.*\b(complete|published|posted|final|https?://)\b", re.IGNORECASE),
        ),
    ),
    ClaimGate(
        key="social_post",
        label="Social post",
        evidence_summary="final social post link",
        patterns=(
            re.compile(r"\bsocial post\b.*\b(complete|published|posted|final|https?://)\b", re.IGNORECASE),
        ),
    ),
)


def audit_claims(repo_root: Path = REPO_ROOT, files: tuple[Path, ...] = AUDITED_FILES) -> dict[str, Any]:
    gate_status = evidence_gate_status(repo_root)
    violations = scan_claims(repo_root, files, gate_status)
    return {
        "status": "passed" if not violations else "failed",
        "repo_root": str(repo_root),
        "gate_status": gate_status,
        "audited_files": [str(path) for path in files],
        "violations": violations,
    }


def evidence_gate_status(repo_root: Path = REPO_ROOT) -> dict[str, bool]:
    return {
        "off_grid": _has_no_cloud_evidence(repo_root),
        "llama_champion": _has_local_4b_competence(repo_root),
        "well_tuned": _has_well_tuned_evidence(repo_root),
        "backyard_user_use": _has_user_test_notes(repo_root),
        "local_asr": _has_local_asr_proof(repo_root),
        "local_4b": _has_local_4b_competence(repo_root),
        "demo_video": _checklist_row_has_final_link(repo_root, "Demo video"),
        "social_post": _checklist_row_has_final_link(repo_root, "Social post"),
    }


def scan_claims(repo_root: Path, files: tuple[Path, ...], gate_status: dict[str, bool]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for relative_path in files:
        path = repo_root / relative_path
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            violations.extend(_line_violations(relative_path, line_number, line, gate_status))
    return violations


def scan_text(
    text: str,
    *,
    relative_path: Path = Path("sample.md"),
    gate_status: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    states = gate_status or {gate.key: False for gate in CLAIM_GATES}
    violations: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        violations.extend(_line_violations(relative_path, line_number, line, states))
    return violations


def _line_violations(
    relative_path: Path,
    line_number: int,
    line: str,
    gate_status: dict[str, bool],
) -> list[dict[str, Any]]:
    if SAFE_CONTEXT_RE.search(line):
        return []
    found: list[dict[str, Any]] = []
    for gate in CLAIM_GATES:
        if gate_status.get(gate.key) is True:
            continue
        for pattern in gate.patterns:
            if pattern.search(line):
                found.append(
                    {
                        "file": str(relative_path),
                        "line": line_number,
                        "gate": gate.key,
                        "claim": gate.label,
                        "required_evidence": gate.evidence_summary,
                        "text": line.strip(),
                    }
                )
                break
    return found


def _has_local_4b_competence(repo_root: Path) -> bool:
    for summary_path in repo_root.glob("traces/local_4b_evidence_*/summary.json"):
        summary = _read_json(summary_path)
        if (
            summary.get("counts_as_50_case_local_llm_competence") is True
            and int(summary.get("total_cases") or 0) >= 50
        ):
            return True
    return False


def _has_local_asr_proof(repo_root: Path) -> bool:
    for summary_path in repo_root.glob("traces/local_asr_parakeet_evidence_*/summary.json"):
        summary = _read_json(summary_path)
        if summary.get("counts_as_local_asr_proof") is True:
            return True
    return False


def _has_no_cloud_evidence(repo_root: Path) -> bool:
    for summary_path in repo_root.glob("traces/local_4b_evidence_*/summary.json"):
        summary = _read_json(summary_path)
        if summary.get("counts_as_no_cloud_route_proof") is True:
            return True
    return False


def _has_well_tuned_evidence(repo_root: Path) -> bool:
    ledger = (repo_root / "docs/model_parameter_evidence_ledger.md").read_text(encoding="utf-8")
    lowered = ledger.lower()
    return (
        "build-small-hackathon/figment-finetuned-model-archive" in lowered
        and "figment_sft_v14p" in lowered
        and "v14p repair-union" in lowered
        and "150/150 competence" in lowered
        and "do not imply the public no-secret space is serving" in lowered
        and "not trained, published, or measured" not in lowered
    )


def _has_user_test_notes(repo_root: Path) -> bool:
    path = repo_root / "docs/user_test_notes.md"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8").lower()
    if "template only" in text or "no outcome recorded" in text:
        return False
    return "pending" not in text


def _checklist_row_has_final_link(repo_root: Path, artifact_label: str) -> bool:
    path = repo_root / "docs/submission_checklist.md"
    if not path.exists():
        return False
    marker = f"| {artifact_label} |"
    for line in path.read_text(encoding="utf-8").splitlines():
        if marker in line:
            lowered = line.lower()
            return "http" in lowered and "pending" not in lowered and "proof needed" not in lowered
    return False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    args = parser.parse_args(argv)

    report = audit_claims(args.repo_root.resolve())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["violations"]:
        print("submission claim audit failed:", file=sys.stderr)
        for violation in report["violations"]:
            print(
                f"{violation['file']}:{violation['line']}: {violation['claim']} needs "
                f"{violation['required_evidence']}: {violation['text']}",
                file=sys.stderr,
            )
    else:
        print("submission claim audit passed")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
