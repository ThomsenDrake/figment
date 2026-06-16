#!/usr/bin/env python3
"""Export local Codex session traces for a project as sanitized JSONL.

The raw Codex Desktop session files include system/developer instructions,
encrypted reasoning payloads, generated environment context, and sometimes
local secrets in command output. This exporter keeps the useful agent trace
shape while removing the material that should not be published directly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HOME_PATH = Path.home()
HOME = str(HOME_PATH)
LOCAL_USERNAME = HOME_PATH.name

IDENTITY_PATTERNS = [
    (re.compile(re.escape(LOCAL_USERNAME)), "$USER"),
    (re.compile(r"\bThomsenDrake\b"), "<USER_HANDLE>"),
    (re.compile(r"\bDrake\b"), "<USER_NAME>"),
    (re.compile(r"\bdrake\b"), "<USER_NAME>"),
]

SECRET_PATTERNS = [
    (re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"), "<REDACTED_OPENAI_KEY>"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "<REDACTED_HF_TOKEN>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"), "<REDACTED_GITHUB_TOKEN>"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "<REDACTED_GITHUB_TOKEN>"),
    (re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"), "<REDACTED_GITLAB_TOKEN>"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"), "<REDACTED_SLACK_TOKEN>"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), "<REDACTED_GOOGLE_KEY>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<REDACTED_AWS_ACCESS_KEY>"),
    (re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]{20,}"), r"\1<REDACTED_BEARER_TOKEN>"),
    (
        re.compile(
            r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)[A-Z0-9_]*"
            r"\s*[:=]\s*)([\"']?)([^\"'\s,}]{8,})([\"']?)"
        ),
        r"\1\2<REDACTED_SECRET>\4",
    ),
]


def looks_like_internal_context_dump(text: str) -> bool:
    internal_markers = (
        '"base_instructions"',
        '\\"base_instructions\\"',
        '"dynamic_tools"',
        '\\"dynamic_tools\\"',
        "========= MEMORY_SUMMARY BEGINS =========",
        "<permissions instructions>",
        "<app-context>",
        "<skills_instructions>",
        "<plugins_instructions>",
        "<collaboration_mode>",
    )
    if any(marker in text for marker in internal_markers):
        return True
    if "<environment_context>" in text and len(text) > 500:
        return True
    if ".codex/sessions" in text and any(
        marker in text for marker in ('"session_meta"', '"turn_context"', '"response_item"')
    ):
        return True
    return False


@dataclass
class SessionStats:
    session_id: str
    timestamp: str | None
    thread_source: str | None
    source_path: str
    source_sha256: str
    source_size_bytes: int
    events_read: int = 0
    records_written: int = 0
    records_dropped: int = 0
    invalid_json_lines: int = 0
    first_user_prompt: str | None = None
    event_type_counts: Counter[str] = field(default_factory=Counter)
    dropped_reason_counts: Counter[str] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=Path(HOME) / ".codex" / "sessions",
        help="Root directory containing Codex session JSONL files.",
    )
    parser.add_argument(
        "--project-cwd",
        type=Path,
        default=Path.cwd(),
        help="Project cwd to match against session_meta.payload.cwd.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL artifact.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Destination JSON manifest.",
    )
    parser.add_argument(
        "--exclude-session-id",
        action="append",
        default=[],
        help="Session id to exclude. Repeatable.",
    )
    parser.add_argument(
        "--user-threads-only",
        action="store_true",
        help="Export only sessions whose first session_meta has thread_source=user.",
    )
    return parser.parse_args()


def redact_text(text: str) -> str:
    if looks_like_internal_context_dump(text):
        return "<OMITTED_INTERNAL_CODEX_CONTEXT_DUMP>"
    redacted = text.replace(HOME, "$HOME")
    for pattern, replacement in IDENTITY_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    for pattern, replacement in SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def sanitize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    return value


def sanitize_any(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        omitted_fields: list[str] = []
        for key, item in value.items():
            clean_key = redact_text(str(key))
            if clean_key == "encrypted_content":
                omitted_fields.append("encrypted reasoning payload")
                continue
            sanitized[clean_key] = sanitize_any(item)
        if omitted_fields:
            existing = sanitized.get("omitted_fields")
            if isinstance(existing, list):
                existing.extend(omitted_fields)
            else:
                sanitized["omitted_fields"] = omitted_fields
        return sanitized
    if isinstance(value, list):
        return [sanitize_any(v) for v in value]
    return sanitize_scalar(value)


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def is_generated_user_message(payload: dict[str, Any]) -> bool:
    if payload.get("type") != "message" or payload.get("role") != "user":
        return False
    text = text_from_content(payload.get("content")).lstrip()
    generated_prefixes = (
        "<environment_context>",
        "<skill>",
        "<permissions instructions>",
        "<app-context>",
        "<collaboration_mode>",
        "<plugins_instructions>",
        "<skills_instructions>",
    )
    return text.startswith(generated_prefixes) or "========= MEMORY_SUMMARY BEGINS =========" in text


def keep_message(payload: dict[str, Any]) -> tuple[bool, str | None]:
    role = payload.get("role")
    if role in {"developer", "system"}:
        return False, f"drop_{role}_message"
    if is_generated_user_message(payload):
        return False, "drop_generated_user_context"
    return True, None


def sanitize_session_meta(payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "timestamp",
        "cwd",
        "originator",
        "cli_version",
        "source",
        "thread_source",
        "model_provider",
    ]
    sanitized = {key: sanitize_any(payload.get(key)) for key in keys if key in payload}
    git = payload.get("git")
    if isinstance(git, dict):
        sanitized["git"] = sanitize_any(
            {
                key: git.get(key)
                for key in ("commit_hash", "branch", "repository_url")
                if key in git
            }
        )
    omitted = []
    for key in ("base_instructions", "dynamic_tools", "memory"):
        if key in payload:
            omitted.append(key)
    if omitted:
        sanitized["omitted_fields"] = omitted
    return sanitized


def sanitize_turn_context(payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "turn_id",
        "cwd",
        "workspace_roots",
        "current_date",
        "timezone",
        "approval_policy",
        "sandbox_policy",
        "model",
        "effort",
        "summary",
        "realtime_active",
        "multi_agent_version",
    ]
    sanitized = {key: sanitize_any(payload.get(key)) for key in keys if key in payload}
    collaboration_mode = payload.get("collaboration_mode")
    if isinstance(collaboration_mode, dict):
        sanitized["collaboration_mode"] = sanitize_any(
            {"mode": collaboration_mode.get("mode")}
        )
    permission_profile = payload.get("permission_profile")
    if isinstance(permission_profile, dict):
        sanitized["permission_profile"] = sanitize_any(
            {"type": permission_profile.get("type")}
        )
    return sanitized


def sanitize_response_item(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    item_type = payload.get("type")
    if item_type == "message":
        keep, reason = keep_message(payload)
        if not keep:
            return None, reason
        return sanitize_any(payload), None

    if item_type == "reasoning":
        sanitized = {
            "type": "reasoning",
            "summary": sanitize_any(payload.get("summary", [])),
        }
        if "encrypted_content" in payload:
            sanitized["omitted_fields"] = ["encrypted reasoning payload"]
        return sanitized, None

    if item_type == "tool_search_output":
        tools = payload.get("tools")
        compact_tools = []
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    compact_tools.append(
                        sanitize_any(
                            {
                                key: tool.get(key)
                                for key in ("type", "name", "description")
                                if key in tool
                            }
                        )
                    )
        return (
            {
                "type": "tool_search_output",
                "call_id": sanitize_any(payload.get("call_id")),
                "status": sanitize_any(payload.get("status")),
                "execution": sanitize_any(payload.get("execution")),
                "tools": compact_tools,
                "omitted_fields": ["tool_schemas"],
            },
            None,
        )

    sanitized = sanitize_any(payload)
    for key in ("encrypted_content",):
        if key in sanitized:
            sanitized.pop(key, None)
            sanitized.setdefault("omitted_fields", []).append("encrypted reasoning payload")
    return sanitized, None


def sanitize_compacted(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = sanitize_any({k: v for k, v in payload.items() if k != "replacement_history"})
    replacement_history = payload.get("replacement_history")
    if isinstance(replacement_history, list):
        filtered = []
        dropped = Counter()
        for item in replacement_history:
            if isinstance(item, dict) and item.get("type") == "message":
                keep, reason = keep_message(item)
                if not keep:
                    dropped[reason or "drop_message"] += 1
                    continue
            filtered.append(sanitize_any(item))
        sanitized["replacement_history"] = filtered
        if dropped:
            sanitized["dropped_replacement_history_counts"] = dict(dropped)
    return sanitized


def sanitize_event(obj: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    event_type = obj.get("type")
    payload = obj.get("payload")

    if event_type == "session_meta" and isinstance(payload, dict):
        return {**obj, "payload": sanitize_session_meta(payload)}, None

    if event_type == "turn_context" and isinstance(payload, dict):
        return {**obj, "payload": sanitize_turn_context(payload)}, None

    if event_type == "response_item" and isinstance(payload, dict):
        clean_payload, reason = sanitize_response_item(payload)
        if clean_payload is None:
            return None, reason
        return {**obj, "payload": clean_payload}, None

    if event_type == "compacted" and isinstance(payload, dict):
        return {**obj, "payload": sanitize_compacted(payload)}, None

    if event_type == "event_msg" and isinstance(payload, dict):
        if payload.get("type") == "user_message":
            message = str(payload.get("message", "")).lstrip()
            if message.startswith("<environment_context>") or message.startswith("<skill>"):
                return None, "drop_generated_event_msg"
        return {**obj, "payload": sanitize_any(payload)}, None

    return sanitize_any(obj), None


def first_session_meta(path: Path) -> dict[str, Any] | None:
    with path.open(errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return None
            if obj.get("type") == "session_meta" and isinstance(obj.get("payload"), dict):
                return obj["payload"]
    return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_session_path(path: Path, sessions_root: Path) -> str:
    try:
        return str(path.relative_to(sessions_root))
    except ValueError:
        return redact_text(str(path))


def discover_sessions(
    sessions_root: Path,
    project_cwd: Path,
    exclude_session_ids: set[str],
    user_threads_only: bool,
) -> list[tuple[Path, dict[str, Any]]]:
    project_key = str(project_cwd.resolve()).lower()
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(sessions_root.rglob("*.jsonl")):
        meta = first_session_meta(path)
        if not meta:
            continue
        session_id = meta.get("id")
        cwd = meta.get("cwd")
        if not isinstance(cwd, str) or cwd.lower() != project_key:
            continue
        if session_id in exclude_session_ids:
            continue
        if user_threads_only and meta.get("thread_source") != "user":
            continue
        matches.append((path, meta))
    matches.sort(key=lambda item: (item[1].get("timestamp") or "", item[1].get("id") or ""))
    return matches


def maybe_capture_first_user_prompt(stats: SessionStats, clean_obj: dict[str, Any]) -> None:
    if stats.first_user_prompt is not None:
        return
    if clean_obj.get("type") != "response_item":
        return
    payload = clean_obj.get("payload")
    if not isinstance(payload, dict):
        return
    if payload.get("type") != "message" or payload.get("role") != "user":
        return
    prompt = text_from_content(payload.get("content")).strip()
    if prompt:
        stats.first_user_prompt = prompt[:1000]


def export_sessions(
    sessions: list[tuple[Path, dict[str, Any]]],
    sessions_root: Path,
    output_path: Path,
) -> tuple[list[SessionStats], Counter[str]]:
    aggregate_dropped = Counter()
    stats_by_session: list[SessionStats] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for session_index, (path, meta) in enumerate(sessions):
            stats = SessionStats(
                session_id=str(meta.get("id")),
                timestamp=meta.get("timestamp"),
                thread_source=meta.get("thread_source"),
                source_path=relative_session_path(path, sessions_root),
                source_sha256=file_sha256(path),
                source_size_bytes=path.stat().st_size,
            )
            with path.open(errors="replace") as handle:
                for event_index, line in enumerate(handle):
                    if not line.strip():
                        continue
                    stats.events_read += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        stats.invalid_json_lines += 1
                        stats.records_dropped += 1
                        stats.dropped_reason_counts["invalid_json"] += 1
                        aggregate_dropped["invalid_json"] += 1
                        continue

                    event_type = str(obj.get("type"))
                    stats.event_type_counts[event_type] += 1
                    clean_obj, drop_reason = sanitize_event(copy.deepcopy(obj))
                    if clean_obj is None:
                        reason = drop_reason or "dropped"
                        stats.records_dropped += 1
                        stats.dropped_reason_counts[reason] += 1
                        aggregate_dropped[reason] += 1
                        continue

                    maybe_capture_first_user_prompt(stats, clean_obj)
                    record = {
                        "schema_version": "figment_codex_agent_trace_event_v1",
                        "project": "figment",
                        "session_id": stats.session_id,
                        "session_index": session_index,
                        "thread_source": stats.thread_source,
                        "source_path": stats.source_path,
                        "source_event_index": event_index,
                        "event": clean_obj,
                    }
                    out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    stats.records_written += 1
            stats_by_session.append(stats)
    return stats_by_session, aggregate_dropped


def write_manifest(
    manifest_path: Path,
    stats: list[SessionStats],
    args: argparse.Namespace,
    aggregate_dropped: Counter[str],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    thread_source_counts = Counter(stat.thread_source for stat in stats)
    event_type_counts = Counter()
    for stat in stats:
        event_type_counts.update(stat.event_type_counts)

    manifest = {
        "schema_version": "figment_codex_agent_trace_manifest_v1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "project": "figment",
        "project_cwd": redact_text(str(args.project_cwd.resolve())),
        "sessions_root": redact_text(str(args.sessions_root.resolve())),
        "output": redact_text(str(args.output.resolve())),
        "selection": {
            "cwd_match": "case-insensitive exact match against session_meta.payload.cwd",
            "excluded_session_ids": args.exclude_session_id,
            "user_threads_only": bool(args.user_threads_only),
        },
        "privacy_transform": {
            "dropped": [
                "system/developer role messages",
                "generated environment-context user messages",
                "generated skill-instruction user messages",
                "generated event_msg copies of environment or skill context",
            ],
            "omitted_fields": [
                "session_meta.payload.base_instructions",
                "session_meta.payload.dynamic_tools",
                "encrypted reasoning content",
                "tool_search_output tool schemas",
                "raw internal Codex context dumps embedded in command output strings",
            ],
            "redactions": [
                "absolute home directory paths -> $HOME",
                "local usernames and public account handles -> <USER...>",
                "common API key/token/secret patterns -> <REDACTED_...>",
            ],
            "note": "This is a sanitized publication candidate, not a substitute for final human privacy review before Hugging Face upload.",
        },
        "counts": {
            "session_files_selected": len(stats),
            "source_size_bytes": sum(stat.source_size_bytes for stat in stats),
            "events_read": sum(stat.events_read for stat in stats),
            "records_written": sum(stat.records_written for stat in stats),
            "records_dropped": sum(stat.records_dropped for stat in stats),
            "invalid_json_lines": sum(stat.invalid_json_lines for stat in stats),
            "thread_source_counts": {str(k): v for k, v in thread_source_counts.items()},
            "event_type_counts": dict(event_type_counts),
            "dropped_reason_counts": dict(aggregate_dropped),
        },
        "sessions": [
            {
                "session_id": stat.session_id,
                "timestamp": stat.timestamp,
                "thread_source": stat.thread_source,
                "source_path": stat.source_path,
                "source_sha256": stat.source_sha256,
                "source_size_bytes": stat.source_size_bytes,
                "events_read": stat.events_read,
                "records_written": stat.records_written,
                "records_dropped": stat.records_dropped,
                "invalid_json_lines": stat.invalid_json_lines,
                "event_type_counts": dict(stat.event_type_counts),
                "dropped_reason_counts": dict(stat.dropped_reason_counts),
                "first_user_prompt": stat.first_user_prompt,
            }
            for stat in stats
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    sessions = discover_sessions(
        sessions_root=args.sessions_root,
        project_cwd=args.project_cwd,
        exclude_session_ids=set(args.exclude_session_id),
        user_threads_only=args.user_threads_only,
    )
    stats, aggregate_dropped = export_sessions(sessions, args.sessions_root, args.output)
    write_manifest(args.manifest, stats, args, aggregate_dropped)
    print(f"selected_sessions={len(stats)}")
    print(f"records_written={sum(stat.records_written for stat in stats)}")
    print(f"records_dropped={sum(stat.records_dropped for stat in stats)}")
    print(f"output={args.output}")
    print(f"manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
