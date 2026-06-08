"""Focused navigator-output repair prompt helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .observation_targets import targets_for_failure_cards


NAVIGATOR_OUTPUT_FIELD_ORDER = (
    "protocol_urgency",
    "red_flags",
    "intake_facts",
    "candidate_protocol_pathways",
    "missing_info_to_collect",
    "next_observations_to_collect",
    "conflicts_or_uncertainties",
    "responder_checklist",
    "do_not_do",
    "source_cards",
    "handoff_note_sbar",
    "responder_plain_language_script",
    "safety_boundary",
)

TEXT_REPAIR_FIELDS = (
    "red_flags",
    "intake_facts",
    "candidate_protocol_pathways",
    "missing_info_to_collect",
    "next_observations_to_collect",
    "conflicts_or_uncertainties",
    "responder_checklist",
    "do_not_do",
    "handoff_note_sbar",
    "responder_plain_language_script",
    "safety_boundary",
)


@dataclass(frozen=True)
class RepairScope:
    """A focused repair target derived from deterministic validation failures."""

    name: str
    fields: tuple[str, ...]
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class FocusedRepairPrompt:
    """A prompt paired with the repair scope it is meant to replace."""

    scope: RepairScope
    prompt: str


def classify_validation_failures(failures: Iterable[str]) -> tuple[RepairScope, ...]:
    """Group navigator validation failure strings into focused repair scopes."""

    groups: dict[str, dict[str, list[str]]] = {}
    scope_order: list[str] = []
    for failure in failures:
        failure_text = str(failure)
        scope_name, fields = _classify_failure(failure_text)
        if scope_name not in groups:
            groups[scope_name] = {"failures": [], "fields": []}
            scope_order.append(scope_name)
        groups[scope_name]["failures"].append(failure_text)
        groups[scope_name]["fields"].extend(fields)

    scopes: list[RepairScope] = []
    for scope_name in scope_order:
        group = groups[scope_name]
        scopes.append(
            RepairScope(
                name=scope_name,
                fields=_fields_for_scope(scope_name, group["fields"]),
                failures=tuple(group["failures"]),
            )
        )
    return tuple(scopes)


def build_focused_repair_prompts(
    *,
    original_prompt: str,
    previous_output: Mapping[str, Any],
    failures: Iterable[str],
    urgency_floor: str,
    required_observation_targets: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[FocusedRepairPrompt, ...]:
    """Build one focused repair prompt for each classified validation scope."""

    return tuple(
        FocusedRepairPrompt(
            scope=scope,
            prompt=build_focused_repair_prompt(
                original_prompt=original_prompt,
                previous_output=previous_output,
                repair_scope=scope,
                urgency_floor=urgency_floor,
                required_observation_targets=required_observation_targets,
            ),
        )
        for scope in classify_validation_failures(failures)
    )


def build_focused_repair_prompt(
    *,
    original_prompt: str,
    previous_output: Mapping[str, Any],
    repair_scope: RepairScope,
    urgency_floor: str,
    required_observation_targets: Iterable[Mapping[str, Any]] | None = None,
) -> str:
    """Build a JSON-only focused repair prompt for one repair scope."""

    allowed_fields = ", ".join(repair_scope.fields)
    selected_previous_values = {field: previous_output.get(field) for field in repair_scope.fields}
    repair_context = {
        "repair_scope": repair_scope.name,
        "allowed_fields": list(repair_scope.fields),
        "deterministic_validation_failures": list(repair_scope.failures),
        "urgency_floor": urgency_floor,
    }
    if repair_scope.name == "missing_observations":
        repair_context["required_observation_targets"] = targets_for_failure_cards(
            required_observation_targets or (),
            repair_scope.failures,
        )
    return (
        f"{original_prompt}\n\n"
        "Your previous navigator JSON failed deterministic validation. Perform focused field repair only.\n"
        "Do not return the whole navigator output. The controller will merge the returned fields into the "
        "previous output and revalidate them before use.\n"
        f"Return JSON with exactly these top-level keys: {allowed_fields}.\n"
        "Do not include markdown, commentary, chain-of-thought, or unrelated fields.\n"
        "Keep all unchanged facts grounded in confirmed intake, deterministic rules, and retrieved protocol cards.\n"
        "Do not diagnose, prescribe, dose, discharge, or override local protocol or deterministic red flags.\n"
        f"{_scope_instruction(repair_scope.name)}\n\n"
        f"FOCUSED_REPAIR_CONTEXT:\n{json.dumps(repair_context, indent=2, sort_keys=True)}\n\n"
        f"PREVIOUS_VALUES_FOR_ALLOWED_FIELDS:\n{json.dumps(selected_previous_values, indent=2, sort_keys=True)}"
    )


def _classify_failure(failure: str) -> tuple[str, tuple[str, ...]]:
    normalized = failure.lower()
    if "handoff_note_sbar" in normalized:
        return "handoff_note_sbar", ("handoff_note_sbar",)
    if "missing_info_to_collect does not reference required observations" in normalized:
        return "missing_observations", ("missing_info_to_collect", "next_observations_to_collect")
    if _is_citation_or_pathway_failure(normalized):
        return "citations_and_pathways", ("source_cards", "candidate_protocol_pathways")
    if normalized.startswith("forbidden clinical language"):
        return "forbidden_clinical_language", TEXT_REPAIR_FIELDS
    if "below deterministic floor" in normalized or normalized == "protocol_urgency is missing or invalid":
        return "protocol_urgency", ("protocol_urgency",)
    if normalized.startswith("missing required schema keys:"):
        return "schema", _fields_from_schema_missing_failure(failure)
    schema_type_field = _field_from_schema_type_failure(failure)
    if schema_type_field:
        return "schema", (schema_type_field,)
    if normalized == "navigator output is not an object":
        return "schema", NAVIGATOR_OUTPUT_FIELD_ORDER
    return "unclassified", NAVIGATOR_OUTPUT_FIELD_ORDER


def _is_citation_or_pathway_failure(normalized_failure: str) -> bool:
    return (
        normalized_failure.startswith("source_cards ")
        or normalized_failure.startswith("unknown source_cards:")
        or normalized_failure.startswith("fired rule card ")
        or normalized_failure.startswith("candidate_protocol_pathways ")
        or normalized_failure.startswith("candidate_protocol_pathway ")
        or normalized_failure.startswith("candidate pathway ")
    )


def _fields_from_schema_missing_failure(failure: str) -> tuple[str, ...]:
    _prefix, _separator, fields_text = failure.partition(":")
    fields = [field.strip() for field in fields_text.split(",")]
    return tuple(field for field in fields if field)


def _field_from_schema_type_failure(failure: str) -> str | None:
    match = re.fullmatch(r"([a-z_]+) must be a (?:list|string)", failure)
    if not match:
        return None
    field = match.group(1)
    if field in {"source_cards", "candidate_protocol_pathways"}:
        return None
    return field


def _ordered_unique_fields(fields: Iterable[str]) -> tuple[str, ...]:
    unique_fields = {field for field in fields if field}
    ordered = [field for field in NAVIGATOR_OUTPUT_FIELD_ORDER if field in unique_fields]
    ordered.extend(sorted(unique_fields - set(NAVIGATOR_OUTPUT_FIELD_ORDER)))
    return tuple(ordered)


def _fields_for_scope(scope_name: str, fields: Iterable[str]) -> tuple[str, ...]:
    if scope_name == "citations_and_pathways":
        return ("source_cards", "candidate_protocol_pathways")
    return _ordered_unique_fields(fields)


def _scope_instruction(scope_name: str) -> str:
    if scope_name == "handoff_note_sbar":
        return (
            "Repair only handoff_note_sbar. Preserve the four SBAR keys: situation, background, "
            "assessment_observations_only, and handoff_request. Then ground handoff_note_sbar only in confirmed "
            "intake and deterministic rules."
        )
    if scope_name == "missing_observations":
        return (
            "Repair only missing_info_to_collect and next_observations_to_collect. Reference required observations "
            "from required_observation_targets by id and display_text; avoid generic placeholders."
        )
    if scope_name == "citations_and_pathways":
        return (
            "Repair only source_cards and candidate_protocol_pathways. Cite only retrieved or otherwise allowed "
            "card IDs, and ensure every candidate pathway card_id also appears in source_cards."
        )
    if scope_name == "forbidden_clinical_language":
        return (
            "Repair only the allowed text-bearing fields. You must remove or rewrite unsafe clinical language while keeping "
            "negated safety-boundary instructions such as do not diagnose or do not prescribe."
        )
    if scope_name == "protocol_urgency":
        return "Repair only protocol_urgency. It must be one of routine, monitor, urgent, emergency and never below urgency_floor."
    if scope_name == "schema":
        return "Repair only the missing or incorrectly typed schema fields. Preserve the required navigator schema shape."
    return "Repair only the allowed fields. Keep the response minimal and deterministic-validation oriented."
