"""Field-level provenance helpers for navigator output merging."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from .prompt_builder import OUTPUT_SCHEMA


MODEL_RAW = "model_raw"
MODEL_REPAIRED = "model_repaired"
DETERMINISTIC_FALLBACK = "deterministic_fallback"

FieldProvenance = Literal["model_raw", "model_repaired", "deterministic_fallback"]
NAVIGATOR_FIELD_NAMES: tuple[str, ...] = tuple(OUTPUT_SCHEMA.keys())


@dataclass(frozen=True)
class FieldMergeResult:
    """Merged navigator output plus per-field provenance labels."""

    output: dict[str, Any]
    provenance: dict[str, FieldProvenance]


def merge_field_provenance(
    raw_model_output: Mapping[str, Any] | None,
    repaired_fields: Mapping[str, Any] | None,
    deterministic_fallback_output: Mapping[str, Any],
    *,
    accepted_raw_fields: Iterable[str] | None = None,
) -> FieldMergeResult:
    """Merge bounded navigator fields and label where each final field came from.

    The caller is responsible for validating which raw or repaired fields are safe
    to retain. This helper only performs schema-bounded precedence:
    repaired field > raw model field > deterministic fallback field.
    """

    raw = _optional_mapping(raw_model_output, "raw_model_output")
    repaired = _optional_mapping(repaired_fields, "repaired_fields")
    fallback = _required_mapping(
        deterministic_fallback_output,
        "deterministic_fallback_output",
    )
    accepted_raw_field_set = set(raw) if accepted_raw_fields is None else set(accepted_raw_fields)
    missing = [field for field in NAVIGATOR_FIELD_NAMES if field not in fallback]
    if missing:
        raise ValueError(
            f"deterministic fallback is missing navigator fields: {', '.join(missing)}"
        )

    output: dict[str, Any] = {}
    provenance: dict[str, FieldProvenance] = {}
    for field in NAVIGATOR_FIELD_NAMES:
        if field in repaired:
            output[field] = deepcopy(repaired[field])
            provenance[field] = MODEL_REPAIRED
        elif field in raw and field in accepted_raw_field_set:
            output[field] = deepcopy(raw[field])
            provenance[field] = MODEL_RAW
        else:
            output[field] = deepcopy(fallback[field])
            provenance[field] = DETERMINISTIC_FALLBACK
    return FieldMergeResult(output=output, provenance=provenance)


def deterministic_field_provenance() -> dict[str, FieldProvenance]:
    """Return provenance labels for a fully deterministic fallback output."""

    return {field: DETERMINISTIC_FALLBACK for field in NAVIGATOR_FIELD_NAMES}


def model_raw_field_provenance() -> dict[str, FieldProvenance]:
    """Return provenance labels for a fully accepted raw model output."""

    return {field: MODEL_RAW for field in NAVIGATOR_FIELD_NAMES}


def accepted_raw_fields_from_failures(failures: Iterable[str]) -> set[str]:
    """Return schema fields that are not implicated by validation failures."""

    failed_fields: set[str] = set()
    try:
        from .focused_repair import classify_validation_failures
    except ImportError:
        return set(NAVIGATOR_FIELD_NAMES)
    for scope in classify_validation_failures(failures):
        if scope.name == "forbidden_clinical_language":
            return set()
        failed_fields.update(scope.fields)
    return set(NAVIGATOR_FIELD_NAMES) - failed_fields


def has_deterministic_patches(provenance: Mapping[str, str]) -> bool:
    """Return whether any final field came from deterministic fallback."""

    return any(value == DETERMINISTIC_FALLBACK for value in provenance.values())


def summarize_field_provenance(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return compact counts that make hybrid outputs visible in traces and UI."""

    if not isinstance(provenance, Mapping):
        return {
            "counts": {},
            "total_fields": 0,
            "deterministic_patch_count": 0,
            "model_retained_count": 0,
        }
    counts = Counter(str(value) for value in provenance.values())
    return {
        "counts": dict(sorted(counts.items())),
        "total_fields": sum(counts.values()),
        "deterministic_patch_count": counts[DETERMINISTIC_FALLBACK],
        "model_retained_count": counts[MODEL_RAW] + counts[MODEL_REPAIRED],
    }


def _optional_mapping(value: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _required_mapping(value, name)


def _required_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


__all__ = [
    "DETERMINISTIC_FALLBACK",
    "FieldMergeResult",
    "FieldProvenance",
    "MODEL_RAW",
    "MODEL_REPAIRED",
    "NAVIGATOR_FIELD_NAMES",
    "accepted_raw_fields_from_failures",
    "deterministic_field_provenance",
    "has_deterministic_patches",
    "merge_field_provenance",
    "model_raw_field_provenance",
    "summarize_field_provenance",
]
