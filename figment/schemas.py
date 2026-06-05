"""Lightweight data contracts for Figment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Urgency = Literal["routine", "monitor", "urgent", "emergency"]
URGENCY_ORDER = {"routine": 0, "monitor": 1, "urgent": 2, "emergency": 3}


@dataclass
class StructuredIntake:
    setting: str = "disaster site"
    patient_age: str = ""
    pregnancy_status: str = "not_applicable"
    chief_concern: str = ""
    symptoms: str = ""
    vitals: str = ""
    allergies: str = ""
    medications: str = ""
    available_supplies: str = ""
    responder_note: str = ""
    confirmed: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "StructuredIntake":
        defaults = cls()
        allowed = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{key: data.get(key, getattr(defaults, key)) for key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuleResult:
    rule_id: str
    label: str
    urgency: Urgency
    evidence: str
    card_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AudioFieldSuggestion:
    field: str
    draft_value: str
    source_snippet: str = ""
    source_timecode: str = ""
    status: Literal["audio_draft", "accepted", "edited", "rejected"] = "audio_draft"
    needs_confirmation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AudioDraft:
    audio_intake_path: str
    audio_model_id: str | None
    field_fill_model_id: str | None
    audio_runtime: str
    transcript: str = ""
    unclear_spans: list[str] = field(default_factory=list)
    suggested_fields: list[AudioFieldSuggestion] = field(default_factory=list)
    missing_or_unclear_fields: list[str] = field(default_factory=list)
    provisional_red_flag_mentions: list[str] = field(default_factory=list)
    confirmed_intake_required: bool = True
    confirmation_status: Literal["unconfirmed", "confirmed"] = "unconfirmed"
    raw_audio_stored: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["task"] = "audio_intake_draft"
        return data


@dataclass
class ValidationResult:
    passed: bool
    failures: list[str] = field(default_factory=list)

    def add(self, failure: str) -> None:
        self.passed = False
        self.failures.append(failure)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def urgency_at_least(value: str, floor: str) -> bool:
    return URGENCY_ORDER.get(value, -1) >= URGENCY_ORDER.get(floor, 0)


def highest_urgency(values: list[str], default: Urgency = "routine") -> Urgency:
    best = default
    for value in values:
        if URGENCY_ORDER.get(value, -1) > URGENCY_ORDER[best]:
            best = value  # type: ignore[assignment]
    return best
