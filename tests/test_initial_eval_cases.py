import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


EVAL_CASES_PATH = Path("data/eval/initial_handwritten_cases.jsonl")
PROTOCOL_CARDS_DIR = Path("data/protocol_cards")

URGENCY_ORDER = ("routine", "monitor", "urgent", "emergency")
REQUIRED_INTAKE_FIELDS = {
    "setting",
    "patient_age",
    "pregnancy_status",
    "chief_concern",
    "symptoms",
    "vitals",
    "allergies",
    "medications",
    "available_supplies",
    "responder_note",
    "confirmed",
}
REQUIRED_CASE_FIELDS = {
    "case_id",
    "target_protocol_card_id",
    "structured_intake",
    "expected_red_flag_rule_ids",
    "expected_min_protocol_urgency",
    "expected_source_card_ids",
    "expected_missing_observations",
    "safety_notes",
    "forbidden_behavior",
}
RAW_AUDIO_KEY_MARKERS = (
    "raw_audio",
    "audio_bytes",
    "audio_data",
    "audio_blob",
    "base64",
    "data_url",
    "dataurl",
)
RAW_AUDIO_TEXT_MARKERS = ("data:audio", "base64", "raw audio bytes")
PHI_KEY_MARKERS = ("patient_name", "full_name", "address", "phone", "email", "dob", "date_of_birth")
PHI_VALUE_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"\b\d{3}[-.\s]\d{2}[-.\s]\d{4}\b"),
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    re.compile(r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b"),
)


def _known_card_ids() -> set[str]:
    return {
        json.loads(path.read_text(encoding="utf-8"))["card_id"]
        for path in PROTOCOL_CARDS_DIR.glob("*.json")
    }


def _load_cases() -> list[dict[str, Any]]:
    lines = EVAL_CASES_PATH.read_text(encoding="utf-8").splitlines()
    assert lines, "initial eval fixture must not be empty"
    return [json.loads(line) for line in lines]


def _walk_keys(payload: Any) -> Iterable[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield str(key)
            yield from _walk_keys(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _walk_keys(value)


def test_initial_handwritten_eval_cases_cover_protocol_cards() -> None:
    cases = _load_cases()
    known_card_ids = _known_card_ids()

    assert len(cases) == 10
    assert {case["target_protocol_card_id"] for case in cases} == known_card_ids


def test_initial_handwritten_eval_cases_have_valid_shape_and_references() -> None:
    cases = _load_cases()
    known_card_ids = _known_card_ids()
    case_ids = [case["case_id"] for case in cases]

    assert len(case_ids) == len(set(case_ids))
    for case in cases:
        assert set(case) == REQUIRED_CASE_FIELDS
        assert case["target_protocol_card_id"] in known_card_ids
        assert isinstance(case["structured_intake"], dict)
        assert REQUIRED_INTAKE_FIELDS <= set(case["structured_intake"])
        assert case["structured_intake"]["confirmed"] is True
        assert isinstance(case["expected_red_flag_rule_ids"], list)
        assert all(isinstance(rule_id, str) and rule_id for rule_id in case["expected_red_flag_rule_ids"])
        assert case["expected_min_protocol_urgency"] in URGENCY_ORDER
        assert isinstance(case["expected_source_card_ids"], list)
        assert case["expected_source_card_ids"]
        assert set(case["expected_source_card_ids"]) <= known_card_ids
        assert case["target_protocol_card_id"] in case["expected_source_card_ids"]
        assert isinstance(case["expected_missing_observations"], list)
        assert all(
            isinstance(observation, str) and observation
            for observation in case["expected_missing_observations"]
        )
        assert isinstance(case["safety_notes"], list)
        assert case["safety_notes"]
        assert isinstance(case["forbidden_behavior"], list)
        assert case["forbidden_behavior"]


def test_initial_handwritten_eval_cases_exclude_raw_audio_and_phi() -> None:
    cases = _load_cases()
    serialized = json.dumps(cases)
    serialized_lower = serialized.lower()

    for key in _walk_keys(cases):
        normalized = key.lower()
        assert not any(marker in normalized for marker in RAW_AUDIO_KEY_MARKERS), key
        assert not any(marker in normalized for marker in PHI_KEY_MARKERS), key

    assert not any(marker in serialized_lower for marker in RAW_AUDIO_TEXT_MARKERS)
    for pattern in PHI_VALUE_PATTERNS:
        assert not pattern.search(serialized), pattern.pattern
