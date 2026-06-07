import json
from pathlib import Path


COMPREHENSIVE_CASES_PATH = Path("data/eval/comprehensive_hosted_cases.jsonl")
PROTOCOL_CARDS_DIR = Path("data/protocol_cards")
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
PHI_MARKERS = ("patient_name", "full_name", "address", "phone", "email", "dob", "date_of_birth")


def test_comprehensive_hosted_eval_cases_have_expected_size_and_card_coverage() -> None:
    cases = _load_cases(COMPREHENSIVE_CASES_PATH)
    known_card_ids = _known_card_ids()

    assert len(cases) == 37
    assert {case["target_protocol_card_id"] for case in cases} == known_card_ids
    assert len({case["case_id"] for case in cases}) == len(cases)


def test_comprehensive_hosted_eval_cases_have_valid_shape_and_no_phi() -> None:
    cases = _load_cases(COMPREHENSIVE_CASES_PATH)
    known_card_ids = _known_card_ids()
    serialized = json.dumps(cases).lower()

    for case in cases:
        assert set(case) == REQUIRED_CASE_FIELDS
        assert case["target_protocol_card_id"] in known_card_ids
        assert set(case["expected_source_card_ids"]) <= known_card_ids
        assert case["target_protocol_card_id"] in case["expected_source_card_ids"]
        assert REQUIRED_INTAKE_FIELDS <= set(case["structured_intake"])
        assert case["structured_intake"]["confirmed"] is True
        assert case["expected_min_protocol_urgency"] in {"routine", "monitor", "urgent", "emergency"}
        assert isinstance(case["expected_red_flag_rule_ids"], list)
        assert isinstance(case["expected_missing_observations"], list) and case["expected_missing_observations"]
        assert isinstance(case["safety_notes"], list) and case["safety_notes"]
        assert isinstance(case["forbidden_behavior"], list) and case["forbidden_behavior"]

    assert "data:audio" not in serialized
    assert "base64" not in serialized
    for marker in PHI_MARKERS:
        assert marker not in serialized


def _load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _known_card_ids() -> set[str]:
    return {
        json.loads(path.read_text(encoding="utf-8"))["card_id"]
        for path in PROTOCOL_CARDS_DIR.glob("*.json")
    }
