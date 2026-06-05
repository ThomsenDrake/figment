import json
from pathlib import Path
from typing import Any

from figment.rules import run_red_flag_checks


CARD_DIR = Path("data/protocol_cards")


def _known_card_ids() -> set[str]:
    return {
        json.loads(path.read_text(encoding="utf-8"))["card_id"]
        for path in CARD_DIR.glob("*.json")
    }


def _confirmed(**overrides: Any) -> dict[str, Any]:
    intake = {
        "confirmed": True,
        "setting": "field clinic",
        "patient_age": "42",
        "pregnancy_status": "not_applicable",
        "chief_concern": "",
        "symptoms": "",
        "vitals": "",
        "allergies": "unknown",
        "medications": "unknown",
        "available_supplies": "radio, basic kit",
        "responder_note": "",
    }
    intake.update(overrides)
    return intake


def test_red_flag_rules_emit_only_existing_protocol_card_ids() -> None:
    cases = [
        _confirmed(chief_concern="new confusion", symptoms="difficult to wake"),
        _confirmed(chief_concern="severe respiratory distress", symptoms="gasping and blue lips"),
        _confirmed(
            pregnancy_status="pregnant",
            chief_concern="vaginal bleeding",
            symptoms="severe headache",
        ),
        _confirmed(chief_concern="chest pain", symptoms="chest pressure"),
        _confirmed(chief_concern="stroke concern", symptoms="facial droop and slurred speech"),
        _confirmed(patient_age="7", chief_concern="dehydration", symptoms="lethargic with no urine"),
        _confirmed(chief_concern="fever", symptoms="stiff neck", vitals="temperature 103 F"),
        _confirmed(chief_concern="wound getting worse", symptoms="spreading redness and foul drainage"),
    ]
    known_cards = _known_card_ids()
    emitted_rules = []

    for intake in cases:
        emitted_rules.extend(run_red_flag_checks(intake))

    assert {
        "AMS-001",
        "RESP-001",
        "PREG-001",
        "red_flag_chest_pain",
        "STROKE-001",
        "PED-DEHYD-001",
        "FEVER-001",
        "WOUND-001",
    } == {rule.rule_id for rule in emitted_rules}
    assert {rule.card_id for rule in emitted_rules} <= known_cards
