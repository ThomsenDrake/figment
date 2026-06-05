from figment.rules import run_red_flag_checks


def _rule_ids(intake: dict[str, str | bool]) -> set[str]:
    return {rule.rule_id for rule in run_red_flag_checks(intake)}


def test_non_pregnant_wound_with_no_fever_does_not_create_red_flags() -> None:
    intake = {
        "confirmed": True,
        "setting": "clinic",
        "patient_age": "43",
        "pregnancy_status": "not_applicable",
        "chief_concern": "cut with bleeding",
        "symptoms": "bleeding controlled",
        "vitals": "no fever",
        "allergies": "",
        "medications": "",
        "available_supplies": "",
        "responder_note": "Adult not pregnant with bleeding from cut.",
    }

    assert _rule_ids(intake) == set()


def test_negative_fever_context_does_not_hide_later_measured_fever_red_flag() -> None:
    intake = {
        "confirmed": True,
        "setting": "mobile clinic",
        "patient_age": "43",
        "pregnancy_status": "not_applicable",
        "chief_concern": "fever concern",
        "symptoms": "stiff neck",
        "vitals": "No fever reported yesterday. Temperature now 103 F.",
        "allergies": "",
        "medications": "",
        "available_supplies": "",
        "responder_note": "",
    }

    assert "FEVER-001" in _rule_ids(intake)


def test_pregnancy_danger_sign_still_fires_for_confirmed_pregnancy() -> None:
    intake = {
        "confirmed": True,
        "setting": "rural clinic",
        "patient_age": "31",
        "pregnancy_status": "pregnant, about 32 weeks",
        "chief_concern": "severe headache and vaginal bleeding",
        "symptoms": "vision changes",
        "vitals": "BP elevated",
        "allergies": "",
        "medications": "",
        "available_supplies": "",
        "responder_note": "",
    }

    assert "PREG-001" in _rule_ids(intake)


def test_month_based_pediatric_age_counts_as_pediatric() -> None:
    intake = {
        "confirmed": True,
        "setting": "shelter clinic",
        "patient_age": "18 months",
        "pregnancy_status": "not_applicable",
        "chief_concern": "possible dehydration",
        "symptoms": "no urine and very dry mouth",
        "vitals": "",
        "allergies": "",
        "medications": "",
        "available_supplies": "",
        "responder_note": "",
    }

    assert "PED-DEHYD-001" in _rule_ids(intake)
