from __future__ import annotations

from typing import Any

import pytest

from figment.field_provenance import (
    DETERMINISTIC_FALLBACK,
    MODEL_RAW,
    MODEL_REPAIRED,
    NAVIGATOR_FIELD_NAMES,
    merge_field_provenance,
)


def _fallback_output(**overrides: Any) -> dict[str, Any]:
    output: dict[str, Any] = {
        "protocol_urgency": "urgent",
        "red_flags": [{"rule_id": "rule-1"}],
        "intake_facts": [
            {
                "fact": "fallback concern",
                "status": "reported",
                "source": "structured_field",
            }
        ],
        "candidate_protocol_pathways": [
            {
                "card_id": "SAFETY-BOUNDARIES-v1",
                "reason_relevant": "fallback",
            }
        ],
        "missing_info_to_collect": ["repeat vitals"],
        "next_observations_to_collect": ["work of breathing"],
        "conflicts_or_uncertainties": ["fallback uncertainty"],
        "responder_checklist": ["use local protocol"],
        "do_not_do": ["do not downgrade red flags"],
        "source_cards": ["SAFETY-BOUNDARIES-v1"],
        "handoff_note_sbar": {
            "situation": "fallback situation",
            "background": "fallback background",
            "assessment_observations_only": "fallback assessment",
            "handoff_request": "fallback request",
        },
        "responder_plain_language_script": "fallback script",
        "safety_boundary": "fallback safety boundary",
    }
    output.update(overrides)
    return output


def test_merge_keeps_raw_model_fields_and_fills_missing_fields_from_fallback() -> None:
    raw_output = {
        "protocol_urgency": "emergency",
        "missing_info_to_collect": ["airway check", "repeat blood pressure"],
    }
    fallback_output = _fallback_output()

    result = merge_field_provenance(raw_output, None, fallback_output)

    assert result.output["protocol_urgency"] == "emergency"
    assert result.output["missing_info_to_collect"] == ["airway check", "repeat blood pressure"]
    assert result.output["source_cards"] == ["SAFETY-BOUNDARIES-v1"]
    assert result.provenance["protocol_urgency"] == MODEL_RAW
    assert result.provenance["missing_info_to_collect"] == MODEL_RAW
    assert result.provenance["source_cards"] == DETERMINISTIC_FALLBACK
    assert set(result.output) == set(NAVIGATOR_FIELD_NAMES)
    assert set(result.provenance) == set(NAVIGATOR_FIELD_NAMES)


def test_merge_can_limit_raw_model_fields_to_field_level_acceptance() -> None:
    raw_output = {
        "protocol_urgency": "emergency",
        "source_cards": ["UNRETRIEVED-CARD-v1"],
    }

    result = merge_field_provenance(
        raw_output,
        None,
        _fallback_output(),
        accepted_raw_fields={"protocol_urgency"},
    )

    assert result.output["protocol_urgency"] == "emergency"
    assert result.output["source_cards"] == ["SAFETY-BOUNDARIES-v1"]
    assert result.provenance["protocol_urgency"] == MODEL_RAW
    assert result.provenance["source_cards"] == DETERMINISTIC_FALLBACK


def test_repaired_fields_override_raw_model_fields_and_are_labeled() -> None:
    raw_output = {
        "protocol_urgency": "urgent",
        "source_cards": ["UNRETRIEVED-CARD-v1"],
        "handoff_note_sbar": {
            "situation": "raw situation",
            "background": "raw background",
            "assessment_observations_only": "raw assessment",
            "handoff_request": "raw request",
        },
    }
    repaired_fields = {
        "source_cards": ["SAFETY-BOUNDARIES-v1"],
        "handoff_note_sbar": {
            "situation": "repaired situation",
            "background": "repaired background",
            "assessment_observations_only": "repaired assessment",
            "handoff_request": "repaired request",
        },
    }

    result = merge_field_provenance(raw_output, repaired_fields, _fallback_output())

    assert result.output["protocol_urgency"] == "urgent"
    assert result.output["source_cards"] == ["SAFETY-BOUNDARIES-v1"]
    assert result.output["handoff_note_sbar"]["situation"] == "repaired situation"
    assert result.provenance["protocol_urgency"] == MODEL_RAW
    assert result.provenance["source_cards"] == MODEL_REPAIRED
    assert result.provenance["handoff_note_sbar"] == MODEL_REPAIRED


def test_merge_ignores_unknown_model_fields_and_requires_complete_fallback() -> None:
    raw_output = {
        "protocol_urgency": "routine",
        "diagnosis": "unsupported extra field",
    }
    repaired_fields = {
        "made_up_section": {"unsafe": True},
    }

    result = merge_field_provenance(raw_output, repaired_fields, _fallback_output())

    assert "diagnosis" not in result.output
    assert "made_up_section" not in result.output
    assert "diagnosis" not in result.provenance
    assert "made_up_section" not in result.provenance

    incomplete_fallback = _fallback_output()
    incomplete_fallback.pop("handoff_note_sbar")
    with pytest.raises(
        ValueError,
        match="deterministic fallback is missing navigator fields: handoff_note_sbar",
    ):
        merge_field_provenance({}, None, incomplete_fallback)


def test_merge_returns_deep_copies_so_callers_cannot_mutate_sources_through_result() -> None:
    raw_output = {"missing_info_to_collect": ["repeat vitals"]}
    fallback_output = _fallback_output()

    result = merge_field_provenance(raw_output, None, fallback_output)
    result.output["missing_info_to_collect"].append("mutated")
    result.output["source_cards"].append("mutated-card")

    assert raw_output["missing_info_to_collect"] == ["repeat vitals"]
    assert fallback_output["source_cards"] == ["SAFETY-BOUNDARIES-v1"]
