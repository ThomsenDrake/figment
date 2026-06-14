from figment.focused_repair import (
    RepairScope,
    build_focused_repair_prompt,
    build_focused_repair_prompts,
    classify_validation_failures,
    mandatory_source_card_ids_for_scope,
    missing_mandatory_source_cards,
)


def test_classifies_validator_failures_into_focused_scopes_and_fields() -> None:
    failures = [
        "handoff_note_sbar is missing: assessment_observations_only, handoff_request",
        "missing_info_to_collect does not reference required observations for CHEST-PAIN-ESCALATION-v1",
        "source_cards not in allowed/retrieved card IDs: WOUND-INFECTION-ESCALATION-v1",
        "candidate pathway CHEST-PAIN-ESCALATION-v1 is not cited in source_cards",
        "forbidden clinical language: prescribe",
        "protocol_urgency routine is below deterministic floor emergency",
    ]

    scopes = classify_validation_failures(failures)

    assert [(scope.name, scope.fields) for scope in scopes] == [
        ("handoff_note_sbar", ("handoff_note_sbar",)),
        ("missing_observations", ("missing_info_to_collect", "next_observations_to_collect")),
        ("citations_and_pathways", ("source_cards", "candidate_protocol_pathways")),
        (
            "forbidden_clinical_language",
            (
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
            ),
        ),
        ("protocol_urgency", ("protocol_urgency",)),
    ]
    assert scopes[2].failures == (
        "source_cards not in allowed/retrieved card IDs: WOUND-INFECTION-ESCALATION-v1",
        "candidate pathway CHEST-PAIN-ESCALATION-v1 is not cited in source_cards",
    )


def test_schema_failures_extract_missing_and_type_invalid_fields() -> None:
    scopes = classify_validation_failures(
        [
            "missing required schema keys: red_flags, intake_facts, safety_boundary",
            "responder_plain_language_script must be a string",
            "next_observations_to_collect must be a list",
        ]
    )

    assert scopes == (
        RepairScope(
            name="schema",
            fields=(
                "red_flags",
                "intake_facts",
                "next_observations_to_collect",
                "responder_plain_language_script",
                "safety_boundary",
            ),
            failures=(
                "missing required schema keys: red_flags, intake_facts, safety_boundary",
                "responder_plain_language_script must be a string",
                "next_observations_to_collect must be a list",
            ),
        ),
    )


def test_build_focused_repair_prompt_limits_model_to_selected_fields() -> None:
    scope = RepairScope(
        name="missing_observations",
        fields=("missing_info_to_collect", "next_observations_to_collect"),
        failures=(
            "missing_info_to_collect does not reference required observations for CHEST-PAIN-ESCALATION-v1",
        ),
    )
    previous_output = {
        "protocol_urgency": "emergency",
        "source_cards": ["CHEST-PAIN-ESCALATION-v1"],
        "missing_info_to_collect": ["ask anything else that seems relevant"],
        "next_observations_to_collect": ["keep monitoring"],
    }

    prompt = build_focused_repair_prompt(
        original_prompt="BASE NAVIGATOR PROMPT",
        previous_output=previous_output,
        repair_scope=scope,
        urgency_floor="emergency",
        required_observation_targets=[
            {
                "id": "CHEST-PAIN-ESCALATION-v1::required_observation::1",
                "card_id": "CHEST-PAIN-ESCALATION-v1",
                "display_text": "chest pain description",
                "cue_tokens": ["chest", "pain", "description"],
            }
        ],
    )

    assert "BASE NAVIGATOR PROMPT" in prompt
    assert "Do not return the whole navigator output" in prompt
    assert "exactly these top-level keys: missing_info_to_collect, next_observations_to_collect" in prompt
    assert "protocol_urgency" not in prompt.split("PREVIOUS_VALUES_FOR_ALLOWED_FIELDS:", 1)[1]
    assert "required observations" in prompt
    assert "CHEST-PAIN-ESCALATION-v1" in prompt
    assert "CHEST-PAIN-ESCALATION-v1::required_observation::1" in prompt
    assert "chest pain description" in prompt
    assert "required_display_text_must_copy_exactly" in prompt
    assert "Copy every display_text" in prompt


def test_citation_repair_prompt_names_mandatory_source_cards() -> None:
    scope = classify_validation_failures(
        [
            "fired rule card STROKE-SIGNS-v1 is not cited in source_cards",
            "candidate pathway STROKE-SIGNS-v1 is not cited in source_cards",
        ]
    )[0]

    prompt = build_focused_repair_prompt(
        original_prompt="BASE NAVIGATOR PROMPT",
        previous_output={
            "source_cards": ["SAFETY-BOUNDARIES-v1"],
            "candidate_protocol_pathways": [
                {
                    "card_id": "SAFETY-BOUNDARIES-v1",
                    "reason_relevant": "Existing pathway.",
                }
            ],
        },
        repair_scope=scope,
        urgency_floor="emergency",
    )

    assert scope.name == "citations_and_pathways"
    assert scope.fields == ("source_cards", "candidate_protocol_pathways")
    assert mandatory_source_card_ids_for_scope(scope) == ("STROKE-SIGNS-v1",)
    assert "Mandatory source cards: STROKE-SIGNS-v1" in prompt
    assert "Do not remove any mandatory source card" in prompt
    assert "exactly these top-level keys: source_cards, candidate_protocol_pathways" in prompt


def test_citation_repair_rejects_output_that_omits_mandatory_source_card() -> None:
    scope = classify_validation_failures(
        ["fired rule card PREG-DANGER-SIGNS-v1 is not cited in source_cards"]
    )[0]

    assert missing_mandatory_source_cards(
        scope,
        {
            "source_cards": ["REFERRAL-SBAR-v1", "SAFETY-BOUNDARIES-v1"],
            "candidate_protocol_pathways": [
                {
                    "card_id": "REFERRAL-SBAR-v1",
                    "reason_relevant": "SBAR handoff.",
                }
            ],
        },
    ) == ("PREG-DANGER-SIGNS-v1",)
    assert missing_mandatory_source_cards(
        scope,
        {
            "source_cards": ["PREG-DANGER-SIGNS-v1", "REFERRAL-SBAR-v1"],
            "candidate_protocol_pathways": [
                {
                    "card_id": "PREG-DANGER-SIGNS-v1",
                    "reason_relevant": "Pregnancy danger sign fired deterministically.",
                }
            ],
        },
    ) == ()


def test_forbidden_language_prompt_keeps_safety_boundaries_explicit() -> None:
    scope = classify_validation_failures(["forbidden clinical language: discharge home"])[0]

    prompt = build_focused_repair_prompt(
        original_prompt="BASE NAVIGATOR PROMPT",
        previous_output={"responder_checklist": ["Discharge home if symptoms improve."]},
        repair_scope=scope,
        urgency_floor="urgent",
    )

    assert scope.name == "forbidden_clinical_language"
    assert "remove or rewrite unsafe clinical language" in prompt
    assert "diagnose, prescribe, dose, discharge, or override" in prompt
    assert "urgency_floor" in prompt
    assert "urgent" in prompt


def test_build_focused_repair_prompts_groups_failures_with_scope_metadata() -> None:
    prompts = build_focused_repair_prompts(
        original_prompt="BASE NAVIGATOR PROMPT",
        previous_output={"handoff_note_sbar": {"situation": "Chest pain"}},
        failures=[
            "handoff_note_sbar background is not grounded in confirmed intake or rules",
            "handoff_note_sbar assessment_observations_only has unsupported high-risk handoff facts: skull",
        ],
        urgency_floor="emergency",
    )

    assert len(prompts) == 1
    assert prompts[0].scope == RepairScope(
        name="handoff_note_sbar",
        fields=("handoff_note_sbar",),
        failures=(
            "handoff_note_sbar background is not grounded in confirmed intake or rules",
            "handoff_note_sbar assessment_observations_only has unsupported high-risk handoff facts: skull",
        ),
    )
    assert "ground handoff_note_sbar only in confirmed intake and deterministic rules" in prompts[0].prompt
