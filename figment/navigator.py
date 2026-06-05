"""Protocol navigator orchestration."""

from __future__ import annotations

from typing import Any

from .config import FigmentConfig, load_config
from .model_client import ModelClient, ModelClientError, canned_navigator_output
from .prompt_builder import build_prompt
from .retrieval import known_card_ids, query_from_intake, search_protocol_cards
from .trace import FigmentTrace, scrub_audio_metadata, stable_hash, write_trace
from .validators import urgency_floor_from_rules, validate_audio_ready, validate_confirmed_intake, validate_navigator_output


def run_navigation(
    intake: dict[str, Any],
    rule_results: list[dict[str, Any]],
    *,
    audio_draft: dict[str, Any] | None = None,
    config: FigmentConfig | None = None,
    retrieved_cards: list[dict[str, Any]] | None = None,
    trace_path: str | None = None,
) -> tuple[dict[str, Any], FigmentTrace]:
    config = (config or load_config()).validated()
    intake_validation = validate_confirmed_intake(intake)
    if not intake_validation.passed:
        raise ValueError("; ".join(intake_validation.failures))
    audio_validation = validate_audio_ready(audio_draft)
    if not audio_validation.passed:
        raise ValueError("; ".join(audio_validation.failures))

    trace_audio = scrub_audio_metadata(audio_draft)
    prompt_audio = _audio_context_for_prompt(trace_audio)
    query = query_from_intake(intake)
    retrieved = retrieved_cards if retrieved_cards is not None else search_protocol_cards(query, limit=6)
    floor = urgency_floor_from_rules(rule_results)
    prompt, prompt_hash = build_prompt(intake, retrieved, rule_results, floor, audio_draft=prompt_audio)
    client = ModelClient(config)
    events = ["input captured", "rules evaluated", "cards retrieved", "navigator output generated"]
    fallback_reason: str | None = None
    try:
        output = client.generate_json(
            prompt,
            {
                "intake": intake,
                "rule_results": rule_results,
                "retrieved_cards": retrieved,
                "urgency_floor": floor,
            },
        )
    except ModelClientError:
        output = canned_navigator_output(intake, rule_results, retrieved, floor)
        fallback_reason = "model_backend_error"
        events.append("model backend failed; deterministic fallback applied")

    card_ids = known_card_ids()
    if not card_ids:
        card_ids = {
            str(item.get("card_id") or item.get("card", {}).get("card_id"))
            for item in retrieved
            if item.get("card_id") or item.get("card", {}).get("card_id")
        }
        card_ids.update(str(card_id) for card_id in output.get("source_cards", []))
    validation = validate_navigator_output(output, card_ids, floor, confirmed_intake=intake, rule_results=rule_results)
    if not validation.passed:
        output = canned_navigator_output(intake, rule_results, retrieved, floor)
        validation = validate_navigator_output(output, card_ids, floor, confirmed_intake=intake, rule_results=rule_results)
        fallback_reason = fallback_reason or "navigator_validation_failure"
        events.append("navigator output failed validation; deterministic fallback applied")
    events.append("validation complete")
    trace = FigmentTrace(
        input_captured={
            "structured_intake": intake,
            "confirmed_intake_hash": stable_hash(intake),
        },
        audio=trace_audio,
        red_flags=rule_results,
        retrieved_card_ids=[item["card_id"] for item in retrieved],
        prompt_template_hash=prompt_hash,
        model_route={
            "model_stack": config.model_stack,
            "model_backend": config.model_backend,
            "model_id": config.active_model_id,
            "fallback_tier": "canned" if config.model_backend == "canned" or fallback_reason else "configured",
            "fallback_reason": fallback_reason,
        },
        navigator_output=output,
        validator_result=validation.to_dict(),
        raw_audio_stored=False,
        events=events,
    )
    if trace_path:
        write_trace(trace, trace_path)
    return output, trace


def _audio_context_for_prompt(audio: dict[str, Any] | None) -> dict[str, Any] | None:
    if not audio:
        return None
    confirmed_fields = []
    for suggestion in audio.get("suggested_fields", []):
        if suggestion.get("status") not in {"accepted", "edited"}:
            continue
        confirmed_fields.append(
            {
                "field": suggestion.get("field"),
                "status": suggestion.get("status"),
            }
        )
    return {
        "audio_intake_path": audio.get("audio_intake_path"),
        "audio_runtime": audio.get("audio_runtime"),
        "confirmation_status": audio.get("confirmation_status"),
        "confirmed_intake_required": audio.get("confirmed_intake_required"),
        "accepted_or_edited_fields": confirmed_fields,
        "raw_audio_stored": False,
    }
