"""Protocol navigator orchestration."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from .config import FigmentConfig, load_config
from .field_provenance import (
    DETERMINISTIC_FALLBACK,
    accepted_raw_fields_from_failures,
    deterministic_field_provenance,
    has_deterministic_patches,
    merge_field_provenance,
    model_raw_field_provenance,
)
from .focused_repair import build_focused_repair_prompts
from .model_client import ModelClient, ModelClientError, canned_navigator_output
from .observation_targets import apply_navigation_scaffolding, required_observation_targets
from .prompt_builder import build_prompt
from .retrieval import known_card_ids, query_from_intake, search_protocol_cards
from .trace import FigmentTrace, scrub_audio_metadata, stable_hash, write_trace
from .validators import urgency_floor_from_rules, validate_audio_ready, validate_confirmed_intake, validate_navigator_output


MAX_FOCUSED_REPAIR_ATTEMPTS = 2


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
    field_provenance: dict[str, str] = {}
    scaffold_patched_fields: set[str] = set()
    repair_metrics: dict[str, Any] = _empty_repair_metrics()
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
        field_provenance = deterministic_field_provenance()
        events.append("model backend failed; deterministic fallback applied")

    card_ids = known_card_ids()
    if not card_ids:
        card_ids = {
            str(item.get("card_id") or item.get("card", {}).get("card_id"))
            for item in retrieved
            if item.get("card_id") or item.get("card", {}).get("card_id")
        }
        card_ids.update(str(card_id) for card_id in output.get("source_cards", []))
    scaffold_result = apply_navigation_scaffolding(
        output,
        retrieved_cards=retrieved,
        rule_results=rule_results,
        urgency_floor=floor,
    )
    output = scaffold_result.output
    scaffold_patched_fields = scaffold_result.patched_fields
    if scaffold_result.filled_required_observation_ids:
        events.append("required-observation targets filled deterministically")
    validation = _validate_output(output, card_ids, floor, intake, rule_results, retrieved)
    if validation.passed and not field_provenance:
        field_provenance = (
            deterministic_field_provenance() if config.model_backend == "canned" else model_raw_field_provenance()
        )
        _mark_deterministic_patch_fields(field_provenance, scaffold_patched_fields)
    if not validation.passed and fallback_reason is None and config.model_backend != "canned":
        field_result = _try_field_level_model_output(
            client=client,
            prompt=prompt,
            raw_output=output,
            validation_failures=validation.failures,
            floor=floor,
            intake=intake,
            rule_results=rule_results,
            retrieved=retrieved,
            known_cards=card_ids,
            scaffold_patched_fields=scaffold_patched_fields,
            events=events,
            repair_metrics=repair_metrics,
        )
        if field_result is not None:
            output, validation, field_provenance = field_result
    if not validation.passed:
        output = canned_navigator_output(intake, rule_results, retrieved, floor)
        fallback_scaffold = apply_navigation_scaffolding(
            output,
            retrieved_cards=retrieved,
            rule_results=rule_results,
            urgency_floor=floor,
        )
        output = fallback_scaffold.output
        scaffold_patched_fields.update(fallback_scaffold.patched_fields)
        validation = _validate_output(output, card_ids, floor, intake, rule_results, retrieved)
        fallback_reason = fallback_reason or "navigator_validation_failure"
        field_provenance = deterministic_field_provenance()
        events.append("navigator output failed validation; deterministic fallback applied")
    events.append("validation complete")
    field_level_fallback_used = has_deterministic_patches(field_provenance)
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
            "field_level_fallback_used": field_level_fallback_used,
            "strict_validation": True,
            "deterministic_scaffold_patched_fields": sorted(scaffold_patched_fields),
            **repair_metrics,
        },
        navigator_output=output,
        validator_result=validation.to_dict(),
        field_provenance=field_provenance,
        raw_audio_stored=False,
        events=events,
    )
    if trace_path:
        write_trace(trace, trace_path)
    return output, trace


def _validate_output(
    output: dict[str, Any],
    card_ids: set[str],
    floor: str,
    intake: dict[str, Any],
    rule_results: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
):
    retrieved_ids = {
        str(item.get("card_id") or item.get("card", {}).get("card_id"))
        for item in retrieved
        if item.get("card_id") or item.get("card", {}).get("card_id")
    }
    return validate_navigator_output(
        output,
        card_ids,
        floor,
        confirmed_intake=intake,
        rule_results=rule_results,
        retrieved_card_ids=retrieved_ids,
        retrieved_cards=retrieved,
        strict_schema=True,
    )


def _try_field_level_model_output(
    *,
    client: ModelClient,
    prompt: str,
    raw_output: dict[str, Any],
    validation_failures: list[str],
    floor: str,
    intake: dict[str, Any],
    rule_results: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
    known_cards: set[str],
    scaffold_patched_fields: set[str],
    events: list[str],
    repair_metrics: dict[str, Any],
) -> tuple[dict[str, Any], Any, dict[str, str]] | None:
    fallback_output = canned_navigator_output(intake, rule_results, retrieved, floor)
    fallback_scaffold = apply_navigation_scaffolding(
        fallback_output,
        retrieved_cards=retrieved,
        rule_results=rule_results,
        urgency_floor=floor,
    )
    fallback_output = fallback_scaffold.output
    accepted_raw_fields = accepted_raw_fields_from_failures(validation_failures)
    repair_context = {
        "intake": intake,
        "rule_results": rule_results,
        "retrieved_cards": retrieved,
        "urgency_floor": floor,
        "previous_output": raw_output,
        "validation_failures": validation_failures,
    }
    repaired_fields: dict[str, Any] = {}
    focused_prompts = build_focused_repair_prompts(
        original_prompt=prompt,
        previous_output=raw_output,
        failures=validation_failures,
        urgency_floor=floor,
        required_observation_targets=required_observation_targets(retrieved),
    )
    repair_metrics["repair_scope_count"] = len(focused_prompts)
    repair_metrics["repair_scopes"] = [focused_prompt.scope.name for focused_prompt in focused_prompts]
    repair_metrics["repair_capped"] = len(focused_prompts) > MAX_FOCUSED_REPAIR_ATTEMPTS
    for focused_prompt in focused_prompts[:MAX_FOCUSED_REPAIR_ATTEMPTS]:
        repair_metrics["repair_attempt_count"] += 1
        started = perf_counter()
        try:
            repair_output = client.generate_json(focused_prompt.prompt, repair_context | {"repair_scope": focused_prompt.scope.name})
        except ModelClientError:
            repair_metrics["repair_latency_ms"] = round(
                repair_metrics["repair_latency_ms"] + ((perf_counter() - started) * 1000),
                3,
            )
            events.append(f"navigator focused repair failed for {focused_prompt.scope.name}")
            continue
        repair_metrics["repair_latency_ms"] = round(
            repair_metrics["repair_latency_ms"] + ((perf_counter() - started) * 1000),
            3,
        )
        if not isinstance(repair_output, dict):
            events.append(f"navigator focused repair for {focused_prompt.scope.name} returned non-object output")
            continue
        for field in focused_prompt.scope.fields:
            if field in repair_output:
                repaired_fields[field] = repair_output[field]

    merge_candidates = []
    if repaired_fields:
        merge_candidates.append((repaired_fields, "navigator output repaired by hosted retry"))
    merge_candidates.append(({}, "navigator output retained with field-level deterministic patches"))
    for candidate_repaired_fields, event_text in merge_candidates:
        merge_result = merge_field_provenance(
            raw_output,
            candidate_repaired_fields,
            fallback_output,
            accepted_raw_fields=accepted_raw_fields,
        )
        merged_validation = _validate_output(merge_result.output, known_cards, floor, intake, rule_results, retrieved)
        if merged_validation.passed:
            if merge_result.provenance == deterministic_field_provenance():
                continue
            _mark_deterministic_patch_fields(
                merge_result.provenance,
                scaffold_patched_fields | fallback_scaffold.patched_fields,
            )
            events.append(event_text)
            return merge_result.output, merged_validation, merge_result.provenance
    events.append("navigator retry failed validation")
    return None


def _empty_repair_metrics() -> dict[str, Any]:
    return {
        "repair_attempt_count": 0,
        "repair_attempt_cap": MAX_FOCUSED_REPAIR_ATTEMPTS,
        "repair_scope_count": 0,
        "repair_capped": False,
        "repair_latency_ms": 0.0,
        "repair_scopes": [],
    }


def _mark_deterministic_patch_fields(provenance: dict[str, str], fields: set[str]) -> None:
    for field in fields:
        if field in provenance:
            provenance[field] = DETERMINISTIC_FALLBACK


def _repair_prompt(
    original_prompt: str,
    previous_output: dict[str, Any],
    failures: list[str],
    urgency_floor: str,
) -> str:
    repair_context = {
        "deterministic_validation_failures": failures,
        "urgency_floor": urgency_floor,
        "previous_output": previous_output,
    }
    return (
        f"{original_prompt}\n\n"
        "Your previous JSON failed deterministic validation. Return corrected JSON only.\n"
        "Keep protocol_urgency at or above the urgency_floor, cite only retrieved source_cards, "
        "ground SBAR fields in confirmed intake/rules, and avoid diagnosis, prescription, dosing, "
        "or autonomous routing language.\n\n"
        f"REPAIR_CONTEXT:\n{json.dumps(repair_context, indent=2, sort_keys=True)}"
    )


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
