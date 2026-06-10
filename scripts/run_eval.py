"""Deterministic Figment eval runner."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figment.config import FigmentConfig  # noqa: E402
from figment.eval_metrics import score_expected_labels, summarize_eval_records  # noqa: E402
from figment.field_provenance import (  # noqa: E402
    DETERMINISTIC_FALLBACK,
    accepted_raw_fields_from_failures,
    deterministic_field_provenance,
    has_deterministic_patches,
    merge_field_provenance,
    model_raw_field_provenance,
)  # noqa: E402
from figment.focused_repair import build_focused_repair_prompts  # noqa: E402
from figment.model_client import ModelClient, ModelClientError, canned_navigator_output  # noqa: E402
from figment.observation_targets import (  # noqa: E402
    NavigationScaffoldResult,
    apply_navigation_scaffolding,
    required_observation_targets,
)
from figment.prompt_builder import build_prompt  # noqa: E402
from figment.retrieval import known_card_ids, query_from_intake, search_protocol_cards  # noqa: E402
from figment.rules import run_red_flag_checks  # noqa: E402
from figment.trace import stable_hash  # noqa: E402
from figment.validators import urgency_floor_from_rules, validate_navigator_output  # noqa: E402


DEFAULT_CASE_GLOB = "data/eval/*.jsonl"
REAL_LLAMA_CPP_EVAL_COMMAND = (
    "FIGMENT_MODE=local MODEL_STACK=local_4b_parakeet MODEL_BACKEND=llama_cpp "
    "LOCAL_MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 "
    "LLAMA_BASE_URL=http://127.0.0.1:8001/v1 PYTHON_DOTENV_DISABLED=true "
    "python3 scripts/run_eval.py --backend llama_cpp --model-stack local_4b_parakeet "
    "--cases data/eval/initial_handwritten_cases.jsonl "
    "--cases data/eval/adversarial_strict_cases.jsonl "
    "--cases data/eval/comprehensive_hosted_cases.jsonl "
    "--output traces/local_llama_cpp_eval_$(date -u +%Y%m%dT%H%M%SZ).jsonl"
)


def load_cases(case_paths: list[Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in case_paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            case["_case_path"] = str(path)
            case["_case_line"] = line_number
            cases.append(case)
    return cases


def run_eval(
    *,
    case_paths: list[Path],
    output_path: Path | None,
    config: FigmentConfig,
    limit: int | None = None,
) -> dict[str, Any]:
    cases = load_cases(case_paths)
    if limit is not None:
        cases = cases[: max(0, limit)]
    records = [_evaluate_case(case, config) for case in cases]
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records),
            encoding="utf-8",
        )
    else:
        for record in records:
            sys.stdout.write(f"{json.dumps(record, sort_keys=True)}\n")
    return _summarize(records, config, case_paths, output_path)


def _evaluate_case(case: dict[str, Any], config: FigmentConfig) -> dict[str, Any]:
    started = perf_counter()
    intake = case["structured_intake"]
    rule_results = [rule.to_dict() for rule in run_red_flag_checks(intake)]
    floor = urgency_floor_from_rules(rule_results)
    query = query_from_intake(intake)
    retrieved = search_protocol_cards(query, limit=6)
    retrieved_ids = [str(item.get("card_id", "")) for item in retrieved if item.get("card_id")]
    prompt, prompt_hash = build_prompt(intake, retrieved, rule_results, floor)
    known_cards = known_card_ids()

    raw_output: dict[str, Any] | None = None
    repaired_output: dict[str, Any] | None = None
    fallback_output: dict[str, Any] | None = None
    raw_validation = {"passed": False, "failures": ["configured model not attempted for canned backend"]}
    repair_validation = {"passed": False, "failures": ["repair not attempted"]}
    fallback_validation = {"passed": False, "failures": ["fallback not used"]}
    raw_attempted = config.model_backend != "canned"
    repair_attempted = False
    fallback_used = False
    fallback_reason: str | None = None
    final_output: dict[str, Any]
    final_validation: dict[str, Any]
    field_provenance: dict[str, str] = {}
    scaffold_patched_fields: set[str] = set()
    filled_required_observation_ids: list[str] = []
    model_selected_required_observation_ids: list[str] = []
    invalid_selected_required_observation_ids: list[str] = []
    stripped_trace_only_fields: list[str] = []

    context = {
        "intake": intake,
        "rule_results": rule_results,
        "retrieved_cards": retrieved,
        "urgency_floor": floor,
    }

    if config.model_backend == "canned":
        fallback_reason = "canned_backend"
        fallback_used = True
        fallback_output, fallback_validation, fallback_scaffold = _run_fallback(
            intake,
            rule_results,
            retrieved,
            floor,
            known_cards,
            retrieved_ids,
        )
        _absorb_scaffold_trace(
            fallback_scaffold,
            scaffold_patched_fields=scaffold_patched_fields,
            filled_required_observation_ids=filled_required_observation_ids,
            model_selected_required_observation_ids=model_selected_required_observation_ids,
            invalid_selected_required_observation_ids=invalid_selected_required_observation_ids,
            stripped_trace_only_fields=stripped_trace_only_fields,
        )
        final_output = fallback_output
        final_validation = fallback_validation
        field_provenance = deterministic_field_provenance()
    else:
        client = ModelClient(config)
        try:
            raw_output = client.generate_json(prompt, context)
            scaffold_result = apply_navigation_scaffolding(
                raw_output,
                retrieved_cards=retrieved,
                rule_results=rule_results,
                urgency_floor=floor,
            )
            raw_output = scaffold_result.output
            _absorb_scaffold_trace(
                scaffold_result,
                scaffold_patched_fields=scaffold_patched_fields,
                filled_required_observation_ids=filled_required_observation_ids,
                model_selected_required_observation_ids=model_selected_required_observation_ids,
                invalid_selected_required_observation_ids=invalid_selected_required_observation_ids,
                stripped_trace_only_fields=stripped_trace_only_fields,
            )
            raw_validation = _validate_output(raw_output, known_cards, floor, intake, rule_results, retrieved, retrieved_ids)
        except ModelClientError as exc:
            raw_validation = {"passed": False, "failures": [f"model backend error: {exc}"]}
            fallback_reason = "model_backend_error"

        if raw_output is not None and raw_validation["passed"]:
            final_output = raw_output
            final_validation = raw_validation
            field_provenance = model_raw_field_provenance()
            _mark_deterministic_patch_fields(field_provenance, scaffold_patched_fields)
        else:
            if raw_output is not None:
                fallback_output, fallback_validation, fallback_scaffold = _run_fallback(
                    intake,
                    rule_results,
                    retrieved,
                    floor,
                    known_cards,
                    retrieved_ids,
                )
                (
                    repaired_output,
                    repair_validation,
                    repair_attempted,
                    merged_output,
                    merged_validation,
                    merged_field_provenance,
                ) = _try_field_level_model_output(
                    client=client,
                    prompt=prompt,
                    context=context,
                    raw_output=raw_output,
                    validation_failures=raw_validation["failures"],
                    fallback_output=fallback_output,
                    known_cards=known_cards,
                    floor=floor,
                    intake=intake,
                    rule_results=rule_results,
                    retrieved=retrieved,
                    retrieved_ids=retrieved_ids,
                    scaffold_patched_fields=scaffold_patched_fields,
                )
                if merged_output is not None and merged_validation is not None:
                    final_output = merged_output
                    final_validation = merged_validation
                    field_provenance = merged_field_provenance
                    if (
                        field_provenance.get("missing_info_to_collect") == DETERMINISTIC_FALLBACK
                        or field_provenance.get("next_observations_to_collect") == DETERMINISTIC_FALLBACK
                    ):
                        _absorb_scaffold_trace(
                            fallback_scaffold,
                            scaffold_patched_fields=scaffold_patched_fields,
                            filled_required_observation_ids=filled_required_observation_ids,
                            model_selected_required_observation_ids=model_selected_required_observation_ids,
                            invalid_selected_required_observation_ids=invalid_selected_required_observation_ids,
                            stripped_trace_only_fields=stripped_trace_only_fields,
                        )
                else:
                    fallback_reason = fallback_reason or "navigator_validation_failure"
                    fallback_used = True
                    final_output = fallback_output
                    final_validation = fallback_validation
                    field_provenance = deterministic_field_provenance()
                    _absorb_scaffold_trace(
                        fallback_scaffold,
                        scaffold_patched_fields=scaffold_patched_fields,
                        filled_required_observation_ids=filled_required_observation_ids,
                        model_selected_required_observation_ids=model_selected_required_observation_ids,
                        invalid_selected_required_observation_ids=invalid_selected_required_observation_ids,
                        stripped_trace_only_fields=stripped_trace_only_fields,
                    )
            else:
                fallback_used = True
                fallback_output, fallback_validation, fallback_scaffold = _run_fallback(
                    intake,
                    rule_results,
                    retrieved,
                    floor,
                    known_cards,
                    retrieved_ids,
                )
                _absorb_scaffold_trace(
                    fallback_scaffold,
                    scaffold_patched_fields=scaffold_patched_fields,
                    filled_required_observation_ids=filled_required_observation_ids,
                    model_selected_required_observation_ids=model_selected_required_observation_ids,
                    invalid_selected_required_observation_ids=invalid_selected_required_observation_ids,
                    stripped_trace_only_fields=stripped_trace_only_fields,
                )
                final_output = fallback_output
                final_validation = fallback_validation
                field_provenance = deterministic_field_provenance()

    field_level_fallback_used = has_deterministic_patches(field_provenance)

    raw_success = raw_attempted and raw_validation["passed"] and not scaffold_patched_fields
    repair_success = repair_attempted and repair_validation["passed"]
    fallback_success = fallback_used and fallback_validation["passed"]
    fallback_tier = "canned" if fallback_used else "configured"
    competence_success = bool(raw_success or repair_success)
    model_route = {
        "model_stack": config.model_stack,
        "model_backend": config.model_backend,
        "model_id": config.active_model_id,
        "fallback_tier": fallback_tier,
        "fallback_reason": fallback_reason,
        "field_level_fallback_used": field_level_fallback_used,
        "deterministic_scaffold_patched_fields": sorted(scaffold_patched_fields),
        "filled_required_observation_ids": filled_required_observation_ids,
        "model_selected_required_observation_ids": model_selected_required_observation_ids,
        "invalid_selected_required_observation_ids": invalid_selected_required_observation_ids,
        "stripped_trace_only_fields": stripped_trace_only_fields,
    }
    trace_payload = {
        "case_id": case["case_id"],
        "input_hash": stable_hash(intake),
        "red_flags": rule_results,
        "retrieved_card_ids": retrieved_ids,
        "prompt_template_hash": prompt_hash,
        "model_route": model_route,
        "navigator_output": final_output,
        "validator_result": final_validation,
        "field_provenance": field_provenance,
    }

    actual_source_card_ids = [
        str(card_id) for card_id in final_output.get("source_cards", []) if str(card_id)
    ]
    actual_candidate_pathway_card_ids = _candidate_pathway_card_ids(
        final_output.get("candidate_protocol_pathways")
    )
    record = {
        "case_id": case["case_id"],
        "case_path": case.get("_case_path"),
        "case_line": case.get("_case_line"),
        "target_protocol_card_id": case.get("target_protocol_card_id"),
        "expected_min_protocol_urgency": case.get("expected_min_protocol_urgency"),
        "expected_red_flag_rule_ids": case.get("expected_red_flag_rule_ids", []),
        "expected_source_card_ids": case.get("expected_source_card_ids", []),
        "expected_missing_observations": case.get("expected_missing_observations", []),
        "forbidden_behavior": case.get("forbidden_behavior", []),
        "actual_red_flag_rule_ids": [rule["rule_id"] for rule in rule_results],
        "actual_protocol_urgency": final_output.get("protocol_urgency"),
        "actual_source_card_ids": actual_source_card_ids,
        "actual_candidate_pathway_card_ids": actual_candidate_pathway_card_ids,
        "retrieved_card_ids": retrieved_ids,
        "model_backend": config.model_backend,
        "model_stack": config.model_stack,
        "active_model_id": config.active_model_id,
        "fallback_tier": fallback_tier,
        "fallback_reason": fallback_reason,
        "field_level_fallback_used": field_level_fallback_used,
        "deterministic_scaffold_patched_fields": sorted(scaffold_patched_fields),
        "filled_required_observation_ids": filled_required_observation_ids,
        "model_selected_required_observation_ids": model_selected_required_observation_ids,
        "invalid_selected_required_observation_ids": invalid_selected_required_observation_ids,
        "stripped_trace_only_fields": stripped_trace_only_fields,
        "raw_configured_model_attempted": raw_attempted,
        "raw_configured_model_success": raw_success,
        "repair_attempted": repair_attempted,
        "repair_success": repair_success,
        "canned_fallback_used": fallback_used,
        "canned_fallback_success": fallback_success,
        "competence_success": competence_success,
        "raw_validation": raw_validation,
        "repair_validation": repair_validation,
        "fallback_validation": fallback_validation,
        "validation_result": final_validation,
        "final_validation": final_validation,
        "raw_model_output": raw_output,
        "repaired_output": repaired_output,
        "fallback_output": fallback_output,
        "final_output": final_output,
        "field_provenance": field_provenance,
        "latency_ms": round((perf_counter() - started) * 1000, 3),
        "trace_hash": stable_hash(trace_payload),
    }
    record["expected_label_score"] = score_expected_labels(record)
    return record


def _run_fallback(
    intake: dict[str, Any],
    rule_results: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
    floor: str,
    known_cards: set[str],
    retrieved_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any], NavigationScaffoldResult]:
    output = canned_navigator_output(intake, rule_results, retrieved, floor)
    scaffold = apply_navigation_scaffolding(
        output,
        retrieved_cards=retrieved,
        rule_results=rule_results,
        urgency_floor=floor,
    )
    output = scaffold.output
    validation = _validate_output(output, known_cards, floor, intake, rule_results, retrieved, retrieved_ids)
    return output, validation, scaffold


def _absorb_scaffold_trace(
    result: NavigationScaffoldResult,
    *,
    scaffold_patched_fields: set[str],
    filled_required_observation_ids: list[str],
    model_selected_required_observation_ids: list[str],
    invalid_selected_required_observation_ids: list[str],
    stripped_trace_only_fields: list[str],
) -> None:
    scaffold_patched_fields.update(result.patched_fields)
    _extend_unique(filled_required_observation_ids, result.filled_required_observation_ids)
    _extend_unique(model_selected_required_observation_ids, result.model_selected_required_observation_ids)
    _extend_unique(invalid_selected_required_observation_ids, result.invalid_selected_required_observation_ids)
    _extend_unique(stripped_trace_only_fields, result.stripped_trace_only_fields)


def _extend_unique(items: list[str], values: list[str]) -> None:
    for value in values:
        if value not in items:
            items.append(value)


def _validate_output(
    output: dict[str, Any],
    known_cards: set[str],
    floor: str,
    intake: dict[str, Any],
    rule_results: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
    retrieved_ids: list[str],
) -> dict[str, Any]:
    return validate_navigator_output(
        output,
        known_cards,
        floor,
        confirmed_intake=intake,
        rule_results=rule_results,
        retrieved_card_ids=set(retrieved_ids),
        retrieved_cards=retrieved,
        strict_schema=True,
    ).to_dict()


def _try_field_level_model_output(
    *,
    client: ModelClient,
    prompt: str,
    context: dict[str, Any],
    raw_output: dict[str, Any],
    validation_failures: list[str],
    fallback_output: dict[str, Any],
    known_cards: set[str],
    floor: str,
    intake: dict[str, Any],
    rule_results: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
    retrieved_ids: list[str],
    scaffold_patched_fields: set[str],
) -> tuple[dict[str, Any] | None, dict[str, Any], bool, dict[str, Any] | None, dict[str, Any] | None, dict[str, str]]:
    accepted_raw_fields = accepted_raw_fields_from_failures(validation_failures)
    repaired_fields: dict[str, Any] = {}
    repair_attempted = False
    repair_validation = {"passed": False, "failures": ["repair not attempted"]}
    for focused_prompt in build_focused_repair_prompts(
        original_prompt=prompt,
        previous_output=raw_output,
        failures=validation_failures,
        urgency_floor=floor,
        required_observation_targets=required_observation_targets(retrieved),
    ):
        repair_attempted = True
        try:
            repair_output = client.generate_json(
                focused_prompt.prompt,
                {
                    **context,
                    "previous_output": raw_output,
                    "validation_failures": validation_failures,
                    "repair_scope": focused_prompt.scope.name,
                },
            )
        except ModelClientError as exc:
            repair_validation = {"passed": False, "failures": [f"repair backend error: {exc}"]}
            continue
        if not isinstance(repair_output, dict):
            repair_validation = {"passed": False, "failures": ["repair output was not an object"]}
            continue
        for field in focused_prompt.scope.fields:
            if field in repair_output:
                repaired_fields[field] = repair_output[field]

    merge_candidates = []
    if repaired_fields:
        merge_candidates.append(repaired_fields)
    merge_candidates.append({})
    for candidate_repaired_fields in merge_candidates:
        merge_result = merge_field_provenance(
            raw_output,
            candidate_repaired_fields,
            fallback_output,
            accepted_raw_fields=accepted_raw_fields,
        )
        merged_validation = _validate_output(
            merge_result.output,
            known_cards,
            floor,
            intake,
            rule_results,
            retrieved,
            retrieved_ids,
        )
        if merged_validation["passed"]:
            if merge_result.provenance == deterministic_field_provenance():
                continue
            _mark_deterministic_patch_fields(merge_result.provenance, scaffold_patched_fields)
            if candidate_repaired_fields:
                repair_validation = merged_validation
            return (
                candidate_repaired_fields or None,
                repair_validation,
                repair_attempted,
                merge_result.output,
                merged_validation,
                merge_result.provenance,
            )
        if candidate_repaired_fields:
            repair_validation = merged_validation
    return None, repair_validation, repair_attempted, None, None, {}


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
        "cite every fired rule card, ground SBAR fields in confirmed intake/rules, and avoid diagnosis, "
        "prescription, dosing, autonomous routing, or treatment language.\n\n"
        f"REPAIR_CONTEXT:\n{json.dumps(repair_context, indent=2, sort_keys=True)}"
    )


def _candidate_pathway_card_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    card_ids: list[str] = []
    for item in value:
        if isinstance(item, dict):
            card_id = item.get("card_id")
        else:
            card_id = item
        if card_id:
            card_ids.append(str(card_id))
    return card_ids


def _summarize(
    records: list[dict[str, Any]],
    config: FigmentConfig,
    case_paths: list[Path],
    output_path: Path | None,
) -> dict[str, Any]:
    summary = summarize_eval_records(records)
    summary.update(
        {
            "model_backend": config.model_backend,
            "model_stack": config.model_stack,
            "active_model_id": config.active_model_id,
            "case_paths": [str(path) for path in case_paths],
            "output_path": str(output_path) if output_path else None,
        }
    )
    if config.model_backend == "llama_cpp":
        summary["local_llm_evidence"] = _local_llm_evidence_summary(summary, config)
    return summary


def _local_llm_evidence_summary(summary: dict[str, Any], config: FigmentConfig) -> dict[str, Any]:
    total_cases = int(summary.get("total_cases", 0))
    competence_successes = int(summary.get("competence_successes", 0))
    return {
        "proof_status": "eval_records_summarized",
        "model_backend": config.model_backend,
        "model_stack": config.model_stack,
        "model_id": config.active_model_id,
        "llama_base_url": config.llama_base_url,
        "total_cases": total_cases,
        "competence_successes": competence_successes,
        "raw_configured_model_successes": summary.get("raw_configured_model_successes", 0),
        "repair_successes": summary.get("repair_successes", 0),
        "fallback_uses": summary.get("fallback_uses", 0),
        "final_validation_successes": summary.get("final_validation_successes", 0),
        "counts_as_50_case_local_llm_eval": total_cases >= 50,
        "counts_as_50_case_local_llm_competence": total_cases >= 50 and competence_successes > 0,
        "no_cloud_note": (
            "MODEL_BACKEND=llama_cpp calls the configured local OpenAI-compatible LLAMA_BASE_URL. "
            "Record server /v1/models metadata and network isolation evidence beside the trace."
        ),
        "real_eval_command": REAL_LLAMA_CPP_EVAL_COMMAND,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["canned", "hosted_omni", "llama_cpp"], default="canned")
    parser.add_argument("--model-stack", choices=["omni_native", "local_4b_parakeet"], default=None)
    parser.add_argument("--cases", action="append", default=None, help="JSONL eval case path. Repeatable.")
    parser.add_argument("--output", default="-", help="JSONL result path, or '-' for stdout.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    case_paths = [Path(path) for path in args.cases] if args.cases else sorted(Path().glob(DEFAULT_CASE_GLOB))
    if not case_paths:
        raise SystemExit(f"no eval case files matched {DEFAULT_CASE_GLOB}")
    output_path = None if args.output == "-" else Path(args.output)
    config = _config_for_backend(args.backend, args.model_stack)
    summary = run_eval(case_paths=case_paths, output_path=output_path, config=config, limit=args.limit)
    if output_path is None:
        print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _config_for_backend(backend: str, model_stack: str | None) -> FigmentConfig:
    if backend == "canned":
        return FigmentConfig(model_backend="canned", model_stack=model_stack or "omni_native").validated()
    stack = model_stack or ("local_4b_parakeet" if backend == "llama_cpp" else "omni_native")
    base = FigmentConfig.from_env()
    return replace(base, model_backend=backend, model_stack=stack).validated()


if __name__ == "__main__":
    raise SystemExit(main())
