import json
from pathlib import Path
from typing import Any

from figment.config import FigmentConfig
from scripts import run_eval


INITIAL_CASES = Path("data/eval/initial_handwritten_cases.jsonl")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class _FakeRule:
    def __init__(self, payload: dict[str, str]) -> None:
        self.payload = payload

    def to_dict(self) -> dict[str, str]:
        return dict(self.payload)


class _FiredCardOmittedModelClient:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def generate_json(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {
            "protocol_urgency": "emergency",
            "red_flags": [_stroke_rule()],
            "intake_facts": [
                {
                    "fact": "Sudden one-sided weakness and trouble speaking.",
                    "status": "reported",
                    "source": "structured_field",
                }
            ],
            "candidate_protocol_pathways": [
                {
                    "card_id": "SAFETY-BOUNDARIES-v1",
                    "reason_relevant": "Safety boundaries are always relevant.",
                }
            ],
            "missing_info_to_collect": ["blood pressure if available"],
            "next_observations_to_collect": ["speech and one-sided weakness status"],
            "conflicts_or_uncertainties": ["Blood pressure not yet measured."],
            "responder_checklist": ["Keep deterministic red flags visible."],
            "do_not_do": ["Do not diagnose.", "Do not prescribe."],
            "source_cards": ["SAFETY-BOUNDARIES-v1", "REFERRAL-SBAR-v1"],
            "handoff_note_sbar": {
                "situation": "one-sided weakness",
                "background": "Age 56. Not pregnant.",
                "assessment_observations_only": "Sudden one-sided weakness and trouble speaking. Stroke sign red flag fired.",
                "handoff_request": "Request emergency review per cited local protocol cards.",
            },
            "responder_plain_language_script": "I am going to keep the stroke red flag visible and request emergency review.",
            "safety_boundary": "Prototype protocol navigation only; trained responder review required.",
        }


class _ObservationPatchRepairModelClient:
    calls = 0

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def generate_json(self, _prompt: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        self.__class__.calls += 1
        if context and context.get("repair_scope") == "missing_observations":
            return {
                "missing_info_to_collect": [
                    "pregnancy or postpartum status",
                    "bleeding report",
                    "abdominal pain report",
                    "headache or vision symptoms",
                    "seizure or fainting report",
                    "fever report",
                ],
                "next_observations_to_collect": [
                    "pregnancy or postpartum status",
                    "bleeding report",
                    "abdominal pain report",
                    "headache or vision symptoms",
                    "seizure or fainting report",
                    "fever report",
                ],
            }
        rules = _postpartum_fever_rules()
        return {
            "protocol_urgency": "emergency",
            "red_flags": rules,
            "intake_facts": [
                {
                    "fact": "Postpartum fever with chills; blood pressure pending.",
                    "status": "reported",
                    "source": "structured_field",
                }
            ],
            "candidate_protocol_pathways": [
                {
                    "card_id": "FEVER-RED-FLAGS-v1",
                    "reason_relevant": "Fever during postpartum period fired the fever card.",
                },
                {
                    "card_id": "PREG-DANGER-SIGNS-v1",
                    "reason_relevant": "Postpartum fever also fired the pregnancy danger-sign card.",
                },
            ],
            "missing_info_to_collect": [
                "temperature if available",
                "age or pregnancy status",
                "mental status",
                "neck stiffness report",
                "rash report",
                "hydration observations",
                "available vital signs",
            ],
            "next_observations_to_collect": [
                "Check temperature if available.",
                "Assess mental status now.",
                "age or pregnancy status",
            ],
            "conflicts_or_uncertainties": ["Blood pressure is still pending."],
            "responder_checklist": ["Keep emergency escalation active per local protocol."],
            "do_not_do": ["Do not diagnose.", "Do not prescribe."],
            "source_cards": [
                "PREG-DANGER-SIGNS-v1",
                "FEVER-RED-FLAGS-v1",
                "SAFETY-BOUNDARIES-v1",
                "REFERRAL-SBAR-v1",
            ],
            "handoff_note_sbar": {
                "situation": "postpartum fever",
                "background": "Setting: flood shelter. Age: 44 years. Pregnancy status: postpartum two weeks.",
                "assessment_observations_only": (
                    "Symptoms: fever with chills. Vitals: temperature 101.5 F; pulse fast; "
                    "blood pressure pending. Red flags: Pregnancy danger sign; Fever escalation cue."
                ),
                "handoff_request": "Request emergency review/escalation per cited local protocol cards.",
            },
            "responder_plain_language_script": (
                "We need emergency review through the local pathway while we document the missing observations."
            ),
            "safety_boundary": "Prototype protocol navigation only; trained responder review required.",
            "selected_required_observation_ids": [
                "FEVER-RED-FLAGS-v1::required_observation::1",
                "FEVER-RED-FLAGS-v1::required_observation::2",
                "FEVER-RED-FLAGS-v1::required_observation::3",
                "FEVER-RED-FLAGS-v1::required_observation::4",
                "FEVER-RED-FLAGS-v1::required_observation::5",
                "FEVER-RED-FLAGS-v1::required_observation::6",
                "FEVER-RED-FLAGS-v1::required_observation::7",
            ],
        }


def _stroke_rule() -> dict[str, str]:
    return {
        "rule_id": "STROKE-001",
        "label": "Stroke sign",
        "urgency": "emergency",
        "evidence": "one-sided weakness",
        "card_id": "STROKE-SIGNS-v1",
    }


def _retrieved_without_stroke_cards() -> list[dict[str, Any]]:
    return [
        {
            "card_id": "SAFETY-BOUNDARIES-v1",
            "title": "Safety boundaries",
            "score": 1.0,
            "source": "test",
            "card": {
                "card_id": "SAFETY-BOUNDARIES-v1",
                "title": "Safety boundaries",
                "required_observations": [],
            },
        },
        {
            "card_id": "REFERRAL-SBAR-v1",
            "title": "Referral SBAR",
            "score": 0.9,
            "source": "test",
            "card": {
                "card_id": "REFERRAL-SBAR-v1",
                "title": "Referral SBAR",
                "required_observations": [],
            },
        },
    ]


def _postpartum_fever_rules() -> list[dict[str, str]]:
    return [
        {
            "rule_id": "PREG-001",
            "label": "Pregnancy danger sign",
            "urgency": "emergency",
            "evidence": "fever",
            "card_id": "PREG-DANGER-SIGNS-v1",
        },
        {
            "rule_id": "FEVER-001",
            "label": "Fever escalation cue",
            "urgency": "urgent",
            "evidence": "pregnancy/infant fever context",
            "card_id": "FEVER-RED-FLAGS-v1",
        },
    ]


def _retrieved_postpartum_fever_cards() -> list[dict[str, Any]]:
    return [
        {
            "card_id": "FEVER-RED-FLAGS-v1",
            "score": 1.0,
            "source": "test",
            "card": {
                "card_id": "FEVER-RED-FLAGS-v1",
                "title": "Fever escalation red flags",
                "required_observations": [
                    "temperature if available",
                    "age or pregnancy status",
                    "mental status",
                    "neck stiffness report",
                    "rash report",
                    "hydration observations",
                    "available vital signs",
                ],
                "red_flags": ["fever during pregnancy or postpartum"],
            },
        },
        {
            "card_id": "PREG-DANGER-SIGNS-v1",
            "score": 0.95,
            "source": "test",
            "card": {
                "card_id": "PREG-DANGER-SIGNS-v1",
                "title": "Pregnancy danger signs",
                "required_observations": [
                    "pregnancy or postpartum status",
                    "bleeding report",
                    "abdominal pain report",
                    "headache or vision symptoms",
                    "seizure or fainting report",
                    "fever report",
                    "available vital signs",
                ],
                "red_flags": ["fever with pregnancy or postpartum concern"],
            },
        },
        {
            "card_id": "SAFETY-BOUNDARIES-v1",
            "score": 0.8,
            "source": "test",
            "card": {
                "card_id": "SAFETY-BOUNDARIES-v1",
                "title": "Safety boundaries",
                "required_observations": ["confirmed intake status"],
            },
        },
        {
            "card_id": "REFERRAL-SBAR-v1",
            "score": 0.7,
            "source": "test",
            "card": {
                "card_id": "REFERRAL-SBAR-v1",
                "title": "Referral and SBAR format",
                "required_observations": ["situation or reason for handoff"],
            },
        },
    ]


def test_canned_eval_runner_keeps_fallback_out_of_model_competence(tmp_path: Path) -> None:
    output_path = tmp_path / "eval-results.jsonl"

    summary = run_eval.run_eval(
        case_paths=[INITIAL_CASES],
        output_path=output_path,
        config=FigmentConfig(model_backend="canned"),
    )

    records = _jsonl(output_path)
    assert summary["total_cases"] == 10
    assert len(records) == 10
    assert summary["raw_configured_model_successes"] == 0
    assert summary["repair_successes"] == 0
    assert summary["canned_fallback_successes"] == 10
    assert summary["competence_successes"] == 0
    assert summary["final_validation_successes"] == 10
    assert "expected_label_successes" in summary
    assert "expected_label_check_successes" in summary

    first = records[0]
    assert first["case_id"] == "initial-ams-confusion-001"
    assert first["model_backend"] == "canned"
    assert first["model_stack"] == "omni_native"
    assert first["active_model_id"]
    assert first["fallback_tier"] == "canned"
    assert first["fallback_reason"] == "canned_backend"
    assert first["raw_configured_model_attempted"] is False
    assert first["raw_configured_model_success"] is False
    assert first["repair_attempted"] is False
    assert first["repair_success"] is False
    assert first["canned_fallback_used"] is True
    assert first["canned_fallback_success"] is True
    assert first["competence_success"] is False
    assert first["final_validation"]["passed"] is True
    assert first["expected_source_card_ids"] == [
        "AMS-RED-FLAGS-v1",
        "SAFETY-BOUNDARIES-v1",
        "REFERRAL-SBAR-v1",
    ]
    assert first["expected_missing_observations"]
    assert first["forbidden_behavior"]
    assert first["actual_protocol_urgency"] == first["final_output"]["protocol_urgency"]
    assert first["actual_source_card_ids"] == first["final_output"]["source_cards"]
    assert "expected_candidate_pathway_card_ids" in first
    assert first["harness_evidence"]["validator_status"] == "passed"
    assert first["harness_evidence"]["fallback_tier"] == "canned"
    assert first["final_output"]["harness_evidence"] == first["harness_evidence"]
    assert "expected_label_score" in first
    assert first["expected_label_score"]["red_flags_match"] is True
    assert first["expected_label_score"]["min_urgency_met"] is True
    assert "harness_evidence_cues_visible" in first["expected_label_score"]
    assert first["field_provenance"]["protocol_urgency"] == "deterministic_fallback"
    assert summary["records_with_field_provenance"] == 10
    assert summary["model_field_pass_rate"] == 0.0
    assert summary["model_visible_fields_retained"] == 0.0
    assert summary["deterministic_patch_count"] == len(first["field_provenance"]) * 10
    assert first["latency_ms"] >= 0
    assert isinstance(first["trace_hash"], str)
    assert len(first["trace_hash"]) >= 12
    assert first["raw_model_output"] is None
    assert first["repaired_output"] is None
    assert isinstance(first["fallback_output"], dict)
    assert (output_path.parent / "eval_summary.json").exists()
    assert (output_path.parent / "eval_evidence_manifest.json").exists()
    manifest = json.loads((output_path.parent / "eval_evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["all_trace_hashes_present"] is True
    assert manifest["scored_reporting_eligible"] is True


def test_eval_runner_repairs_known_fired_card_when_retrieval_missed_it(monkeypatch) -> None:
    monkeypatch.setattr(run_eval, "ModelClient", _FiredCardOmittedModelClient)
    monkeypatch.setattr(run_eval, "run_red_flag_checks", lambda _: [_FakeRule(_stroke_rule())])
    monkeypatch.setattr(run_eval, "search_protocol_cards", lambda *_args, **_kwargs: _retrieved_without_stroke_cards())

    record = run_eval._evaluate_case(
        {
            "case_id": "unit-stroke-retrieval-miss",
            "structured_intake": {
                "setting": "mobile clinic",
                "patient_age": "56",
                "pregnancy_status": "not_pregnant",
                "chief_concern": "one-sided weakness",
                "symptoms": "Sudden one-sided weakness and trouble speaking",
                "vitals": "blood pressure not yet measured; pulse fast",
                "responder_note": "Adult with acute stroke-sign concern.",
                "confirmed": True,
            },
            "target_protocol_card_id": "STROKE-SIGNS-v1",
            "expected_min_protocol_urgency": "emergency",
            "expected_red_flag_rule_ids": ["STROKE-001"],
            "expected_source_card_ids": ["STROKE-SIGNS-v1"],
            "expected_candidate_pathway_card_ids": ["STROKE-SIGNS-v1"],
        },
        FigmentConfig(model_backend="hosted_omni", nvidia_api_key="test-nvidia-key"),
    )

    assert record["final_validation"]["passed"] is True
    assert record["competence_success"] is False
    assert "STROKE-SIGNS-v1" not in record["raw_model_output"]["source_cards"]
    assert "STROKE-SIGNS-v1" not in {
        pathway["card_id"] for pathway in record["raw_model_output"]["candidate_protocol_pathways"]
    }
    assert "STROKE-SIGNS-v1" in record["scaffolded_model_output"]["source_cards"]
    assert "STROKE-SIGNS-v1" in {
        pathway["card_id"] for pathway in record["scaffolded_model_output"]["candidate_protocol_pathways"]
    }
    assert "STROKE-SIGNS-v1" in record["final_output"]["source_cards"]
    assert "STROKE-SIGNS-v1" in record["actual_candidate_pathway_card_ids"]
    assert record["field_provenance"]["source_cards"] == "deterministic_fallback"
    assert record["field_provenance"]["candidate_protocol_pathways"] == "deterministic_fallback"
    assert record["expected_label_score"]["target_card_in_source_cards"] is True
    assert record["expected_label_score"]["target_card_in_candidate_pathways"] is True


def test_eval_runner_repairs_model_observation_patch_fields(monkeypatch) -> None:
    _ObservationPatchRepairModelClient.calls = 0
    monkeypatch.setattr(run_eval, "ModelClient", _ObservationPatchRepairModelClient)
    monkeypatch.setattr(
        run_eval,
        "run_red_flag_checks",
        lambda _: [_FakeRule(rule) for rule in _postpartum_fever_rules()],
    )
    monkeypatch.setattr(run_eval, "search_protocol_cards", lambda *_args, **_kwargs: _retrieved_postpartum_fever_cards())

    record = run_eval._evaluate_case(
        {
            "case_id": "unit-postpartum-fever-observation-repair",
            "structured_intake": {
                "setting": "flood shelter",
                "patient_age": "44 years",
                "pregnancy_status": "postpartum two weeks",
                "chief_concern": "postpartum fever",
                "symptoms": "fever with chills during postpartum period",
                "vitals": "temperature 101.5 F; pulse fast; blood pressure pending",
                "responder_note": "Confirmed postpartum fever concern.",
                "confirmed": True,
            },
            "target_protocol_card_id": "FEVER-RED-FLAGS-v1",
            "expected_min_protocol_urgency": "emergency",
            "expected_red_flag_rule_ids": ["PREG-001", "FEVER-001"],
            "expected_source_card_ids": ["PREG-DANGER-SIGNS-v1", "FEVER-RED-FLAGS-v1"],
            "expected_candidate_pathway_card_ids": ["FEVER-RED-FLAGS-v1"],
        },
        FigmentConfig(model_backend="hosted_omni", nvidia_api_key="test-nvidia-key"),
    )

    assert _ObservationPatchRepairModelClient.calls == 2
    assert record["final_validation"]["passed"] is True
    assert record["raw_configured_model_success"] is False
    assert record["repair_attempted"] is True
    assert record["repair_success"] is True
    assert record["competence_success"] is True
    assert record["field_level_fallback_used"] is False
    assert record["deterministic_scaffold_patched_fields"] == [
        "missing_info_to_collect",
        "next_observations_to_collect",
    ]
    assert record["field_provenance"]["missing_info_to_collect"] == "model_repaired"
    assert record["field_provenance"]["next_observations_to_collect"] == "model_repaired"
    assert "PREG-DANGER-SIGNS-v1::required_observation::2" in record["filled_required_observation_ids"]
    assert "bleeding report" in record["final_output"]["missing_info_to_collect"]
    assert "selected_required_observation_ids" not in record["final_output"]


def test_eval_cli_runs_initial_cases_against_canned_without_network(tmp_path: Path) -> None:
    output_path = tmp_path / "cli-results.jsonl"

    exit_code = run_eval.main(
        [
            "--backend",
            "canned",
            "--cases",
            str(INITIAL_CASES),
            "--output",
            str(output_path),
        ]
    )

    records = _jsonl(output_path)
    assert exit_code == 0
    assert len(records) == 10
    assert {record["raw_configured_model_success"] for record in records} == {False}
    assert {record["canned_fallback_used"] for record in records} == {True}
    assert {record["final_validation"]["passed"] for record in records} == {True}
    assert {record["field_provenance"]["source_cards"] for record in records} == {"deterministic_fallback"}
    assert all("expected_label_score" in record for record in records)


def test_llama_eval_summary_describes_real_eval_evidence_scope(tmp_path: Path) -> None:
    summary = run_eval._summarize(
        [
            {
                "raw_configured_model_success": True,
                "repair_success": False,
                "canned_fallback_used": False,
                "canned_fallback_success": False,
                "competence_success": True,
                "final_validation": {"passed": True},
            }
        ],
        FigmentConfig(model_backend="llama_cpp", model_stack="local_4b_parakeet"),
        [INITIAL_CASES],
        tmp_path / "local-eval.jsonl",
    )

    assert summary["local_llm_evidence"]["proof_status"] == "eval_records_summarized"
    assert summary["local_llm_evidence"]["model_backend"] == "llama_cpp"
    assert summary["local_llm_evidence"]["counts_as_50_case_local_llm_competence"] is False
    assert summary["local_llm_evidence"]["competence_successes"] == 1
    assert summary["local_llm_evidence"]["scored_reporting_eligible"] is True
    assert summary["local_llm_evidence"]["models_endpoint"]["available"] is False
    assert "MODEL_BACKEND=llama_cpp" in summary["local_llm_evidence"]["real_eval_command"]


def test_runtime_errors_mark_local_eval_ineligible_for_scored_reporting(tmp_path: Path) -> None:
    summary = run_eval._summarize(
        [
            {
                "raw_configured_model_success": False,
                "repair_success": False,
                "canned_fallback_used": True,
                "canned_fallback_success": True,
                "competence_success": False,
                "raw_validation": {
                    "passed": False,
                    "failures": ["model backend error: http_status=500 reason=failed to find free space in the KV cache"],
                },
                "final_validation": {"passed": True},
            }
        ],
        FigmentConfig(model_backend="llama_cpp", model_stack="local_4b_parakeet"),
        [INITIAL_CASES],
        tmp_path / "local-eval.jsonl",
    )

    assert summary["scored_reporting_eligible"] is False
    assert summary["runtime_error_summary"]["server_http_500"] is True
    assert summary["runtime_error_summary"]["kv_cache_failure"] is True
