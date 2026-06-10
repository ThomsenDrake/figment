# Local 4B V4 Scaffolding And Eval Shape Plan

Date: 2026-06-10

## Purpose

This plan defines the scaffolding and evaluation-shape fixes to make before any v4 LoRA training run.

The v3 model is not generally broken. The latest field-workflow holdout shows strong safety and protocol-navigation behavior, but weak handoff usefulness and weak observation-cue scoring. Some of the weakness is model-owned, especially `REFERRAL-SBAR-v1` and radio handoff behavior. Some is harness-owned: the evaluator currently expects metadata cues that the app can deterministically know, such as retrieved card IDs and validation status, to appear inside model-authored missing-observation fields.

The goal is to stop asking the model to memorize app metadata, then measure the remaining model-owned gap clearly.

## Current Evidence

Primary trace:

- `traces/local_4b_finetuned_v3_field_holdout_sequential_20260610T102450Z/local_4b_eval.jsonl`
- `traces/local_4b_finetuned_v3_field_holdout_sequential_20260610T102450Z/eval_summary.json`
- `traces/local_4b_finetuned_v3_field_holdout_sequential_20260610T102450Z/eval_evidence_manifest.json`

V3 field-workflow holdout result:

- `total_cases`: `150`
- `competence_successes`: `107`
- `raw_configured_model_successes`: `93`
- `repair_successes`: `14`
- `fallback_uses`: `2`
- `final_validation_successes`: `148`
- `model_visible_fields_retained`: `0.9415`

Strong areas:

- `min_urgency_met`: `150/150`
- `red_flags_match`: `150/150`
- `forbidden_behavior_absent`: `150/150`
- `target_card_in_candidate_pathways`: `149/150`
- `target_card_in_source_cards`: `149/150`
- `expected_source_cards_present`: `144/150`

Weak areas:

- `missing_observation_cues_present`: `0/150`
- `REFERRAL-SBAR-v1`: `0/27` competence
- `radio_handoff`: `0/16` competence
- `sbar_handoff_usefulness`: `0/10` competence
- `source_card_discipline`: `2/6` competence

Top missing observation cues among failed competence cases:

- `navigator validation result`: `40`
- `manual correction status for audio-derived fields`: `40`
- `retrieved protocol card IDs`: `37`
- `deterministic rule results`: `23`
- `objective observations only`: `18`
- `relevant background and timeline`: `17`
- `specific request or receiving pathway`: `17`
- `situation or reason for handoff`: `15`
- `red flags already fired`: `15`
- `confirmed intake status`: `12`
- `source protocol card IDs`: `12`

## Diagnosis

The current eval mixes three different things in `expected_missing_observations`:

1. Clinical or workflow observations the medic may need to collect.
2. Handoff content cues the model should include when the target is SBAR or radio handoff.
3. App/harness metadata the model should not need to invent, such as validation status, retrieved card IDs, deterministic rule results, and manual correction status.

This makes the score hard to interpret. A model can be safe, cite the right cards, preserve red flags, and produce useful protocol navigation while still failing every expected-label row because it did not phrase app metadata as a missing observation.

The fix is not to train the 4B model to recite every metadata cue. The fix is to move deterministic metadata into deterministic output surfaces and reserve model training for bounded, model-owned text.

## Fix 1: Split Observation Cues By Ownership

Add an ownership split to generated eval cases and scoring:

- `expected_model_observation_cues`: facts or observations the responder may need to collect or confirm.
- `expected_handoff_cues`: SBAR/radio facts that should appear in the handoff fields.
- `expected_harness_evidence_cues`: deterministic app metadata that should be visible in trace or UI, not authored as missing observations.

Initial harness-owned cues:

- `navigator validation result`
- `manual correction status for audio-derived fields`
- `retrieved protocol card IDs`
- `deterministic rule results`
- `confirmed intake status`
- `source protocol card IDs`

Implementation targets:

- `scripts/generate_field_workflow_holdout.py`
- `scripts/generate_finetune_data.py`
- `figment/eval_metrics.py`
- `scripts/run_eval.py`

Do not rewrite the frozen `field_workflow_holdout_v1` cases in place. Preserve them and add a derived scoring view or v1.1 manifest that maps existing `expected_missing_observations` into ownership buckets.

## Fix 2: Add Deterministic Evidence Badges

Add deterministic evidence fields to the trace and app-facing navigator output so the UI can show metadata without asking the model to write it:

- intake confirmation status,
- retrieved protocol card IDs,
- fired deterministic rule IDs,
- urgency floor,
- validator status,
- audio/manual correction status,
- source-card set used by final output,
- fallback or repair tier.

Preferred shape:

```json
{
  "harness_evidence": {
    "confirmed_intake": true,
    "retrieved_card_ids": ["..."],
    "deterministic_rule_ids": ["..."],
    "urgency_floor": "emergency",
    "validator_status": "passed",
    "audio_correction_status": "not_applicable",
    "final_route": "configured"
  }
}
```

This object should be deterministic and excluded from model-retained-field credit unless the model actually authored it. The UI can render it as compact badges beside the handoff rather than burying it in `missing_info_to_collect`.

## Fix 3: Make Handoff A First-Class Eval Surface

Today the SBAR/radio misses are visible mostly through target card and observation-cue failures. Add explicit handoff metrics:

- `sbar_situation_present`
- `sbar_background_present`
- `sbar_assessment_observation_only`
- `sbar_request_present`
- `sbar_source_card_cited`
- `sbar_red_flags_visible`
- `handoff_brevity_ok`
- `handoff_unsupported_fact_count`
- `handoff_readiness_passed`

For `radio_handoff` and `sbar_handoff_usefulness`, these metrics should drive competence more than generic missing-observation cue coverage.

Acceptance target after scaffolding, before v4 training:

- `radio_handoff` and `sbar_handoff_usefulness` should no longer fail only because of harness-owned metadata cues.
- SBAR failures should report a specific missing handoff slot or unsupported fact, not a generic observation-cue miss.

## Fix 4: Add A Deterministic SBAR Draft Scaffold

Build a deterministic SBAR draft before the model writes final text.

Inputs:

- confirmed intake,
- deterministic red flags,
- urgency floor,
- retrieved card IDs,
- target protocol card,
- high-value observation cues,
- source-card titles.

The scaffold should produce slot-limited draft facts:

- `situation`: patient, chief concern, target pathway or reason for handoff.
- `background`: only confirmed context and timeline.
- `assessment`: observation-only summary plus fired rule IDs, no diagnosis.
- `request`: the specific review, transport, callback, or protocol-navigation ask.

Then ask the model to rewrite only within those facts. If the model fails SBAR grounding, repair only the SBAR fields rather than falling back the whole navigator output.

Implementation targets:

- `figment/prompt_builder.py`
- `figment/navigator.py`
- `figment/focused_repair.py`
- `scripts/run_eval.py`

## Fix 5: Trigger Competence Repair For Safe But Weak Outputs

The current repair path is mostly validation-driven. Many v3 SBAR misses are safe and schema-valid, so repair never fires.

Add an optional eval and app repair mode for model-owned competence failures:

- If final validation passes but `handoff_readiness_passed` fails, run a focused `handoff_note_sbar` repair.
- If source-card discipline fails but validation passes, run a focused source-card/candidate-pathway repair.
- If model-owned observation cue coverage fails, run a focused missing-observation repair.

This should not hide safety failures. Safety validation still wins. The competence repair should be reported separately:

- `validation_repair_attempted`
- `competence_repair_attempted`
- `competence_repair_success`
- `competence_repair_scope`

## Fix 6: Normalize Cue Matching

For model-owned cues, add alias-aware matching so the eval rewards equivalent field phrasing:

- `red flags already fired` can match `fired deterministic red flags`, `rule ids triggered`, or a direct rule ID mention.
- `source protocol card IDs` can match a cited source-card ID in SBAR or source fields.
- `specific request or receiving pathway` can match a direct callback/transport/receiving-clinician request.
- `objective observations only` can match absence of diagnosis plus observation-only assessment language.

Do not relax safety, urgency, red-flag, source-card, or forbidden-behavior checks. Only relax brittle cue phrase matching for useful equivalent language.

## Fix 7: Add Runtime Evidence To Prevent Bad Parallel Evals

The attempted parallel v3 holdout run produced invalid records because `llama-server` split the context across parallel slots and hit KV/cache overflow. Keep those records quarantined.

Add run metadata to every local eval bundle:

- server command,
- GGUF path and SHA-256,
- `n_ctx`,
- `n_parallel`,
- prompt cache settings,
- endpoint `/v1/models` payload,
- whether any server HTTP 500s occurred,
- whether the run is eligible for scored reporting.

Eval runner should refuse to mark a run clean if backend errors include `Context size has been exceeded` or `failed to find free space in the KV cache`.

## Implementation Order

1. Add ownership bucketing for expected cues and update summary metrics.
2. Add deterministic `harness_evidence` to traces and UI-facing output.
3. Add explicit handoff-readiness metrics.
4. Add SBAR scaffold and focused SBAR repair.
5. Add competence repair for safe but weak outputs.
6. Add alias-aware cue matching for model-owned cues.
7. Add local runtime clean-run metadata and invalid-run detection.
8. Rerun v3 on the 150-case holdout and the locked 50-case regression before training v4.

## Acceptance Gates

Before starting v4 training:

- The new eval report separates:
  - validation success,
  - model-owned observation cue coverage,
  - harness-owned evidence visibility,
  - handoff readiness,
  - competence repair success.
- The v3 holdout rerun no longer has `expected_label_successes = 0/150` purely because of harness-owned metadata cues.
- `REFERRAL-SBAR-v1` failures identify concrete handoff defects instead of only missing metadata cues.
- No regression on:
  - urgency floor,
  - red-flag match,
  - forbidden behavior,
  - final validation,
  - source-card validity.

Expected scaffolding-only improvement:

- Better interpretability immediately.
- Some increase in competence from deterministic handoff scaffolding and competence repair.
- Remaining SBAR/radio gaps become cleaner targets for v4 training.

Do not train v4 until this pass is done. Otherwise the v4 dataset will teach the model to satisfy a muddled scorer rather than to help field medics.
