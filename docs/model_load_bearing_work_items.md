# Model load-bearing work items

Date: 2026-06-07

This note turns the hosted Omni eval result into four implementation workstreams. The goal is to make the configured model more visibly load-bearing without weakening deterministic safety controls.

Current hosted evidence:

- Hosted eval follow-up trace: `traces/hosted_omni_eval_load_bearing_20260607T210047Z.jsonl`
- Whole-output hosted competence: 31/50
- Full deterministic fallback use: 8/50
- Field-level model retention: 480/650 fields, or 73.8%
- Deterministic patches: 170/650 fields
- Final validation: 50/50

Use these follow-up metrics for current README, checklist, and submission scorecard wording. Keep the baseline below only as comparison evidence. Final validation is application safety; whole-output competence and field retention are the model-load-bearing metrics.

Baseline evidence:

- Hosted eval trace: `traces/hosted_omni_eval_20260607T194833Z.jsonl`
- Model: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
- Corpus: 50 synthetic cases across `data/eval/*.jsonl`
- Whole-output hosted competence: 28/50
- Deterministic fallback use: 22/50
- Final validation: 50/50

Safety principle: do not loosen validators to raise the score. Increase model contribution by narrowing the model's tasks, validating its outputs field-by-field, and preserving deterministic fallback for any unsafe, missing, or ungrounded field.

## 1. Field-level model acceptance and provenance

Current problem: a single invalid field can force full deterministic fallback, even when other model-generated fields are useful and safe.

Implementation direction:

- Validate navigator output at the field or section level where possible.
- Retain validated model-generated fields.
- Deterministically patch or replace only failing fields.
- Record field provenance as `model_raw`, `model_repaired`, or `deterministic_fallback`.
- Surface provenance in trace and UI so fallback is never counted as model competence.

Done when:

- A trace can show which bounded navigator fields were model-generated versus repaired or fallback-filled.
- Eval metrics can distinguish whole-output competence from field-level model contribution.
- Existing final safety validation still passes.

## 2. More constrained prompt contract

Current problem: hosted outputs commonly miss SBAR fields, omit required observations, introduce unsupported handoff facts, or use forbidden clinical phrasing.

Implementation direction:

- Give the model a literal JSON skeleton with every required key.
- Provide an explicit allowed-facts inventory derived from confirmed intake, deterministic rules, and retrieved cards.
- Provide a required-observations inventory derived from retrieved protocol cards.
- Add concise examples for routine or negated cases where emergency cards are nearby but red flags are absent.
- Keep diagnosis, prescription, dosing, discharge, and autonomous routing prohibitions explicit.

Done when:

- Raw hosted output pass rate improves without validator relaxation.
- Missing SBAR key failures decrease.
- Missing-observation grounding failures decrease.
- Negated routine cases are less likely to be escalated or contaminated by nearby emergency card language.

## 3. Focused field repair before fallback

Current problem: repair currently asks the model to regenerate the whole navigator output after validation failure. That is larger than necessary and can reintroduce unrelated defects.

Implementation direction:

- Classify validation failures by field or section.
- For isolated SBAR failures, repair only `handoff_note_sbar`.
- For missing-observation failures, repair only `missing_info_to_collect` and `next_observations_to_collect`.
- For source-card or pathway failures, repair only citations and candidate pathways.
- Revalidate repaired fields before merging them into the final output.
- Fall back deterministically for fields that still fail after focused repair.

Done when:

- Repair attempts are smaller and traceable by field.
- Successful focused repair increases retained model contribution.
- Repeated repair failure still produces safe deterministic final output.

## 4. Eval metrics for model load-bearing behavior

Current problem: the eval has a strict whole-output competence metric, but it cannot yet quantify partial model contribution.

Implementation direction:

- Add field-level metrics such as `model_field_pass_rate`, `model_visible_fields_retained`, `deterministic_patch_count`, and per-field provenance counts.
- Keep existing metrics: raw model success, repair success, fallback use, final validation success.
- Report field-level contribution separately from whole-output competence.
- Update the hosted eval result note after running the new metrics.

Done when:

- The eval can report how much of the visible navigator output came from the model.
- Fallback cannot inflate model competence metrics.
- The project can honestly claim partial or improved model load-bearing behavior with trace evidence.

## Implementation checkpoint

Checkpoint trace: `traces/hosted_omni_eval_load_bearing_20260607T210047Z.jsonl`

Implemented changes:

- `figment/prompt_builder.py` now includes a literal required JSON skeleton, an allowed-facts inventory, required-observations inventory, and routine/negated-case guidance.
- `figment/field_provenance.py` provides schema-bounded field merging and provenance labels.
- `figment/focused_repair.py` classifies validation failures into scoped repair prompts.
- `figment/eval_metrics.py` reports field-level model retention and deterministic patch counts.
- `figment/navigator.py` and `scripts/run_eval.py` now retain validated model fields, patch unsafe or missing fields deterministically, and record `field_provenance`.

Measured result:

- Whole-output hosted competence improved from 28/50 to 31/50.
- Full deterministic fallback dropped from 22/50 to 8/50.
- Final validation stayed at 50/50.
- Field-level model retention is 480/650 fields, or 73.8%.
- Deterministic patches remain visible and are not counted as model competence.

Open follow-up:

- Focused repair increases latency; consider capping repair scopes or batching repair prompts.
- SBAR and referral cases still need targeted prompt or validator UX work.
