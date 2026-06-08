# Local 4B Prompting And Scaffolding Fixes

Date: 2026-06-07

This note captures the prompt and controller changes I would make before reaching for a larger model. The goal is to make `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` more load-bearing on the local `llama.cpp` route while keeping Figment's deterministic validators and safety boundaries intact.

## Evidence Snapshot

Primary trace:

- `traces/local_4b_evidence_20260607T231248Z/`
- `traces/local_4b_evidence_20260607T231248Z/local_4b_eval.jsonl`
- `traces/local_4b_evidence_20260607T231248Z/eval_summary.json`

Observed local 4B behavior:

- 50/50 final validation successes.
- 18/50 competence successes.
- 13/50 raw configured-model successes.
- 5/50 repair successes.
- 9/50 full deterministic fallbacks.
- 499/650 visible fields retained from model output, or 76.8%.
- Expected-label full success was 2/50.

The failure pattern looks fixable with better scaffolding and fine-tuning. The model is not generally missing urgency: `min_urgency_met` passed 50/50. It is mostly missing exact required-observation cues, candidate pathway/source-card rubric details, negation discipline, and grounded SBAR phrasing.

Expected-label failure counts:

- `missing_observation_cues_present`: 47 failures.
- `target_card_in_candidate_pathways` / `expected_candidate_pathways_present`: 15 failures.
- `expected_source_cards_present`: 12 failures.
- `red_flags_match`: 7 failures.
- `forbidden_behavior_absent`: 4 failures.
- `target_card_in_source_cards`: 1 failure.
- `min_urgency_met`: 0 failures.

## Current Anchors

Current prompt scaffolding already exists but is too advisory:

- `figment/prompt_builder.py` builds `allowed_facts_inventory`, `required_observations_inventory`, `routine_or_negated_case_guidance`, and a required JSON skeleton.
- `figment/validators.py` enforces source-card constraints, urgency floors, missing-observation grounding, SBAR grounding, and forbidden clinical language.
- `figment/focused_repair.py` already has a `missing_observations` repair scope, but the repair prompt still asks the model to infer the exact observation language.
- `figment/eval_metrics.py` scores expected labels separately from safety validation, which is the right separation.

## Fix 1: Promote Required Observations From Context To Targets

Problem: `required_observations_inventory` is present, but the 4B model treats it like optional supporting context.

Change:

- Add a compact `required_observation_targets` payload to the prompt context.
- Give each target a stable id, card id, normalized cue tokens, and display text.
- Tell the model that `missing_info_to_collect` and `next_observations_to_collect` must include at least one target cue for every cited non-exempt card that has required observations.
- After model output, deterministically patch missing target cues into those two fields before falling back.
- Trace each patch as `deterministic_required_observation_fill`, not model competence.

The key distinction is that the LLM can phrase the responder-facing sentence, but the app owns the checklist target. This should attack the largest failure class directly without weakening the validator.

Done when:

- `missing_observation_cues_present` improves from 3/50 passing to at least 45/50 passing.
- The trace shows whether each required-observation cue was model-written, repaired, or deterministically filled.
- Whole-output competence and field-retention metrics do not count deterministic fills as raw model success.

## Fix 2: Pre-Fill Non-Creative Control Fields

Problem: the model is being asked to regenerate facts the deterministic system already knows.

Change:

- Pre-fill `protocol_urgency` from the deterministic urgency floor.
- Pre-fill fired `red_flags` from deterministic rule results.
- Pre-fill mandatory `source_cards` from fired rule card ids plus the retrieved/selected card ids.
- Pre-fill candidate pathway options from retrieval before asking the model to write reasons.
- Ask the model to write bounded text for `reason_relevant`, `responder_checklist`, `do_not_do`, SBAR slots, plain language, and uncertainty handling.

This narrows the 4B model's job from "reconstruct the whole navigation state" to "explain and operationalize already-bounded state." That is a better match for a small model.

Done when:

- Source-card and candidate-pathway expected-label failures drop sharply.
- Fallbacks caused by sparse or malformed source-card fields become rare.
- The model is still visibly load-bearing on prose, checklists, and pathway reasons.

## Fix 3: Add A Negation Ledger

Problem: routine or denied-symptom cases still sometimes inherit nearby emergency-card language.

Change:

- Add a `case_fact_ledger` to the prompt with three explicit buckets: `present`, `absent_or_denied`, and `unclear`.
- Add `must_not_fire_rule_ids` when a rule's trigger terms are absent or denied.
- Put the ledger before the protocol cards in the prompt.
- Require any red flag to cite a `present` fact or deterministic rule result.
- For SBAR and checklist text, forbid copying high-risk card language unless supported by `present` facts or deterministic red flags.

Done when:

- `red_flags_match` failures drop from 7/50 to 1/50 or less.
- Routine negated cases stay routine when the deterministic urgency floor is routine.

## Fix 4: Make SBAR A Filled Template

Problem: SBAR failures are not mainly creative-writing failures. They are slot-grounding failures.

Change:

- Build an SBAR draft template with fixed slots:
  - `situation`: chief concern plus setting, if confirmed.
  - `background`: confirmed age, pregnancy status, and relevant context only.
  - `assessment_observations_only`: confirmed symptoms, vitals, and deterministic red flags only.
  - `handoff_request`: local protocol, supervisor, clinician, or emergency pathway request only.
- Ask the model to lightly rewrite the template, not invent SBAR content.
- Keep unsupported high-risk facts out of the template before the model sees it.

Done when:

- SBAR grounding failures do not force full fallback when the rest of the model output is usable.
- Unsupported high-risk SBAR terms such as pregnancy, oxygen, or pressure only appear when present in confirmed intake, deterministic rules, or card text that is allowed for that slot.

## Fix 5: Separate Observations From Clinical Actions

Problem: the expected-label scorer and validators need sharper language around terms that can be either safe observations or unsafe instructions.

Change:

- Split observation cues from intervention cues.
- Treat `oxygen saturation`, `SpO2`, and `room-air saturation` as observation language.
- Treat `administer oxygen`, `start oxygen`, oxygen-flow settings, dosing, and medication instructions as intervention language.
- Update expected-label forbidden checks so observation requests do not get penalized as unsafe oxygen instructions.
- Keep forbidden clinical action patterns strict.

Done when:

- The model can ask for oxygen saturation as missing information without being pushed toward oxygen administration language.
- `forbidden_behavior_absent` failures are true safety failures, not observation/action ambiguity.

## Fix 6: Make Focused Repair Deterministic-Target-Aware

Problem: the current focused repair scope for missing observations asks the model to repair only the two observation arrays, but it does not force exact target coverage.

Change:

- For the `missing_observations` repair scope, include only:
  - the failed card ids,
  - the missing target cue ids,
  - the allowed display text,
  - the previous two observation arrays.
- Require the repair output to return exactly `missing_info_to_collect` and `next_observations_to_collect`.
- Reject repairs that omit target cue ids.
- If repair still omits a cue, patch deterministically rather than asking for another broad repair.

Done when:

- Missing-observation repairs become short, low-latency, and predictable.
- Repaired observation fields count as `model_repaired`; deterministic fills count separately.

## Fix 7: Add A Structured Output Contract For Target Coverage

Problem: natural-language arrays are hard to audit for exact expected-label coverage.

Change:

- Add an internal-only field during model generation, such as `selected_required_observation_ids`.
- Strip it from the user-facing navigator output after validation.
- Validate that every cited required-observation card has at least one selected observation id.
- Use the selected ids to prove why a natural-language observation sentence satisfies the target.

Done when:

- Expected-label scoring can distinguish "model selected the right cue but phrased it differently" from "model missed the cue."
- The visible output remains clean while trace evidence becomes more exact.

## Suggested Implementation Order

1. Add `required_observation_targets` and `case_fact_ledger` in `figment/prompt_builder.py`.
2. Add deterministic target-fill helpers shared by navigator finalization and tests.
3. Update focused repair for the `missing_observations` scope to use target ids.
4. Pre-fill or lock control fields before the LLM call.
5. Convert SBAR generation to a slot template plus bounded rewrite.
6. Split observation/action forbidden-language scoring for oxygen-like terms.
7. Rerun `scripts/run_local_4b_evidence.py` and compare against the 2026-06-07 trace.

## Non-Negotiables

- Do not weaken deterministic red-flag rules.
- Do not loosen validators to inflate model competence.
- Do not count deterministic target fills as raw model output.
- Do not claim Parakeet ASR proof from typed transcripts or local artifact presence.
- Keep the final local artifact full-weight/BF16 for the local route; no quantized local model claim.
