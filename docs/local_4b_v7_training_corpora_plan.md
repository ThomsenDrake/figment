# Figment Local 4B V7 Training Corpora Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a v7 SFT corpus that keeps v6's large reduction in scaffold dependence while restoring source-card closure and improving the model's native ability to operate inside the Figment field-workflow harness.

**Architecture:** Treat v6 as the behavior anchor, reuse only audited v6 and historical replay rows, and add a focused v7 delta for source-card closure, joint source-card plus observation ownership, and distractor-card resistance. The corpus remains aligned to the exact navigator and focused-repair prompt shapes used by the harness, with verifier gates that reject rows teaching deterministic patch artifacts or eval leakage.

**Tech Stack:** Python, JSONL SFT corpora, Figment eval harness, OpenRouter/NVIDIA Nemotron teacher, Modal LoRA SFT, H100 training, `llama.cpp`/GGUF evaluation, pytest.

---

## Current Evidence

Primary v6 eval trace:

- `traces/figment_sft_v6_field_workflow_holdout_modal_gpu_20260611_h100_gguf/local_4b_eval.jsonl`
- `traces/figment_sft_v6_field_workflow_holdout_modal_gpu_20260611_h100_gguf/eval_summary.json`
- `traces/figment_sft_v6_field_workflow_holdout_modal_gpu_20260611_h100_gguf/summary.json`

Observed v6 result:

- `142/150` configured-model competence successes.
- `150/150` final validation successes.
- `146/150` expected-label successes.
- `0` fallback uses.
- `21` deterministic field patches across `1950` scored visible fields.
- Field provenance:
  - `source_cards`: `144/150` model raw, `6/150` deterministic fallback.
  - `missing_info_to_collect`: `143/150` model raw, `7/150` deterministic fallback.
  - `next_observations_to_collect`: `143/150` model raw, `7/150` deterministic fallback.
  - `candidate_protocol_pathways`: `149/150` model raw, `1/150` deterministic fallback.

Primary v5 comparison trace:

- `traces/figment_sft_v5_field_workflow_holdout_modal_gpu_20260611_h100_gguf/local_4b_eval.jsonl`
- `traces/figment_sft_v5_field_workflow_holdout_modal_gpu_20260611_h100_gguf/eval_summary.json`

Observed v5 comparison:

- `2/150` configured-model competence successes.
- `150/150` final validation successes.
- `150/150` expected-label successes.
- `0` fallback uses.
- `302` deterministic field patches.

Interpretation:

V6 is the right base. It made the model far less scaffold-dependent than v5 while preserving final safety. The v7 corpus should not revert to the v5 shape, where final outputs looked good because scaffolding was doing too much. The v7 target is to keep v6's model-owned observation behavior and close the small source-card exactness gap.

## V6 Failure Shape

Configured-model competence failures:

- `field_workflow_holdout_v1-000019`
- `field_workflow_holdout_v1-000050`
- `field_workflow_holdout_v1-000055`
- `field_workflow_holdout_v1-000067`
- `field_workflow_holdout_v1-000078`
- `field_workflow_holdout_v1-000091`
- `field_workflow_holdout_v1-000099`
- `field_workflow_holdout_v1-000115`

Expected-label failures:

- `field_workflow_holdout_v1-000067`
- `field_workflow_holdout_v1-000091`
- `field_workflow_holdout_v1-000099`
- `field_workflow_holdout_v1-000115`

Source-card misses inside expected-label failures:

- `field_workflow_holdout_v1-000067`: missing `REFERRAL-SBAR-v1` and `SAFETY-BOUNDARIES-v1`; actual source cards were `PREG-DANGER-SIGNS-v1`, `CHEST-PAIN-ESCALATION-v1`, and `FEVER-RED-FLAGS-v1`.
- `field_workflow_holdout_v1-000091`: missing `REFERRAL-SBAR-v1` and `SAFETY-BOUNDARIES-v1`; actual source cards were `PREG-DANGER-SIGNS-v1`, `CHEST-PAIN-ESCALATION-v1`, and `FEVER-RED-FLAGS-v1`.
- `field_workflow_holdout_v1-000099`: missing `SAFETY-BOUNDARIES-v1`; actual source cards were `PREG-DANGER-SIGNS-v1`, `CHEST-PAIN-ESCALATION-v1`, `FEVER-RED-FLAGS-v1`, and `REFERRAL-SBAR-v1`.
- `field_workflow_holdout_v1-000115`: missing `REFERRAL-SBAR-v1` and `SAFETY-BOUNDARIES-v1`; actual source cards were `PREG-DANGER-SIGNS-v1`, `CHEST-PAIN-ESCALATION-v1`, and `FEVER-RED-FLAGS-v1`.

Patch fields across the eight competence failures:

- `source_cards`: `6`
- `missing_info_to_collect`: `7`
- `next_observations_to_collect`: `7`
- `candidate_protocol_pathways`: `1`

Diagnosis:

The main v7 data need is not broad medical reasoning. It is harness-native closure behavior:

- If the answer uses SBAR handoff structure, cite `REFERRAL-SBAR-v1`.
- If the answer emits safety boundaries, forbidden-action constraints, or protocol-only disclaimers, cite `SAFETY-BOUNDARIES-v1`.
- If multiple clinical protocol cards are relevant, do not let the support cards disappear.
- If retrieval includes distractor protocol cards, cite the mandatory support cards without over-citing irrelevant clinical cards.
- Preserve v6's model-owned observation fields while improving source-card closure in the same output.

## Existing Corpora

Canonical local files:

- `data/finetune/figment_sft_v3.jsonl`: `3000` rows.
- `data/finetune/figment_sft_v4.jsonl`: `1650` rows.
- `data/finetune/figment_sft_v5.jsonl`: `1300` rows.
- `data/finetune/figment_sft_v6_delta.jsonl`: `1430` rows.
- `data/finetune/figment_sft_v6_replay.jsonl`: `570` rows.
- `data/finetune/figment_sft_v6.jsonl`: `2000` rows.

V6 merged corpus shape:

- `1430` v6 delta rows.
- `570` audited historical replay rows.
- Replay source counts:
  - `figment_sft_v3`: `330`
  - `figment_sft_v4`: `120`
  - `figment_sft_v5`: `120`
- Task type counts:
  - `navigator_full`: `1180`
  - `focused_repair`: `820`
- V6 verifier result: `2000` rows, `1413` case specs, `0` issues.

V6 category counts:

- `required_observation_ownership`: `879`
- `observation_correction`: `218`
- `v6_preservation`: `83`
- `focused_repair:missing_observations`: `250`
- `focused_repair:handoff_note_sbar`: `233`
- `focused_repair:citations_and_pathways`: `163`
- `focused_repair:protocol_urgency`: `87`
- `focused_repair:schema`: `87`

## Reuse Decision

Do not fully regenerate v7 from scratch.

Reuse v6 heavily because v6 is the first adapter that made model-owned harness behavior plausible. Blindly adding older corpora would risk reintroducing v5's failure mode, where the output passed only because deterministic scaffolding repaired hundreds of fields.

Safe reuse:

- Reuse all `570` rows from `data/finetune/figment_sft_v6_replay.jsonl`.
- Reuse all `1430` rows from `data/finetune/figment_sft_v6_delta.jsonl`.
- Prefer v6 delta rows that already passed source-card, observation, SBAR, urgency, schema, and forbidden-behavior checks.
- Keep category balance so v7 does not overfit to source cards and forget required-observation ownership.

Replay audit update:

- `scripts/build_v7_replay_corpus.py` selected `2000` reusable rows.
- Selected by source bucket:
  - `figment_sft_v6_delta`: `1430`
  - `figment_sft_v6_replay`: `570`
- Selected by task type:
  - `navigator_full`: `1180`
  - `focused_repair`: `820`
- Selected by category:
  - `required_observation_ownership`: `879`
  - `observation_correction`: `218`
  - `v6_preservation`: `83`
  - `focused_repair:missing_observations`: `250`
  - `focused_repair:handoff_note_sbar`: `233`
  - `focused_repair:citations_and_pathways`: `163`
  - `focused_repair:protocol_urgency`: `87`
  - `focused_repair:schema`: `87`

Interpretation:

The plan no longer needs `1200` newly generated rows. All v6 anchor rows pass the v7 replay gate, so v7 should preserve the full v6 behavior base and add a smaller, sharper source-card closure delta.

Selective historical reuse:

- Do not directly append v3, v4, or v5 rows beyond the already audited v6 replay rows.
- If more historical diversity is needed, run the v7 replay selector against v3-v6 and select only rows passing the v7 policy.
- Rejected v3-v5 rows may be used as teacher rewrite seeds, not as positive assistant targets.

Hard non-reuse:

- Do not train on exact `field_workflow_holdout_v1` examples as assistant targets.
- Do not use rows where `source_cards` were correct only after deterministic patching.
- Do not use rows with harness-owned metadata in responder observation fields.
- Do not use rows with duplicated long `missing_info_to_collect` and `next_observations_to_collect`.
- Do not use rows that teach arbitrary over-citation of every retrieved card.

## V7 Target Corpus Shape

Target total: `2800` rows.

Reused anchor rows: `2000`.

- `570` rows from `data/finetune/figment_sft_v6_replay.jsonl`.
- `1430` rows from `data/finetune/figment_sft_v6_delta.jsonl`.

New v7 delta rows: `800`.

- `240` `source_card_closure` navigator rows.
- `160` `focused_repair:source_card_closure` rows.
- `140` `observation_source_joint` navigator rows.
- `100` `distractor_card_resistance` navigator rows.
- `80` `sbar_source_coupling` navigator rows.
- `50` `source_card_negative_correction` focused-repair rows.
- `30` `observation_patch_repair` focused-repair rows.

Final v7 task type target:

- `navigator_full`: about `1740` rows.
- `focused_repair`: about `1060` rows.

Final v7 source target:

- `v6_delta_reuse`: `1430`
- `v6_replay_reuse`: `570`
- `v7_delta`: `800`

## New Data Category Definitions

### `source_card_closure`

Purpose: Teach the model to include mandatory support cards in `source_cards` when the output relies on their content.

Each row must include:

- at least one clinical target card;
- `SAFETY-BOUNDARIES-v1` whenever `safety_boundary` or `do_not_do` uses protocol-only, no-orders, no-diagnosis, no-treatment, or local-protocol language;
- `REFERRAL-SBAR-v1` whenever `handoff_note_sbar` is present and actionable;
- at least one multi-card scenario where clinical cards compete for attention;
- complete source-card list in the assistant output before any deterministic repair.

Oversample source-card combinations matching the v6 misses:

- `PREG-DANGER-SIGNS-v1` plus `CHEST-PAIN-ESCALATION-v1` plus `FEVER-RED-FLAGS-v1` plus `SAFETY-BOUNDARIES-v1` plus `REFERRAL-SBAR-v1`.
- The same clinical triad plus only one support-card distractor, requiring the teacher to add the missing support card.
- Clinical card plus `SAFETY-BOUNDARIES-v1` plus `REFERRAL-SBAR-v1`, with irrelevant retrieved distractors excluded.

### `focused_repair:source_card_closure`

Purpose: Teach minimal repair behavior for source-card exactness without requiring a full navigator regeneration.

Prompt shape:

- Use the same focused-repair path as `focused_repair:citations_and_pathways`.
- Input includes a flawed model output whose `source_cards` omit one or both support cards.
- Assistant target returns only the repaired fields requested by the focused-repair prompt.

Each accepted row must:

- add missing mandatory source cards;
- preserve valid existing clinical source cards;
- avoid adding irrelevant retrieved cards;
- preserve candidate pathway card IDs when they are already correct;
- avoid changing urgency, handoff, observation, or safety text unless the focused-repair scope asks for it.

### `observation_source_joint`

Purpose: Preserve v6's observation gains while teaching source-card closure in the same full output.

Each row must require:

- valid `selected_required_observation_ids`;
- responder-facing text for every selected ID;
- non-identical `missing_info_to_collect` and `next_observations_to_collect`;
- complete `source_cards` including relevant clinical and support cards;
- SBAR handoff that cites or depends on `REFERRAL-SBAR-v1`;
- safety boundary that cites or depends on `SAFETY-BOUNDARIES-v1`.

### `distractor_card_resistance`

Purpose: Prevent v7 from solving closure by over-citing every retrieved card.

Each row must include:

- retrieved cards containing at least one irrelevant clinical protocol;
- expected source cards that include the target clinical card and mandatory support cards;
- explicit exclusion of irrelevant retrieved card IDs from `source_cards`;
- normal final validation behavior.

The verifier must reject rows where the teacher adds every retrieved card to `source_cards` without justification.

### `sbar_source_coupling`

Purpose: Make the SBAR card load-bearing when the answer includes SBAR structure.

Each row must include:

- an SBAR handoff with situation, background, objective assessment observations, and request;
- `REFERRAL-SBAR-v1` in `source_cards`;
- no unsupported facts in the SBAR assessment;
- no treatment orders, dosing, or autonomous disposition.

### `source_card_negative_correction`

Purpose: Use bad source-card outputs as correction seeds.

Input flaws to seed:

- missing `SAFETY-BOUNDARIES-v1`;
- missing `REFERRAL-SBAR-v1`;
- missing both support cards;
- over-citing unrelated clinical cards;
- replacing the target clinical card with a support card;
- placing source-card IDs in `missing_info_to_collect` or `next_observations_to_collect`.

Assistant target:

- a corrected focused-repair JSON answer that fixes source-card fields only.

### `observation_patch_repair`

Purpose: Keep pressure on the remaining v6 observation patch cases without rebuilding the whole corpus around observations.

Input flaws to seed:

- missing selected required-observation IDs;
- selected IDs not visible in observation text;
- duplicate long missing and next observation lists;
- generic filler such as `monitor closely` as a standalone observation;
- known observations repeated as missing.

Assistant target:

- corrected observation fields in the exact focused-repair prompt shape.

### `v7_preservation`

Purpose: Guard against regressions in the behavior already working in v6.

Rows should cover:

- red-flag urgency floors;
- protocol-only safety boundaries;
- no diagnosis or treatment instructions;
- noisy/radio-style intake;
- rural clinic and disaster first-response constraints;
- short SBAR handoffs with objective observations only;
- target clinical card retained in both candidate pathways and source cards.

## V7 Gold Output Policy

For full navigator rows, accepted assistant output must:

- emit complete navigator JSON in the current Figment shape;
- include valid `candidate_protocol_pathways`;
- include complete `source_cards` with all cards used by the output;
- include `SAFETY-BOUNDARIES-v1` when safety boundary or forbidden-action content is present;
- include `REFERRAL-SBAR-v1` when SBAR handoff structure is present;
- include the target protocol card in `source_cards`;
- include any fired deterministic rule card in `source_cards`;
- optionally include trace-only `selected_required_observation_ids`;
- select required observation IDs only from `required_observation_targets`;
- include every metadata-required ID listed in `must_include_selected_required_observation_ids`;
- express every selected required-observation ID as recognizable responder-facing text;
- keep `next_observations_to_collect` as a prioritized subset or near-subset of `missing_info_to_collect`, not a full copy;
- preserve protocol-only, no-treatment, no-diagnosis, and no-autonomous-disposition boundaries.

For focused repair rows, accepted assistant output must:

- return only the fields requested by the focused-repair prompt;
- repair the targeted failure;
- preserve valid existing fields;
- avoid expanding into a full navigator answer;
- avoid visible reasoning tags, teacher notes, or commentary.

## V7 Replay Policy

Use the v6 replay policy as the base and add source-card closure checks.

Hard rejects:

- Any row failing the existing v6 policy in `scripts/build_v6_replay_corpus.py`.
- Any row with `metadata.expected_label_score.all_expected_labels_passed == false`.
- Any row where `metadata.validation_result.passed == false`.
- Any row where source-card provenance says deterministic fallback authored `source_cards`.
- Any full navigator row with `handoff_note_sbar` present and `REFERRAL-SBAR-v1` absent from `source_cards`.
- Any full navigator row with safety-boundary text present and `SAFETY-BOUNDARIES-v1` absent from `source_cards`.
- Any full navigator row where the target protocol card is absent from `source_cards`.
- Any row that adds all retrieved cards when irrelevant distractors are marked in metadata.
- Any row overlapping locked eval signatures from:
  - `data/eval/field_workflow_holdout_v1.jsonl`
  - `data/eval/adversarial_strict_cases.jsonl`
  - `data/eval/comprehensive_hosted_cases.jsonl`
  - `data/eval/initial_handwritten_cases.jsonl`

Soft preferences:

- Prefer v6 rows where all scored fields were `model_raw`.
- Prefer rows with source-card lists of length `3` to `5`.
- Prefer multi-card cases with one target clinical card plus support cards.
- Prefer concise observation fields and short SBAR handoffs.
- Prefer rows with no deterministic patches in their originating eval metadata.

## Teacher Generation Strategy

Teacher model:

- Primary: `nvidia/nemotron-3-ultra-550b-a55b:free` through the existing OpenRouter endpoint.
- Alternate: `nvidia/nemotron-3-ultra-550b-a55b` through the NVIDIA-compatible endpoint if quota and reliability are healthy.

Teacher prompt requirements:

- Build prompts through the existing Figment case/preparation path, not a generic clinical note format.
- Include the exact current navigator JSON schema.
- Include retrieved card IDs and card summaries.
- Include required-observation targets for full navigator rows.
- Include closure rules for `SAFETY-BOUNDARIES-v1` and `REFERRAL-SBAR-v1`.
- Include distractor-card instructions where relevant.
- Tell the teacher to produce only JSON.

Generation must use near-neighbor variants of failures, not copied holdout cases.

Variant knobs:

- rural clinic versus disaster first-response setting;
- adult, pediatric, pregnant, fever, chest pain, respiratory distress, stroke, wound infection, and altered mental status presentations;
- partial vitals, noisy intake, radio-style handoff, missing transport availability, language uncertainty, and power/network constraints;
- retrieved card order permutations;
- support cards placed first, middle, or last in retrieval context;
- irrelevant clinical distractors included but not cited.

## File Map

Create:

- `docs/local_4b_v7_training_corpora_plan.md`: this plan.
- `scripts/summarize_v7_corpus_needs.py`: extracts v6 failure IDs, source-card misses, deterministic patch fields, and source-card closure combinations.
- `scripts/build_v7_replay_corpus.py`: selects clean v6 and historical replay rows under the v7 policy.
- `scripts/generate_v7_full_corpus.py`: generates v7 delta rows, verifies them, and prepares Modal train/validation splits.
- `scripts/merge_v7_training_corpus.py`: merges v7 delta rows with v7 replay rows and writes manifest plus Modal split artifacts.
- `tests/test_finetune_v7_data_plan.py`: verifies v7 category counts, replay policy, closure rules, and wrapper defaults.
- `data/finetune/figment_sft_v7_replay.jsonl`: selected reusable rows.
- `data/finetune/figment_sft_v7_replay_manifest.json`: replay selection evidence.
- `data/finetune/figment_sft_v7_delta.jsonl`: newly generated v7 rows.
- `data/finetune/figment_sft_v7_delta_case_specs.jsonl`: case specs for new v7 navigator rows.
- `data/finetune/figment_sft_v7_delta_manifest.json`: v7 delta generation evidence.
- `data/finetune/figment_sft_v7.jsonl`: final merged corpus.
- `data/finetune/figment_sft_v7_case_specs.jsonl`: merged case specs.
- `data/finetune/figment_sft_v7_manifest.json`: final corpus manifest.
- `data/finetune/modal/figment_sft_v7/train.jsonl`: Modal training split.
- `data/finetune/modal/figment_sft_v7/validation.jsonl`: Modal validation split.
- `data/finetune/modal/figment_sft_v7/manifest.json`: Modal split manifest.

Modify:

- `scripts/generate_finetune_data.py`: add v7 failure classes, v7 scoring checks, and source-card closure policy.
- `scripts/augment_finetune_repair_rows.py`: add `source_card_closure` and `observation_patch_repair` repair scopes.
- `scripts/verify_finetune_harness_alignment.py`: add v7 closure verifier checks.
- `scripts/prepare_modal_finetune_dataset.py`: no behavioral change expected; use existing split command and verify group balance.

## Implementation Tasks

### Task 1: Summarize V6 Corpus Needs

**Files:**

- Create: `scripts/summarize_v7_corpus_needs.py`
- Read: `traces/figment_sft_v6_field_workflow_holdout_modal_gpu_20260611_h100_gguf/local_4b_eval.jsonl`
- Write: `traces/figment_sft_v6_field_workflow_holdout_modal_gpu_20260611_h100_gguf/v7_corpus_needs_summary.json`

- [ ] Add a script that reads the v6 eval JSONL and writes:
  - `total_cases`
  - `competence_failure_case_ids`
  - `expected_label_failure_case_ids`
  - `expected_label_check_failures`
  - `missing_source_card_ids_by_case`
  - `deterministic_patch_fields_by_case`
  - `deterministic_patch_field_counts`
  - `actual_source_card_sets_for_failures`

- [ ] Run:

```bash
PYTHONPATH=. .venv/bin/python scripts/summarize_v7_corpus_needs.py \
  --eval-jsonl traces/figment_sft_v6_field_workflow_holdout_modal_gpu_20260611_h100_gguf/local_4b_eval.jsonl \
  --output traces/figment_sft_v6_field_workflow_holdout_modal_gpu_20260611_h100_gguf/v7_corpus_needs_summary.json
```

Expected:

- `total_cases` is `150`.
- `competence_failure_case_ids` has `8` items.
- `expected_label_failure_case_ids` has `4` items.
- `missing_source_card_ids_by_case` includes `SAFETY-BOUNDARIES-v1` and `REFERRAL-SBAR-v1`.
- `deterministic_patch_field_counts.source_cards` is `6`.

### Task 2: Add V7 Source-Card Closure Policy

**Files:**

- Modify: `scripts/generate_finetune_data.py`
- Modify: `scripts/verify_finetune_harness_alignment.py`
- Test: `tests/test_finetune_v7_data_plan.py`

- [ ] Add a helper in `scripts/generate_finetune_data.py`:

```python
def v7_source_card_closure_issues(output: dict, *, target_protocol_card_id: str | None = None) -> list[str]:
    issues: list[str] = []
    source_cards = {str(card_id) for card_id in output.get("source_cards") or []}
    if target_protocol_card_id and target_protocol_card_id not in source_cards:
        issues.append(f"missing_target_source_card:{target_protocol_card_id}")
    if output.get("handoff_note_sbar") and "REFERRAL-SBAR-v1" not in source_cards:
        issues.append("missing_referral_sbar_source_card")
    safety_text = json.dumps(
        {
            "safety_boundary": output.get("safety_boundary"),
            "do_not_do": output.get("do_not_do"),
            "responder_plain_language_script": output.get("responder_plain_language_script"),
        },
        sort_keys=True,
    ).lower()
    safety_terms = (
        "local protocol",
        "do not diagnose",
        "do not provide clinical orders",
        "do not provide treatment instructions",
        "safety boundary",
    )
    if any(term in safety_text for term in safety_terms) and "SAFETY-BOUNDARIES-v1" not in source_cards:
        issues.append("missing_safety_boundaries_source_card")
    return issues
```

- [ ] Wire the helper into the v7 candidate scorer so v7 rows are rejected on any closure issue.

- [ ] Add verifier checks that report issue types:
  - `v7_missing_referral_sbar_source_card`
  - `v7_missing_safety_boundaries_source_card`
  - `v7_missing_target_source_card`

- [ ] Add tests that construct one row per failure type and assert the verifier rejects them.

Run:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_finetune_v7_data_plan.py tests/test_finetune_v6_data_plan.py -q
```

Expected:

- v7 tests pass.
- v6 tests still pass.

### Task 3: Build V7 Replay Selector

**Files:**

- Create: `scripts/build_v7_replay_corpus.py`
- Test: `tests/test_finetune_v7_data_plan.py`
- Write: `data/finetune/figment_sft_v7_replay.jsonl`
- Write: `data/finetune/figment_sft_v7_replay_manifest.json`

- [ ] Start from `scripts/build_v6_replay_corpus.py`.

- [ ] Set default inputs:

```python
DEFAULT_INPUTS = [
    Path("data/finetune/figment_sft_v6_delta.jsonl"),
    Path("data/finetune/figment_sft_v6_replay.jsonl"),
    Path("data/finetune/figment_sft_v5.jsonl"),
    Path("data/finetune/figment_sft_v4.jsonl"),
    Path("data/finetune/figment_sft_v3.jsonl"),
]
```

- [ ] Set default targets:

```python
DEFAULT_TARGETS = {
    "figment_sft_v6_delta": 1430,
    "figment_sft_v6_replay": 570,
    "figment_sft_v5": 0,
    "figment_sft_v4": 0,
    "figment_sft_v3": 0,
}
```

- [ ] Apply v6 replay policy first, then apply v7 closure policy.

- [ ] Annotate selected rows with:

```json
{
  "v7_replay_audit": {
    "source_dataset_version": "figment_sft_v6_delta",
    "policy_version": 1,
    "accepted": true
  }
}
```

- [ ] Run:

```bash
PYTHONPATH=. .venv/bin/python scripts/build_v7_replay_corpus.py
```

Expected:

- `data/finetune/figment_sft_v7_replay.jsonl` has `2000` rows.
- Manifest `selected_by_source_bucket` includes `figment_sft_v6_delta: 1430` and `figment_sft_v6_replay: 570`.
- Manifest includes nonzero rejected counts for v7 closure issues if any candidate rows fail closure.

### Task 4: Add V7 Generation Categories

**Files:**

- Modify: `scripts/generate_finetune_data.py`
- Modify: `scripts/augment_finetune_repair_rows.py`
- Test: `tests/test_finetune_v7_data_plan.py`

- [ ] Add v7 navigator counts:

```python
V7_NAVIGATOR_COUNTS = {
    "source_card_closure": 240,
    "observation_source_joint": 140,
    "distractor_card_resistance": 100,
    "sbar_source_coupling": 80,
}
```

- [ ] Add v7 repair counts:

```python
V7_REPAIR_COUNTS = {
    "source_card_closure": 160,
    "source_card_negative_correction": 50,
    "observation_patch_repair": 30,
}
```

- [ ] Add `_failure_class_for_index(..., dataset_version="figment_sft_v7_delta")` coverage so the first `20` rows interleave all v7 navigator categories.

- [ ] Add repair scope scheduling so `figment_sft_v7_delta` produces:

```python
{
    "source_card_closure": 160,
    "source_card_negative_correction": 50,
    "observation_patch_repair": 30,
}
```

- [ ] Add tests asserting the category counts exactly match the constants.

Run:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_finetune_v7_data_plan.py tests/test_finetune_v6_data_plan.py -q
```

Expected:

- v7 category-count tests pass.
- v6 category-count tests still pass.

### Task 5: Add V7 Full-Corpus Wrapper

**Files:**

- Create: `scripts/generate_v7_full_corpus.py`
- Test: `tests/test_finetune_v7_data_plan.py`

- [ ] Create a wrapper patterned after `scripts/generate_v6_full_corpus.py`.

- [ ] Pin defaults:

```python
DEFAULT_OUTPUT_VERSION = "figment_sft_v7_delta"
DEFAULT_TEACHER_MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_NAVIGATOR_COUNT = 560
DEFAULT_REPAIR_COUNT = 240
DEFAULT_BASE_START_INDEX = "80000"
DEFAULT_SHARD_PREFIX = "data/finetune/shards/figment_sft_v7_delta_full_shard"
```

- [ ] Ensure the default output paths are:

```text
data/finetune/figment_sft_v7_delta.jsonl
data/finetune/figment_sft_v7_delta_case_specs.jsonl
data/finetune/figment_sft_v7_delta_manifest.json
data/finetune/modal/figment_sft_v7_delta
```

- [ ] Add a wrapper test that calls:

```python
args = build_corpus_args(["--navigator-count", "5", "--repair-count", "3", "--dry-run"])
```

and asserts:

- dataset version is `figment_sft_v7_delta`;
- teacher model is `nvidia/nemotron-3-ultra-550b-a55b:free`;
- output path is `data/finetune/figment_sft_v7_delta.jsonl`;
- dry-run flag is preserved.

Run:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_finetune_v7_data_plan.py -q
```

Expected:

- wrapper defaults are pinned by tests.

### Task 6: Smoke Generate V7 Delta

**Files:**

- Write: `/tmp/figment_v7_smoke/figment_sft_v7_delta.jsonl`
- Write: `/tmp/figment_v7_smoke/figment_sft_v7_delta_case_specs.jsonl`
- Write: `/tmp/figment_v7_smoke/figment_sft_v7_delta_manifest.json`

- [ ] Run a deterministic dry-run smoke:

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_v7_full_corpus.py \
  --navigator-count 8 \
  --repair-count 4 \
  --rows-per-shard 2 \
  --parallelism 1 \
  --base-start-index 88000 \
  --shard-prefix /tmp/figment_v7_smoke/shard \
  --output /tmp/figment_v7_smoke/figment_sft_v7_delta.jsonl \
  --case-specs /tmp/figment_v7_smoke/figment_sft_v7_delta_case_specs.jsonl \
  --manifest /tmp/figment_v7_smoke/figment_sft_v7_delta_manifest.json \
  --modal-output-dir /tmp/figment_v7_smoke/modal \
  --dry-run
```

Expected:

- `12` total rows.
- At least one row from each v7 navigator category appears in the manifest or smoke distribution.
- Harness verifier reports `issue_count=0`.

- [ ] Run a real-teacher smoke:

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_v7_full_corpus.py \
  --navigator-count 8 \
  --repair-count 4 \
  --rows-per-shard 2 \
  --parallelism 1 \
  --teacher-error-retries 3 \
  --teacher-error-sleep-seconds 10 \
  --base-start-index 88100 \
  --shard-prefix /tmp/figment_v7_teacher_smoke/shard \
  --output /tmp/figment_v7_teacher_smoke/figment_sft_v7_delta.jsonl \
  --case-specs /tmp/figment_v7_teacher_smoke/figment_sft_v7_delta_case_specs.jsonl \
  --manifest /tmp/figment_v7_teacher_smoke/figment_sft_v7_delta_manifest.json \
  --modal-output-dir /tmp/figment_v7_teacher_smoke/modal \
  --log-rejections
```

Expected:

- `12` accepted rows.
- Verifier reports `issue_count=0`.
- Teacher rejection reasons, if any, are source-card closure, JSON parsing, or policy rejections recorded in the manifest.

### Task 7: Generate Full V7 Delta

**Files:**

- Write: `data/finetune/figment_sft_v7_delta.jsonl`
- Write: `data/finetune/figment_sft_v7_delta_case_specs.jsonl`
- Write: `data/finetune/figment_sft_v7_delta_manifest.json`
- Write: `data/finetune/modal/figment_sft_v7_delta/train.jsonl`
- Write: `data/finetune/modal/figment_sft_v7_delta/validation.jsonl`
- Write: `data/finetune/modal/figment_sft_v7_delta/manifest.json`

- [ ] Run:

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_v7_full_corpus.py \
  --parallelism 4 \
  --teacher-error-retries 3 \
  --teacher-error-sleep-seconds 10 \
  --log-rejections
```

Expected:

- `data/finetune/figment_sft_v7_delta.jsonl` has `800` rows.
- `data/finetune/figment_sft_v7_delta_case_specs.jsonl` has case specs for all navigator rows.
- Manifest category counts match the v7 target.
- Harness verifier reports `issue_count=0`.

### Task 8: Merge V7 Corpus

**Files:**

- Create: `scripts/merge_v7_training_corpus.py`
- Read: `data/finetune/figment_sft_v7_delta.jsonl`
- Read: `data/finetune/figment_sft_v7_replay.jsonl`
- Write: `data/finetune/figment_sft_v7.jsonl`
- Write: `data/finetune/figment_sft_v7_case_specs.jsonl`
- Write: `data/finetune/figment_sft_v7_manifest.json`
- Write: `data/finetune/modal/figment_sft_v7/train.jsonl`
- Write: `data/finetune/modal/figment_sft_v7/validation.jsonl`
- Write: `data/finetune/modal/figment_sft_v7/manifest.json`

- [ ] Create merge logic patterned after `scripts/merge_v6_training_corpus.py`.

- [ ] Set defaults:

```python
DEFAULT_DELTA = Path("data/finetune/figment_sft_v7_delta.jsonl")
DEFAULT_DELTA_CASE_SPECS = Path("data/finetune/figment_sft_v7_delta_case_specs.jsonl")
DEFAULT_REPLAY = Path("data/finetune/figment_sft_v7_replay.jsonl")
DEFAULT_OUTPUT = Path("data/finetune/figment_sft_v7.jsonl")
DEFAULT_CASE_SPECS = Path("data/finetune/figment_sft_v7_case_specs.jsonl")
DEFAULT_MANIFEST = Path("data/finetune/figment_sft_v7_manifest.json")
DEFAULT_MODAL_DIR = Path("data/finetune/modal/figment_sft_v7")
```

- [ ] Run:

```bash
PYTHONPATH=. .venv/bin/python scripts/merge_v7_training_corpus.py
```

Expected:

- `data/finetune/figment_sft_v7.jsonl` has `2800` rows.
- Manifest records `delta_rows=800`.
- Manifest records `replay_rows=2000`.
- Harness verifier reports `issue_count=0`.
- Modal split has `2520` train rows and `280` validation rows.

### Task 9: Final Verification

**Files:**

- Read: `data/finetune/figment_sft_v7.jsonl`
- Read: `data/finetune/figment_sft_v7_case_specs.jsonl`
- Read: `data/finetune/figment_sft_v7_manifest.json`
- Read: `data/finetune/modal/figment_sft_v7/manifest.json`

- [ ] Count rows:

```bash
wc -l data/finetune/figment_sft_v7.jsonl \
  data/finetune/figment_sft_v7_delta.jsonl \
  data/finetune/figment_sft_v7_replay.jsonl \
  data/finetune/modal/figment_sft_v7/train.jsonl \
  data/finetune/modal/figment_sft_v7/validation.jsonl
```

Expected:

- `2800 data/finetune/figment_sft_v7.jsonl`
- `800 data/finetune/figment_sft_v7_delta.jsonl`
- `2000 data/finetune/figment_sft_v7_replay.jsonl`
- `2520 data/finetune/modal/figment_sft_v7/train.jsonl`
- `280 data/finetune/modal/figment_sft_v7/validation.jsonl`

- [ ] Run verifier:

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_finetune_harness_alignment.py \
  --dataset data/finetune/figment_sft_v7.jsonl \
  --case-specs data/finetune/figment_sft_v7_case_specs.jsonl
```

Expected:

- `passed` is `true`.
- `issue_count` is `0`.
- `rows` is `2800`.

- [ ] Run tests:

```bash
PYTHONPATH=. .venv/bin/pytest \
  tests/test_finetune_v7_data_plan.py \
  tests/test_finetune_v6_data_plan.py \
  tests/test_v6_replay_selection.py \
  tests/test_modal_finetune_prep.py \
  -q
```

Expected:

- all tests pass.

## Training Readiness Gates

Do not launch v7 training until all gates pass:

- `data/finetune/figment_sft_v7.jsonl` exists and has `2800` rows.
- `data/finetune/figment_sft_v7_manifest.json` exists.
- `data/finetune/modal/figment_sft_v7/train.jsonl` has `2520` rows.
- `data/finetune/modal/figment_sft_v7/validation.jsonl` has `280` rows.
- Harness alignment verifier reports `issue_count=0`.
- V7 tests pass.
- The manifest shows nonzero counts for:
  - `source_card_closure`
  - `focused_repair:source_card_closure`
  - `observation_source_joint`
  - `distractor_card_resistance`
  - `sbar_source_coupling`
- The manifest shows the corpus includes all `570` v6 replay rows and all `1430` v6 delta rows.
- No holdout eval case IDs appear as training `case_id` values.

## Post-Training Acceptance Gates

Run v7 against `data/eval/field_workflow_holdout_v1.jsonl` on Modal GPU using the same harness/scoring path as v5 and v6.

V7 should beat v6 if:

- `final_validation_successes == 150`
- `fallback_uses == 0`
- `expected_label_successes == 150`
- `expected_label_check_failures.expected_source_cards_present == 0`
- `competence_successes >= 146`
- `deterministic_patch_count <= 15`
- `field_provenance_by_field.source_cards.deterministic_fallback <= 1`
- `field_provenance_by_field.missing_info_to_collect.deterministic_fallback <= 5`
- `field_provenance_by_field.next_observations_to_collect.deterministic_fallback <= 5`
- `handoff_unsupported_fact_total == 0`

V7 should be rejected or rolled into a v8 data plan if:

- expected-label success stays below `150/150`;
- source-card deterministic fallback remains above `1/150`;
- observation deterministic fallback rises above v6's `7/150` for either observation field;
- final validation fails on any case;
- fallback uses become nonzero;
- source-card closure improves only by over-citing irrelevant retrieved cards.

## Notes For Training

Recommended training posture:

- Start from the v6 adapter, not from base, because v6 already internalized most of the harness behavior.
- Use H100 for the full run because the user has already chosen faster Modal iteration.
- Keep the learning rate conservative relative to v6 if the trainer supports it, because v7 is a targeted continuation rather than a broad skill rebuild.
- After training, upload the merged BF16 artifact to the existing Hugging Face archive path for v7.
- Evaluate on Modal as a detached batch job that exits after writing artifacts.

Suggested output names:

- Dataset version: `figment_sft_v7`
- Adapter output name: `figment-sft-v7-lora`
- Merged output name: `figment-sft-v7-lora-merged-bf16`
- Modal checkpoint path: `figment-checkpoints:/figment_sft_v7/figment-sft-v7-lora`
- Modal merged path: `figment-checkpoints:/figment_sft_v7/figment-sft-v7-lora-merged-bf16`
- Eval result name: `figment_sft_v7_field_workflow_holdout_modal_gpu_20260611_h100_gguf`

## Self-Review Checklist

- [ ] The plan reuses v6 heavily instead of regenerating the entire corpus.
- [ ] The plan does not directly train on holdout eval examples.
- [ ] The new data categories target the actual v6 failures.
- [ ] The replay policy blocks v5-style scaffold dependence.
- [ ] The corpus shape preserves observation ownership instead of over-rotating on source cards.
- [ ] The final gates can prove whether v7 improved over v6.
