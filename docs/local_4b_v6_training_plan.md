# Figment Local 4B V6 Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a v6 local 4B adapter that makes required-observation planning model-owned instead of scaffold-authored, while preserving the v5 safety, source-card, SBAR, and validation behavior.

**Architecture:** Reuse v3-v5 rows as filtered replay, then add a focused v6 delta dataset for required-observation ownership inside the exact Figment harness prompt shape. Keep deterministic scaffolding as the product safety layer, but make the model reliably emit valid `selected_required_observation_ids`, `missing_info_to_collect`, and `next_observations_to_collect` before the scaffold patches them.

**Tech Stack:** Python, JSONL SFT corpora, Figment eval harness, OpenRouter/NVIDIA teacher model, Modal LoRA SFT, H100/L40S GPU training, `llama.cpp`/GGUF evaluation, pytest.

---

## Current Evidence

Primary v5 eval trace:

- `traces/figment_sft_v5_field_workflow_holdout_modal_gpu_20260611_h100_gguf/local_4b_eval.jsonl`
- `traces/figment_sft_v5_field_workflow_holdout_modal_gpu_20260611_h100_gguf/eval_summary.json`
- `traces/figment_sft_v5_field_workflow_holdout_modal_gpu_20260611_h100_gguf/modal_eval_manifest.json`

Observed v5 result:

- `150/150` final harness validation.
- `150/150` expected-label success.
- `0` global/canned fallback uses.
- `0` unsupported handoff facts.
- `1648/1950` model-visible fields retained, or `84.5%`.
- Only `2/150` configured-model outputs passed without deterministic field-level patching.
- Deterministic field patches appeared in `148/150` cases, primarily:
  - `missing_info_to_collect`: `148/150`
  - `next_observations_to_collect`: `148/150`
  - `candidate_protocol_pathways`: `6/150`

Diagnosis:

V5 is not broadly unsafe or broadly confused. The product path is strong because the scaffold catches and repairs the weak field. The model-specific gap is narrow and important: the model does not reliably turn `required_observation_targets` into valid, responder-facing `missing_info_to_collect` and `next_observations_to_collect` fields.

## Reuse Decision

Do not fully regenerate v6 from scratch.

Reuse previous datasets as replay because v3-v5 already contain valuable behavior:

- v3 contains broad rural, disaster, low-resource, and field-workflow diversity.
- v4 contains stronger radio/SBAR/handoff usefulness examples.
- v5 contains source-card invariants, selected-observation-id scaffolding, and general regression examples.

But do not blindly append old observation rows. The v5 result shows that the current observation examples were not sharp enough, were underweighted, or taught the wrong distinction between medic-owned observations and harness-owned metadata.

Existing local corpora:

- `data/finetune/figment_sft_v3.jsonl`: `3000` rows.
- `data/finetune/figment_sft_v4.jsonl`: `1650` rows.
- `data/finetune/figment_sft_v5.jsonl`: `1300` rows.
- `data/finetune/figment_sft_v5.jsonl` includes `242` `required_observation_id_selection` rows and `55` `focused_repair:missing_observations` rows.

## Replay Audit Update

The first v6 replay audit changed the corpus shape.

Artifacts:

- `scripts/build_v6_replay_corpus.py`
- `tests/test_v6_replay_selection.py`
- `data/finetune/figment_sft_v6_replay.jsonl`
- `data/finetune/figment_sft_v6_replay_manifest.json`

Audit result:

- `570` direct replay rows passed the v6 cleanliness policy.
- Selected replay rows by source:
  - `figment_sft_v3`: `330`
  - `figment_sft_v4`: `120`
  - `figment_sft_v5`: `120`
- Selected replay rows by category:
  - `focused_repair:handoff_note_sbar`: `233`
  - `focused_repair:citations_and_pathways`: `163`
  - `focused_repair:protocol_urgency`: `87`
  - `focused_repair:schema`: `87`
- Selected replay rows by task type:
  - `focused_repair`: `570`

Interpretation:

The usable replay pool is smaller than planned, and it contains no clean old full-navigator rows. Most old full-navigator rows teach at least one behavior v6 is supposed to stop: duplicated long `missing_info_to_collect` / `next_observations_to_collect` lists, harness metadata inside medic observation fields, or observation-focused rows without clean selected required-observation IDs.

Rejected rows are still useful as negative/correction seeds, but not as positive replay targets. For v6 SFT, only the teacher-rewritten corrected output should be used as the assistant target.

## V6 Dataset Shape

Target total: `2000` rows.

New v6 delta: `1430` rows.

- `900` full navigator rows focused on required-observation ownership.
- `250` focused repair rows for `missing_info_to_collect` and `next_observations_to_collect`.
- `180` contrastive correction rows seeded from rejected old outputs, where the teacher rewrites the output into the v6 shape.
- `100` preservation rows for SBAR, source-card discipline, urgency floors, red flags, noisy intake, and low-resource constraints.

Filtered replay: `570` rows.

- `330` v3 focused-repair replay rows.
- `120` v4 focused-repair replay rows.
- `120` v5 focused-repair replay rows.

Do not force the original `900` replay quota. If a row fails the v6 replay policy, either reject it outright or use it only as a seed for a teacher-generated correction example.

Modal split target:

- `1800` train rows.
- `200` validation rows.
- Preserve category balance in validation so v6 cannot hide observation failure in the train split.

## V6 Gold Output Policy

The teacher output must be aligned to the real harness prompt and schema.

For full navigator rows, each accepted assistant output must:

- emit complete navigator JSON in the current Figment shape;
- optionally emit trace-only `selected_required_observation_ids` for training, knowing the runtime strips it from final user-visible output;
- select required observation IDs only from `required_observation_targets`;
- include every metadata-required ID listed in `must_include_selected_required_observation_ids`;
- express each selected required observation ID as recognizable responder-facing text;
- keep `missing_info_to_collect` as the broader list of still-needed observations;
- keep `next_observations_to_collect` as the prioritized next 3-5 observations, not a copy of every missing item;
- avoid treating harness metadata as medic observations;
- preserve source-card discipline, urgency floors, red flags, SBAR grounding, and forbidden-behavior constraints.

For focused repair rows, each accepted assistant output must:

- return only `missing_info_to_collect` and `next_observations_to_collect`;
- repair validator-style missing-observation failures from the exact `build_focused_repair_prompts(...)` prompt shape;
- reference required observations by ID and display text;
- preserve valid existing clinical workflow content;
- avoid expanding into a full navigator answer.

## V6 Observation Policy

Reject any new or replay row that violates these rules.

Hard rejects:

- `missing_info_to_collect` and `next_observations_to_collect` are identical non-empty lists with more than three items.
- Observation fields include harness-owned metadata phrases such as:
  - `source card IDs`
  - `source protocol card IDs`
  - `retrieved protocol card IDs`
  - `deterministic rule results`
  - `navigator validation result`
  - `confirmed intake status`
  - `manual correction status for audio-derived fields`
- `selected_required_observation_ids` is missing for a v6 full navigator row.
- Any selected required-observation ID is not in the provided `required_observation_targets`.
- Any ID in `must_include_selected_required_observation_ids` is absent from the assistant output.
- A selected required-observation ID has no matching responder-facing text in either observation field.
- Observation text gives diagnosis, medication, dosing, procedure, disposition, or autonomous routing instructions.
- The row overlaps the locked eval signatures from:
  - `data/eval/field_workflow_holdout_v1.jsonl`
  - `data/eval/adversarial_strict_cases.jsonl`
  - `data/eval/comprehensive_hosted_cases.jsonl`
  - `data/eval/initial_handwritten_cases.jsonl`

Soft preferences:

- `next_observations_to_collect` should usually be a prioritized subset of `missing_info_to_collect`.
- Prefer concrete field language: `count respiratory rate`, `measure blood pressure if cuff available`, `confirm bleeding amount`, `check current mental status`.
- Avoid generic filler: `monitor closely`, `collect more information`, `follow up`, `assess patient`.
- Preserve uncertainty explicitly when intake is unclear or conflicting.

## Teacher Generation Strategy

Use the same teacher route as v5 unless the primary NVIDIA endpoint is healthy:

- Preferred teacher if available: `nvidia/nemotron-3-ultra-550b-a55b`
- Working fallback teacher: `nvidia/nemotron-3-ultra-550b-a55b:free` via OpenRouter

Generate new v6 cases as near-neighbor-free variants, not copies of holdout cases.

Each new case spec should include:

- setting,
- responder constraints,
- confirmed intake facts,
- denied or absent symptoms,
- retrieved card set,
- fired deterministic rules,
- urgency floor,
- required observation targets,
- expected selected required-observation IDs,
- expected model-owned observation cue phrases,
- forbidden behavior.

Failure classes to oversample:

- missing required-observation IDs;
- invalid selected required-observation IDs;
- generic observation filler;
- duplicate missing/next observation lists;
- harness metadata incorrectly placed in observation fields;
- unknown observation target omitted from text;
- known observation incorrectly repeated as missing;
- treatment advice disguised as observation collection.

## Implementation Tasks

### Task 1: Summarize V5 Observation Failures

**Files:**

- Read: `traces/figment_sft_v5_field_workflow_holdout_modal_gpu_20260611_h100_gguf/local_4b_eval.jsonl`
- Create: `traces/figment_sft_v5_field_workflow_holdout_modal_gpu_20260611_h100_gguf/v6_observation_failure_summary.json`

- [ ] Count deterministic patches by field.
- [ ] Count missing, invalid, and unused `selected_required_observation_ids`.
- [ ] Extract top required-observation target IDs that were scaffold-filled.
- [ ] Extract bad model phrasings that caused patching.
- [ ] Save a compact JSON summary for v6 corpus generation.

Suggested command:

```bash
PYTHONPATH=. .venv/bin/python scripts/summarize_v6_observation_failures.py \
  --eval-jsonl traces/figment_sft_v5_field_workflow_holdout_modal_gpu_20260611_h100_gguf/local_4b_eval.jsonl \
  --output traces/figment_sft_v5_field_workflow_holdout_modal_gpu_20260611_h100_gguf/v6_observation_failure_summary.json
```

Expected result:

- `total_cases` is `150`.
- `missing_info_to_collect` deterministic patch count is near `148`.
- `next_observations_to_collect` deterministic patch count is near `148`.
- The summary names exact observation target IDs and phrase families to generate against.

### Task 2: Add V6 Observation Filters

**Files:**

- Modify: `scripts/verify_finetune_harness_alignment.py`
- Modify: `scripts/generate_finetune_data.py`
- Test: `tests/test_finetune_v5_data_plan.py`
- Create: `tests/test_finetune_v6_data_plan.py`

- [ ] Add a `uses_v6_observation_policy(dataset_version: str) -> bool` helper.
- [ ] Reject duplicate non-empty `missing_info_to_collect` and `next_observations_to_collect` lists with more than three items.
- [ ] Reject harness-owned metadata cues in observation fields.
- [ ] Reject missing or invalid `selected_required_observation_ids`.
- [ ] Reject rows where selected IDs are not visible as responder-facing text.
- [ ] Add tests for each reject reason.

Suggested verification:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_finetune_v6_data_plan.py tests/test_finetune_v5_data_plan.py -q
```

Expected result:

- v5 tests still pass.
- v6 tests prove every hard reject is enforced.

### Task 3: Build Filtered Replay Corpus

**Files:**

- Create: `scripts/build_v6_replay_corpus.py`
- Read: `data/finetune/figment_sft_v3.jsonl`
- Read: `data/finetune/figment_sft_v4.jsonl`
- Read: `data/finetune/figment_sft_v5.jsonl`
- Create: `data/finetune/figment_sft_v6_replay.jsonl`
- Create: `data/finetune/figment_sft_v6_replay_manifest.json`

- [x] Audit v3-v5 rows with the v6 replay policy.
- [x] Select only rows that avoid duplicate long observation lists and harness metadata in observation fields.
- [x] Preserve only clean direct replay rows instead of filling quota with bad rows.
- [x] Run every candidate through the v6 observation policy.
- [x] Preserve original row metadata with `source_dataset_version`.
- [x] Add `replay_reason` metadata to each retained row.
- [x] Save a manifest with source counts, rejected counts, and SHA256.

Suggested command:

```bash
PYTHONPATH=. .venv/bin/python scripts/build_v6_replay_corpus.py \
  --input data/finetune/figment_sft_v5.jsonl \
  --input data/finetune/figment_sft_v4.jsonl \
  --input data/finetune/figment_sft_v3.jsonl \
  --output data/finetune/figment_sft_v6_replay.jsonl \
  --manifest data/finetune/figment_sft_v6_replay_manifest.json \
  --figment-sft-v5-target 450 \
  --figment-sft-v4-target 300 \
  --figment-sft-v3-target 150
```

Expected result:

- `data/finetune/figment_sft_v6_replay.jsonl` contains `570` clean direct replay rows.
- The manifest reports `0` v6 policy issues among retained rows.
- The manifest records replay shortages rather than filling the planned quota with bad rows.
- Selected rows contain `0` duplicate long missing/next observation lists.
- Selected rows contain `0` harness-metadata cue hits in assistant observation fields.

### Task 4: Generate New V6 Delta Rows

**Files:**

- Modify: `scripts/generate_finetune_data.py`
- Create: `scripts/generate_v6_full_corpus.py`
- Create: `data/finetune/figment_sft_v6_delta.jsonl`
- Create: `data/finetune/figment_sft_v6_delta_case_specs.jsonl`
- Create: `data/finetune/figment_sft_v6_delta_manifest.json`

- [ ] Add v6 failure classes to the case-spec scheduler.
- [ ] Oversample required-observation targets that v5 scaffold-filled.
- [ ] Use rejected old full-navigator rows as negative/correction seeds, not as positive SFT targets.
- [ ] Ask the teacher for full navigator output in the real prompt shape.
- [ ] Ask the teacher for focused repair output in the real repair prompt shape.
- [ ] Ask the teacher to rewrite rejected prior outputs into clean v6 full-navigator targets.
- [ ] Reject rows that fail v6 observation policy.
- [ ] Reject rows that fail existing harness alignment checks.
- [ ] Save accepted rows and case specs with anti-overfit signatures enabled.

Suggested smoke:

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_v6_full_corpus.py \
  --new-delta-count 10 \
  --repair-count 3 \
  --correction-count 2 \
  --parallelism 1 \
  --output /tmp/figment_sft_v6_delta_smoke.jsonl \
  --case-specs /tmp/figment_sft_v6_delta_smoke_case_specs.jsonl \
  --manifest /tmp/figment_sft_v6_delta_smoke_manifest.json
```

Expected smoke result:

- `15` accepted rows.
- `0` verifier issues.
- At least one accepted row for duplicate-list correction.
- At least one accepted row for harness-owned metadata exclusion.

Suggested full generation:

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_v6_full_corpus.py \
  --new-delta-count 1000 \
  --repair-count 250 \
  --correction-count 180 \
  --parallelism 4 \
  --teacher-error-retries 3 \
  --teacher-error-sleep-seconds 10 \
  --output data/finetune/figment_sft_v6_delta.jsonl \
  --case-specs data/finetune/figment_sft_v6_delta_case_specs.jsonl \
  --manifest data/finetune/figment_sft_v6_delta_manifest.json
```

Expected full result:

- `1430` accepted delta rows.
- Delta includes `900` required-observation full-navigator rows, `250` focused missing-observation repair rows, `180` teacher-rewritten correction rows, and `100` preservation rows.
- `0` verifier issues.
- Delta manifest records category counts and rejected row reasons.

### Task 5: Merge, Verify, And Split V6

**Files:**

- Create: `data/finetune/figment_sft_v6.jsonl`
- Create: `data/finetune/figment_sft_v6_manifest.json`
- Create: `data/finetune/modal/figment_sft_v6/train.jsonl`
- Create: `data/finetune/modal/figment_sft_v6/validation.jsonl`
- Create: `data/finetune/modal/figment_sft_v6/manifest.json`

- [ ] Merge `figment_sft_v6_delta.jsonl` and `figment_sft_v6_replay.jsonl`.
- [ ] Shuffle deterministically with a fixed seed.
- [ ] Verify the full merged dataset.
- [ ] Split into `1800` train rows and `200` validation rows.
- [ ] Verify train and validation SHA256 hashes in the Modal manifest.

Suggested commands:

```bash
PYTHONPATH=. .venv/bin/python scripts/merge_v6_training_corpus.py \
  --delta data/finetune/figment_sft_v6_delta.jsonl \
  --replay data/finetune/figment_sft_v6_replay.jsonl \
  --output data/finetune/figment_sft_v6.jsonl \
  --manifest data/finetune/figment_sft_v6_manifest.json \
  --modal-output-dir data/finetune/modal/figment_sft_v6 \
  --train-count 1800 \
  --validation-count 200

PYTHONPATH=. .venv/bin/python scripts/verify_finetune_harness_alignment.py \
  --dataset data/finetune/figment_sft_v6.jsonl \
  --case-specs data/finetune/figment_sft_v6_delta_case_specs.jsonl
```

Expected result:

- Full dataset has `2000` rows.
- Full dataset is `1430` new/corrected delta rows plus `570` clean direct replay rows.
- Modal train split has `1800` rows.
- Modal validation split has `200` rows.
- Verifier reports `issue_count=0`.

### Task 6: Train V6 On Modal

**Files:**

- Use: `modal/finetune_figment_nemotron.py`
- Use: `data/finetune/modal/figment_sft_v6/train.jsonl`
- Use: `data/finetune/modal/figment_sft_v6/validation.jsonl`
- Output: `figment-checkpoints:/figment_sft_v6/figment-sft-v6-lora`

- [ ] Run a short smoke training job.
- [ ] Verify finite train and eval loss.
- [ ] Verify adapter artifacts.
- [ ] Launch full detached training.
- [ ] Prefer continuation from the v5 adapter with a lower learning rate.
- [ ] Keep a fallback option to resume from v4 if v5 continuation shows observation overfitting or safety regression in smoke eval.

Suggested smoke:

```bash
PYTHONPATH=. .venv/bin/modal run modal/finetune_figment_nemotron.py::train \
  --dataset-version figment_sft_v6 \
  --output-name figment-sft-v6-lora-smoke \
  --max-steps 20 \
  --resume-adapter-name figment-sft-v5-lora \
  --resume-adapter-dataset-version figment_sft_v5
```

Suggested full detached run:

```bash
PYTHONPATH=. .venv/bin/modal run --detach modal/finetune_figment_nemotron.py::train \
  --dataset-version figment_sft_v6 \
  --output-name figment-sft-v6-lora \
  --resume-adapter-name figment-sft-v5-lora \
  --resume-adapter-dataset-version figment_sft_v5
```

Expected result:

- Adapter artifacts exist under `/checkpoints/figment_sft_v6/figment-sft-v6-lora`.
- `adapter_model.safetensors`, `adapter_config.json`, tokenizer files, `chat_template.jinja`, and `figment_training_manifest.json` are present.

### Task 7: Merge, Convert, And Evaluate V6

**Files:**

- Use: `modal/eval_figment_nemotron.py`
- Use: `data/eval/field_workflow_holdout_v1.jsonl`
- Output: `traces/figment_sft_v6_field_workflow_holdout_modal_gpu_<timestamp>/`

- [ ] Merge the v6 adapter into BF16 weights.
- [ ] Convert to GGUF if the eval path requires it.
- [ ] Run the full `150`-case `field_workflow_holdout_v1` suite.
- [ ] Save JSONL traces, summaries, route smoke, endpoint metadata, and manifest.
- [ ] Verify result count is exactly `150`.
- [ ] Compare field provenance against v5.

Suggested eval:

```bash
PYTHONPATH=. .venv/bin/modal run modal/eval_figment_nemotron.py::run_batch_eval \
  --dataset-version figment_sft_v6 \
  --model-artifact figment-checkpoints:/figment_sft_v6/figment-sft-v6-lora-merged-bf16 \
  --cases data/eval/field_workflow_holdout_v1.jsonl \
  --output-name figment_sft_v6_field_workflow_holdout
```

Expected result:

- Eval JSONL contains `150` records.
- Summary includes raw, repair, fallback, field-provenance, latency, and trace-hash counts.

## Acceptance Gates

V6 is accepted only if it beats v5 on model ownership without losing safety.

Required:

- `150/150` final validation successes.
- `150/150` trace hashes present.
- `0` global/canned fallback uses.
- `0` unsupported handoff facts.
- `0` invalid selected required-observation IDs.
- `missing_info_to_collect` model-owned in `>=140/150`.
- `next_observations_to_collect` model-owned in `>=140/150`.
- Deterministic patches for `missing_info_to_collect` are `<=10/150`.
- Deterministic patches for `next_observations_to_collect` are `<=10/150`.
- `raw_configured_model_successes >=125/150`.
- No regression in red flags, urgency floors, source-card discipline, or SBAR handoff grounding.

Nice to have:

- Mean latency stays below v5 by avoiding repair calls.
- Field retention improves from v5's `84.5%` to `>=93%`.
- `candidate_protocol_pathways` deterministic patches remain `<=6/150`.

## Stop Conditions

Do not proceed to full training if:

- v6 smoke rows fail harness verification;
- the replay builder cannot produce more than `570` clean rows without weakening filters and the plan still assumes old direct replay can fill the gap;
- teacher output repeatedly treats harness metadata as medic observation text;
- smoke training shows non-finite loss;
- smoke eval regresses safety, source cards, or unsupported handoff facts.

If those happen, fix the v6 data policy or generator first. Do not solve this by broadening the corpus or relaxing the eval.
