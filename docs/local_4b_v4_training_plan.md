# Local 4B V4 Training Plan

Date: 2026-06-10

## Purpose

Train one focused v4 LoRA only after the scaffolding and eval-shape fixes in `docs/local_4b_v4_scaffolding_eval_shape_plan.md` are implemented and rerun.

The v4 goal is not broad assistant quality. The goal is to improve the local 4B model at the model-owned parts of Figment's field workflow:

- radio and runner handoff,
- concise SBAR referral support,
- source-card discipline,
- high-value next observations,
- low-resource constraints,
- safe protocol-navigation language.

The v3 result is good enough to be worth refining, not bad enough to restart from scratch.

## Current Evidence

Primary v3 trace:

- `traces/local_4b_finetuned_v3_field_holdout_sequential_20260610T102450Z/`

Published trace dataset:

- `https://huggingface.co/datasets/ThomsenDrake/figment-eval-traces`
- `local_4b_clean_scored_records`: `350`
- `hosted_omni_scored_records`: `100`
- `scored_eval_records`: `450`
- `useful_trace_records`: `455`

V3 field-workflow holdout:

- `150/150` cases completed
- `107/150` competence successes
- `93/150` raw model successes
- `14/150` focused repair successes
- `2/150` full fallbacks
- `148/150` final validation successes
- `0/150` strict expected-label successes due to `missing_observation_cues_present`

Failure concentration:

- `REFERRAL-SBAR-v1`: `0/27`
- `radio_handoff`: `0/16`
- `sbar_handoff_usefulness`: `0/10`
- `source_card_discipline`: `2/6`
- `rural_clinic_intake`: `33/36`
- `disaster_triage`: `30/32`

Interpretation:

- V3 is strong enough on safety and protocol navigation to keep.
- V3 is not strong enough on the handoff layer that matters to the field workflow.
- The v4 dataset should be narrow and high-signal, not another broad corpus.

## Prerequisite

Do not start the v4 training job until these are complete:

1. Eval cue ownership is split into model-owned, handoff-owned, and harness-owned cues.
2. Deterministic harness evidence is visible outside model-authored missing-observation text.
3. SBAR/radio handoff metrics report concrete failures.
4. V3 is rerun with the updated scoring.
5. The remaining v3 failures are exported as v4 teacher prompts or repair seeds.

This prevents v4 from learning to recite deterministic metadata instead of improving handoff usefulness.

## Training Strategy

Use a targeted continuation from v3 as the primary run.

Primary run:

- Dataset version: `figment_sft_v4`
- Output adapter name: `figment-sft-v4-lora`
- Base model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`
- Starting point: continue from the v3 behavior if the Modal script is extended to load an existing adapter; otherwise train a focused v4 LoRA from the BF16 base with replay rows.
- Method: LoRA SFT, BF16 base, merge back to BF16, convert to GGUF, evaluate locally through llama.cpp.
- Context length: `16384`
- Target GPU: `L40S` first, `A100-80GB` only if the run hits memory or sequence-length failures.

If the current Modal trainer cannot resume from an existing adapter, patch it before v4 or run a fresh LoRA with enough v2/v3 replay to preserve schema behavior.

## Dataset Size And Mix

Target accepted rows: `1200` to `1800`.

Recommended mix:

- `350` to `450` radio handoff rows.
- `300` to `400` SBAR handoff usefulness rows.
- `175` to `250` source-card discipline rows.
- `150` to `225` low-resource constraint rows.
- `125` to `175` missing-observation prioritization rows focused on first-five usefulness, not every cue.
- `100` to `150` focused competence-repair rows from v3 safe-but-weak outputs.
- `100` to `150` clinical-protocol replay rows from high-quality v2/v3 data.
- `75` to `125` hard negative or safety-boundary rows to preserve refusal and no-treatment behavior.

Replay rows should be high quality only:

- validation passed,
- no full fallback,
- no forbidden behavior,
- correct target card,
- correct source-card set,
- strong field provenance,
- no close neighbor of locked eval or holdout rows.

## What To Generate

Every v4 row should match the exact harness prompt and response format. Do not generate generic clinical conversations.

### Full Navigator Rows

Generate full assistant outputs where the model must:

- preserve deterministic red flags,
- keep urgency at or above the deterministic floor,
- cite only retrieved source cards,
- include `REFERRAL-SBAR-v1` when the task is handoff-focused,
- produce compact SBAR fields grounded only in confirmed intake, rules, and retrieved cards,
- prioritize the next observations that would actually help the responder move the case forward.

### Focused Repair Rows

Generate repair rows for safe-but-weak outputs, not only invalid outputs.

Repair scopes:

- `handoff_note_sbar`
- `source_cards`
- `candidate_protocol_pathways`
- `missing_observations`
- `responder_checklist`
- `safety_boundary`

Each repair row should include:

- previous weak output,
- deterministic validation result,
- competence metric failures,
- scope name,
- corrected assistant output or corrected fields,
- provenance metadata saying this is a competence repair.

### Preference Pairs

Only add preference data after the SFT row set exists.

Preferred outputs:

- concise,
- grounded,
- high-value next observations first,
- correct source cards,
- safe SBAR,
- useful to a field medic under radio or paper constraints.

Rejected outputs:

- schema-valid but generic,
- overlong,
- metadata-stuffed,
- unsupported,
- target-card correct but handoff useless,
- observation list repeats the prompt without prioritization.

Preference tuning is optional. Use it only if v4 SFT improves format but still leaves SBAR/radio output operationally weak.

## Teacher Model

Use the existing stronger teacher path:

- Teacher model: `nvidia/nemotron-3-ultra-550b-a55b`
- Preferred endpoint: existing hosted OpenAI-compatible endpoint.
- Fallback endpoint: OpenRouter if needed.
- Secrets: use `.env` locally and Modal secrets remotely. Do not write keys into dataset rows, manifests, traces, or docs.

Teacher instructions should make the field workflow explicit:

- "You are generating training targets for a bounded protocol-navigation harness, not medical advice."
- "The model output must be JSON only and match Figment's current navigator schema."
- "Optimize for a trained field responder who needs faster intake, escalation, and handoff, under low-resource constraints."
- "Do not copy locked eval rows or close paraphrases."
- "Do not add diagnosis, treatment, dosing, discharge, or autonomous routing language."

## Validators

Keep all v3 validators:

- JSON/schema validation,
- known-card validation,
- retrieved-card validation,
- urgency floor,
- red-flag match,
- source-card coverage,
- candidate-pathway coverage,
- forbidden behavior,
- no teacher notes,
- no locked eval or holdout near-neighbor.

Add v4 validators:

- `handoff_readiness_passed`
- `sbar_slot_coverage`
- `sbar_unsupported_fact_count`
- `radio_brevity_ok`
- `first_five_observation_usefulness`
- `source_card_discipline_passed`
- `competence_repair_scope_valid`
- `harness_owned_metadata_not_required_in_model_text`

Reject any row that only wins by stuffing deterministic metadata into prose.

## Modal Work Needed

Patch `modal/finetune_figment_nemotron.py` before v4 if needed:

- expose `learning_rate`,
- expose `lora_r`,
- expose `lora_alpha`,
- expose `lora_dropout`,
- expose `gradient_accumulation_steps`,
- expose `validation_steps`,
- expose `save_steps`,
- optionally support `resume_adapter_name` or `adapter_init_path`.

The current entrypoint accepts `max_steps`, `max_seq_length`, `gpu`, `dataset_version`, `dataset`, and `output_name`, but the lower learning rate and rank controls are hard-coded through `build_train_config`.

Recommended SFT config:

- `max_seq_length`: `16384`
- `lora_r`: `16` first, `32` only if v4 underfits the targeted handoff tasks
- `lora_alpha`: `32` for rank 16, `64` for rank 32
- `lora_dropout`: `0.05`
- `learning_rate`: `2e-5` for continuation from v3, `5e-5` if training fresh from base with replay
- `gradient_accumulation_steps`: `8`
- `validation_fraction`: `0.10`
- `max_steps`: choose from staged row count and tokenized row count, not a fixed old pilot value

## Runbook

1. Implement and rerun the scaffolding/eval-shape plan.
2. Export v3 failures with ownership labels and handoff metrics.
3. Generate v4 candidate specs from those failures and nearby synthetic siblings.
4. Use the teacher to produce JSON-only target outputs.
5. Validate and reject rows until `1200` to `1800` accepted rows remain.
6. Prepare Modal train/validation split:

```bash
.venv/bin/python scripts/prepare_modal_finetune_dataset.py \
  --dataset data/finetune/figment_sft_v4.jsonl \
  --dataset-version figment_sft_v4
```

7. Run a smoke job:

```bash
.venv/bin/modal run modal/finetune_figment_nemotron.py \
  --dataset-version figment_sft_v4 \
  --dataset data/finetune/figment_sft_v4.jsonl \
  --output-name figment-sft-v4-lora-smoke \
  --smoke true \
  --gpu L40S
```

8. Run the full detached job:

```bash
.venv/bin/modal run modal/finetune_figment_nemotron.py \
  --dataset-version figment_sft_v4 \
  --dataset data/finetune/figment_sft_v4.jsonl \
  --output-name figment-sft-v4-lora \
  --max-steps <computed_steps> \
  --gpu L40S \
  --spawn-train
```

9. Merge adapter:

```bash
.venv/bin/modal run modal/finetune_figment_nemotron.py \
  --merge-only \
  --dataset-version figment_sft_v4 \
  --adapter-name figment-sft-v4-lora \
  --merged-name figment-sft-v4-lora-merged-bf16 \
  --gpu L40S
```

10. Pull merged BF16 weights, convert to GGUF, serve locally through `llama-server`, smoke route, and run:

- locked 50-case regression,
- field-workflow holdout with updated scoring,
- old v3 scoring for comparison only.

## Acceptance Gates

Primary gate:

- field-workflow holdout competence at least `125/150`.
- `REFERRAL-SBAR-v1` at least `20/27`.
- `radio_handoff` at least `12/16`.
- `sbar_handoff_usefulness` at least `8/10`.
- `source_card_discipline` at least `5/6`.

Safety gates:

- `final_validation_successes` at least `148/150`.
- `forbidden_behavior_absent` remains `150/150`.
- `red_flags_match` remains `150/150`.
- `min_urgency_met` remains `150/150`.
- full fallbacks no more than `2/150`.

Regression gate:

- locked 50-case competence must not drop below the v2 result of `33/50` unless the miss is only a newly separated non-safety cue metric.
- no increase in unsafe or unsupported clinical language.
- no loss of local/no-cloud route proof.

Operational gate:

- local GGUF hash recorded,
- `/v1/models` metadata recorded,
- llama.cpp run uses `n_parallel=1` or otherwise proves enough KV context for the prompt length,
- eval manifest has all trace hashes,
- invalid parallel/runtime records are excluded from scored reporting.

## Ship Decision

Train v4 if the scaffolding rerun still shows a real model-owned SBAR/radio gap.

Ship v3 plus scaffolding if:

- scaffolding alone gets field holdout competence close to the target,
- v4 regresses safety or validation,
- v4 improves scorer numbers by stuffing metadata rather than improving handoff usefulness,
- the remaining failures are mostly evaluator wording artifacts.

With roughly 8.5 days left, the recommended path is one focused v4 swing, not an open-ended training campaign.
