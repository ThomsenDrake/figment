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

Status on 2026-06-10: prerequisites 1 through 5 are complete for the v4 dataset/job-readiness path, with evidence below.

## Implementation Evidence

Current v4-readiness work on 2026-06-10:

- Updated scoring now splits model-owned, handoff-owned, and harness-owned evidence cues.
- Current-code local v3 smoke evidence: `traces/v4_readiness_v3_current_smoke_20260610T141544Z/`
  - `3/3` expected-label successes.
  - `3/3` handoff-readiness successes.
  - No fallback use and no context/KV/HTTP-500 runtime errors.
- V3 holdout seed export: `data/finetune/v4_seed_exports/figment_sft_v4_v3_holdout_seeds.jsonl`
  - `8` model/handoff/source failure seeds.
  - `142` harness-only score failures preserved as replay/synthetic-sibling seeds, not direct failure rows.
  - Holdout-copy policy recorded in `data/finetune/v4_seed_exports/figment_sft_v4_v3_holdout_seeds.manifest.json`.
- V4 corpus wrapper: `scripts/generate_v4_full_corpus.py`
  - Defaults: `1500` navigator rows plus `150` focused repair rows.
  - Dataset/output paths default to `figment_sft_v4`.
  - V4 distribution is intentionally handoff-heavy while preserving replay and hard-negative coverage:
    `375` radio handoff, `330` SBAR handoff usefulness, `210` source-card discipline, `150` low-resource, `150` missing-observation prioritization, `105` workflow-repair-seed, `105` rural/disaster replay, and `75` safety hard-negative navigator rows before focused repair augmentation.
- Teacher-backed v4 smoke corpus: `data/finetune/figment_sft_v4_smoke.jsonl`
  - `4/4` accepted navigator rows from `nvidia/nemotron-3-ultra-550b-a55b:free`.
  - `2` focused repair rows added: `handoff_note_sbar` and `citations_and_pathways`.
  - Harness verification passed with `0` issues.
  - Modal smoke split prepared at `data/finetune/modal/figment_sft_v4_smoke/`.
- Full teacher-backed v4 corpus: `data/finetune/figment_sft_v4.jsonl`
  - `1500` navigator rows plus `150` focused repair rows, `1650` total.
  - `1500` case specs at `data/finetune/figment_sft_v4_case_specs.jsonl`.
  - Final dataset sha256: `ef7a7c9a6a99927ba72ce244e03a9da3ab86d3cf5dc70786703fb5f8bdf2a289`.
  - Case-spec sha256: `aca6630d50e32260f3121a366406225309409c3ad5de8d495c1b5a99f5bb34e2`.
  - Standalone harness verification passed with `0` issues:
    `.venv/bin/python scripts/verify_finetune_harness_alignment.py --dataset data/finetune/figment_sft_v4.jsonl --case-specs data/finetune/figment_sft_v4_case_specs.jsonl`.
  - Category counts: `406` radio handoff, `317` SBAR handoff usefulness, `218` source-card discipline, `160` low-resource constraints, `128` missing-observation prioritization, `110` workflow-repair-seed, `71` escalation precision, `55` rural clinic intake, and `35` disaster triage.
  - Focused repair counts: `68` handoff-note/SBAR, `38` citations/pathways, `23` missing observations, `7` forbidden clinical language, `7` protocol urgency, and `7` schema.
  - Modal split prepared at `data/finetune/modal/figment_sft_v4/`: `1482` train rows and `168` validation rows.
  - Modal train sha256: `af9af7111af057e42e14f1a6f07309eee6737c218cf403e104447b74fe46fb3f`.
  - Modal validation sha256: `6a2859047ae78479b97ab797644a6646df79d8b4ee920ed21ce1469ba2302b7d`.
  - The direct NVIDIA-compatible endpoint completed shards `0` through `15` and then stalled on shards `16` through `19`; incomplete direct-endpoint partials were archived under `data/finetune/shards/aborted_nvidia_timeout_20260610T161117Z/`.
  - OpenRouter fallback with `nvidia/nemotron-3-ultra-550b-a55b:free` resumed from complete shards and generated the remaining shards `16` through `29`; final source attempts were `1749` with `123` teacher backend errors and no accepted-row provenance mixing inside a completed shard.
  - Focused regression suite passed after generation: `.venv/bin/python -m pytest tests/test_prompt_builder_contract.py tests/test_focused_repair.py tests/test_navigator_safety.py tests/test_eval_runner.py tests/test_eval_metrics.py tests/test_finetune_v2_data_plan.py tests/test_runtime_honesty.py tests/test_modal_finetune_prep.py tests/test_v4_training_seed_export.py -q` -> `82 passed`.
- Modal v4 smoke job passed:
  - Command: `.venv/bin/modal run modal/finetune_figment_nemotron.py --dataset-version figment_sft_v4 --dataset data/finetune/figment_sft_v4.jsonl --output-name figment-sft-v4-lora-smoke --smoke --gpu L40S --learning-rate 2e-5 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --gradient-accumulation-steps 8 --validation-steps 2 --save-steps 5`.
  - Modal app: `ap-J7w1D5j8VwZ1S9CuF4mwzN`.
  - Staged rows: `1482` train, `168` validation.
  - Tokenized rows: `1482` train, `168` validation.
  - Adapter path: `/checkpoints/figment_sft_v4/figment-sft-v4-lora-smoke`.
  - Smoke config: `max_steps=5`, `max_seq_length=2048`, `learning_rate=2e-5`, `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `gradient_accumulation_steps=8`.
  - Metrics: `train_loss=14.122270011901856`, `train_runtime=148.2881`, `epoch=0.02699055330634278`; eval loss was `1.741158127784729` at step 2 and `1.7384405136108398` at step 4.
  - Verified Modal volume artifacts include `adapter_model.safetensors`, `adapter_config.json`, tokenizer files, `chat_template.jinja`, `figment_training_manifest.json`, and `checkpoint-5/`.

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

Status on 2026-06-10:

- The entrypoint now accepts `learning_rate`, `lora_r`, `lora_alpha`, `lora_dropout`, `gradient_accumulation_steps`, `validation_steps`, and `save_steps`.
- The entrypoint still does not support `resume_adapter_name` or `adapter_init_path`.
- The ready full-run path is therefore fresh from `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` with replay-heavy v4 data, not continuation from the v3 adapter.

Recommended SFT config:

- `max_seq_length`: `16384`
- `lora_r`: `16` first, `32` only if v4 underfits the targeted handoff tasks
- `lora_alpha`: `32` for rank 16, `64` for rank 32
- `lora_dropout`: `0.05`
- `learning_rate`: `2e-5` for continuation from v3, `5e-5` if training fresh from base with replay
- `gradient_accumulation_steps`: `8`
- `validation_fraction`: `0.10`
- `max_steps`: `372` for the current fresh-from-base v4 run, approximately `2.0` epochs over `1482` train rows at batch size `1` and gradient accumulation `8`

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
  --max-steps 372 \
  --learning-rate 5e-5 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --gradient-accumulation-steps 8 \
  --validation-steps 25 \
  --save-steps 50 \
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
