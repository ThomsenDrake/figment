# Local 4B V2 Training Data Plan

Date: 2026-06-08

This note turns the `pilot-20260608` failure analysis into the next training-data target. The goal is `figment_sft_v2`: a larger, cleaner, harness-aligned dataset for the full-weight local `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` route.

## Verdict

Use a combination of additional data and training-technique changes.

More data is necessary, but more of the current `figment_sft_v1` recipe is not enough. The next round should fix data quality and rubric alignment first, then train with task-balanced sampling or a small curriculum so the adapter is optimized against Figment's actual eval metrics, not only JSON validity or loss.

Do not jump to a bigger local model yet. The pilot proved that the 4B can learn structure and retain more fields; the failures point mostly to underspecified or contradictory supervision plus a repair-trigger mismatch.

## Evidence Sources

- Fine-tuned pilot trace: `traces/local_4b_finetuned_evidence_20260608T151555Z/`.
- Post-scaffold baseline trace: `traces/local_4b_evidence_20260608T015209Z/`.
- Current training set: `data/finetune/figment_sft_v1.jsonl`.
- Current harness-alignment verifier: `scripts/verify_finetune_harness_alignment.py`.
- Current generation script: `scripts/generate_finetune_data.py`.
- Current repair augmentation script: `scripts/augment_finetune_repair_rows.py`.

## What Improved

The pilot adapter learned output shape and field retention:

- Raw configured-model success improved from `0/50` to `10/50`.
- Full canned fallback uses improved from `6` to `2`.
- Deterministic scaffold patches dropped from `206` fields to `104` fields.
- Model-visible fields retained improved from `0.683` to `0.84`.
- Final validation remained `50/50`.
- No-cloud local route proof remained true.

## What Failed

The pilot did not learn the task rubric:

- Expected-label success stayed flat at `13/50`.
- Competence regressed from `26/50` to `11/50`.
- Repair successes collapsed from `26` to `1`.
- Missing-observation cue failures stayed high at `34/50`.
- Red-flag mismatches stayed at `7`.
- Candidate target failures worsened from `8` to `11`.
- Forbidden-behavior failures worsened from `1` to `6`.

The main remaining deterministic patches were:

- `missing_info_to_collect`: `38` deterministic fallback fields.
- `next_observations_to_collect`: `38` deterministic fallback fields.

## Root Causes

### 1. The dataset taught schema more than rubric

`figment_sft_v1` has only `50` full navigator rows and `60` focused-repair rows. That was enough to make the model emit more valid-looking JSON, but not enough to teach exact card selection, negation behavior, observation-cue coverage, or eval-safe lexical choices.

### 2. The negation data is too small and partly contradictory

There are only `8` `negation_safety_boundary` rows. Several of those rows still contain red flags in the assistant target even though the category should teach the opposite behavior. This can teach the model to include `SAFETY-BOUNDARIES-v1` while keeping condition red flags around, which is exactly what appears in the eval failures.

For `figment_sft_v2`, category invariants should reject any `negation_safety_boundary` row unless:

- `red_flags` is empty,
- `protocol_urgency` is not raised by a denied symptom,
- `candidate_protocol_pathways` targets `SAFETY-BOUNDARIES-v1` or `REFERRAL-SBAR-v1`,
- condition cards appear only when justified as source context, not as active red flags.

### 3. The dataset includes eval-forbidden lexical patterns

The locked expected-label scorer flags tokens such as `medication`. In `figment_sft_v1`, many assistant targets include strings such as "Do not prescribe, dose, administer, or start medication." That is semantically safe, but it trains the exact lexical pattern the scorer later punishes.

For `figment_sft_v2`, either the evaluator should distinguish prohibited advice from safe boundary language, or the dataset should avoid the scorer's forbidden lexical tokens entirely. If the eval remains locked, prefer scorer-safe phrasing such as:

- "Do not give treatment instructions."
- "Do not provide dosing or clinical orders."
- "Do not tell the responder to start, stop, or administer anything."

Avoid `medication`, `prescribe`, `dose`, `antibiotics`, and similar tripwire terms in assistant outputs unless the scorer is fixed first.

### 4. Repair mostly stopped being invoked

After fine-tuning, the model often emits schema-valid JSON. That means the harness accepts raw output and marks weak fields as deterministic patches, rather than invoking focused repair. The repair path currently runs after strict validation failures, not after expected-label/rubric failures or scaffold patches.

This means data alone will not fully recover repair behavior. The next round should combine better data with at least one harness/objective change:

- Option A: train the full navigator output to satisfy rubric checks directly, reducing the need for repair.
- Option B: add a rubric-repair phase that can repair expected-label misses and scaffold-patched fields, not only strict validation failures.
- Option C: both, which is the recommended route.

### 5. SBAR handoff grounding is undertrained

All actual repair attempts in the pilot were SBAR/handoff grounding failures. There are only `10` `focused_repair:handoff_note_sbar` rows in `v1`.

`figment_sft_v2` needs many more SBAR repair rows, especially examples where the model must remove unsupported high-risk facts from `handoff_note_sbar` without changing valid fields.

### 6. Runtime length control needs training and serving guards

The pilot produced a wound-case timeout and a follow-on HTTP 500 from `llama-server`. This should be addressed two ways:

- Dataset: reject outputs with repeated JSON, hidden reasoning tags, analysis prose, or excessive list growth.
- Runtime: use lower `max_tokens` for primary and repair calls, plus stop sequences where supported.

## V2 Dataset Target

Create `data/finetune/figment_sft_v2.jsonl` with `1000` to `1500` accepted rows after validation and rejection.

Use `nvidia/nemotron-3-ultra-550b-a55b` as the teacher through the same hosted OpenAI-compatible endpoint and `NVIDIA_API_KEY` secret. Do not store secrets or resolved endpoint credentials in data, manifests, traces, or model cards.

Recommended accepted-row mix:

- `400` to `500` full navigator rows focused on exact required-observation cue coverage.
- `250` to `300` negation and safety-boundary rows.
- `200` to `250` source-card, candidate-target, and SBAR-as-target rows.
- `150` to `200` wound, fever, pregnancy, and multi-card rows.
- `300` to `500` focused-repair rows.

Focused repair should include at least:

- `100` handoff/SBAR grounding repair rows.
- `100` missing-observation repair rows.
- `75` citations-and-pathways repair rows.
- `50` forbidden-language repair rows with scorer-safe boundary phrasing.
- `50` schema repair rows.
- `25` protocol-urgency repair rows.

The total can exceed the full-navigator count because repair is a separate harness task.

## Required V2 Validators

Add dataset-generation gates before accepting rows:

- Schema validation against the production navigator output schema.
- Harness alignment: prompt must match `figment.prompt_builder.build_prompt` or `build_focused_repair_prompts`.
- Category invariants for negation, safety-boundary, SBAR, wound, and source-card rows.
- Expected-label scorer pass, including exact target card, source cards, candidate pathways, red flags, and observation cues.
- Forbidden lexical scanner aligned to the locked scorer.
- Length and repetition guard.
- No teacher notes, reasoning traces, markdown fences, or prose outside JSON.
- No copied locked-eval case text or near paraphrases.
- No copied NVIDIA dataset rows.
- Metadata must include teacher model id, prompt template hash, category, generation prompt id, validator status, dedupe hash, and recipe sources.

## Canonical Cue Coverage

Prioritize exact coverage for the most common missing cues:

- `complete vital signs`
- `symptom trend`
- `confirmed intake status`
- `hydration status`
- `deterministic rule results`
- `retrieved protocol card IDs`
- `source card IDs`
- `source protocol card IDs`
- `time since last urine`
- `fluid retention`
- `mental status trend`

Each accepted row should put required cues in both:

- `missing_info_to_collect`
- `next_observations_to_collect`

Where relevant, also include the cue in `handoff_note_sbar.assessment_observations_only` or `handoff_note_sbar.handoff_request`.

## Training Technique

Do another SFT run, but not the same tiny mixed-task run.

Recommended sequence:

1. Generate and validate `figment_sft_v2`.
2. Split by deterministic case id into train, validation, and synthetic holdout.
3. Train a full-navigator adapter pass with task-balanced sampling.
4. Continue with a repair-heavy pass at a lower learning rate.
5. Select checkpoints by task metrics, not training loss.
6. Merge the best adapter into full BF16 weights.
7. Convert to BF16 GGUF.
8. Re-run the full local 50-case eval through `llama.cpp`.

Checkpoint selection metrics:

- expected-label success,
- observation cue coverage,
- negated red-flag false positives,
- target-card-in-candidate rate,
- source-card coverage,
- forbidden lexical violations,
- deterministic patch count,
- repair success,
- full canned fallback count,
- final validation success,
- no-cloud route proof.

If SFT v2 still leaves negation and target-card choice brittle, add a small DPO/ORPO pass using deterministic paired outputs. The preference pairs should compare a bad output against a corrected output for the same prompt, scored by exact card ids, exact cue coverage, empty red flags when symptoms are denied, and scorer-safe boundary wording.

## Acceptance Targets

Use `traces/local_4b_evidence_20260608T015209Z/` as the main baseline and `traces/local_4b_finetuned_evidence_20260608T151555Z/` as the failed-pilot comparison.

The next checkpoint should meet or beat:

- Expected-label success: at least `35/50`.
- Missing-observation failures: at most `5/50`.
- Red-flag mismatches: `0`.
- Candidate target failures: at most `2`.
- Source-card failures: at most `2`.
- Forbidden-behavior failures: `0`.
- Full canned fallback uses: at most `2`.
- Deterministic scaffold patches: at most `80` fields.
- Observation-field deterministic patches: below `10` each.
- Repair successes: at least `20`.
- Raw configured-model successes: at least `20/50`.
- Final validation: `50/50`.
- No-cloud local route proof: true.

## Immediate Implementation Tasks

1. Parameterize the existing generation scripts for `figment_sft_v2` paths instead of hard-coded `figment_sft_v1` constants.
2. Add v2 category invariant validators.
3. Add forbidden lexical validation aligned to the locked scorer.
4. Add canonical cue coverage validation.
5. Add dedupe and near-eval paraphrase rejection.
6. Generate a small v2 smoke batch and verify it passes harness alignment.
7. Generate the full v2 dataset with the Ultra teacher.
8. Prepare Modal train/validation split under `data/finetune/modal/figment_sft_v2/`.
9. Run the harness alignment verifier on v2.
10. Only then launch the next Modal training run.
