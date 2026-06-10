# Local 4B Fine-Tuning Plan

Date: 2026-06-08

This note captures the fine-tuning strategy I would use after the prompting and scaffolding fixes in `docs/local_4b_prompting_scaffolding_fixes.md`. The goal is a small, local, full-weight Figment navigator that is more load-bearing on the 50-case eval without weakening deterministic safety gates.

## Training Hypothesis

The local 4B model is probably big enough for this task if the task is framed as bounded protocol navigation instead of open-ended clinical reasoning.

The evidence points toward trainable rubric-following gaps:

- Urgency floors are already reliable: `min_urgency_met` passed 50/50.
- After the prompting/scaffolding pass, expected-label success improved from 2/50 to 13/50 and local competence improved from 18/50 to 26/50.
- The biggest remaining miss is still exact required-observation cue coverage: 35/50 failures.
- Forbidden behavior is nearly solved but not fully solved: 1/50 still included a forbidden medication mention.
- Negated red-flag handling is a trainable false-positive problem: all 7 red-flag mismatches are unexpected red flags on negated/safety-boundary cases.
- Source-card, candidate-pathway, target-card selection, and SBAR failures are still format and grounding behaviors that SFT can teach.
- The model is less load-bearing than the final validation score suggests: every record needed at least one deterministic scaffold patch, `red_flags` needed deterministic patching in 45/50 records, and `missing_info_to_collect` / `next_observations_to_collect` each needed deterministic patching in 37/50 records.

I would not jump to a larger model unless Figment must rely on raw model output without scaffolding, field repair, or deterministic fallback. That is not the current architecture.

## Locked Test Set

Do not train on the current 50-case eval.

Keep this as the locked regression test:

- `data/eval/initial_handwritten_cases.jsonl`
- `data/eval/adversarial_strict_cases.jsonl`
- `data/eval/comprehensive_hosted_cases.jsonl`
- Pre-scaffold baseline: `traces/local_4b_evidence_20260607T231248Z/`
- Current post-scaffold baseline: `traces/local_4b_evidence_20260608T015209Z/`

Use those cases only for checkpoint comparison and final evidence.

## Failure Analysis From The Current Trace

Current trace: `traces/local_4b_evidence_20260608T015209Z/`.

Headline comparison against `traces/local_4b_evidence_20260607T231248Z/`:

- `competence_successes`: 18 -> 26.
- `repair_successes`: 5 -> 26.
- `fallback_uses`: 9 -> 6.
- `expected_label_successes`: 2 -> 13.
- `expected_label_failures`: 48 -> 37.
- `final_validation_successes`: 50 -> 50.
- `raw_configured_model_successes`: 13 -> 0, because the stricter post-scaffold scorer only counts raw success when no deterministic scaffold field is patched.

Failure breakdown in the current trace:

- `missing_observation_cues_present`: 35 failures.
- `expected_source_cards_present`: 11 failures.
- `expected_candidate_pathways_present`: 8 failures.
- `target_card_in_candidate_pathways`: 8 failures.
- `red_flags_match`: 7 failures.
- `target_card_in_source_cards`: 2 failures.
- `forbidden_behavior_absent`: 1 failure.
- `min_urgency_met`: 0 failures.

The failures are not evenly distributed. The finetune should target these clusters:

- Required-observation cue lexicalization. The model often includes a nearby observation but misses the evaluator-recognized cue. The most common missing cues are `symptom trend`, `complete vital signs`, `confirmed intake status`, `deterministic rule results`, `retrieved protocol card IDs`, `hydration status`, `source card IDs`, `source protocol card IDs`, `time since last urine`, and `fluid retention`.
- Negation and safety-boundary target selection. The 7 red-flag mismatches are all unexpected red flags where the expected rule set is empty: negated chest pain, AMS, respiratory distress, stroke, pediatric dehydration, wound infection, or fever-like cases should land on `SAFETY-BOUNDARIES-v1` / `REFERRAL-SBAR-v1`, not the condition red-flag pathway.
- Source-card omissions. The model still drops `REFERRAL-SBAR-v1` and/or `SAFETY-BOUNDARIES-v1` in pregnancy, AMS, respiratory-negated, pediatric, fever-infant, wound, and negated-safety cases.
- SBAR target-card behavior. SBAR-focused cases sometimes include `REFERRAL-SBAR-v1` in source cards but fail to put it in `candidate_protocol_pathways` as the target pathway.
- Canned fallback cases. Six cases still needed full canned fallback: respiratory gasping, safety-boundary injection, wound red streaking, pediatric lethargy, pregnancy SBAR handoff, and safety-ignore-cards.

Interpretation:

- This is a good SFT target, not a pure model-size blocker. The model can generally preserve schema, urgency floors, and safety phrasing, but it needs repeated examples of exact target-field coverage.
- The training objective should not reward prettier prose. It should reward exact field membership, exact observation-cue coverage, and correct abstention from red flags when facts are negated.
- Continue to measure final validation separately from model competence. Final validation is protected by deterministic scaffolding; the finetune should reduce deterministic patches and fallbacks.

## Teacher Model For Synthetic Data

Do not rely on handmaking the training data. Use a significantly larger teacher model to generate and critique synthetic SFT rows:

- Teacher model: `nvidia/nemotron-3-ultra-550b-a55b`.
- Use the same OpenAI-compatible hosted route Figment already uses for the hosted model.
- Endpoint config: use `OMNI_ENDPOINT_URL` or `HF_ENDPOINT_URL` if that is the active hosted endpoint, otherwise use `NVIDIA_BASE_URL` with the default `https://integrate.api.nvidia.com/v1`.
- API key: use the existing `NVIDIA_API_KEY` secret. Never write the key into datasets, logs, model cards, manifests, or Modal artifacts.
- Keep teacher selection separate from app runtime selection. Use a dedicated generation variable such as `SFT_TEACHER_MODEL_ID=nvidia/nemotron-3-ultra-550b-a55b`; do not overwrite the production `NVIDIA_MODEL_ID` just to generate data.

The teacher model is only for data generation, repair suggestions, and critique. The target artifact remains the full-weight local 4B route: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` plus a Figment adapter merged back into BF16 weights.

The teacher is not trusted blindly. Every accepted training row must pass deterministic schema validation, product-contract validation, safety validation, card-id validation, cue-coverage validation, and a rubric check before it is admitted to `data/finetune/figment_sft_v1.jsonl`.

## NVIDIA Training-Data Priors

Use NVIDIA's published Nemotron datasets as recipe priors for the Figment SFT set. The Hugging Face dataset pages list `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` among models trained or fine-tuned on these datasets, so the plan should match the kind of supervision the base model already knows.

Authenticated Hugging Face CLI probe on 2026-06-08:

- `hf datasets info nvidia/Nemotron-Post-Training-Dataset-v2` showed a gated dataset with parquet shards for `chat`, `code`, `math`, and multilingual splits.
- `hf datasets sql` over the Post-Training `chat` parquet showed row columns `uuid`, `license`, `generator`, `version`, `category`, `reasoning`, and `messages`.
- Downloading the small Post-Training multilingual shard showed the same columns plus a JSON `metadata` field with values such as `sub_category`, `dataset_name`, `source_file`, and `lang_id`.
- `hf download nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1 train.jsonl` produced rows with top-level keys `trajectory_id`, `responses_create_params`, `expected_action`, `scenario`, `num_unique_actions`, `meta_info`, `qwen_235b_info`, `agent_ref`, `pass_rate`, `pass_rate_total`, and `pass_rate_passed`. In the first 1000 rows, `expected_action` used `type`, `content`, `name`, and `arguments`.
- `hf datasets list nvidia/Nemotron-CC-v2 -R` and authenticated README/LICENSE downloads exposed the CC-v2 file layout and source recipe, but `hf download ... Diverse-QA/part_000000.parquet --dry-run` returned access denied because the repo still requires approval for data-file access in this token. Until that approval clears, treat CC-v2 as README/file-layout-informed rather than row-schema-confirmed.

Dataset-specific lessons:

- `nvidia/Nemotron-Post-Training-Dataset-v2`: SFT/RL-style post-training data with public/open or synthetic prompts, synthetic responses from larger public/open models, quality and complexity filtering, and multiple response modes. Figment should copy the shape: synthetic prompts, teacher-generated gold, explicit validation, and mode discipline. The target mode is "final navigator JSON only"; any teacher reasoning or critique stays in metadata and never becomes assistant output.
- `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1`: structured conversational tool-use trajectories where each assistant step is treated as a behavior-cloning problem with an `expected_action`, pass-rate fields, and reward metadata. Figment should copy this more than generic chat SFT: each row should have verifiable expected actions for card selection, red-flag membership, required observations, SBAR fields, and refusal boundaries.
- `nvidia/Nemotron-CC-v2`: broad pretraining data with high-value math/code preservation, synthetic rephrasing, multilingual QA, filtering, and global deduplication. Figment should copy the data hygiene, not the raw corpus: diverse paraphrases, hard examples, dedupe hashes, source metadata, and synthetic-only/de-identified healthcare scenarios.

Do not ingest NVIDIA dataset rows directly into the Figment fine-tune unless license and redistribution terms are reviewed for this project. Use them to set the data-generation recipe:

- Generate synthetic public-style prompts and synthetic domain cases instead of handmaking one-off examples.
- Use multiple generation prompts and temperatures with the Ultra teacher to create diversity, then filter heavily.
- Reject easy cases that do not exercise a real failure mode, just as the post-training dataset filtered easy-to-guess or low-quality prompts.
- Store structured row fields inspired by Post-Training: `uuid`, `license`, `generator`, `version`, `category`, `reasoning`, `messages`, and `metadata`.
- Store structured metadata inspired by the RL dataset: `expected_action`, `reward_components`, `pass_rate_total`, `pass_rate_passed`, `teacher_model_id`, `critic_model_id`, `generation_prompt_id`, `dedupe_hash`, and `recipe_sources`.
- Globally deduplicate by normalized intake text, protocol-card set, expected target card, and embedding similarity so the synthetic set does not become 1500 near-copies of the locked eval.
- Keep legal and ethical metadata explicit: source is synthetic, no PHI, no copied clinical transcript, no NVIDIA row copied, license review status recorded.
- Reject assistant outputs containing `<think>`, hidden reasoning tags, or visible teacher critique. The local model should learn the final navigator artifact, not the teacher's reasoning trace.

## Dataset

Create `data/finetune/figment_sft_v1.jsonl` with 500 to 1500 synthetic sibling cases generated by the Ultra teacher from existing protocol cards, synthetic case specs, and failure-class templates.

Recommended distribution:

- 40% required-observation exactness and cue lexicalization.
- 20% negation, denied symptoms, routine near-miss cases, and safety-boundary target selection.
- 15% source-card, target-card, candidate-pathway, and citation repair cases.
- 15% grounded SBAR slot filling, including SBAR-as-target-pathway cases.
- 5% forbidden clinical instruction avoidance.
- 5% repair-and-fallback-rescue cases based on the six current canned-fallback failure shapes.

Each example should use the exact production prompt shape, including:

- confirmed structured intake,
- deterministic red flags,
- urgency floor,
- retrieved protocol cards,
- allowed facts inventory,
- required observation targets,
- fact ledger,
- required JSON skeleton.

The target output should be ideal navigator JSON, not a cleaned-up local-model sample. Generate the gold JSON with the Ultra teacher, then accept it only after deterministic validation and, where useful, a second teacher critique pass.

Generation pipeline:

1. Generate a synthetic case spec from protocol cards and a failure class such as `missing_observation_cues`, `negated_red_flag`, `sbar_target_pathway`, or `source_card_coverage`.
2. Avoid copying locked eval cases or near-paraphrasing their free-text intake. The locked eval can define failure classes, not training examples.
3. Run the same deterministic retrieval, red-flag rules, urgency floors, required-observation extraction, fact ledger construction, and JSON skeleton construction used by production.
4. Ask `nvidia/nemotron-3-ultra-550b-a55b` to produce concise, observation-only semantic notes over a bounded JSON contract. The accepted SFT row still pairs the final assistant label with the exact production prompt shape.
5. Assemble the navigator JSON from the teacher-authored notes plus deterministic fixed fields for urgency floors, red flags, source cards, and candidate pathway ids, then validate the output with Figment's deterministic validators and eval-label scorer.
6. Reject, repair, or regenerate rows that miss required cues, cite unavailable cards, add unsupported red flags, lower urgency below the floor, leak unsafe clinical instruction, or omit SBAR grounding.
7. Optionally run a second Ultra pass as a critic that checks field membership against the rubric. Do not use the critic's approval as a replacement for deterministic validators.
8. Generate 4 to 8 candidate outputs for ordinary cases and 16 to 32 candidate outputs for high-risk failure classes such as negated red flags, forbidden behavior, and fallback-rescue cases. Score candidates with reward components, keep the best passing row, and store the candidate pass rate. This mirrors NVIDIA's agentic dataset pattern without requiring a full RL environment for the first run.
9. Deduplicate globally by case text, required facts, expected target card, and embedding similarity.
10. Write only accepted rows, with metadata for `teacher_model_id`, `teacher_label_mode`, endpoint variable names, prompt hash, protocol card ids, failure class, validation result, pass-rate metadata, dedupe hash, recipe-source links, and generation timestamp. Do not store secrets.

Current implementation note, 2026-06-08:

- `scripts/generate_finetune_data.py` uses streamed teacher calls to `nvidia/nemotron-3-ultra-550b-a55b` with `reasoning_effort="none"`.
- Non-streaming full-output teacher calls were not reliable for this prompt family; they produced long waits/hangs. The working route asks the teacher for concise semantic notes and rejects malformed/missing note payloads.
- The stream call runs in a child process with a parent-enforced timeout, so wedged teacher requests become `teacher_backend_error` rejections instead of blocking the dataset run.
- The first live artifact is a 50-row validated seed set at `data/finetune/figment_sft_v1.jsonl` with manifest `data/finetune/figment_sft_v1_manifest.json`; it is `dry_run=false`, generated by `nvidia/nemotron-3-ultra-550b-a55b`, and can be extended with `--resume`.
- The seed set is aligned to the actual local 4B harness: each row uses the same single user-message prompt shape that `ModelClient(... model_backend="llama_cpp")` sends to `/v1/chat/completions`, and each prompt is built with plain `search_protocol_cards(query_from_intake(...), limit=6)` retrieval rather than teacher-only forced cards.
- The seed set now covers both local-4B chat-completion tasks in the harness:
  - `navigator_full`: the primary production navigator prompt sent by `ModelClient.generate_json(prompt, context)`.
  - `focused_repair`: the field-repair prompts produced by `build_focused_repair_prompts(...)` when deterministic validation fails.
- `scripts/augment_finetune_repair_rows.py` adds repair-task rows from teacher-gold navigator outputs by corrupting a previous output in the same ways the harness sees, rebuilding the exact focused repair prompt, and using only the relevant teacher-gold fields as the assistant target.
- `scripts/verify_finetune_harness_alignment.py` verifies this contract by rebuilding every navigator prompt from `figment.prompt_builder.build_prompt`, rebuilding every focused repair prompt from `build_focused_repair_prompts`, validating navigator assistant JSON against the same retrieved cards, checking expected-label success, and rejecting teacher-facing artifacts such as `teacher_note` sources or teacher-specific pathway reasons.
- Audio intake is not currently a separate 4B chat-completion task in this harness. The local audio path is Parakeet/provider payload plus deterministic draft-field extraction, with the 4B route recorded as the field-fill model id but not called through `ModelClient.generate_json` for audio drafting.

Create a cue alias table before generating examples. The teacher can propose aliases, but deterministic code should canonicalize and filter them. Each required observation target should have:

- canonical cue text from the protocol card,
- 3 to 6 allowed natural-language variants,
- one short responder-facing phrase that should appear in `missing_info_to_collect` or `next_observations_to_collect`,
- a negative example where the cue is absent or contradicted.

For the first SFT dataset, oversample examples whose gold outputs cover every required observation cue in both the structured observation fields and the SBAR handoff. Then add ablations where the same intake should not trigger a condition pathway because the symptom is denied or historical.

## Data Split

Use a deterministic split by case id:

- 80% train.
- 10% validation during training.
- 10% synthetic holdout.

Also keep the current 50-case eval as a separate locked test set. It should never be mixed into train or validation.

## Gold Output Rules

Gold outputs must preserve Figment's product contract:

- `protocol_urgency` never below deterministic floor.
- `red_flags` only from deterministic fired rules or confirmed present facts.
- When deterministic red flags are empty, `red_flags` must be empty and the target pathway should usually be `SAFETY-BOUNDARIES-v1` or `REFERRAL-SBAR-v1`, not the condition card for the denied symptom.
- `source_cards` must cite every fired rule card and every candidate pathway card.
- `candidate_protocol_pathways` may only use allowed/retrieved card ids.
- In SBAR-focused tasks, `REFERRAL-SBAR-v1` must be present in both `source_cards` and `candidate_protocol_pathways`.
- `missing_info_to_collect` and `next_observations_to_collect` must cover required observation target ids.
- Observation coverage must use evaluator-recognizable cue wording, especially for `symptom trend`, `complete vital signs`, `confirmed intake status`, `deterministic rule results`, `retrieved protocol card IDs`, and `source card IDs`.
- SBAR must be grounded in confirmed intake, deterministic rules, and allowed slot sources.
- No diagnosis, prescribing, dosing, discharge, autonomous routing, or unsafe treatment instructions.
- Audio-derived facts must only appear when accepted or edited by the responder.

## Training Format

Use supervised fine-tuning where each row contains:

```json
{
  "case_id": "figment-sft-v1-000123",
  "uuid": "figment-sft-v1-000123",
  "license": "synthetic internal training data",
  "generator": "nvidia/nemotron-3-ultra-550b-a55b",
  "version": "figment_sft_v1",
  "category": "missing_observation_cues",
  "reasoning": "off",
  "messages": [
    {"role": "system", "content": "Figment system prompt..."},
    {"role": "user", "content": "CONTEXT JSON..."},
    {"role": "assistant", "content": "{...ideal navigator JSON...}"}
  ],
  "tags": ["missing_observations", "negation"],
  "metadata": {
    "teacher_model_id": "nvidia/nemotron-3-ultra-550b-a55b",
    "critic_model_id": "nvidia/nemotron-3-ultra-550b-a55b",
    "teacher_base_url_env": "NVIDIA_BASE_URL",
    "teacher_api_key_env": "NVIDIA_API_KEY",
    "failure_class": "missing_observation_cues",
    "expected_action": {
      "target_card": "SAFETY-BOUNDARIES-v1",
      "required_observation_cues": ["complete vital signs", "symptom trend"]
    },
    "reward_components": {
      "schema_valid": 1,
      "source_cards_present": 1,
      "required_observation_cues_present": 1,
      "red_flags_match": 1,
      "forbidden_behavior_absent": 1
    },
    "pass_rate_total": 8,
    "pass_rate_passed": 6,
    "dedupe_hash": "sha256:...",
    "recipe_sources": [
      "nvidia/Nemotron-Post-Training-Dataset-v2",
      "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
      "nvidia/Nemotron-CC-v2"
    ],
    "validator_passed": true,
    "license_review": "synthetic_figment_row_no_nvidia_rows_copied"
  }
}
```

If the training stack expects a single `text` column, serialize the same chat template into one string and keep `case_id`, `tags`, and non-secret teacher metadata as metadata.

Disable sequence packing for the first run. Exact prompt-to-output boundaries matter more than throughput for this task.

## Modal Job Shape

Use Modal for data generation and the training run because it gives code-defined images, GPU selection, secrets, and persistent volumes. Data generation is API-bound and can run on CPU; training needs GPU.

Reference docs:

- Modal guide: https://modal.com/docs/guide
- GPUs: https://modal.com/docs/guide/gpu
- Volumes: https://modal.com/docs/guide/volumes
- Secrets: https://modal.com/docs/guide/secrets
- Unsloth fine-tuning example: https://modal.com/docs/examples/unsloth_finetune
- LLM fine-tuning example: https://modal.com/docs/examples/llm-finetuning

Create a generation script such as `modal/generate_figment_sft_data.py` or `scripts/generate_finetune_data.py` that calls the same hosted endpoint shape as `figment/model_client.py`:

```python
import os
import modal

app = modal.App("figment-sft-data-generation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("openai", "pydantic")
)

data_volume = modal.Volume.from_name("figment-sft-data", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("nvidia-api")],
    timeout=6 * 60 * 60,
)
def generate(config: dict) -> dict:
    teacher_model_id = os.environ.get(
        "SFT_TEACHER_MODEL_ID",
        "nvidia/nemotron-3-ultra-550b-a55b",
    )
    base_url = (
        os.environ.get("OMNI_ENDPOINT_URL")
        or os.environ.get("HF_ENDPOINT_URL")
        or os.environ.get("NVIDIA_BASE_URL")
        or "https://integrate.api.nvidia.com/v1"
    )
    api_key = os.environ["NVIDIA_API_KEY"]
    ...
```

The generation job should record `teacher_model_id` and the endpoint environment variable name that was used, but it should never record `api_key` or the resolved secret value.

The generation job should also write a manifest inspired by NVIDIA's dataset cards:

- row counts by failure class,
- candidate counts and pass rates,
- rejection counts by validator,
- dedupe counts,
- teacher and critic model ids,
- source recipe links,
- license/PHI assertions,
- prompt-template hashes.

Create a new script such as `modal/finetune_figment_nemotron.py`:

```python
import modal

app = modal.App("figment-nemotron-lora")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate",
        "datasets",
        "peft",
        "sentencepiece",
        "torch",
        "transformers",
        "trl",
        "wandb",
    )
)

model_cache = modal.Volume.from_name("figment-model-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("figment-sft-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("figment-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    volumes={
        "/model_cache": model_cache,
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-token"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=12 * 60 * 60,
)
def train(config: dict) -> dict:
    ...
```

Start with `gpu="L40S"` for LoRA if 12k to 16k context fits with gradient checkpointing and batch size 1. Move to `gpu="H100"` or `gpu="A100-80GB"` if the long-context run is memory-bound. Do not make the final local model quantized just because training used memory-saving tricks.

Suggested setup commands:

```bash
modal volume create figment-model-cache
modal volume create figment-sft-data
modal volume create figment-checkpoints
modal secret create nvidia-api NVIDIA_API_KEY=... NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1 SFT_TEACHER_MODEL_ID=nvidia/nemotron-3-ultra-550b-a55b
modal secret create huggingface-token HF_TOKEN=...
modal secret create wandb-secret WANDB_API_KEY=...
modal run modal/generate_figment_sft_data.py --dataset-version figment_sft_v1
modal run modal/finetune_figment_nemotron.py --dataset-version figment_sft_v1
```

If the hosted model is currently using `OMNI_ENDPOINT_URL` or `HF_ENDPOINT_URL` instead of `NVIDIA_BASE_URL`, put that same endpoint variable in the `nvidia-api` Modal secret as well. The point is to reuse the hosted route, not to create a second provider configuration for the teacher.

## Base Model And Adapter

Base model:

- `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`
- Use the full-weight BF16 base.
- Train a LoRA adapter.
- Merge the adapter back into BF16 weights for the final local route.
- Convert the merged BF16 weights to BF16 GGUF for `llama.cpp`.

Training parameters for the first run:

- LoRA rank: 16.
- LoRA alpha: 32.
- LoRA dropout: 0.05.
- Learning rate: `1e-4`.
- Warmup ratio: 0.05.
- Epochs: 2 to 3.
- Effective batch size: 8 to 16 via gradient accumulation.
- Per-device batch size: 1.
- Max sequence length: 16384 if memory allows. The post-scaffold prompts reached roughly 12k to 14.5k tokens before completion, so a 12000-token first run would teach the wrong truncated task.
- Packing: false.
- Precision: BF16.
- Gradient checkpointing: true.
- Save every 50 steps.
- Evaluate every 25 to 50 steps.
- Early stop on validation loss plus task metrics, not loss alone.

For target modules, start with PEFT all-linear targeting if supported by the installed stack. If not, inspect the base model's `named_modules()` and include attention and MLP projection linears such as `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`, plus any Nemotron hybrid projection modules exposed as linear layers. Do not guess silently; log the matched trainable modules in the Modal run artifact.

If 16k context is memory-bound on `L40S`, prefer moving to `A100-80GB` or `H100` over silently truncating examples. If a smaller pilot is needed, build a separate short-context ablation dataset and label it as such; do not compare it directly to the locked 50-case eval.

Use oversampling before introducing more complex objectives:

- 3x oversample required-observation cue examples.
- 3x oversample negated-red-flag false-positive examples.
- 2x oversample SBAR target-pathway and source-card omission examples.
- Keep forbidden-behavior examples present but do not let them dominate, since the current failure rate is 1/50.

If SFT reduces loss but red-flag false positives persist, add a second small preference-tuning pass using paired outputs: one output that incorrectly fires the condition red flag from a negated fact, and one output that stays on the safety-boundary pathway with empty `red_flags`.

If SFT improves field imitation but still leaves tool-like decisions brittle, add a small RLVR-style or preference-tuning stage after SFT. Keep it simple on Modal first:

- Build paired outputs from the same synthetic cases.
- Score each pair with deterministic reward components.
- Train with DPO/ORPO in TRL if it is enough.
- Move to a NeMo Gym-style environment only if simple preference tuning cannot reduce deterministic patch counts.

The reward should be verifiable and product-shaped, not subjective: exact card ids, exact required observation cues, empty red flags when negated, no forbidden instruction, and SBAR groundedness.

## Training Metrics

During each checkpoint, run a lightweight held-out scorer that measures:

- exact JSON parse rate,
- required schema pass rate,
- required-observation target coverage,
- required-observation cue lexical coverage,
- source-card coverage,
- candidate pathway coverage,
- target-card-in-source and target-card-in-candidate rates,
- negation/red-flag match,
- unexpected-red-flag rate on negated cases,
- deterministic patch counts by field,
- full canned fallback count,
- model-visible-fields-retained,
- forbidden clinical language rate,
- SBAR grounding violations.

After promising checkpoints, run the full local 50-case eval against the checkpoint through the same `llama.cpp` route used for evidence.

## Acceptance Targets

Use the 2026-06-08 post-scaffold local 4B trace as the baseline:

- Expected-label successes: improve from 13/50 to at least 35/50.
- Missing-observation failures: reduce from 35 to 5 or fewer.
- Unexpected red flags on negated cases: reduce from 7 to 0.
- Source-card failures: reduce from 11 to 2 or fewer.
- Candidate-pathway target failures: reduce from 8 to 2 or fewer.
- Forbidden-behavior failures: reduce from 1 to 0.
- Full canned fallback uses: reduce from 6 to 2 or fewer.
- Deterministic scaffold patches: reduce from 206 patched fields to 80 or fewer, with `red_flags` deterministic patches below 10 and observation-field deterministic patches below 10 each.
- Model-visible-fields-retained: improve from 0.683 to at least 0.85.
- Raw configured-model successes under the strict no-patch definition: improve from 0/50 to at least 20/50.
- Final validation: remain 50/50.
- No-cloud local proof: remain true.

If the model improves loss but fails those task metrics, discard the checkpoint. Figment needs protocol-rubric behavior, not prettier prose.

## Modal Pilot Run

Status as of 2026-06-08:

- Modal app: `figment-nemotron-4b-lora`.
- Dataset staged in Modal volume `figment-sft-data` at `/data/figment_sft_v1`.
- Dataset split: 99 train rows, 11 validation rows.
- Training image: PyTorch 2.7.1 CUDA 12.6 devel base, PEFT LoRA, Transformers, `mamba-ssm==2.2.6.post3`, and `causal-conv1d==1.6.2.post1`.
- Smoke run: `initial-smoke`, 1 step at 2048 context, passed and saved adapter.
- Full-context smoke: `16k-context-smoke`, 1 step at 16384 context, passed and saved adapter.
- Pilot run: Modal run `ap-rocN1bpDhAMYPBLVE9Dfrg`, adapter path `/checkpoints/figment_sft_v1/pilot-20260608`.
- Pilot config: full BF16 base `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, LoRA rank 16, alpha 32, dropout 0.05, learning rate `1e-4`, max sequence length 16384, gradient accumulation 8, 40 optimizer steps.
- Pilot result: 99 train rows and 11 validation rows tokenized; train runtime 1297.0099 seconds; train loss 3.4215212553739547; checkpoint artifacts include `adapter_model.safetensors`, `adapter_config.json`, tokenizer files, and `figment_training_manifest.json`.
- Local manifest copy: `data/finetune/modal/figment_sft_v1/pilot-20260608-training_manifest.json`.
- Local adapter copy: `artifacts/modal_checkpoints/pilot-20260608/`.
- Modal merge run: `ap-D2ySCj6r9jRo8zQwUGBc6b`, saved merged BF16 Hugging Face weights to `/checkpoints/figment_sft_v1/pilot-20260608-merged-bf16`.
- Local merged BF16 copy: `artifacts/modal_checkpoints/pilot-20260608-merged-bf16/`.
- Local BF16 GGUF: `artifacts/modal_checkpoints/pilot-20260608-merged-bf16.gguf`, SHA-256 `85d92bc721a6f6d04c8f656ea3d32ff7c2714eef500f6cd0b8227e53268fc6d2`.
- GGUF conversion used `tools/llama.cpp/convert_hf_to_gguf.py`; the local `NemotronHModel` converter needed a repo-local patch so dense 4B configs are detected from raw `config.json` rather than the `AutoConfig` MoE defaults.
- Local server route was proved through `llama-server` on `http://127.0.0.1:8001/v1`, advertising `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` with `n_params=3973556832`, `n_ctx=16384`, and `n_ctx_train=1048576`.

Pilot eval evidence:

- Fine-tuned trace: `traces/local_4b_finetuned_evidence_20260608T151555Z/`.
- Baseline trace: `traces/local_4b_evidence_20260608T015209Z/`.
- Final validation remained 50/50 and no-cloud local route proof remained true.
- Raw configured-model successes improved from 0/50 to 10/50.
- Full canned fallback uses improved from 6 to 2.
- Deterministic scaffold patches improved from 206 fields to 104 fields.
- Model-visible fields retained improved from 0.683 to 0.84.
- Expected-label successes did not improve: 13/50 before, 13/50 after.
- Competence successes regressed from 26/50 to 11/50 because the previous focused-repair path collapsed from 26 repair successes to 1.
- The main remaining patched fields were `missing_info_to_collect` and `next_observations_to_collect`, 36 deterministic patches each.
- The pilot also produced a wound-case timeout and a follow-on HTTP 500 from `llama-server`; the eval recovered through canned fallback, but this is evidence that stop discipline and runtime token caps still need work.

This pilot checkpoint is not accepted as the final local model. It is useful because it proves the Modal train, merge, BF16 GGUF conversion, local serve, and no-cloud eval loop, and because it identifies the next training target: preserve the raw JSON/schema gains while restoring focused-repair behavior and improving expected-label cue coverage.

Next training iteration:

- Increase focused-repair rows substantially. The adapter must learn the exact `build_focused_repair_prompts(...)` task, including returning only the requested field subset and preserving valid existing fields.
- Add hard negative and ablation examples where the model must not drift from repair into full navigator output.
- Oversample accepted rows for `missing_info_to_collect` and `next_observations_to_collect` until every required observation target is expressed in evaluator-recognizable language.
- Add explicit wound, source-card, and target-card coverage rows because the local timeout happened in the wound cluster and source/candidate coverage remains a task metric.
- Add length-control examples and reject any training output with hidden reasoning tags, analysis prose, repeated JSON, or completion text outside the requested JSON object.
- Add an eval-time runtime guard for the local route, such as a lower `max_tokens` for primary and repair calls plus stop sequences where supported, so a single runaway generation cannot consume the full 240-second timeout.
- Keep the locked 50-case eval out of training data. Use these results only to define failure classes and acceptance metrics.

## Artifact Flow

1. Train LoRA adapter on Modal.
2. Save adapter, training config, data manifest, metrics, and module-match log to the checkpoint volume.
3. Pull the chosen adapter locally.
4. Merge adapter into the full BF16 base weights.
5. Save merged Hugging Face weights with a model card.
6. Convert merged safetensors to BF16 GGUF with the same `llama.cpp` conversion path used for the current local model.
7. Start local `llama-server` with the merged BF16 GGUF.
8. Rerun:

```bash
PYTHON_DOTENV_DISABLED=true FIGMENT_MODEL_TIMEOUT_SECONDS=180 .venv/bin/python scripts/run_local_4b_evidence.py --base-url http://127.0.0.1:8001/v1 --timeout-seconds 180 --force-eval
```

9. Update evidence docs only if the new run passes the gates. The `pilot-20260608` run does not pass the task-quality gates even though it passes final validation and no-cloud route proof.

## What Not To Do

- Do not train on the locked 50-case eval.
- Do not copy rows from NVIDIA's published datasets into Figment SFT without a deliberate license and redistribution review.
- Do not handwave teacher output into the dataset without deterministic validation and rejection.
- Do not store `NVIDIA_API_KEY` or any resolved hosted endpoint secret in training artifacts.
- Do not change the production hosted model config to Ultra 550B just to generate SFT data; use a dedicated teacher variable.
- Do not replace deterministic safety gates with model judgment.
- Do not use final validation success as model competence.
- Do not count deterministic fallback or deterministic target fills as model output.
- Do not publish a Well-Tuned claim until the adapter is trained, published or otherwise evidence-linked, loaded by the app, and verified by eval traces.
- Do not describe the final local route as quantized if the artifact is BF16/full-weight.
