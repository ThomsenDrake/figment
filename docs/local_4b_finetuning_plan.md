# Local 4B Fine-Tuning Plan

Date: 2026-06-07

This note captures the fine-tuning strategy I would use after the prompting and scaffolding fixes in `docs/local_4b_prompting_scaffolding_fixes.md`. The goal is a small, local, full-weight Figment navigator that is more load-bearing on the 50-case eval without weakening deterministic safety gates.

## Training Hypothesis

The local 4B model is probably big enough for this task if the task is framed as bounded protocol navigation instead of open-ended clinical reasoning.

The evidence points toward trainable rubric-following gaps:

- Urgency floors are already reliable: `min_urgency_met` passed 50/50.
- The model retained 499/650 visible fields from its own or repaired output.
- The biggest miss was exact required-observation coverage: 47/50 failures.
- Source-card, candidate-pathway, negation, and SBAR failures are all format and grounding behaviors that SFT can teach.

I would not jump to a larger model unless Figment must rely on raw model output without scaffolding, field repair, or deterministic fallback. That is not the current architecture.

## Locked Test Set

Do not train on the current 50-case eval.

Keep this as the locked regression test:

- `data/eval/initial_handwritten_cases.jsonl`
- `data/eval/adversarial_strict_cases.jsonl`
- `data/eval/comprehensive_hosted_cases.jsonl`
- Evidence baseline: `traces/local_4b_evidence_20260607T231248Z/`

Use those cases only for checkpoint comparison and final evidence.

## Dataset

Create `data/finetune/figment_sft_v1.jsonl` with 500 to 1500 synthetic sibling cases generated from the existing protocol cards and eval case families.

Recommended distribution:

- 35% required-observation exactness.
- 25% negation, denied symptoms, and routine near-miss cases.
- 20% grounded SBAR slot filling.
- 10% forbidden clinical instruction avoidance.
- 10% source-card, pathway, and citation repair cases.

Each example should use the exact production prompt shape, including:

- confirmed structured intake,
- deterministic red flags,
- urgency floor,
- retrieved protocol cards,
- allowed facts inventory,
- required observation targets,
- fact ledger,
- required JSON skeleton.

The target output should be ideal navigator JSON, not a cleaned-up model sample. Generate the gold JSON from deterministic scaffolding plus hand-authored responder-facing phrasing.

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
- `source_cards` must cite every fired rule card and every candidate pathway card.
- `candidate_protocol_pathways` may only use allowed/retrieved card ids.
- `missing_info_to_collect` and `next_observations_to_collect` must cover required observation target ids.
- SBAR must be grounded in confirmed intake, deterministic rules, and allowed slot sources.
- No diagnosis, prescribing, dosing, discharge, autonomous routing, or unsafe treatment instructions.
- Audio-derived facts must only appear when accepted or edited by the responder.

## Training Format

Use supervised fine-tuning where each row contains:

```json
{
  "case_id": "figment-sft-v1-000123",
  "messages": [
    {"role": "system", "content": "Figment system prompt..."},
    {"role": "user", "content": "CONTEXT JSON..."},
    {"role": "assistant", "content": "{...ideal navigator JSON...}"}
  ],
  "tags": ["missing_observations", "negation"]
}
```

If the training stack expects a single `text` column, serialize the same chat template into one string and keep `case_id` plus `tags` as metadata.

Disable sequence packing for the first run. Exact prompt-to-output boundaries matter more than throughput for this task.

## Modal Job Shape

Use Modal for the training run because it gives code-defined images, GPU selection, secrets, and persistent volumes.

Reference docs:

- Modal guide: https://modal.com/docs/guide
- GPUs: https://modal.com/docs/guide/gpu
- Volumes: https://modal.com/docs/guide/volumes
- Secrets: https://modal.com/docs/guide/secrets
- Unsloth fine-tuning example: https://modal.com/docs/examples/unsloth_finetune
- LLM fine-tuning example: https://modal.com/docs/examples/llm-finetuning

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
modal secret create huggingface-token HF_TOKEN=...
modal secret create wandb-secret WANDB_API_KEY=...
modal run modal/finetune_figment_nemotron.py --dataset-version figment_sft_v1
```

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
- Max sequence length: 12000 first, then 16384 if memory allows.
- Packing: false.
- Precision: BF16.
- Gradient checkpointing: true.
- Save every 50 steps.
- Evaluate every 25 to 50 steps.
- Early stop on validation loss plus task metrics, not loss alone.

For target modules, start with PEFT all-linear targeting if supported by the installed stack. If not, inspect the base model's `named_modules()` and include attention and MLP projection linears such as `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`, plus any Nemotron hybrid projection modules exposed as linear layers. Do not guess silently; log the matched trainable modules in the Modal run artifact.

## Training Metrics

During each checkpoint, run a lightweight held-out scorer that measures:

- exact JSON parse rate,
- required schema pass rate,
- required-observation target coverage,
- source-card coverage,
- candidate pathway coverage,
- negation/red-flag match,
- forbidden clinical language rate,
- SBAR grounding violations.

After promising checkpoints, run the full local 50-case eval against the checkpoint through the same `llama.cpp` route used for evidence.

## Acceptance Targets

Use the 2026-06-07 local 4B trace as the baseline:

- Expected-label successes: improve from 2/50 to at least 35/50.
- Missing-observation failures: reduce from 47 to 5 or fewer.
- Forbidden-behavior failures: reduce from 4 to 0.
- Red-flag mismatches: reduce from 7 to 1 or fewer.
- Full deterministic fallback uses: reduce from 9 to 3 or fewer.
- Final validation: remain 50/50.
- No-cloud local proof: remain true.

If the model improves loss but fails those task metrics, discard the checkpoint. Figment needs protocol-rubric behavior, not prettier prose.

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

9. Update evidence docs only if the new run passes the gates.

## What Not To Do

- Do not train on the locked 50-case eval.
- Do not replace deterministic safety gates with model judgment.
- Do not use final validation success as model competence.
- Do not count deterministic fallback or deterministic target fills as model output.
- Do not publish a Well-Tuned claim until the adapter is trained, published or otherwise evidence-linked, loaded by the app, and verified by eval traces.
- Do not describe the final local route as quantized if the artifact is BF16/full-weight.
