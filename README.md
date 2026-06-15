---
title: Figment
emoji: 📉
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
python_version: 3.12.12
preload_from_hub:
  - build-small-hackathon/figment-finetuned-model-archive figment_sft_v14p/figment-sft-v14p-lora-merged-bf16/chat_template.jinja,figment_sft_v14p/figment-sft-v14p-lora-merged-bf16/config.json,figment_sft_v14p/figment-sft-v14p-lora-merged-bf16/configuration_nemotron_h.py,figment_sft_v14p/figment-sft-v14p-lora-merged-bf16/generation_config.json,figment_sft_v14p/figment-sft-v14p-lora-merged-bf16/model-00001-of-00002.safetensors,figment_sft_v14p/figment-sft-v14p-lora-merged-bf16/model-00002-of-00002.safetensors,figment_sft_v14p/figment-sft-v14p-lora-merged-bf16/model.safetensors.index.json,figment_sft_v14p/figment-sft-v14p-lora-merged-bf16/modeling_nemotron_h.py,figment_sft_v14p/figment-sft-v14p-lora-merged-bf16/special_tokens_map.json,figment_sft_v14p/figment-sft-v14p-lora-merged-bf16/tokenizer.json,figment_sft_v14p/figment-sft-v14p-lora-merged-bf16/tokenizer_config.json
---

# Figment

**Protocol navigation for trained responders in low-connectivity clinics and disaster response.**

Figment turns messy field intake into a card-cited protocol workflow: confirm the facts, run deterministic danger-sign rules, retrieve local protocol cards, ask a small model for bounded navigation fields, validate or repair the output, and show a trace of what happened.

> **Safety boundary:** Figment is a prototype, not a medical device. It does not diagnose, prescribe, dose medication, autonomously triage, or replace a trained responder, supervisor, clinician, or local protocol.

## Current Snapshot

| Surface | Current evidence | What it means | Boundary |
| --- | --- | --- | --- |
| Public Space | [build-small-hackathon/figment](https://huggingface.co/spaces/build-small-hackathon/figment) was `RUNNING` on `zero-a10g` at Space commit `45df7643e9e592f8496214c436532a3ade3cfdfc` on 2026-06-15. A synthetic `/run_navigator` call returned `raw_route=hf_zerogpu`, `fallback_tier=configured`, `field_level_fallback_used=true`, `final_route=model_with_deterministic_patches`, and `validation_status=passed` in 41.87 seconds. | The public Space reaches the published v14p BF16 model archive through HF ZeroGPU and the validator passed on the returned response. | This is one live synthetic route check. Deterministic patches still contributed, so it is live serving proof, not pure model-only competence proof. |
| Hosted Omni eval | `31/50` whole-output competence, `8/50` full fallback, `480/650` model-retained fields, `170/650` deterministic patches, and `50/50` final validation. | Hosted Omni can carry bounded fields, and the app can keep outputs inside the safety contract. | `50/50` final validation is app safety after validation, repair, and fallback. It is not pure model performance. |
| 4B LoRA system eval | v14p repair-union on the corrected 150-case field-workflow holdout: `150/150` competence, `150/150` expected labels, `150/150` final validation, `0` deterministic patches, `0` fallback. Raw first-pass success is `146/150`; `4/150` cases close through focused model repair. | The strongest documented small-model result is model-owned output plus model repair on a synthetic/de-identified holdout. | This is not clinical validation, target-user validation, local ASR proof, or proof that raw first-pass output solved every case. |
| Public artifacts | [model archive](https://huggingface.co/build-small-hackathon/figment-finetuned-model-archive) and [eval/training dataset](https://huggingface.co/datasets/build-small-hackathon/figment-eval-traces). | Versioned BF16/GGUF model artifacts, synthetic corpora, eval traces, and summaries are inspectable outside this checkout. | Generated `traces/`, `data/finetune/`, weights, and checkpoint folders are intentionally not part of a clean clone. |

Final submission claims are evidence-gated. Before changing public copy, run:

```bash
make audit-claims PYTHON=.venv/bin/python
make evidence-gates PYTHON=.venv/bin/python
```

## Why Figment Exists

When a rural clinic, mobile unit, shelter, or disaster site loses connectivity, the work does not become simpler. Protocol binders still matter, but they do not ask follow-up questions, organize missing observations, or draft a clean handoff.

Figment is built as a restrained protocol binder that can talk back. It does not try to be an AI clinician. Its job is narrower:

- preserve deterministic red-flag floors;
- cite the protocol cards it used;
- ask for missing observations;
- produce a responder checklist;
- draft an SBAR-style handoff;
- expose whether each field came from raw model output, model repair, or deterministic fallback.

That separation is the core project claim: useful small-model systems get safer and easier to improve when the model's job is narrow enough to inspect.

## User Workflow

Figment's Gradio Server app is organized around the field workflow:

1. **Intake** captures setting, age, pregnancy status, chief concern, symptoms, vitals, allergies, medications, available supplies, and a free-text responder note. Audio intake is only a draft layer; typed or edited facts must be confirmed before rules or navigation run.
2. **Risk Check** runs deterministic red-flag rules before model navigation and sets the minimum urgency floor.
3. **Protocol Guidance** retrieves 3-6 local protocol cards through SQLite FTS/BM25, with JSON fallback search.
4. **Navigator Output + Handoff** returns candidate pathways, uncertainty notes, missing observations, responder checklist, source cards, plain-language script, and SBAR handoff.
5. **Trace** shows input, rules, retrieval, prompt context, raw output, repair, fallback, validation, route labels, field provenance, and trace hashes.

Three included demo scenarios cover pediatric dehydration, wound infection after disaster injury, and pregnancy danger signs. The demo audio clips are synthetic and are not real patient audio.

## Architecture

```text
app.py
  -> confirmed structured intake
  -> figment/rules.py              deterministic danger-sign rules
  -> figment/retrieval.py          local protocol-card retrieval
  -> figment/prompt_builder.py     bounded navigator prompt
  -> figment/model_client.py       hosted Omni, HF ZeroGPU, local OpenAI-compatible, or canned route
  -> figment/navigator.py          raw output, scaffold, repair, fallback orchestration
  -> figment/validators.py         schema, citations, urgency floor, safety checks
  -> figment/field_provenance.py   model_raw / model_repaired / deterministic_fallback labels
  -> figment/eval_metrics.py       app-safety and model-contribution metrics
  -> figment/trace.py              auditable route and trace export
```

The safety pattern is deliberate:

- **Rules before model:** danger signs set an urgency floor the model cannot lower.
- **Cards as source of truth:** the model must stay inside retrieved protocol cards and cite card IDs.
- **Human confirmation:** audio-derived fields are provisional until the responder confirms them.
- **Scoped repair:** when an output fails validation, focused repair targets a bounded failure class rather than asking the model to improvise a new answer.
- **Visible fallback:** deterministic patches and full fallback are counted separately from model competence.

## Models

Figment supports four runtime routes:

| Route | Backend | Use |
| --- | --- | --- |
| Canned fallback | `MODEL_BACKEND=canned` | No-secret app smoke, UI development, honest fallback traces. |
| Hosted Omni | `MODEL_BACKEND=hosted_omni` with `NVIDIA_API_KEY` | Live hosted demo and hosted eval path using `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`. |
| HF ZeroGPU v14p | `MODEL_BACKEND=hf_zerogpu` with `ZEROGPU_MODEL_REPO` / `ZEROGPU_MODEL_SUBFOLDER` | Public Space route using the published v14p BF16 merged model on Hugging Face ZeroGPU. |
| Local OpenAI-compatible | `MODEL_BACKEND=llama_cpp` with `LLAMA_BASE_URL` | Local text-navigation route for the 4B BF16/GGUF artifacts and local evidence bundles. |

The Build Small constraint is <=32B total parameters. The hosted Omni path is tracked with a parameter-count caveat: the NVIDIA model-card body reports 31B total parameters, while sidebar counts have differed. The 4B BF16 base model, `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, is the local text-navigation training target.

Parakeet ASR remains a gated local-audio path. Artifact presence alone is not enough; local ASR needs provider-output evidence before any local-audio claim is upgraded.

## Evaluation

Figment reports app safety and model contribution separately.

| Metric | Meaning |
| --- | --- |
| Final validation | Did the final app output satisfy schema, citations, urgency floors, and safety checks? |
| Competence success | Did the configured model path, including allowed model repair, produce a competent case result? |
| Raw configured-model success | Did first-pass model output work without repair? |
| Focused repair success | Did a scoped model repair close a bounded failure? |
| Deterministic patch count | How many final fields came from code scaffolding rather than model output? |
| Full fallback use | Did the app abandon the model route and use deterministic fallback output? |
| Expected-label success | Did the final output preserve case-level target labels such as urgency, source cards, and red flags? |

Selected lineage on the 150-case field-workflow holdout:

| Run | Competence | Raw success | Repair | Expected labels | Final validation | Fallback | Deterministic patches | Lesson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v3 | 107/150 | 93/150 | 14 | 0/150 | 148/150 | 2 | 114 | First strong field-workflow jump, but weak observation ownership. |
| v5 | 2/150 | 2/150 | 0 | 150/150 | 150/150 | 0 | 302 | The app passed; deterministic scaffolding carried too much. |
| v6 | 142/150 | 142/150 | 0 | 146/150 | 150/150 | 0 | 21 | Targeted replay and delta rows moved model-owned behavior. |
| v7 corrected | 148/150 | 148/150 | 0 | 147/150 | 150/150 | 0 | 3 | Remaining failures became narrow and inspectable. |
| v10 | 147/150 | 147/150 | 0 | 150/150 | 150/150 | 0 | 6 | Some misses resisted generic corpus growth. |
| v14p repair-union | 150/150 | 146/150 | 4 | 150/150 | 150/150 | 0 | 0 | Focused model repair closed the remaining corrected-holdout cases. |

The corrected scoring view changes 6 cases from the original frozen holdout and preserves the correction manifest in `data/eval/field_workflow_holdout_v1_corrected_scoring_manifest.json`. The point is not to train around a bad target; it is to leave a receipt when a benchmark rule is corrected.

## Public Artifacts

- Demo Space: [build-small-hackathon/figment](https://huggingface.co/spaces/build-small-hackathon/figment)
- Runtime URL: [build-small-hackathon-figment.hf.space](https://build-small-hackathon-figment.hf.space/)
- Model archive: [build-small-hackathon/figment-finetuned-model-archive](https://huggingface.co/build-small-hackathon/figment-finetuned-model-archive)
- Eval traces and SFT corpora: [build-small-hackathon/figment-eval-traces](https://huggingface.co/datasets/build-small-hackathon/figment-eval-traces)
- Safety statement: [docs/safety_statement.md](docs/safety_statement.md)
- Submission gates: [docs/submission_checklist.md](docs/submission_checklist.md)
- Build Small org card: [docs/build-small-hackathon-org-card.md](docs/build-small-hackathon-org-card.md)

The model archive contains the v1 pilot, v2-v4 checkpoints, and versioned v5-v14p BF16/GGUF artifacts. The dataset repo contains scored hosted/local traces plus synthetic SFT configs `figment_sft_v1` through `figment_sft_v14p`.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

Or use the Makefile after creating the venv:

```bash
make install PYTHON=.venv/bin/python
```

Run the no-secret app path:

```bash
MODEL_BACKEND=canned make run PYTHON=.venv/bin/python
```

Run the hosted Omni demo path:

```bash
NVIDIA_API_KEY=nvapi-... make run-hosted-demo PYTHON=.venv/bin/python
```

Run tests:

```bash
PYTHONPATH=. .venv/bin/pytest tests -q
```

## Local Model Route

Start a local OpenAI-compatible server, for example with a downloaded GGUF:

```bash
llama-server \
  -m /path/to/figment-sft-v14p-lora-merged-bf16.bf16.gguf \
  --host 127.0.0.1 \
  --port 8001 \
  -c 16384
```

Point Figment at it:

```dotenv
FIGMENT_MODE=local
MODEL_STACK=local_4b_parakeet
MODEL_BACKEND=llama_cpp
LOCAL_MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
LLAMA_BASE_URL=http://127.0.0.1:8001/v1
AUDIO_BACKEND=none
```

Capture an evidence bundle once the endpoint is live:

```bash
make smoke-local-model-route PYTHON=.venv/bin/python
make local-4b-evidence PYTHON=.venv/bin/python
```

These commands write local evidence under `traces/`, which is generated and ignored by git.

## Training And Modal Eval

Modal scripts are included for the full train, merge, upload, and eval loop. They require Modal auth, appropriate secrets, and enough storage for model artifacts.

```bash
.venv/bin/modal run modal/finetune_figment_nemotron.py --smoke true
.venv/bin/modal run modal/finetune_figment_nemotron.py
.venv/bin/modal run modal/eval_figment_nemotron.py
```

The high-level loop is:

1. generate or replay synthetic harness-shaped rows;
2. verify rows against the real prompt, validators, retrieval, and expected-label rules;
3. stage train/validation splits for Modal;
4. train a LoRA adapter on H100;
5. merge into BF16, convert to GGUF, and serve locally;
6. rerun the field-workflow holdout;
7. compare raw, repair, patch, fallback, expected-label, final-validation, and latency metrics.

## Repository Layout

```text
app.py                  Gradio Server app and API surface
figment/                config, schemas, rules, retrieval, prompt, model clients,
                        navigator, validators, repair, provenance, traces
data/protocol_cards/    10 prototype protocol cards
data/eval/              hosted and field-workflow eval cases plus manifests
data/demo_audio/        synthetic dictated-intake demo clips
scripts/                eval, smoke, evidence, generation, merge, and claim audit helpers
modal/                  Modal training, merge, upload, and H100 eval entrypoints
tests/                  regression tests for runtime, safety, eval, data plans, and gates
docs/                   plans, evidence notes, safety, submission, and public drafts
```

Generated or heavyweight paths such as `traces/`, `data/finetune/`, `tools/`, checkpoints, weights, and local artifacts are intentionally ignored. Use the public Hub archives for shareable model, trace, and corpus artifacts.

## Data Handling

- Demo and eval scenarios are synthetic or de-identified.
- Do not enter real PHI into the hosted demo.
- Hosted mode may send synthetic or de-identified text/audio to the configured hosted endpoint.
- Local mode is intended to keep runtime inputs on the local machine.
- Figment traces do not retain raw audio bytes, uploaded filenames, local secrets, or unnecessary identifying details.

## Safety And Non-Goals

Figment will not:

- diagnose a condition as fact;
- prescribe medication or provide doses beyond cited protocol-card content;
- replace clinician, supervisor, or trained responder judgment;
- hide fallback, deterministic patches, or model repair;
- use unconfirmed audio fields for final navigation;
- present local/off-grid, local ASR, target-user, or final submission claims without the corresponding evidence gate.

See [docs/safety_statement.md](docs/safety_statement.md) for the fuller intended-use and non-goal statement.

## License

| Artifact | License |
| --- | --- |
| Code | [Apache-2.0](LICENSE) |
| Synthetic/de-identified dataset artifacts | CC-BY-4.0 where published |
| Model artifacts | NVIDIA Nemotron Open Model License inherited from `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` |

## Acknowledgements

Figment was built for the [Build Small Hackathon](docs/build-small-hackathon-org-card.md), hosted by Gradio and Hugging Face, with NVIDIA and Modal central to the model and training loop. It also depends on Gradio Server, Hugging Face Hub, Modal, llama.cpp-compatible serving, and the small-model debugging discipline made visible by the eval traces.
