---
title: Figment
emoji: 📉
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
python_version: 3.11
---

# Figment

**Protocol support for low-connectivity field clinics and disaster response.**

Figment uses deterministic rules for danger signs and an AI protocol navigator for messy field notes, missing-observation planning, card-cited responder checklists, and SBAR handoffs. The frozen primary model path is NVIDIA Nemotron 3 Nano Omni: hosted Omni powers live-model demos when configured, and self-hosted Omni can technically support an Off the Grid run if it is served on adequate local hardware with no runtime cloud APIs. The current local/off-grid gap is hardware and recorded evidence, not an architecture impossibility; the smaller proof path targets Nemotron 3 Nano 4B for text navigation plus Parakeet RNNT ASR for dictated intake after verification. (The app scaffold is runnable and still under active development — see **Status** below.)

> ⚠️ **Figment is not a medical device.** It does not diagnose, prescribe, or replace a clinician. It is a prototype for protocol navigation, escalation support, and documentation in low-connectivity environments, for use by trained responders. See [Safety & non-goals](#safety--non-goals).

- **Status:** In active development for the [Build Small Hackathon](docs/build-small-hackathon-org-card.md) (build window **June 5-15, 2026**). The Gradio scaffold, deterministic rules, hosted NVIDIA Omni client, local OpenAI-compatible client, canned fallback, traces, and tests run locally; the hosted NVIDIA API smoke test is green. The public Hugging Face Space now boots with app files present at commit `5dcfc5c830de7331eca9020b17e1c571a8619654`; a no-secret public workflow smoke loaded a demo case, confirmed typed intake, fired deterministic pediatric-dehydration escalation, retrieved protocol cards, and produced an honestly labeled `canned_backend` trace with `validation_status=passed`, `raw_audio_stored=false`, and zero model-retained fields. The 50-case hosted Omni eval has run: baseline whole-output model competence was **28/50**, and the load-bearing follow-up reached **31/50** with **480/650** model-retained fields, **170/650** deterministic patches, **8/50** full fallback, and **50/50** final validation. Local 4B runtime evidence, Parakeet ASR evidence, demo video, social post, and user-test notes are still proof-needed items tracked in the [adversarial review action items](docs/adversarial-review-action-items.md), [hosted eval results](docs/hosted_omni_eval_results.md), [parameter/evidence ledger](docs/model_parameter_evidence_ledger.md), and [submission checklist](docs/submission_checklist.md).
- **Track target:** Backyard AI (solve a real problem for a specific, real person you know). Final evidence still needs a real trained responder using synthetic or de-identified scenarios; see [user test notes](docs/user_test_notes.md).
- **Built for:** a real disaster-response volunteer trained in disaster-response first aid and local protocol use; name withheld for privacy.
- **Model:** NVIDIA **Nemotron 3 Nano Omni 30B-A3B Reasoning** as the v1 default. The model-card body reports 31B total parameters; the workback plan and [parameter/evidence ledger](docs/model_parameter_evidence_ledger.md) track the HF-sidebar count ambiguity, local 4B + Parakeet story, adapter count status, and organizer-confirmation status.

---

## Why Figment

> What happens when the clinic loses internet?

Rural clinics, mobile units, and disaster sites lose connectivity exactly when decisions get hardest. Cloud medical assistants stop working; paper protocol binders don't talk back. Figment is built toward an **offline** mode: with a verified local model route, it can read the same protocol cards a responder would, apply hard-coded danger-sign rules, and turn a messy field note into a structured handoff on the machine in front of you. Until a no-cloud run is recorded, hosted mode and off-grid mode are labeled separately.

The design goal is restraint. Figment is **a field protocol binder that can talk, cite itself, and knows when to shut up** — not an "AI doctor."

---

## What it does

Figment is a [Gradio Server](https://www.gradio.app/guides/server-mode) app with five frozen workflow views:

1. **Intake** — structured capture of setting, patient age, pregnancy status, chief concern, symptoms, vitals, allergies, medications, available supplies, and a free-text responder note. Optional audio intake drafts fields only; typed/edited values must be confirmed before rules or navigation run.
2. **Risk Check** — deterministic red-flag rules fire **before** the LLM and set the minimum urgency floor (e.g. altered mental status, severe respiratory distress, chest pain, stroke signs, pregnancy bleeding, pediatric lethargy, severe dehydration signs, fever escalation criteria, wound infection escalation criteria).
3. **Protocol Guidance** — local retrieval returns 3–6 relevant protocol cards via SQLite FTS/BM25; the AI navigator selects candidate pathways, flags uncertainty, and plans missing observations.
4. **Navigator Output + Handoff** — shows candidate protocol pathways, a responder checklist, missing observations, an SBAR note, a referral summary, and source protocol-card IDs.
5. **Trace** — shows the full pipeline (input → rules → retrieval → prompt → output → validation) so judges and users can see *why*, not just *what*. This is the "show, don't tell" engine.

---

## How it works

```text
Gradio Server custom frontend + Gradio API endpoints
  → structured intake schema
  → rules.py        (deterministic red-flag engine)
  → retrieval.py    (SQLite FTS protocol search)
  → prompt_builder.py (constrained navigator prompt; cards + rules injected)
  → navigator.py      (AI protocol navigator)
  → model_client.py   (hosted/self-hosted Omni, local 4B OpenAI-compatible server, or canned fallback)
  → validators.py   (output validator: JSON, citations, safety checks)
  → sbar.py         (referral note renderer)
  → trace.py        (trace export)
```

Two principles make this safe rather than chatty:

- **Rules before the model.** Danger-sign detection is deterministic code, not a model guess, so a red flag can't be "reasoned away."
- **The cards are the source of truth; the model is a behavior harness.** The base hosted/local model is prompted and validated to stay inside retrieved cards, cite card IDs, ask for missing observations, preserve deterministic red-flag floors, build checklists, and refuse out-of-scope requests — not to memorize medical facts. Fine-tuning is deferred unless the runtime demo is already safe and reliable.

---

## The model & the ≤32B constraint

The Build Small Hackathon caps models at **32B total parameters**. Figment's primary path is **NVIDIA Nemotron 3 Nano Omni 30B-A3B Reasoning** — a multimodal MoE hybrid Mamba-Transformer with an integrated speech encoder and roughly 3B active parameters per token.

> **Compliance note:** NVIDIA's model-card body reports **31B total parameters**, which fits the 32B cap. The Hugging Face sidebar count has differed, so the workback plan keeps this as a submission risk to verify with organizers. The ~3B *active* figure is **not** the compliance number — the limit is on *total* parameters.

The live parameter and proof status is tracked in the [model parameter/evidence ledger](docs/model_parameter_evidence_ledger.md). It separates hosted Omni evidence from the unproven local 4B + Parakeet path, and it keeps adapter counts and organizer confirmation explicit before any badge or compliance claim is upgraded.

Omni can satisfy an off-grid claim if it is self-hosted on sufficient local hardware and the demo uses no runtime cloud APIs. This repo has not yet recorded that proof. The nearer local/off-grid proof path targets a smaller split stack after verification:

| Artifact | Use |
| -------- | --- |
| `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` | primary hosted/self-hosted Omni model ID |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | NVIDIA API Catalog / NIM chat-completions model ID |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | local text-navigation and first fine-tuning target |
| `nvidia/parakeet-rnnt-1.1b` | local/offline ASR target, enabled only after the local ASR gate passes |

Reference dev/demo machine: an M4 Pro MacBook Pro with 48 GB RAM. Hosted Omni is the intended public Space story; local audio is Parakeet-only after verification, and the safe local proof may use typed intake or a canned transcript if ASR is not stable. The full 4B BF16 artifact and Parakeet `.nemo` artifact are present locally, but the local 50-case text eval and local ASR provider proof still need to be run before any local badge claim is upgraded.

---

## Getting Started

Start with the full [prerequisites checklist](docs/prerequisites.md). The short version:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

### 1. Run the app with the hosted NVIDIA API

Copy `.env.example` to `.env`, set the hosted model variables, and add `NVIDIA_API_KEY`. The hosted route uses the NVIDIA API Catalog OpenAI-compatible endpoint:

```dotenv
FIGMENT_MODE=hosted
MODEL_BACKEND=hosted_omni
MODEL_STACK=omni_native
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL_ID=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
NVIDIA_API_KEY=nvapi-...
AUDIO_BACKEND=omni_native
ENABLE_AUDIO_INTAKE=true
```

Then run:

```bash
make run-hosted-demo PYTHON=.venv/bin/python
```

If the hosted model is unavailable or returns invalid JSON, Figment falls back to the deterministic canned navigator output and still validates the result.

### 2. Run against a local OpenAI-compatible server

To target a local OpenAI-compatible server after the Nemotron 3 Nano 4B path is verified, use the full BF16 4B model as the canonical local text artifact:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --served-model-name nemotron3-nano-4b-bf16 \
  --trust-remote-code \
  --max-model-len 16384
```

The full-weight snapshot has been downloaded locally at:

```text
/Users/drake.thomsen/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f
```

For a local run, expose that full-weight runtime through an OpenAI-compatible `/v1/chat/completions` endpoint and point Figment at it. Set `MODEL_BACKEND=llama_cpp`, `MODEL_STACK=local_4b_parakeet`, `LLAMA_BASE_URL=<local-openai-compatible-endpoint>/v1`, and `LOCAL_MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` in `.env`. Do not count the local route as model competence until the 50-case eval records configured-model successes rather than full deterministic fallback.

To capture the local evidence bundle once the endpoint is live:

```bash
PYTHON_DOTENV_DISABLED=true \
python3 scripts/run_local_4b_evidence.py \
  --base-url http://127.0.0.1:8001/v1
```

The helper writes `/v1/models` metadata, one-case route smoke, eval JSONL, eval summary, and `eval_evidence_manifest.json` under `traces/local_4b_evidence_*`. The manifest captures model/server metadata, no-cloud route flags, raw/repair/full-fallback counts, field provenance, latency, and trace hashes. If the endpoint is unavailable or the route smoke falls back deterministically, it records that state and does not upgrade the local model claim.

To capture the local Parakeet ASR evidence bundle once a real local ASR adapter or device runtime produces provider output:

```bash
PYTHON_DOTENV_DISABLED=true \
python3 scripts/run_local_asr_evidence.py \
  --provider-payload <local-parakeet-provider-output.json> \
  --audio <optional-source-audio.wav>
```

The ASR helper records Parakeet artifact metadata, optional audio metadata without copying raw audio, provider-output hash, Figment draft checks, and `asr_evidence_manifest.json`. Artifact presence alone does not count as local ASR proof.

### 3. Canned fallback

The scaffold can still run without any live model:

```dotenv
MODEL_BACKEND=canned
```

### 4. Hosted demo target

The submission Space target is under the **build-small-hackathon** Hugging Face org:

[build-small-hackathon/figment](https://huggingface.co/spaces/build-small-hackathon/figment)

The submission target is a live Gradio demo powered by a hosted or self-hosted Nemotron Omni endpoint. Canned responses and traces are fallback only if hosted model, quota, or cold-start reliability fails. The public Space is now runnable in no-secret mode: the Hugging Face Space API reports `runtime.stage=RUNNING`, `app.py` is present, the Space serves HTTP 200, and the public Gradio API completed a typed pediatric-dehydration workflow with deterministic escalation and an honestly labeled `canned_backend` trace. Live hosted Omni generation in the public Space still needs a final demo trace if secrets are configured for judging; keep local/off-grid and small-model badge claims gated by the [submission checklist](docs/submission_checklist.md).

---

## Repository layout

This repo now holds the runnable scaffold plus the planning docs. Current structure:

```text
figment/
  app.py                # Gradio Server app and custom frontend
  figment/              # config, schemas, rules, retrieval, model_client,
                        #   prompt_builder, validators, trace, sbar
  data/
    protocol_cards/     # 10 prototype cards (JSON)
    demo_audio/         # three synthetic dictated-intake WAV clips for the demo
  scripts/              # FTS build, smoke, and eval helpers
  traces/               # exported demo traces
  docs/                 # field notes, model/dataset/safety cards, this plan
```

Available now:

```text
app.py                                        # Gradio app scaffold
figment/                                      # protocol engine, model/audio adapters, trace/validators
data/protocol_cards/                          # 10 prototype protocol cards
data/demo_audio/                              # click-to-load hosted audio demo clips
traces/                                       # regenerated demo traces
tests/                                        # regression tests for safety, audio, rules, app smoke
docs/figment-workback-plan.md                 # the full day-by-day build plan
docs/build-small-hackathon-org-card.md         # hackathon rules (source of truth)
docs/prerequisites.md                          # setup contract for local, hosted, and Modal work
docs/superpowers/specs/  docs/superpowers/plans/  # design spec + implementation plan for plan additions
requirements.txt / requirements-dev.txt / .env.example
```

Key docs: [workback plan](docs/figment-workback-plan.md) · [adversarial review action items](docs/adversarial-review-action-items.md) · [hosted eval results](docs/hosted_omni_eval_results.md) · [parameter/evidence ledger](docs/model_parameter_evidence_ledger.md) · [local llama evidence](docs/local_llama_eval_evidence.md) · [local Parakeet ASR evidence](docs/local_parakeet_asr_evidence.md) · [submission checklist](docs/submission_checklist.md) · [safety statement](docs/safety_statement.md) · [user test notes](docs/user_test_notes.md) · [prerequisites](docs/prerequisites.md) · [hackathon rules](docs/build-small-hackathon-org-card.md) · [design spec](docs/superpowers/specs/2026-06-05-figment-plan-additions-design.md) · [implementation plan](docs/superpowers/plans/2026-06-05-figment-plan-additions.md).

---

## Data & evaluation

- **Synthetic data, not memorized facts.** Future 5,000–10,000 candidate cases are generated by teacher models (Mistral/MiniMax, build-time only), cross-critiqued, and filtered by a deterministic validator down to ~2,000–4,000 kept examples. No real PHI is used.
- **Behavior, not knowledge.** Training teaches the model to cite cards, ask for missing info, escalate red flags, produce SBAR, and refuse unsafe requests.
- **Eval before training.** A 50-case hosted Omni eval now scores the model on measurable behavior, while the larger 50-100 case target thresholds remain the quality bar:

| Metric | Target |
| ------ | -----: |
| Valid JSON | ≥ 98% |
| Source-card citation rate | ≥ 95% |
| Red-flag recall | ≥ 90% |
| Unsupported diagnosis rate | 0% |
| Unsupported medication/dose rate | 0% |
| Missing-info question rate | ≥ 85% |
| SBAR factuality | ≥ 95% |
| Prompt-injection compliance failure | 0 critical |

Current measured hosted Omni results are in [hosted_omni_eval_results.md](docs/hosted_omni_eval_results.md). The baseline run reached **28/50** whole-output model competence with **22/50** full deterministic fallback and **50/50** final validation. The load-bearing follow-up reached **31/50** whole-output model competence, **8/50** full fallback, **480/650** model-retained fields, **170/650** deterministic patches, and **50/50** final validation. Final validation is application safety, not pure model competence; deterministic fallback and deterministic patches are reported separately and cannot inflate model scores.

Local 4B + Parakeet eval, no-cloud/off-grid proof, and any fine-tuned adapter eval are still unmeasured.

The current eval harness records strict validation, repair/fallback, field provenance, and latency. Judgment metrics can still be added with a held-out judge model once the larger gold set exists.

---

## Safety & non-goals

Figment is deliberately scoped. **It will not:**

- **diagnose** — it surfaces protocol cards and danger signs; it does not name a condition as fact;
- **prescribe or dose medication** — doses appear only if a cited card contains them;
- **replace a clinician** — the trained responder remains the decision-maker;
- **serve untrained users** — it is a tool for trained responders;
- **store PHI or raw audio** — traces scrub raw audio-like payloads and uploaded filenames;
- **hide hosted-mode data flow** — hosted Space mode may send synthetic or de-identified text/audio to the configured Omni endpoint, while local mode keeps runtime inputs on the local machine;
- **act autonomously** — every output is advisory and requires human judgment.

This posture reflects real risk: the WHO has warned that authoritative-sounding health AI can create automation bias, and the FDA regulates clinical-decision-support software depending on its claims and users. Figment makes no clinical claims. See the fuller [safety statement](docs/safety_statement.md).

---

## Licensing & data handling

| Artifact | License |
| -------- | ------- |
| Model / adapter | inherits the NVIDIA Nemotron model license; cite exact upstream terms in the model card |
| Synthetic dataset | CC-BY-4.0 |
| Code | [Apache-2.0](LICENSE) |

Data handling: local mode keeps runtime inputs on the local machine; hosted mode is for synthetic or de-identified demo inputs only; traces do not retain raw audio.

---

## Demo cases

Three canonical cases drive the demo:

1. **Pediatric dehydration** — missing vitals, urgent red flags, asks next questions, produces a referral note.
2. **Wound infection after disaster injury** — protocol retrieval, avoids antibiotic overreach, recommends escalation criteria, clean documentation.
3. **Pregnancy danger sign** — deterministic red-flag override, immediate escalation, minimal model freelancing.

The Intake tab includes click-to-load audio examples for all three cases when `data/demo_audio/*.wav` is present. These are synthetic Voxtral-generated dictated-intake clips; they are not real patient audio.

---

## Hackathon

Built for the **[Build Small Hackathon](docs/build-small-hackathon-org-card.md)** (Gradio · Hugging Face), which caps models at 32B parameters and requires a Gradio app hosted as a Hugging Face Space plus a demo video and social post.

Submission claims are evidence-gated:

| Claim / badge area | Current status | Proof needed before claiming achieved |
| ------------------ | -------------- | ------------------------------------- |
| Hosted Gradio Space | Runnable in no-secret canned-fallback mode; live hosted-model demo trace still proof-needed | Public Space app files present, cold boot, typed intake run, trace showing `raw_route=canned`, `final_route=canned_backend`, `validation_status=passed`, and `raw_audio_stored=false` |
| Backyard AI | Targeted / proof-needed | A real trained responder using synthetic or de-identified scenarios, recorded in [user test notes](docs/user_test_notes.md) |
| Off the Grid | Targeted, not yet proven | Recorded no-cloud run using either self-hosted Omni on adequate local hardware or the smaller verified local stack |
| Llama Champion | Targeted, not yet proven | Working eligible local model route with trace/eval evidence |
| Sharing is Caring | Targeted / proof-needed | Public Space, repo, demo video, and social post links |
| Well-Tuned | Stretch / proof-needed | Eval harness plus measured improvement from tuning or an adapter, not fallback output |
| Field Notes | Tentative / proof-needed | Submission rules confirmation plus field-note artifact |
| Off-Brand | Targeted / proof-needed | Final demo/story asset aligned to organizer criteria |

---

## Acknowledgements

- **NVIDIA** — Nemotron 3 Nano Omni model · **Modal** — fine-tune/eval compute · **Gradio** & **Hugging Face** — app framework and hosting · **llama.cpp** — local inference.

---

## Disclaimer

Figment is a **prototype for trained responders**, not medical advice and not a medical device. It does not diagnose or prescribe. Protocol cards are prototypes derived from public guideline concepts, **not** clinical guidelines. Always rely on qualified clinical judgment and local protocols.

<!-- TODO before submission:
  - Add a LICENSE file.
  - Add the demo video + social post links.
-->
