# Figment

**Offline protocol support for field clinics and disaster response.**

Figment uses deterministic rules for danger signs and an AI protocol navigator for messy field notes, missing-observation planning, card-cited responder checklists, and SBAR handoffs. The frozen primary model path is NVIDIA Nemotron 3 Nano Omni: hosted or self-hosted Omni powers the live Space demo, while local/off-grid mode targets Nemotron 3 Nano 4B for text navigation plus Parakeet RNNT ASR for dictated intake after verification. (The app scaffold is runnable and still under active development — see **Status** below.)

> ⚠️ **Figment is not a medical device.** It does not diagnose, prescribe, or replace a clinician. It is a prototype for protocol navigation, escalation support, and documentation in low-connectivity environments, for use by trained responders. See [Safety & non-goals](#safety--non-goals).

- **Status:** 🚧 In active development for the [Build Small Hackathon](docs/build-small-hackathon-org-card.md) (build window **June 5–15, 2026**). The Gradio scaffold, deterministic rules, hosted NVIDIA Omni client, local OpenAI-compatible client, canned fallback, traces, and tests run locally; the hosted NVIDIA API smoke test is green, while the local 4B runtime and Parakeet ASR still need real model boot tests.
- **Track:** 🏡 Backyard AI (solve a real problem for a specific, real person you know).
- **Built for:** a real disaster-response volunteer trained in disaster-response first aid and local protocol use; name withheld for privacy.
- **Model:** NVIDIA **Nemotron 3 Nano Omni 30B-A3B Reasoning** as the v1 default. The model-card body reports 31B total parameters; the workback plan tracks the HF-sidebar count ambiguity and fallback story.

---

## Why Figment

> What happens when the clinic loses internet?

Rural clinics, mobile units, and disaster sites lose connectivity exactly when decisions get hardest. Cloud medical assistants stop working; paper protocol binders don't talk back. Figment keeps working **offline**: it reads the same protocol cards a responder would, applies hard-coded danger-sign rules, and turns a messy field note into a structured handoff — all on the machine in front of you.

The design goal is restraint. Figment is **a field protocol binder that can talk, cite itself, and knows when to shut up** — not an "AI doctor."

---

## What it does

Figment is a [Gradio](https://www.gradio.app/) app with five frozen tabs:

1. **Intake** — structured capture of setting, patient age, pregnancy status, chief concern, symptoms, vitals, allergies, medications, available supplies, and a free-text responder note. Optional audio intake drafts fields only; typed/edited values must be confirmed before rules or navigation run.
2. **Risk Check** — deterministic red-flag rules fire **before** the LLM and set the minimum urgency floor (e.g. altered mental status, severe respiratory distress, chest pain, stroke signs, pregnancy bleeding, pediatric lethargy, severe dehydration signs, fever escalation criteria, wound infection escalation criteria).
3. **Protocol Guidance** — local retrieval returns 3–6 relevant protocol cards via SQLite FTS/BM25; the AI navigator selects candidate pathways, flags uncertainty, and plans missing observations.
4. **Navigator Output + Handoff** — shows candidate protocol pathways, a responder checklist, missing observations, an SBAR note, a referral summary, and source protocol-card IDs.
5. **Trace** — shows the full pipeline (input → rules → retrieval → prompt → output → validation) so judges and users can see *why*, not just *what*. This is the "show, don't tell" engine.

---

## How it works

```text
Gradio Blocks UI
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

Local/off-grid proof targets a smaller split stack after verification:

| Artifact | Use |
| -------- | --- |
| `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` | primary hosted/self-hosted Omni model ID |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | NVIDIA API Catalog / NIM chat-completions model ID |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | local text-navigation and first fine-tuning target |
| `nvidia/parakeet-rnnt-1.1b` | local/offline ASR target, enabled only after the local ASR gate passes |

Reference dev/demo machine: an M4 Pro MacBook Pro with 48 GB RAM. Hosted Omni remains the public Space story; local audio is Parakeet-only after verification, and the safe local proof may use typed intake or a canned transcript if ASR is not stable.

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

To target a local OpenAI-compatible server after the Nemotron 3 Nano 4B path is verified:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --served-model-name nemotron3-nano-4b-bf16 \
  --trust-remote-code \
  --max-model-len 16384

# Or, on the Mac/off-grid path, use a verified 4B llama.cpp-compatible quantization:
llama-server \
  -hf <verified-nemotron-3-nano-4b-gguf> \
  --ctx-size 16384 \
  --port 8001 \
  --host 127.0.0.1 \
  --temp 0.4 \
  --top-p 0.9
```

Set `MODEL_BACKEND=llama_cpp`, `MODEL_STACK=local_4b_parakeet`, `LLAMA_BASE_URL=http://127.0.0.1:8001/v1`, and `LOCAL_MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` in `.env`.

### 3. Canned fallback

The scaffold can still run without any live model:

```dotenv
MODEL_BACKEND=canned
```

### 4. Hosted demo

A Gradio Space is hosted under the **build-small-hackathon** Hugging Face org:

[build-small-hackathon/figment](https://huggingface.co/spaces/build-small-hackathon/figment)

The primary hosted path is a live Gradio demo powered by a hosted or self-hosted Nemotron Omni endpoint. Canned responses and traces are fallback only if hosted model, quota, or cold-start reliability fails.

---

## Repository layout

This repo now holds the runnable scaffold plus the planning docs. Current structure:

```text
figment/
  app.py                # Gradio Blocks UI
  figment/              # config, schemas, rules, retrieval, model_client,
                        #   prompt_builder, validators, trace, sbar
  data/
    protocol_cards/     # 10 prototype cards (JSON)
    demo_audio/         # three synthetic dictated-intake WAV clips for the demo
  scripts/              # build_fts now; generation/eval scripts later
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

Key docs: [workback plan](docs/figment-workback-plan.md) · [prerequisites](docs/prerequisites.md) · [hackathon rules](docs/build-small-hackathon-org-card.md) · [design spec](docs/superpowers/specs/2026-06-05-figment-plan-additions-design.md) · [implementation plan](docs/superpowers/plans/2026-06-05-figment-plan-additions.md).

---

## Data & evaluation

- **Synthetic data, not memorized facts.** Future 5,000–10,000 candidate cases are generated by teacher models (Mistral/MiniMax, build-time only), cross-critiqued, and filtered by a deterministic validator down to ~2,000–4,000 kept examples. No real PHI is used.
- **Behavior, not knowledge.** Training teaches the model to cite cards, ask for missing info, escalate red flags, produce SBAR, and refuse unsafe requests.
- **Eval before training.** A 50–100 case gold set scores the model on measurable behavior:

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

> These are **target thresholds, not measured results** — no full evaluation has been run yet. The scaffold and safety regressions are covered by local tests; model evaluation is still planned.

Deterministic eval scripts are planned in the workback plan; judgment metrics use a held-out judge model once the gold set exists.

---

## Safety & non-goals

Figment is deliberately scoped. **It will not:**

- **diagnose** — it surfaces protocol cards and danger signs; it does not name a condition as fact;
- **prescribe or dose medication** — doses appear only if a cited card contains them;
- **replace a clinician** — the trained responder remains the decision-maker;
- **serve untrained users** — it is a tool for trained responders;
- **store PHI or raw audio** — traces scrub raw audio-like payloads and uploaded filenames;
- **hide hosted-mode data flow** — hosted Space mode may send synthetic or de-identified text/audio to the configured Omni endpoint, while local mode keeps runtime inputs on-device;
- **act autonomously** — every output is advisory and requires human judgment.

This posture reflects real risk: the WHO has warned that authoritative-sounding health AI can create automation bias, and the FDA regulates clinical-decision-support software depending on its claims and users. Figment makes no clinical claims. A fuller `docs/safety_statement.md` will be published with the submission.

---

## Licensing & data handling

| Artifact | License |
| -------- | ------- |
| Model / adapter | inherits the NVIDIA Nemotron model license; cite exact upstream terms in the model card |
| Synthetic dataset | CC-BY-4.0 |
| Code | Apache-2.0 |

Data handling: local mode keeps runtime inputs on-device; hosted mode is for synthetic or de-identified demo inputs only; traces do not retain raw audio.

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

Badges targeted (none guaranteed): 🔌 Off the Grid · 🦙 Llama Champion · 📡 Sharing is Caring · 🎯 Well-Tuned · 📓 Field Notes *(marked tentative by the organizers)* · 🎨 Off-Brand.

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
