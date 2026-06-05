# Figment

**An offline field-clinic copilot for rural clinics and disaster-response settings.**

Figment helps a **trained responder** follow local protocol cards, flag danger signs, ask for the information that's missing, and generate a referral note — **with no internet connection**. By design it runs a local ≤32B model, retrieves protocol cards from an on-device database, and fires deterministic red-flag rules *before* the model ever speaks. (Today the local model runs; the app that orchestrates retrieval and rules is in active development — see **Status** below.)

> ⚠️ **Figment is not a medical device.** It does not diagnose, prescribe, or replace a clinician. It is a prototype for protocol navigation, escalation support, and documentation in low-connectivity environments, for use by trained responders. See [Safety & non-goals](#safety--non-goals).

- **Status:** 🚧 In active development for the [Build Small Hackathon](docs/build-small-hackathon-org-card.md) (build window **June 5–15, 2026**). The local model runs today; the Figment app is being built per the [workback plan](docs/figment-workback-plan.md).
- **Track:** 🏡 Backyard AI (solve a real problem for a specific, real person you know).
- **Built for:** _a specific real responder — name & role to be filled in before submission_ (Backyard AI is judged on a real, named person who actually used it).
- **Model:** NVIDIA **Nemotron 3 Nano 30B-A3B** (30B total params ≤ 32B limit), run locally through **llama.cpp**.

---

## Why Figment

> What happens when the clinic loses internet?

Rural clinics, mobile units, and disaster sites lose connectivity exactly when decisions get hardest. Cloud medical assistants stop working; paper protocol binders don't talk back. Figment keeps working **offline**: it reads the same protocol cards a responder would, applies hard-coded danger-sign rules, and turns a messy field note into a structured handoff — all on the machine in front of you.

The design goal is restraint. Figment is **a field protocol binder that can talk, cite itself, and knows when to shut up** — not an "AI doctor."

---

## What it does

Figment is being built as a [Gradio](https://www.gradio.app/) app with five tabs. This is the **target feature set** — only the local model runs today (see the Status note above); the tabs below describe the intended behavior:

1. **Field Intake** — structured capture of setting, patient age, pregnancy status, chief concern, symptoms, vitals, allergies, medications, available supplies, and a free-text responder note.
2. **Risk Check** — deterministic red-flag rules fire **before** the LLM (e.g. altered mental status, severe respiratory distress, chest pain, stroke signs, pregnancy bleeding, pediatric lethargy, severe dehydration signs, anaphylaxis, uncontrolled bleeding, suspected sepsis).
3. **Protocol Guidance** — local retrieval returns 3–6 relevant protocol cards via SQLite FTS/BM25 (no embedding model needed).
4. **Handoff Note** — generates an SBAR note, a referral summary, a missing-info checklist, and the source protocol-card IDs.
5. **Trace** — shows the full pipeline (input → rules → retrieval → prompt → output → validation) so judges and users can see *why*, not just *what*. This is the "show, don't tell" engine.

---

## How it works

```text
Gradio Blocks UI
  → structured intake schema
  → rules.py        (deterministic red-flag engine)
  → retrieval.py    (SQLite FTS protocol search)
  → prompt_builder.py (constrained prompt; cards + rules injected)
  → llama.cpp local server → Nemotron 3 Nano 30B-A3B
  → validators.py   (output validator: JSON, citations, safety checks)
  → sbar.py         (referral note renderer)
  → trace.py        (trace export)
```

Two principles make this safe rather than chatty:

- **Rules before the model.** Danger-sign detection is deterministic code, not a model guess, so a red flag can't be "reasoned away."
- **The cards are the source of truth; the fine-tune is a behavior harness.** The model is taught to stay inside retrieved cards, cite card IDs, ask for missing info, escalate red flags, and refuse out-of-scope requests — not to memorize medical facts.

---

## The model & the ≤32B constraint

The Build Small Hackathon caps models at **32B total parameters**. Figment uses **NVIDIA Nemotron 3 Nano 30B-A3B** — a MoE hybrid Mamba-Transformer with **30B total** parameters (≈3B active per forward pass).

> **Compliance:** total parameters = **30B ≤ 32B** (~2B headroom). The ~3B *active* figure is **not** the compliance number — the limit is on *total* parameters. A LoRA adapter adds far less than 1B trainable params, so base+adapter stays ≤ 32B.

Local quantization (from the [Bartowski GGUF repo](https://huggingface.co/bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF), verified June 2026):

| Quant | Size | Use |
| ----- | ---- | --- |
| Q4_K_M | 24.66 GB | **primary local demo** |
| Q5_K_M | 26.15 GB | viable alternative |
| Q6_K | 33.43 GB | tight on 48 GB unified memory |
| Q8_0 | 33.51 GB | tight |
| BF16 | 63.18 GB | training base (not for local inference) |

> Q6_K and Q8_0 are near-identical in size — a genuine quantization artifact of this MoE/hybrid model, not a typo.

Reference dev/demo machine: an M4 Pro MacBook Pro with 48 GB RAM (24.66 GB Q4_K_M weights leave ~23 GB for runtime, KV cache, Gradio, and the OS).

---

## Getting started

### 1. Run the model locally (works today)

**Prerequisites:** [llama.cpp](https://github.com/ggml-org/llama.cpp) and ~25 GB free RAM for the Q4_K_M weights (Python 3.10+ for the app once it lands).

```bash
brew install llama.cpp

llama-server \
  -hf bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF:Q4_K_M \
  --ctx-size 16384 \
  --port 8001 \
  --host 127.0.0.1 \
  --temp 0.4 \
  --top-p 0.9
```

This exposes an OpenAI-compatible endpoint at `http://127.0.0.1:8001`.

### 2. Run the Figment app (in development)

> The app (`app.py` + the `figment/` package) is being built per the [workback plan](docs/figment-workback-plan.md). When it lands:

```bash
pip install -r requirements.txt
python app.py          # launches the Gradio app against the local llama-server
```

### 3. Hosted demo

A Gradio Space will be hosted under the **build-small-hackathon** Hugging Face org (link added at submission). The hosted Space runs self-contained — a smaller quant or a canned-response fallback — so it works without the author's laptop.

---

## Repository layout

This repo currently holds the planning docs; the application code is added during the build window. Target structure (see the [workback plan §4](docs/figment-workback-plan.md)):

```text
figment/
  app.py                # Gradio Blocks UI
  figment/              # config, schemas, rules, retrieval, model_client,
                        #   prompt_builder, validators, trace, sbar
  data/
    protocol_cards/     # 10 prototype cards (JSON)
    synthetic/          # train / validation / test / rejected (JSONL)
    eval/               # gold + adversarial cases, eval results
  scripts/              # build_fts, generate/critique/validate, make_sft, run_eval, export_traces
  modal/                # fine-tune + eval jobs (finetune_4b, finetune_30b, eval_batch, export_adapter)
  traces/               # exported demo traces
  docs/                 # field notes, model/dataset/safety cards, this plan
  release/              # demo video, social post
```

Available now:

```text
docs/figment-workback-plan.md                 # the full day-by-day build plan
docs/build-small-hackathon-org-card.md         # hackathon rules (source of truth)
docs/superpowers/specs/  docs/superpowers/plans/  # design spec + implementation plan for plan additions
modal-starter.py                               # Modal smoke-test stub
```

Key docs: [workback plan](docs/figment-workback-plan.md) · [hackathon rules](docs/build-small-hackathon-org-card.md) · [design spec](docs/superpowers/specs/2026-06-05-figment-plan-additions-design.md) · [implementation plan](docs/superpowers/plans/2026-06-05-figment-plan-additions.md).

---

## Data & evaluation

- **Synthetic data, not memorized facts.** 5,000–10,000 candidate cases are generated by teacher models (Mistral/MiniMax, build-time only), cross-critiqued, and filtered by a deterministic validator down to ~2,000–4,000 kept examples. No real PHI is used.
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

> These are **target thresholds, not measured results** — no evaluation has been run yet (the model + app are still being built).

Deterministic metrics run in `scripts/run_eval.py`; judgment metrics use a held-out judge model.

---

## Safety & non-goals

Figment is deliberately scoped. **It will not:**

- **diagnose** — it surfaces protocol cards and danger signs; it does not name a condition as fact;
- **prescribe or dose medication** — doses appear only if a cited card contains them;
- **replace a clinician** — the trained responder remains the decision-maker;
- **serve untrained users** — it is a tool for trained responders;
- **store or transmit PHI** — patient inputs stay on-device and are never logged or sent off-device;
- **act autonomously** — every output is advisory and requires human judgment.

This posture reflects real risk: the WHO has warned that authoritative-sounding health AI can create automation bias, and the FDA regulates clinical-decision-support software depending on its claims and users. Figment makes no clinical claims. A fuller `docs/safety_statement.md` will be published with the submission.

---

## Licensing & data handling

| Artifact | License |
| -------- | ------- |
| Model / adapter | inherits the NVIDIA Nemotron model license *(confirm exact terms)* |
| Synthetic dataset | CC-BY-4.0 *(to confirm)* |
| Code | Apache-2.0 *(to confirm)* |

Data handling: the app is local-only; patient inputs are never logged or transmitted; training data is synthetic with no real PHI.

---

## Demo cases

Three canonical cases drive the demo:

1. **Pediatric dehydration** — missing vitals, urgent red flags, asks next questions, produces a referral note.
2. **Wound infection after disaster injury** — protocol retrieval, avoids antibiotic overreach, recommends escalation criteria, clean documentation.
3. **Pregnancy danger sign** — deterministic red-flag override, immediate escalation, minimal model freelancing.

---

## Hackathon

Built for the **[Build Small Hackathon](docs/build-small-hackathon-org-card.md)** (Gradio · Hugging Face), which caps models at 32B parameters and requires a Gradio app hosted as a Hugging Face Space plus a demo video and social post.

Badges targeted (none guaranteed): 🔌 Off the Grid · 🦙 Llama Champion · 📡 Sharing is Caring · 🎯 Well-Tuned · 📓 Field Notes *(marked tentative by the organizers)* · 🎨 Off-Brand.

---

## Acknowledgements

- **NVIDIA** — Nemotron 3 Nano model · **Modal** — fine-tune/eval compute · **Gradio** & **Hugging Face** — app framework and hosting · **llama.cpp** — local inference.

---

## Disclaimer

Figment is a **prototype for trained responders**, not medical advice and not a medical device. It does not diagnose or prescribe. Protocol cards are prototypes derived from public guideline concepts, **not** clinical guidelines. Always rely on qualified clinical judgment and local protocols.

<!-- TODO before submission:
  - Name the specific real responder Figment was built for and tested with (Backyard AI requirement).
  - Add the hosted HF Space URL.
  - Confirm the three license choices above and add a LICENSE file.
  - Add the demo video + social post links.
-->
