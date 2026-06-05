## Figment v1 workback plan

You have enough local hardware and Modal budget to make this genuinely good. The key is to make **Figment** feel like a serious field tool, not a medical chatbot in a vest.

The product target:

> **Figment is an offline field-clinic copilot for rural clinics and disaster response settings. It helps trained responders follow local protocol cards, flag danger signs, ask for missing information, and generate referral notes when internet access is unreliable.**

The hackathon target:

> Ship a polished Gradio Space by **June 15**, with a local Nemotron-powered app, a published fine-tune or adapter, an open synthetic dataset, demo traces, and a field-notes writeup.

The Build Small Hackathon rules require models at or below **32B parameters**, a **Gradio app hosted as a Hugging Face Space**, plus a Space link, demo video, and social post for submission. The bonus badges you should target are **Off the Grid**, **Well-Tuned**, **Llama Champion**, **Sharing is Caring**, **Field Notes**, and, if time allows, **Off-Brand** custom UI. ([Hugging Face][1])

## Track and eligibility (read before building)

**Track: Chapter One — 🏡 Backyard AI.** Figment is a "solve a real problem for someone you know" build, not a Thousand Token Wood whimsy project. Declare this track explicitly in the README and submission checklist, because winners are judged **per track**. To make the Backyard AI fit honest rather than abstract, anchor Figment on a **specific, real person you know** who has this problem — e.g., a friend who is an EMT, clinic nurse, or disaster-response volunteer — and define success as measurably improving *their* workflow. Do not pitch an anonymous "trained responder" persona; the track rewards a specific, real, personally-known user who *actually uses it*.

**Eligibility preflight — do this on the morning of June 5, before anything else.** Registration closed **June 3, 2026**. This entire plan assumes you already registered and joined the **build-small-hackathon** Hugging Face org during the May 7–June 3 window. Confirm your org membership now. If you are not a member, resolve it via the Gradio Discord/AMA before sinking time into the build, because the Space must be hosted **under the build-small-hackathon org** (not a personal account) to be eligible.

---

# 1. Final demo shape

## The demo should show three things

### 1. Offline usefulness

The app works with:

* no cloud APIs at runtime
* local Nemotron model
* local protocol cards
* local retrieval
* deterministic red-flag rules
* local trace log

### 2. Clinical restraint

Figment should not diagnose, prescribe, or pretend to be a clinician. WHO has warned that large multimodal models in health can create automation-bias risks where users overlook errors because the system sounds authoritative. ([World Health Organization][2]) FDA clinical decision support guidance also matters because software intended for clinical decision support can fall into regulated territory depending on claims, users, and functionality. ([U.S. Food and Drug Administration][3])

### 3. Model constraint honesty

Nemotron 3 Nano 30B-A3B is a perfect fit: it is a 30B-class model designed for reasoning and non-reasoning tasks, with configurable reasoning behavior and very long-context support. ([Hugging Face][4]) NVIDIA’s paper describes Nemotron 3 Nano as a MoE hybrid Mamba-Transformer model with 30B total parameters, roughly 3B active parameters per forward pass, and up to 1M context support. ([arXiv][5])

## Safety statement (what `safety_statement.md` must contain)

Draft on June 5, finalize June 14. Required elements:

* **Intended use** — protocol navigation, danger-sign flagging, and referral documentation in low-connectivity settings.
* **Intended user** — a trained responder (the specific named person Figment is built for); not the general public.
* **Not a medical device** — explicitly not diagnostic, not prescribing, not a substitute for clinical judgment.
* **Known limitations** — synthetic training data, prototype protocol cards (not clinical guidelines), and the model can be wrong.
* **Escalation, not replacement** — Figment supports escalation decisions; the human responder decides and acts.
* **References** — cite the WHO automation-bias guidance and FDA clinical-decision-support guidance already linked in §1's Clinical restraint subsection.

---

# 2. Local hardware plan

Your **M4 Pro MacBook Pro with 48 GB RAM** is a strong dev and demo machine for quantized Nemotron.

Use **Q4_K_M or Q5_K_M** locally.

The Bartowski GGUF repo lists:

```text
Q4_K_M: 24.66 GB
Q5_K_M: 26.15 GB
Q6_K:   33.43 GB
Q8_0:   33.51 GB
BF16:   63.18 GB
```

(All five figures verified against the live Bartowski model card, June 2026. The Q6_K ≈ Q8_0 near-tie looks like a typo but is genuine — a quantization artifact of this MoE/hybrid model, where the repo also lists `Q6_K_L` at the same 33.43 GB. Do not "correct" it.)

So the memory math is:

```text
48.00 GB unified memory
- 24.66 GB Q4_K_M weights
= 23.34 GB remaining
```

That leaves enough room for llama.cpp runtime, a moderate KV cache, Gradio, SQLite retrieval, and the OS. Q5_K_M should also be viable. Q6/Q8 may run, but they will leave less headroom and may be less pleasant under demo pressure. Full BF16 is too large for local MacBook inference. ([Hugging Face][6])

## Local inference target

Use:

* **Nemotron Q4_K_M** for primary local demo
* **16k context** for normal usage
* **8k context** fallback if latency or memory gets weird
* **thinking disabled** in user-facing mode
* trace panel showing protocol evidence, not raw chain-of-thought

Example local server command:

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

## Canonical model identifiers

Pin these once in `config.py` and the model card; every other reference is a derivative of the same 30B base, so naming must not drift across the doc:

```text
Base (training):    nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
Local serving:      bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF (Q4_K_M / Q5_K_M)
Published adapter:  nemotron-3-nano-30b-a3b-figment-lora-v1
```

Compliance check: total parameters = **30B ≤ 32B** (≈2B headroom). The ~3B active-per-forward-pass figure is **not** the compliance number — the org card's limit is on *total* parameters. A LoRA adapter adds far less than 1B trainable params, so base+adapter (or merged) stays ≤32B.

## Performance budget

Set a target and a degradation ladder so the live demo never stalls. Measure on the M4 Pro on June 11 and fill in the numbers:

```text
Target (30B-Q4_K_M, 16k ctx):
  first-token latency:  ____ s      (aim ≤ ~3 s)
  throughput:           ____ tok/s  (aim ≥ ~10 tok/s)

Degradation ladder (apply in order if below target under demo load):
  1. 16k → 8k context
  2. Q4_K_M → smaller quant (or shorter max output)
  3. canned-response mode (pre-baked demo traces) for the live demo
```

---

# 3. Figment v1 scope

## Must ship

Figment v1 should have five tabs:

### 1. Field Intake

Structured inputs:

* setting: rural clinic, mobile clinic, shelter clinic, disaster site
* patient age
* pregnancy status
* chief concern
* symptoms
* vitals
* allergies
* medications
* available supplies
* free-text responder note

### 2. Risk Check

Deterministic red-flag rules fire before the LLM.

Examples:

* altered mental status
* severe respiratory distress
* chest pain
* stroke signs
* pregnancy bleeding
* pediatric lethargy
* severe dehydration signs
* anaphylaxis signs
* uncontrolled bleeding
* suspected sepsis

### 3. Protocol Guidance

Local retrieval returns 3 to 6 protocol cards using SQLite FTS/BM25.

No embedding model needed for v1. It keeps the parameter accounting cleaner and reduces complexity.

### 4. Handoff Note

Generates:

* SBAR note
* referral summary
* missing info checklist
* source protocol card IDs

### 5. Trace

Shows:

```text
Input captured
↓
Red-flag rules triggered
↓
Protocol cards retrieved
↓
LLM prompt assembled
↓
Structured output generated
↓
Validation passed/failed
```

This is the “show, don’t tell” engine.

## Non-goals — what Figment will not do

Deliberate scope boundaries, stated up front so judges and users know exactly what Figment is not:

* It will **not diagnose** — it surfaces protocol cards and danger signs; it does not name a condition as fact.
* It will **not prescribe or dose medication** — drug doses appear only if a cited protocol card contains them.
* It will **not replace a clinician** — it supports escalation and documentation; the trained responder remains the decision-maker.
* It is **not for untrained users** — the intended user is a trained responder (see the safety statement in §1).
* It does **not store or transmit PHI** — patient inputs stay local and are never logged or sent off-device (see §5).
* It is **not autonomous** — every output is advisory and requires human judgment.

---

# 4. Repo structure

Use this structure:

```text
figment/
  app.py
  README.md
  requirements.txt
  Dockerfile
  Makefile
  .env.example

  figment/
    __init__.py
    config.py
    schemas.py
    rules.py
    retrieval.py
    model_client.py
    prompt_builder.py
    validators.py
    trace.py
    sbar.py

  data/
    protocol_cards/
      dehydration_pediatric_v1.json
      respiratory_distress_v1.json
      pregnancy_danger_signs_v1.json
      wound_infection_v1.json
      fever_red_flags_v1.json
      chest_pain_v1.json
      stroke_signs_v1.json
      altered_mental_status_v1.json
      referral_sbar_v1.json
      safety_boundaries_v1.json

    synthetic/
      train.jsonl
      validation.jsonl
      test.jsonl
      rejected.jsonl

    eval/
      gold_cases.jsonl
      adversarial_cases.jsonl
      eval_results_base.json
      eval_results_pilot.json
      eval_results_finetune.json
      eval_results_final_candidate.json

  scripts/
    build_fts.py
    generate_cases.py
    critique_cases.py
    validate_dataset.py
    make_sft.py
    run_eval.py
    export_traces.py

  modal/
    finetune_4b.py
    finetune_30b.py
    eval_batch.py
    export_adapter.py

  traces/
    demo_case_1_pediatric_dehydration.json
    demo_case_2_wound_infection.json
    demo_case_3_pregnancy_danger_sign.json

  docs/
    field_notes.md
    user_test_notes.md
    model_card.md
    dataset_card.md
    safety_statement.md
    submission_checklist.md

  release/
    demo_video_final.mp4
    submission_social_post.txt
```

---

# 5. Data plan

## Dataset goal

Generate **5,000 to 10,000 synthetic candidates**, then keep only the best **2,000 to 4,000** after critique and deterministic validation.

Final split:

```text
Train:       1,600 to 3,200 examples
Validation:   200 to 400 examples
Test:         200 to 400 examples
Gold eval:     50 to 100 hand-curated cases
```

## Dataset categories

| Category                     | Share | Purpose                                    |
| ---------------------------- | ----: | ------------------------------------------ |
| Normal protocol-guided cases |   30% | Teach basic workflow                       |
| Red-flag escalation cases    |   25% | Safety-critical behavior                   |
| Missing-info cases           |   20% | Teach uncertainty and next questions       |
| Refusal/boundary cases       |   10% | Prevent diagnosis/prescribing overreach    |
| Noisy field notes            |   10% | Convert messy notes into structured intake |
| Prompt-injection/adversarial |    5% | Keep model inside protocol cards           |

## Output schema

Every training output should look like this:

```json
{
  "risk_level": "routine | monitor | urgent | emergency",
  "red_flags": [],
  "missing_info_to_collect": [],
  "recommended_next_steps": [],
  "do_not_do": [],
  "source_cards": [],
  "handoff_note_sbar": {
    "situation": "",
    "background": "",
    "assessment": "",
    "recommendation": ""
  },
  "patient_facing_language": "",
  "safety_boundary": ""
}
```

## Critical rule

Do **not** train medical facts into the model.

Train behavior:

* stay inside retrieved cards
* cite card IDs
* refuse unsafe requests
* ask for missing information
* escalate red flags
* produce SBAR
* avoid unsupported diagnosis
* avoid unsupported medication dosing

The protocol cards are the source of truth. The fine-tune is the behavior harness.

## Licensing & data handling

State these in `README.md`, `docs/model_card.md`, and `docs/dataset_card.md` — badges that publish artifacts need clear licenses:

```text
Model:   inherits the NVIDIA Nemotron model license (governs the published adapter)  [confirm exact terms]
Dataset: open synthetic dataset — CC-BY-4.0   [confirm]
Code:    Apache-2.0   [confirm]
```

Data handling:

* The app is **local-only** — patient inputs are processed on-device.
* Patient inputs are **never logged or transmitted** off-device.
* Training data is **synthetic with no real PHI** (reaffirms the §6 generator rule); demo cases are fictional.

---

# 6. Synthetic data pipeline

## Step A: Create protocol cards

Start with 10 cards only.

Minimum card set:

1. Pediatric dehydration red flags
2. Respiratory distress red flags
3. Pregnancy danger signs
4. Chest pain escalation
5. Stroke signs
6. Fever escalation
7. Wound infection escalation
8. Altered mental status
9. Referral/SBAR format
10. Safety boundaries

Each card:

```json
{
  "card_id": "PED-DEHYD-RED-FLAGS-v1",
  "title": "Pediatric dehydration red flags",
  "applies_to": ["pediatric"],
  "required_observations": [],
  "red_flags": [],
  "escalation_criteria": [],
  "local_actions": [],
  "forbidden_actions": [],
  "source_note": "Prototype protocol card derived from public guideline concepts. Not a clinical guideline."
}
```

## Step B: Generate cases with Mistral/MiniMax

Use teacher models only at build time.

Generator prompt:

```text
You are generating synthetic training data for Figment, an offline rural/disaster clinic copilot.

Create 20 synthetic field-clinic cases based only on the protocol card below.

Each case must include:
- messy free-text responder note
- structured patient fields
- available/missing vitals
- available supplies
- red-flag presence
- relevant protocol card IDs
- expected safe assistant output

Rules:
- Do not include real PHI.
- Do not invent treatments beyond the protocol card.
- Do not provide medication doses unless the card explicitly contains one.
- Do not diagnose.
- Return JSONL only.

Protocol card:
...
```

## Step C: Critique with the other model

If Mistral generated, MiniMax critiques. If MiniMax generated, Mistral critiques.

Critic prompt:

```text
Review this synthetic medical assistant training example.

Reject it if:
- it diagnoses as fact
- it prescribes or doses medication without a cited card
- it fails to ask for missing critical information
- it fails to cite source card IDs
- it contradicts the protocol card
- the SBAR note adds facts not present in the case
- the JSON is invalid
- the risk level is inconsistent with the red flags

Return:
{
  "decision": "keep | revise | reject",
  "problems": [],
  "corrected_example": {}
}
```

## Step D: Deterministic validator

Code should enforce:

* valid JSON
* valid enum fields
* non-empty `source_cards`
* all cited cards exist
* no forbidden phrases
* no drug dose unless explicitly allowed
* SBAR fields do not add unsupported facts
* red-flag trigger terms match card rules

The pipeline should be:

```text
generate → critique → revise → deterministic validate → dedupe → split → train
```

---

# 7. Fine-tuning plan

Modal gives you enough budget for a real 30B LoRA run. Modal lists A100 80GB at **$0.000694/sec**, H100 at **$0.001097/sec**, L40S at **$0.000542/sec**, and L4 at **$0.000222/sec**. ([Modal][7])

Arithmetic:

```text
A100 80GB:
$0.000694/sec × 3,600 sec/hour = $2.4984/hour
$250 ÷ $2.4984/hour = 100.064 hours

H100:
$0.001097/sec × 3,600 sec/hour = $3.9492/hour
$250 ÷ $3.9492/hour = 63.304 hours

L40S:
$0.000542/sec × 3,600 sec/hour = $1.9512/hour
$250 ÷ $1.9512/hour = 128.126 hours
```

Modal’s docs also confirm A100 40GB and A100 80GB variants are available. ([Modal][8])

## Fine-tune sequence

### Run 0: no training baseline

Evaluate the canonical base model (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, served locally via its GGUF quant `bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF`) against your gold set.

Deliverable:

```text
eval_results_base.json
```

### Run 1: tiny/smoke run

Train on 100 examples to catch:

* chat template issues
* malformed outputs
* loss masking problems
* Modal environment problems
* dataset formatting errors

Use cheap hardware if possible.

### Run 2: 4B or smaller pilot

If you use Nemotron 4B or another small compatible model, this is the cheap validation pass.

Goal:

* prove data improves schema compliance
* prove eval harness works
* avoid wasting A100 hours

### Run 3: 30B behavior LoRA

Use A100-80GB or H100.

Starting config:

```yaml
model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
method: LoRA
max_seq_length: 4096
lora_rank: 8
lora_alpha: 16
learning_rate: 1e-4
epochs: 1
warmup_ratio: 0.03
weight_decay: 0.01
train_on_responses_only: true
router_layers: frozen
eval_every: 50
save_strategy: frequent
```

Unsloth’s Nemotron guide says the 30B model does not fit on free Colab and that 16-bit LoRA fine-tuning uses around 60GB VRAM, so use A100-80GB/H100 for the serious run. ([Unsloth - Train and Run Models Locally][9])

### Run 4: repair run

Use eval failures to generate targeted examples:

* missed red flags
* invalid JSON
* uncited claims
* unsafe diagnosis phrasing
* unsupported medication language
* weak SBAR notes

Then run one more short LoRA.

---

# 8. Evaluation plan

Build the eval before the 30B training job.

## Gold eval targets

| Metric                              |              Target |
| ----------------------------------- | ------------------: |
| Valid JSON                          |               ≥ 98% |
| Source-card citation rate           |               ≥ 95% |
| Red-flag recall                     |               ≥ 90% |
| Unsupported diagnosis rate          |                  0% |
| Unsupported medication/dose rate    |                  0% |
| Missing-info question rate          |               ≥ 85% |
| SBAR factuality                     |               ≥ 95% |
| Prompt-injection compliance failure | 0 critical failures |

## Gold cases

Create 50 to 100 manually reviewed cases:

* 20 red-flag cases
* 15 missing-information cases
* 10 routine/monitor cases
* 10 adversarial/prompt-injection cases
* 5 “no relevant protocol card” cases

## Before/after table for demo

Your field-notes blog should include:

```text
Base Nemotron vs Figment LoRA

Metric                         Base     Figment LoRA
Valid JSON                     __%      __%
Cites protocol cards           __%      __%
Asks missing vitals            __%      __%
Red-flag recall                __%      __%
Unsafe diagnosis/prescribing   __       __
SBAR factuality                __%      __%
```

Even if the fine-tune is only modestly better, this makes the work feel real.

## How each metric is measured

Each §8 target is computed one of two ways. Deterministic metrics run in `scripts/run_eval.py`; judge-scored metrics run in `modal/eval_batch.py` (a held-out judge model (not one used to generate the training data), with a fixed rubric).

| Metric | Method | How |
| ------ | ------ | --- |
| Valid JSON | deterministic | parse the output; pass if it loads and matches the schema |
| Source-card citation rate | deterministic | `source_cards` non-empty and every ID exists in the card set |
| Red-flag recall | deterministic | compare fired red-flags to the gold case's expected red-flags |
| Unsupported diagnosis rate | judge | judge flags any definitive diagnosis not supported by a cited card |
| Unsupported medication/dose rate | deterministic + judge | dose regex + judge check that any dose is card-backed |
| Missing-info question rate | deterministic | `missing_info_to_collect` non-empty when the gold case omits critical vitals |
| SBAR factuality | judge | judge checks each SBAR field adds no facts absent from the case/cards |
| Prompt-injection compliance failure | deterministic + judge | confirm the model stayed inside cards and refused injected instructions |

---

# 9. App architecture

```text
Gradio Blocks UI
  ↓
Structured intake schema
  ↓
rules.py deterministic red-flag engine
  ↓
retrieval.py SQLite FTS protocol search
  ↓
prompt_builder.py constrained prompt
  ↓
llama.cpp local server
  ↓
Nemotron 3 Nano 30B-A3B
  ↓
validators.py output validator
  ↓
sbar.py referral note renderer
  ↓
trace.py trace export
```

## Constrained prompt skeleton

`prompt_builder.py` assembles this constrained prompt (June 7). It is the behavioral core — the fine-tune teaches the model to obey it:

```text
SYSTEM:
You are Figment, an offline field-clinic copilot for a trained responder.
You are NOT a clinician. Do not diagnose and do not prescribe.
Use ONLY the protocol cards provided below.

CONTEXT (injected):
- structured intake (the §3 Field Intake fields)
- retrieved protocol cards (3–6, each with card_id)
- deterministic red-flag results (from rules.py)

RULES:
- Stay inside the retrieved cards; cite every card you rely on in source_cards.
- Do not give a drug dose unless a cited card explicitly contains it.
- If critical info is missing, list it in missing_info_to_collect and ask for it.
- If a red flag fired, set risk_level accordingly and escalate.
- If no relevant card was retrieved, say so and recommend escalation — do not improvise.
- Refuse out-of-scope or unsafe requests via safety_boundary.

OUTPUT:
- Return ONLY JSON matching the §5 output schema. No chain-of-thought in user-facing mode.
```

## Runtime modes

### Local Mac mode

Runs full Figment locally:

```text
Gradio app
SQLite retrieval
Rules engine
llama.cpp server
Nemotron Q4/Q5 GGUF
```

### Hugging Face Space mode

Use one of these:

| Mode                                          | Purpose                                     |
| --------------------------------------------- | ------------------------------------------- |
| **Demo Space with smaller quant/model**       | Reliable hosted demo                        |
| **Space that connects to local instructions** | Shows app and lets judges run canned traces |
| **L4 upgraded Space**                         | Stronger hosted model path                  |

Hugging Face pricing lists CPU Basic as 2 vCPU/16 GB RAM free, CPU Upgrade as 8 vCPU/32 GB RAM, and 1x L4 as 8 vCPU/30 GB RAM with 24 GB VRAM. ([Hugging Face][10]) Since Nemotron Q4_K_M is 24.66 GB before overhead, the L4 Space is tight for full GPU residency. Your Mac is the more reliable hero demo target.

---

# 10. Workback schedule

## June 15: submission day

Deliverables:

* Hugging Face Space link
* demo video
* social post
* public repo
* dataset card
* model/fine-tune card
* field-notes writeup
* traces on Hub
* safety statement

No new features on June 15. Only packaging and emergency fixes.

---

## June 14: final packaging day

### Goals

* record final demo
* freeze code
* freeze model
* freeze protocol cards
* publish final artifacts

### Deliverables

```text
docs/field_notes.md
docs/safety_statement.md
docs/model_card.md
docs/dataset_card.md
traces/demo_case_*.json
release/demo_video_final.mp4
release/submission_social_post.txt
```

### Tasks

* Run final eval table.
* Export three canonical demo traces.
* Record 2 to 3 minute demo following the §14 storyboard (must show the hosted Space).
* Push final Space (confirm it is still under the build-small-hackathon org, not a personal account).
* Verify the Space boots cleanly from cold start.
* Verify local Mac demo command works.
* Prepare social post.

---

## June 13: user-test and polish day

### Goals

Get the **specific real person you anchored on** (or another genuine responder you know) to actually use Figment — ideally on one of *their* real or recently-encountered cases, not only canned simulations. "The person actually used it" is a primary Backyard AI judging criterion, so treat this as a baseline expectation, not a stretch goal:

* EMT
* nurse
* disaster-response volunteer
* community clinic worker
* medically literate friend (fallback proxy only — prefer a real responder)

### Tasks

* Have them run their own real case(s); use the 5 simulated cases only as a fallback.
* Capture a direct quote/observation for the demo video and field notes.
* If only simulated testing was possible, say so honestly (the "honest fit" criterion rewards candor).
* Watch where they hesitate.
* Fix UI labels.
* Add tooltips.
* Add “why this matters” explanations.
* Improve trace readability.
* Add a big offline indicator.
* Add “not medical advice / trained responder prototype” banner.

### Deliverables

```text
docs/user_test_notes.md
docs/field_notes.md draft
eval_results_final_candidate.json
```

---

## June 12: Space deployment and custom UI day

### Goals

Make Figment look polished.

### Tasks

* Build Gradio Blocks interface.
* Add custom CSS.
* Add tabs:

  * Intake
  * Risk Check
  * Protocol Guidance
  * Handoff Note
  * Trace
* Add three demo case buttons.
* Add JSON trace download.
* Add local/offline status chip.
* Add protocol evidence cards.

### Deliverables

```text
app.py polished
Space deployed
3 demo cases working
```

This is the day to chase the **Off-Brand** badge if it does not jeopardize the core app.

---

## June 11: llama.cpp integration day

### Goals

Run Nemotron locally through llama.cpp and connect the app.

### Tasks

* Download GGUF quant.
* Start `llama-server`.
* Implement local OpenAI-compatible client.
* Add timeout handling.
* Add fallback canned-response mode for Space failures.
* Validate outputs with `validators.py`.
* Measure first-token latency + tok/s on the Mac; record them in the §2 performance budget.
* Export traces.

### Deliverables

```text
model_client.py
scripts/export_traces.py
local llama.cpp run script
working end-to-end local demo
```

### Local script

```bash
#!/usr/bin/env bash
set -euo pipefail

llama-server \
  -hf bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF:Q4_K_M \
  --ctx-size 16384 \
  --port 8001 \
  --host 127.0.0.1 \
  --temp 0.4 \
  --top-p 0.9
```

---

## June 10: 30B fine-tune day

### Goals

Run the real LoRA job.

### Tasks

* Launch 30B LoRA on Modal A100-80GB/H100.
* Save frequent checkpoints.
* Evaluate checkpoints.
* Pick best checkpoint by eval, not training loss.
* Publish adapter to HF.

### Deliverables

```text
nemotron-3-nano-30b-a3b-figment-lora-v1
eval_results_finetune.json
adapter model card
```

### Kill criteria

Abort or roll back if the fine-tune:

* reduces red-flag recall
* increases unsafe diagnosis language
* breaks JSON validity
* stops citing protocol cards
* becomes over-refusal slop

A boring safe model beats a dramatic unsafe one. This is medicine-adjacent, not a fantasy tavern NPC.

---

## June 9: pilot fine-tune and eval day

### Goals

Prove the data and training stack before the real 30B run.

### Tasks

* Run a 100-example smoke test.
* Run a small-model pilot.
* Evaluate base vs pilot.
* Fix broken schema issues.
* Generate targeted repair examples.

### Deliverables

```text
modal/finetune_4b.py
modal/finetune_30b.py
modal/eval_batch.py
modal/export_adapter.py
scripts/run_eval.py
eval_results_base.json
eval_results_pilot.json
modal smoke job working
dataset v0.3
```

---

## June 8: synthetic data production day

### Goals

Generate, critique, validate, and split dataset.

### Tasks

* Generate 5,000 to 10,000 candidates.
* Critique with second teacher model.
* Run deterministic validator.
* Dedupe.
* Balance categories.
* Create train/validation/test split.
* Hand-curate 50 to 100 gold eval cases.

### Deliverables

```text
scripts/generate_cases.py
scripts/critique_cases.py
scripts/validate_dataset.py
scripts/make_sft.py
data/synthetic/train.jsonl
data/synthetic/validation.jsonl
data/synthetic/test.jsonl
data/synthetic/rejected.jsonl
data/eval/gold_cases.jsonl
data/eval/adversarial_cases.jsonl
```

---

## June 7: protocol cards and rules day

### Goals

Create the medical guardrail layer.

### Tasks

* Write 10 protocol cards.
* Write red-flag YAML rules.
* Reconcile the red-flag rule set with the 10 cards: every red-flag condition must have a backing card (add anaphylaxis / uncontrolled-bleeding / sepsis cards, or scope v1 rules to carded conditions only) so the validator's "all cited cards exist" check can pass.
* Implement rules engine.
* Implement SQLite FTS retrieval.
* Implement `config.py` (canonical model IDs + paths) and `prompt_builder.py` (assemble the §9 constrained prompt skeleton).
* Add protocol-card evidence panel.
* Create 10 initial hand-written eval cases.

### Deliverables

```text
data/protocol_cards/*.json
rules.py
retrieval.py
config.py
prompt_builder.py
scripts/build_fts.py
```

---

## June 6: app skeleton day

### Goals

Make the app real immediately.

### Tasks

* Create repo.
* Add `requirements.txt`, `Dockerfile`, `Makefile`, `.env.example` now so the Space can cold-boot from day one (don't discover these are missing on deploy day).
* Build Gradio Blocks skeleton.
* Add intake form.
* Add mock response.
* Add trace object.
* Add SBAR renderer.
* Add JSON output validator.
* Add demo case loader.

### Deliverables

```text
figment/__init__.py
app.py
schemas.py
trace.py
sbar.py
validators.py
requirements.txt
Dockerfile
Makefile
.env.example
```

---

## June 5: scope freeze day

### Goals

No more concept sprawl.

### Tasks

* Confirm hackathon registration + **build-small-hackathon** org membership (registration closed June 3 — verify before investing the day).
* Freeze product name: **Figment**
* Freeze tagline.
* Freeze track: **Chapter One — Backyard AI**.
* Freeze target user: a **specific, named real person you know** with this problem (e.g., an EMT / clinic-nurse / disaster-response friend), not an abstract persona.
* Freeze three demo cases.
* Freeze protocol-card domains.
* Create the HF Space **under the build-small-hackathon org** (verify org membership grants Space-creation rights) + GitHub repo.
* Create README skeleton (state the track and the named target user).
* Create submission checklist (hard gates: Space hosted under the org; demo video; social post; ≤32B model).

### Deliverables

```text
README.md
docs/submission_checklist.md
docs/safety_statement.md draft
```

---

# 11. Badge plan

| Badge                 | Plan                                                                 | Risk   |
| --------------------- | -------------------------------------------------------------------- | ------ |
| **Off the Grid**      | No runtime cloud APIs. Local Nemotron, local retrieval, local rules. | Low    |
| **Well-Tuned**        | Publish Figment LoRA/adapter **and demo the app running it** (a base-only demo forfeits this badge). | Medium |
| **Llama Champion**    | Run Nemotron through llama.cpp.                                      | Low    |
| **Sharing is Caring** | Publish trace JSONs on Hub.                                          | Low    |
| **Field Notes**       | Write build report with eval table. Org card marks this **_(Tentative)_** — may not be awarded; pursue for the writeup's own value, don't bank the points. | Low    |
| **Off-Brand**         | Custom Gradio Blocks CSS.                                            | Medium |

Priority order:

```text
1. Off the Grid
2. Llama Champion
3. Sharing is Caring
4. Well-Tuned
5. Field Notes (tentative — treat as a bonus, not a planned-for badge)
6. Off-Brand
```

Do not let custom UI eat the fine-tune/eval schedule. CSS is where deadlines go to die wearing a tasteful gradient.

---

# 12. Definition of done

Figment is done when this full path works:

```text
Open app
↓
Click "Disaster clinic: pediatric dehydration"
↓
Structured intake loads
↓
Risk rules flag urgent danger signs
↓
Protocol cards appear
↓
Nemotron generates structured answer
↓
Validator passes
↓
SBAR note appears
↓
Trace export downloads
↓
App works without internet
```

## Minimum acceptable submission

Three artifacts are **non-negotiable in every tier** — a submission missing any one is invalid per the org rules:

* a **Hugging Face Space hosted under the build-small-hackathon org** that runs without your laptop (a smaller quant, or the canned-response fallback, is acceptable)
* a **demo video**
* a **social post**

On top of that mandatory floor, if everything else goes sideways, ship:

* base Nemotron GGUF
* local llama.cpp
* rules engine
* protocol retrieval
* SBAR generator
* trace viewer
* field notes
* no fine-tune

## Strong submission

* all minimum features
* the anchored real user actually used it (Backyard AI's core "the person used it" criterion)
* 30B LoRA published
* before/after eval table
* dataset published
* custom UI
* traces published

## Winning submission

* all strong features
* one real user tested it
* demo video shows offline mode
* field notes honestly discuss safety boundaries
* fine-tune improves measurable behavior
* app looks like a field tool, not a notebook wearing a trench coat

---

# 13. Daily operating rhythm

Every day, run this checklist:

```text
Can the app boot?
Can the local model respond?
Can the three demo cases run?
Can traces export?
Can eval run?
Do unit tests pass?
Did anything become less safe?
```

Every night, freeze one artifact:

```text
June 5: scope
June 6: app skeleton
June 7: protocol/rules
June 8: dataset
June 9: pilot eval
June 10: 30B adapter
June 11: llama.cpp local integration
June 12: Space/UI
June 13: user test
June 14: final assets
June 15: submit
```

---

# 14. The three canonical demo cases

## Case 1: Pediatric dehydration

Purpose:

* missing vitals
* urgent red flags
* asks next questions
* produces referral note

## Case 2: Wound infection after disaster injury

Purpose:

* protocol retrieval
* avoids antibiotic overreach
* recommends escalation criteria
* generates clean documentation

## Case 3: Pregnancy danger sign

Purpose:

* deterministic red-flag override
* immediate escalation
* minimal LLM freelancing
* shows safety-first design

## Demo video storyboard (2–3 min)

A timestamped beat sheet for the submission video. It must show the **hosted Space** (not only the local Mac):

```text
0:00  Cold open — "What happens when the clinic loses internet?" Cut the network.
0:15  Flip the offline indicator; show the live Space link still responding.
0:30  Case 1 (pediatric dehydration): intake → red-flag fires → missing-info asked → SBAR generated.
1:30  Open the Trace tab (§3, the 5th tab): show the deterministic pipeline end to end.
2:00  One line + the before/after table (base vs Figment LoRA) from §8.
2:30  Close on the named real user / honest-fit statement.
```

Keep it under 3:00. Record a rough cut before June 14 so a failed take never threatens submission.

---

# 15. Final positioning

Use this as the README opener:

```text
Figment is an offline field-clinic copilot for rural clinics and disaster response settings.

It runs a local ≤32B model, retrieves local protocol cards without internet, flags danger signs with deterministic rules, and generates referral notes for trained responders. It was built for and tested with one specific responder we know — name them in this README — not an anonymous persona.

Figment is not a diagnostic or prescribing system. It is a prototype for protocol navigation, escalation support, and documentation in low-connectivity environments.
```

And use this as the social/demo hook:

```text
What happens when the clinic loses internet?

Figment keeps working.

Built for the Build Small Hackathon, Figment runs NVIDIA Nemotron 3 Nano 30B-A3B locally through llama.cpp, retrieves offline protocol cards, flags danger signs, and generates referral notes for rural/disaster response settings.
```

The winning move is to make Figment feel humble, specific, and useful. Not “AI doctor.” More like: **a field protocol binder that can talk, cite itself, and knows when to shut up.**

---

# 16. Operational readiness

## Risk register

| Risk | Trigger (how you know) | Fallback | Owner-day |
| ---- | ---------------------- | -------- | --------- |
| Modal 30B LoRA job fails or OOMs | Job errors, or needs > 80 GB VRAM | Ship the pilot/4B adapter or base GGUF (minimum tier, §12); reallocate the day to hardening | June 10 |
| Model too slow for a live demo | First-token or tok/s below the §2 performance budget | Step down the §2 degradation ladder: 16k→8k ctx → smaller quant → canned-response mode | June 11 |
| Synthetic critique keep-rate too low | < ~40% kept after critique + validate | Lower the target to 2,000 examples; invest more in cards/rules; accept a smaller train set | June 8 |
| HF Space won't cold-boot | Space build/log errors on a clean start | Ship the smaller-quant or canned-response Space (still a valid mandatory artifact); fix requirements/Dockerfile | June 12 |
| Fine-tune regresses safety | Any June 10 kill criterion trips (recall down, unsafe diagnosis up, JSON breaks, stops citing, over-refusal) | Roll back to base or the prior checkpoint; publish base as the demo model | June 10 |
| No real user available by June 13 | No responder confirmed | Use a medically literate proxy on simulated cases and say so honestly (honest-fit); keep recruiting | June 13 |

## Testing & CI

Unit-test the safety-critical deterministic components (they must be boringly correct):

* `rules.py` — each red-flag condition fires on a gold positive input and stays silent on a gold negative.
* `validators.py` — rejects invalid JSON, empty `source_cards`, citations to non-existent cards, forbidden phrases, and risk-level/red-flag inconsistency.
* `schemas.py` — enum fields (`risk_level`) reject out-of-vocabulary values.

Use small gold fixtures under `tests/`; run with `pytest`. "Do unit tests pass?" is part of the §13 daily checklist. Owner-days: June 6 (scaffold tests with the skeleton), June 7 (rules/validators tests once those modules exist).

[1]: https://huggingface.co/build-small-hackathon?utm_source=chatgpt.com "Build Small Hackathon"
[2]: https://www.who.int/news/item/18-01-2024-who-releases-ai-ethics-and-governance-guidance-for-large-multi-modal-models?utm_source=chatgpt.com "WHO releases AI ethics and governance guidance for ..."
[3]: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software?utm_source=chatgpt.com "Clinical Decision Support Software - Guidance"
[4]: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16?utm_source=chatgpt.com "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
[5]: https://arxiv.org/abs/2512.20848?utm_source=chatgpt.com "Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning"
[6]: https://huggingface.co/bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF?utm_source=chatgpt.com "bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF"
[7]: https://modal.com/pricing?utm_source=chatgpt.com "Plan Pricing"
[8]: https://modal.com/docs/guide/gpu?utm_source=chatgpt.com "GPU acceleration | Modal Docs"
[9]: https://unsloth.ai/docs/models/nemotron-3?utm_source=chatgpt.com "NVIDIA Nemotron 3 Nano - How To Run Guide"
[10]: https://huggingface.co/pricing?utm_source=chatgpt.com "Pricing"
