## Figment v1 workback plan

You have enough local hardware and Modal budget to make this genuinely good. The key is to make **Figment** feel like a serious field tool, not a medical chatbot in a vest.

The product target:

> **Figment is an offline protocol navigator for field clinics and disaster-response settings. Deterministic rules own danger-sign detection; the AI owns messy-note intake, protocol-pathway selection, missing-information planning, protocol-card synthesis, responder checklists, and referral handoffs — without diagnosing, prescribing, or overriding red flags.**
>
> **Audio assists intake only:** Hosted mode starts with NVIDIA Nemotron 3 Nano Omni's native multimodal path for the true Space demo. Local/offline mode uses Parakeet RNNT ASR plus a smaller Nemotron 3 Nano 4B text navigator/fine-tune target. In both modes, the medic must confirm or correct every audio-derived value before red-flag rules or the navigator run.

The hackathon target:

> Ship a polished Gradio Space by **June 15**, with a hosted Nemotron Omni-powered app, a local/offline Nemotron 3 Nano 4B + Parakeet path, an open synthetic dataset if time allows, demo traces, and a field-notes writeup. Fine-tuning should target the smaller 4B local model first; Omni fine-tuning is deferred unless the runtime demo, safety validation, local/offline proof, and 4B adapter story are already green.

Architecture note: **Nemotron Omni remains the hosted v1 default and submission demo story**. The local/off-grid target is now **NVIDIA Nemotron 3 Nano 4B BF16** for text navigation and fine-tuning, paired with **Parakeet RNNT 1.1B** for offline ASR. The old 30B text-plus-Parakeet split is no longer the preferred local path.

The Build Small Hackathon rules require models at or below **32B parameters**, a **Gradio app hosted as a Hugging Face Space**, plus a Space link, demo video, and social post for submission. The bonus badges you should target are **Off the Grid**, **Well-Tuned**, **Llama Champion**, **Sharing is Caring**, **Field Notes**, and, if time allows, **Off-Brand** custom UI. ([Hugging Face][1])

## Track and eligibility (read before building)

**Track: Chapter One — 🏡 Backyard AI.** Figment is a "solve a real problem for someone you know" build, not a Thousand Token Wood whimsy project. Declare this track explicitly in the README and submission checklist, because winners are judged **per track**. To make the Backyard AI fit honest rather than abstract, anchor Figment on a **specific real responder you know** — public role: **a disaster-response volunteer trained in disaster-response first aid and local protocol use; name withheld for privacy** — and define success as measurably improving *their* workflow. Do not pitch an anonymous "trained responder" persona; the track rewards a specific, personally-known user who *actually uses it*.

**Eligibility preflight — do this on the morning of June 5, before anything else.** Registration closed **June 3, 2026**. This entire plan assumes you already registered and joined the **build-small-hackathon** Hugging Face org during the May 7–June 3 window. Confirm your org membership now. If you are not a member, resolve it via the Gradio Discord/AMA before sinking time into the build, because the Space must be hosted **under the build-small-hackathon org** (not a personal account) to be eligible.

---

# 1. Final demo shape

## The demo should show five things

### 1. Offline usefulness

The local/offline mode works with:

* no cloud APIs at runtime
* local Nemotron 3 Nano 4B text navigator once verified
* local protocol cards
* local retrieval
* deterministic red-flag rules
* local Parakeet RNNT ASR for synthetic/offline dictated intake once verified
* typed intake or canned transcript fallback if local ASR is not stable
* local trace log

The hosted Space should still be a true interactive demo, not only a canned trace viewer. In Space mode, Figment can call a hosted or self-hosted Nemotron Omni endpoint so judges can exercise live audio-assisted intake and protocol navigation without your laptop. Label this honestly as hosted live mode; the offline claim belongs to the local 4B + Parakeet path. Do not claim local audio support until Parakeet ASR and local 4B navigation are both proven.

### 2. AI load-bearing protocol navigation

The AI is not decoration and not a clinician. It should visibly do useful protocol-navigation work that deterministic code would make brittle or tedious:

* identify candidate protocol pathways from retrieved cards
* reconcile unclear or conflicting observations instead of smoothing them away
* prioritize the next 3 to 5 observations to collect
* turn protocol cards into a case-specific responder checklist
* parse messy field notes into structured, uncertainty-marked facts
* draft the referral handoff in SBAR form

Deterministic code still owns hard danger-sign detection, validation, and "do not cross this line" safety checks.

### 3. Clinical restraint

Figment should not diagnose, prescribe, or pretend to be a clinician. WHO has warned that large multimodal models in health can create automation-bias risks where users overlook errors because the system sounds authoritative. ([World Health Organization][2]) FDA clinical decision support guidance also matters because software intended for clinical decision support can fall into regulated territory depending on claims, users, and functionality. ([U.S. Food and Drug Administration][3])

### 4. Model constraint honesty

Nemotron 3 Nano Omni 30B-A3B Reasoning is the frozen hosted model for the v1 Space demo. NVIDIA's model card and technical report state **31B total parameters**, roughly **3B active parameters per token**, multimodal input (**video, audio, image, text**), text output, a Mamba2-Transformer Hybrid MoE architecture, an integrated speech encoder, and up to **256k context**. ([Hugging Face][4]) ([NVIDIA][5])

The hosted compliance claim should cite the NVIDIA model-card value: **31B <= 32B**. There is one caveat: the Hugging Face sidebar currently reports 33B params for the same repo, so treat this as an organizer-confirmation risk in the risk register rather than a fact to hand-wave away.

The local/off-grid model story is cleaner: NVIDIA Nemotron 3 Nano 4B BF16 lists **3.97B parameters**, text input/output, a hybrid Mamba2-Transformer architecture, and 262K context on the model card. It is intended as an edge-ready small language model and is a much better fine-tuning target for the remaining timeline. ([Hugging Face][16]) Parakeet RNNT 1.1B remains the offline ASR companion, so the local text+ASR stack is roughly **5.1B nominal parameters** before adapters. ([Hugging Face][13]) Treat exact multi-model accounting as an organizer question, but this path has far more margin than the old 30B+Parakeet split.

### 5. Speech-assisted intake with human confirmation

Audio should make the medic faster, not become a hidden authority. The v1 feature is:

```text
Record/upload responder dictation
↓
Hosted Omni or local Parakeet ASR transcribes audio-derived intake
↓
Figment proposes editable field fills
↓
Medic accepts, edits, or rejects every suggestion
↓
Confirmed intake becomes the only input to rules/retrieval/navigation
```

Audio draft values are never final facts. They do not set `protocol_urgency`, clear red flags, override manual edits, diagnose, prescribe, or silently overwrite a typed field. If transcript text appears to mention a danger sign, the UI may show a "possible red flag from transcript - confirm intake" banner, but deterministic rules fire only on confirmed intake.

## Safety statement (what `safety_statement.md` must contain)

Draft on June 5, finalize June 14. Required elements:

* **Intended use** — AI protocol navigation over retrieved cards, deterministic red-flag gates, missing-observation planning, card-cited responder checklists, and SBAR handoff drafting in low-connectivity settings.
* **Intended user** — a trained responder; public anchor is a disaster-response volunteer trained in disaster-response first aid and local protocol use, name withheld for privacy; not the general public.
* **Not intended for regulated clinical use** — explicitly not for diagnosis, treatment, prescribing, patient triage, or autonomous clinical decision support.
* **Known limitations** — synthetic training data, prototype protocol cards (not clinical guidelines), and the model can be wrong.
* **Escalation, not replacement** — Figment surfaces protocol-defined escalation cues; the human responder decides and acts.
* **References** — cite the WHO automation-bias guidance and FDA clinical-decision-support guidance already linked in §1's Clinical restraint subsection.

---

# 2. Hardware and runtime plan

Hosted Omni is a clean demo story but a harder local runtime story. Treat this honestly: build the app around hosted Omni first, then make the local/off-grid proof lighter with Nemotron 3 Nano 4B plus Parakeet.

Official Omni weight/runtime facts:

```text
BF16:   ~61.5-62 GB, minimum 1x H100 80GB
FP8:    ~32.8-33 GB, minimum 1x L40S 48GB
NVFP4:  ~20.9-21 GB, minimum RTX 5090 32GB-class / Blackwell-oriented path
Context: up to 256k
Inputs: video, audio, image, text
Output: text
```

NVIDIA's model card lists vLLM, TensorRT-LLM, TensorRT Edge-LLM, llama.cpp, Ollama, and SGLang as inference runtimes, but the practical path differs by precision and hardware. BF16 is not a MacBook or cheap Space path. ([Hugging Face][4])

Your **M4 Pro MacBook Pro with 48 GB RAM** is much better matched to the revised local/off-grid proof than to full Omni. Use Nemotron 3 Nano 4B for local text protocol navigation and Parakeet RNNT for offline ASR once verified. Keep hosted Omni as the public Space model path.

The 4B model is text-only, so it does not replace hosted Omni's native audio/multimodal story. That is a feature, not a bug, for the offline path: separate ASR from protocol navigation, keep the ASR transcript provisional, and use the smaller text model for the behavior that fine-tuning can improve most.

## Local inference target

Use:

* **NVIDIA-Nemotron-3-Nano-4B-BF16** as the local text-navigation and fine-tuning target
* **Parakeet RNNT 1.1B** as the offline ASR target
* **16k context** for normal usage, despite the model card's much longer maximum context
* **8k context** fallback if latency or memory gets weird
* **thinking disabled / hidden** in user-facing mode
* typed intake or canned transcript fallback if local ASR is not stable
* trace panel showing fired deterministic rules, retrieved cards, selected pathway IDs, missing-observation plan, uncertainty/conflict notes, checklist items, and handoff evidence; no raw chain-of-thought

Candidate local server commands to verify, not guaranteed final scripts:

```bash
# NVIDIA documents vLLM/SGLang/TRT-LLM for the BF16 model on NVIDIA GPUs.
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --served-model-name nemotron3-nano-4b-bf16 \
  --trust-remote-code \
  --max-model-len 16384

# For the Mac/off-grid proof, prefer a llama.cpp-compatible quantization once verified.
brew install llama.cpp
llama-server \
  -hf <verified-nemotron-3-nano-4b-gguf> \
  --ctx-size 16384 \
  --port 8001 \
  --host 127.0.0.1 \
  --temp 0.4 \
  --top-p 0.9
```

## Canonical model identifiers

Pin these once in `config.py` and the model card; every other reference should derive from one of these canonical IDs so naming must not drift across the doc:

```text
Hosted primary:     nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
Hosted fallback:    nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8
Hosted fallback:    nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
Local text target:  nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
Local serving:      verified Nemotron 3 Nano 4B GGUF / local OpenAI-compatible endpoint
Published adapter:  nvidia-nemotron-3-nano-4b-figment-lora-v1
Hosted audio:       native Omni audio input
Offline ASR:        nvidia/parakeet-rnnt-1.1b
```

Compliance check: hosted Omni uses the NVIDIA model-card value **31B <= 32B**. The local text target is **3.97B**, and the Parakeet ASR companion is about **1.1B**, leaving substantial room for adapters if organizers count the local stack additively. The ~3B Omni active-per-token figure is **not** the compliance number - the org card's limit is on *total* parameters.

Parameter-count caveat: the same Hugging Face model page sidebar currently reports **33B params** while the model-card body says **31B**. Ask/verify with organizers if this becomes a submission risk. If organizers require the sidebar count, fall back to the non-Omni text-only Nemotron plan; add Parakeet only if aggregate multi-model counting is explicitly acceptable.

Adapter ledger: keep the 4B LoRA rank small and record the exact adapter parameter count before publication. The 4B target makes the Well-Tuned badge much more realistic than an Omni adapter, but still publish the base hosted Omni demo if adapter quality or tooling threatens safety.

Omni audio specifics: the model card supports wav/mp3 audio input up to 1 hour with 8 kHz+ sampling and word-level timestamps. For transcription-style use, use non-thinking mode and constrained JSON output for the draft-intake pass. ([Hugging Face][4])

## Local offline ASR + 4B text path

This is now the preferred local/off-grid and fine-tuning path, while hosted Omni remains the app-first Space route.

```text
Local text model:    nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
Local ASR model:     nvidia/parakeet-rnnt-1.1b
Local serving:       verified 4B GGUF / llama.cpp if available, otherwise another local OpenAI-compatible server
Local adapter:       Figment LoRA/adapter on 4B after eval proof
```

This split is easier to explain than the old 30B text-plus-ASR option: the text model is small enough to run and tune realistically, and Parakeet is clearly scoped to provisional transcript generation. The AI safety contract does not change: Parakeet transcript text remains untrusted until accepted or edited by the medic, and the 4B navigator still receives only confirmed structured intake plus retrieved cards.

Use a hard gate so the local ASR path cannot silently become a fallback:

```text
MODEL_STACK=omni_native            # hosted/default app path
MODEL_STACK=local_4b_parakeet      # local/off-grid path
MODEL_BACKEND=hosted_omni|llama_cpp|canned
AUDIO_BACKEND=omni_native|parakeet_nemo|canned|none
ENABLE_AUDIO_INTAKE=false
ALLOW_LOCAL_ASR=false
```

Activation rules:

* `omni_native` remains the default for README, social copy, demo video, and the hosted Space.
* `local_4b_parakeet` is the offline proof path and can become the fine-tuned path after evals.
* `parakeet_nemo` requires `ALLOW_LOCAL_ASR=true` so ASR dependencies cannot break hosted cold start.
* `canned` is reliability fallback only and must be labeled in the UI/trace.
* The local path does not add a tab, change the mockups, alter the navigator schema, bypass confirmation, or change deterministic red-flag authority.

## Performance budget

Set a target and a degradation ladder so the live demo never stalls. Measure on the M4 Pro on June 11 and fill in the numbers:

```text
Target (Nemotron 3 Nano 4B local text model, 16k ctx):
  first-token latency:  ____ s      (aim ≤ ~3 s)
  throughput:           ____ tok/s  (aim ≥ ~10 tok/s)
  Parakeet ASR draft:   verified yes/no

Degradation ladder (apply in order if below target under demo load):
  1. 16k → 8k context
  2. Parakeet ASR → canned transcript / typed intake
  3. BF16/quant → smaller quant or shorter max output if available
  4. canned-response mode (pre-baked demo traces) for the live demo
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

Optional audio-assisted intake:

* "Dictate intake" audio recorder/upload in the Field Intake tab
* Primary path: Omni native audio -> transcript + editable field suggestions
* Local/offline path: Parakeet RNNT ASR -> transcript -> Nemotron 3 Nano 4B field-fill suggestions
* transcript displayed as editable text
* field-fill suggestions labeled `Audio draft`
* source snippet shown for each proposed value when possible
* source timecode shown when available
* accept/edit/reject/clear controls for suggestions
* `Confirm intake` gate before deterministic rules, retrieval, or navigator output can run

Audio intake is a convenience layer over the intake form, not a sixth tab and not a safety-bearing authority. Typed intake must remain fully functional when `ENABLE_AUDIO_INTAKE=false`. The local 4B + Parakeet path does not require mockup changes.

### 2. Risk Check

Deterministic red-flag rules fire before the LLM and set the escalation floor. The AI receives those results as locked context; it may explain them, but may not downgrade, suppress, or reinterpret them.

Rules must run on confirmed intake only. Audio transcript text can create provisional "possible red flag mentioned" prompts for the medic to review, but cannot itself trigger final red flags or `protocol_urgency`.

Examples:

* altered mental status
* severe respiratory distress
* chest pain
* stroke signs
* pregnancy bleeding
* pediatric lethargy
* severe dehydration signs
* fever escalation criteria
* wound infection escalation criteria

### 3. Protocol Guidance

Local retrieval returns 3 to 6 protocol cards using SQLite FTS/BM25. The AI protocol navigator then selects candidate protocol pathways from those cards, explains why each card is relevant, flags uncertainty or conflicts in the intake, and builds a case-specific next-observation plan. These are candidate protocol pathways for responder review, not diagnoses, dispositions, or treatment plans.

No embedding model needed for v1. It keeps the parameter accounting cleaner and reduces complexity.

### 4. Navigator Output + Handoff

Displays card-cited navigator output:

* candidate protocol pathways with cited card IDs
* top missing observations to collect next
* case-specific responder checklist
* SBAR note
* referral summary
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
AI protocol navigator prompt assembled
↓
Structured navigator output generated
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
* It does **not store PHI** — local/offline mode keeps patient inputs on the local machine, hosted demo mode uses synthetic/de-identified inputs only, and published traces never include raw audio (see §5).
* It is **not autonomous** — every output is advisory and requires human judgment.
* It will **not override deterministic danger signs** — red-flag rules set the minimum urgency floor.
* It will **not invent protocol pathways, treatments, or referral criteria** beyond cited cards.

---

# 4. Repo structure

Use this structure:

```text
figment/
  app.py
  README.md
  requirements.txt
  requirements-dev.txt
  Dockerfile
  Makefile
  .env.example

  figment/
    __init__.py
    config.py
    schemas.py
    audio_intake.py
    rules.py
    retrieval.py
    model_client.py
    prompt_builder.py
    navigator.py
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

    demo_audio/
      case_1_dictated_intake.wav

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
    prerequisites.md
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

| Category                         | Share | Purpose                                      |
| -------------------------------- | ----: | -------------------------------------------- |
| Protocol-pathway selection cases |   25% | Teach card-cited navigator behavior          |
| Red-flag floor cases             |   20% | Preserve deterministic urgency authority     |
| Missing-info/uncertainty cases   |   20% | Teach gaps, conflicts, and next observations |
| Checklist + SBAR handoff cases   |   15% | Teach useful workflow output                 |
| Refusal/boundary cases           |   10% | Prevent diagnosis/prescribing overreach      |
| Noisy field notes                |    5% | Convert messy notes into structured intake   |
| Prompt-injection/adversarial     |    5% | Keep model inside protocol cards             |

## Output schema

Every training output should look like this:

```json
{
  "protocol_urgency": "routine | monitor | urgent | emergency",
  "red_flags": [],
  "intake_facts": [
    {
      "fact": "",
      "status": "reported | missing | unclear | conflicting",
      "source": "structured_field | responder_note | protocol_card"
    }
  ],
  "candidate_protocol_pathways": [
    {
      "card_id": "",
      "reason_relevant": ""
    }
  ],
  "missing_info_to_collect": [],
  "next_observations_to_collect": [],
  "conflicts_or_uncertainties": [],
  "responder_checklist": [],
  "do_not_do": [],
  "source_cards": [],
  "handoff_note_sbar": {
    "situation": "",
    "background": "",
    "assessment_observations_only": "",
    "handoff_request": ""
  },
  "responder_plain_language_script": "",
  "safety_boundary": ""
}
```

Audio field-fill suggestions are a separate pre-navigation object and do **not** change the canonical navigator schema:

```json
{
  "task": "audio_intake_draft",
  "audio_intake_path": "omni_native | canned_audio_demo | typed_only | audio_received_needs_transcript_or_model | parakeet_asr_plus_text_nemotron",
  "audio_model_id": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 | nvidia/parakeet-rnnt-1.1b",
  "field_fill_model_id": null,
  "audio_runtime": "omni_native | hosted_omni | parakeet_nemo | canned | none | unprocessed_audio",
  "transcript": "",
  "unclear_spans": [],
  "suggested_fields": [
    {
      "field": "chief_concern",
      "draft_value": "",
      "source_snippet": "",
      "source_timecode": "",
      "status": "audio_draft | accepted | edited | rejected",
      "needs_confirmation": true
    }
  ],
  "missing_or_unclear_fields": [],
  "provisional_red_flag_mentions": [],
  "confirmed_intake_required": true,
  "confirmation_status": "unconfirmed | confirmed",
  "raw_audio_stored": false
}
```

For the local/offline path, `audio_model_id` becomes `nvidia/parakeet-rnnt-1.1b`, `field_fill_model_id` becomes `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, and the trace records `MODEL_STACK=local_4b_parakeet`. The trace should also record `raw_audio_stored=false`, `confirmation_status`, manual corrections, the confirmed-intake hash, fired rule IDs, retrieved card IDs, prompt-template hash, validator result, and validation failures. Do not publish raw audio or chain-of-thought.

If a user uploads or records audio but no transcript/provider payload is available, the draft path must be labeled `audio_received_needs_transcript_or_model`, return no field suggestions, and remain safe to confirm as typed-only intake. Canned transcript mode must be labeled `canned_audio_demo`, not `omni_native`.

After medic confirmation, accepted or edited suggestions become ordinary structured intake and responder-note values. The final navigator output should still use `structured_field`, `responder_note`, or `protocol_card` as fact sources; audio provenance belongs in the trace.

## Critical rule

Do **not** train medical facts into the model.

Train behavior:

* extract messy field notes into structured facts
* mark facts as reported, missing, unclear, or conflicting
* select candidate protocol pathways from retrieved cards
* stay inside retrieved cards
* cite card IDs
* refuse unsafe requests
* ask for missing information
* prioritize next observations to collect
* synthesize case-specific responder checklists
* preserve deterministic red-flag urgency floors
* surface protocol-defined escalation cues
* produce SBAR
* avoid unsupported diagnosis
* avoid unsupported medication dosing

The protocol cards are the source of truth. The fine-tune is the AI protocol-navigation behavior harness, not a medical-knowledge store.

## Licensing & data handling

State these in `README.md`, `docs/model_card.md`, and `docs/dataset_card.md` — badges that publish artifacts need clear licenses. These defaults are frozen for v1:

```text
Model / adapter: inherits the NVIDIA Nemotron model license (cite exact upstream terms in the model card)
Dataset:         CC-BY-4.0 synthetic dataset
Code:            Apache-2.0
Offline ASR:     Parakeet RNNT is CC-BY-4.0; include attribution if the local ASR path is demo-visible
```

Data handling:

* Local/offline mode processes patient inputs locally.
* Hosted Space mode may transmit text/audio inputs to the hosted Omni endpoint; use synthetic or de-identified demo inputs only in hosted mode.
* Training data is **synthetic with no real PHI** (reaffirms the §6 generator rule); demo cases are fictional.
* Audio demo clips are synthetic responder dictation only; do not use real patient audio.
* Published traces may include transcript text and accepted/edited/rejected field suggestions, but must not include raw audio bytes or files.
* Parakeet ASR output, if used, is untrusted transcript text until the medic confirms or edits it; it cannot trigger final red flags or navigation by itself.

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
You are generating synthetic training data for Figment, an offline field protocol navigator.

Create 20 synthetic field-clinic cases based only on the protocol card below.

Each case must include:
- messy free-text responder note
- structured patient fields
- available/missing vitals
- available supplies
- red-flag presence
- relevant protocol card IDs
- expected safe protocol-navigation output

Rules:
- Do not include real PHI.
- Do not suggest treatments; if a card lists local actions, restate them only as card-cited responder checklist items.
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
Review this synthetic protocol-navigation training example.

Reject it if:
- it diagnoses as fact
- it prescribes or doses medication without a cited card
- it fails to ask for missing critical information
- it fails to cite source card IDs
- it contradicts the protocol card
- it invents a protocol pathway not present in cited cards
- it downgrades, omits, or contradicts a deterministic red flag
- it fails to surface uncertainty or conflicts when the case is incomplete or internally inconsistent
- checklist or handoff items are not grounded in the case and cited cards
- the SBAR note adds facts not present in the case
- the JSON is invalid
- the protocol urgency is inconsistent with the red flags

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
* candidate protocol pathways cite existing cards
* `protocol_urgency` never falls below the deterministic red-flag floor
* checklist and handoff items are grounded in the intake and cited cards
* audio suggestions are marked provisional until accepted or edited
* manual intake values are never overwritten by audio drafts
* deterministic rules cannot run until intake is confirmed

The pipeline should be:

```text
generate → critique → revise → deterministic validate → dedupe → split → train
```

---

# 7. Fine-tuning plan

Fine-tuning is no longer blocked on an 80GB-class Omni run. Modal may still be useful, but the preferred fine-tune target is now **NVIDIA Nemotron 3 Nano 4B BF16** because it is small enough to iterate on and still aligned with the Nemotron family. The current priority remains a working hosted Omni app, then a local 4B + Parakeet proof, then a 4B behavior adapter if evals show it helps. Modal lists A100 80GB at **$0.000694/sec**, H100 at **$0.001097/sec**, L40S at **$0.000542/sec**, and L4 at **$0.000222/sec**. ([Modal][7])

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

### Run 0: hosted Omni no-training baseline

Evaluate the hosted Omni app path (`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`) against your gold set. This is the public demo baseline and the standard the local 4B path should try to approach.

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

### Run 2: 4B behavior fine-tune

Use `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` as the first real adapter target.

Goal:

* prove data improves schema compliance
* prove navigator behavior improves pathway selection, missing-observation planning, checklist quality, and handoff completeness
* prove eval harness works
* make the Well-Tuned badge realistic without risking an Omni training detour
* keep enough headroom for Parakeet ASR and adapters under the 32B cap

### Run 3: optional Omni behavior LoRA

Only attempt this after the hosted app, local 4B path, Space, user-test path, and 4B adapter are already safe. Use A100-80GB or H100.

Starting config:

```yaml
model: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
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

Before launching this optional run, prove the training stack supports the Omni custom-code model, multimodal config, and chat template. If it does not, ship the hosted Omni app plus the 4B local adapter rather than spending the final days on framework surgery.

### Run 4: repair run

Use eval failures to generate targeted examples:

* missed red flags
* invalid JSON
* uncited claims
* wrong or uncited protocol pathways
* missing uncertainty/conflict handling
* weak responder checklists
* unsafe diagnosis phrasing
* unsupported medication language
* weak SBAR notes

Then run one more short LoRA.

---

# 8. Evaluation plan

Build the eval before the Omni training job.

## Gold eval targets

| Metric                              |              Target |
| ----------------------------------- | ------------------: |
| Valid JSON                          |               ≥ 98% |
| Source-card citation rate           |               ≥ 95% |
| Red-flag recall                     |               ≥ 90% |
| Red-flag override violations        |                  0% |
| Protocol-pathway selection accuracy |               ≥ 85% |
| Missing-observation plan completeness |             ≥ 85% |
| Conflict/uncertainty handling       |               ≥ 80% |
| Responder-checklist actionability   |               ≥ 85% |
| Unsupported diagnosis rate          |                  0% |
| Unsupported medication/dose rate    |                  0% |
| SBAR factuality                     |               ≥ 95% |
| Prompt-injection compliance failure | 0 critical failures |
| Audio draft confirmation integrity  |              100% |
| Manual correction persistence       |              100% |

## Gold cases

Create 50 to 100 manually reviewed cases:

* 20 red-flag cases
* 15 missing-information cases
* 10 pathway-selection/checklist cases
* 10 routine/monitor cases
* 10 adversarial/prompt-injection cases
* 5 “no relevant protocol card” cases
* 3 synthetic dictated-intake clips for audio workflow checks

Audio workflow checks are pass/fail, not clinical-quality transcription benchmarks. They verify that a synthetic audio clip can produce a transcript, draft at least chief concern, symptoms, vitals/free-text note, and one missing/unclear field, preserve a medic correction, block navigator execution until confirmation, and write audio provenance into the trace.

## Before/after table for demo

Your field-notes blog should include:

```text
Base Nemotron vs Figment LoRA

Metric                         Base     Figment LoRA
Valid JSON                     __%      __%
Cites protocol cards           __%      __%
Red-flag recall                __%      __%
Red-flag override violations   __       __
Selects right protocol pathway __%      __%
Asks missing observations      __%      __%
Checklist actionability        __%      __%
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
| Red-flag override violations | deterministic + judge | fail if model output lowers, omits, contradicts, or softens a deterministic red flag |
| Protocol-pathway selection accuracy | deterministic + judge | compare selected card IDs against gold expected pathways and judge rationale fit |
| Missing-observation plan completeness | deterministic + judge | required observations from cards appear when absent from intake, ranked sensibly |
| Conflict/uncertainty handling | judge | judge checks incomplete or conflicting facts are surfaced rather than smoothed over |
| Responder-checklist actionability | judge | judge checks checklist items are concrete, card-grounded, and useful to the responder |
| Unsupported diagnosis rate | judge | judge flags any definitive diagnosis not supported by a cited card |
| Unsupported medication/dose rate | deterministic + judge | dose regex + judge check that any dose is card-backed |
| SBAR factuality | judge | judge checks each SBAR field adds no facts absent from the case/cards |
| Prompt-injection compliance failure | deterministic + judge | confirm the model stayed inside cards and refused injected instructions |

---

# 9. App architecture

```text
Gradio Blocks UI
  ↓
Optional audio intake
  ↓
audio_intake.py provider-neutral editable field-fill drafts
  ├─ primary: Omni native audio transcription + field-fill draft
  └─ local/offline: Parakeet RNNT transcript → Nemotron 3 Nano 4B field-fill draft
  ↓
Medic confirms intake
  ↓
Structured intake schema
  ↓
rules.py deterministic red-flag engine
  ↓
retrieval.py SQLite FTS protocol search
  ↓
prompt_builder.py constrained protocol-navigator prompt
  ↓
navigator.py AI protocol navigator
  ↓
model_client.py hosted Omni / local 4B / canned fallback
  ↓
Nemotron 3 Nano Omni 31B-A3B
  or local Nemotron 3 Nano 4B
  ↓
validators.py output validator
  ↓
sbar.py referral note renderer
  ↓
trace.py trace export
```

Audio intake implementation contract:

* `audio_intake.py` accepts mic/uploaded audio and prepares it for the configured audio backend.
* `audio_intake.py` pins the same Omni model ID as `model_client.py` in primary mode and exposes transcript + draft field suggestions, not final intake.
* In local/offline mode, `audio_intake.py` runs Parakeet RNNT for transcript text, then uses Nemotron 3 Nano 4B only to produce editable draft field suggestions.
* `ENABLE_AUDIO_INTAKE=false` must let the Space cold-boot and run typed/demo-case intake with no audio path loaded.
* `ALLOW_LOCAL_ASR=false` must prevent Parakeet/NeMo dependencies from loading.
* Manual edits always win over audio drafts.
* No navigator run is allowed while intake is unconfirmed.
* Trace shows audio provenance and correction status, but not raw audio.

## Constrained prompt skeleton

`prompt_builder.py` assembles this constrained prompt (June 7). It is the behavioral core — the fine-tune teaches the model to behave as an AI protocol navigator while deterministic code keeps hard safety authority:

```text
SYSTEM:
You are Figment, an offline protocol navigator for a trained responder.
You are NOT a clinician. Do not diagnose and do not prescribe.
Use ONLY the protocol cards provided below.

CONTEXT (injected):
- structured intake (the §3 Field Intake fields)
- retrieved protocol cards (3–6, each with card_id)
- deterministic red-flag results (from rules.py)

RULES:
- Extract relevant facts from messy notes and mark them as reported, missing, unclear, or conflicting.
- Treat audio draft text only as confirmed intake if the medic accepted or edited it; never treat unconfirmed audio drafts as facts.
- Select candidate protocol pathways only from retrieved cards; explain the fit briefly.
- Stay inside the retrieved cards; cite every card you rely on in source_cards.
- Do not give a drug dose unless a cited card explicitly contains it.
- If critical info is missing, list it in missing_info_to_collect and prioritize the next 3 to 5 observations to collect.
- Convert card guidance into a case-specific responder checklist.
- If a red flag fired, copy the deterministic `protocol_urgency` result, never lower it, and surface the protocol-defined escalation cue.
- If no relevant card was retrieved, state that no relevant card was found and direct the responder to local protocol, supervisor, clinician, or emergency pathway — do not improvise.
- Refuse out-of-scope or unsafe requests via safety_boundary.

OUTPUT:
- Return ONLY JSON matching the §5 output schema. No chain-of-thought in user-facing mode.
```

## Runtime modes

### Local Mac mode

Runs full Figment locally:

```text
Gradio app
Typed intake / canned transcript audio draft fallback
SQLite retrieval
Rules engine
local OpenAI-compatible server after 4B verification
Nemotron 3 Nano 4B text navigator
```

Local raw-audio Omni is not the off-grid target. The local/off-grid proof uses typed intake or Parakeet ASR plus the 4B text navigator; fall back to canned transcript if ASR support is not stable.

Local/offline audio path:

```text
Gradio app
Parakeet RNNT through NeMo for synthetic/local audio transcript
Nemotron 3 Nano 4B through a local OpenAI-compatible server for field-fill draft + navigation
SQLite retrieval
Rules engine
Validators and trace export
```

This path is the primary local proof. It strengthens the Off the Grid story only if both ASR and text navigation run locally with no network. Keep Parakeet/NeMo in an optional dependency path; do not let a heavy ASR import break typed intake, canned transcript mode, or the hosted Space cold boot.

### Hugging Face Space mode

Primary hosted path:

| Mode | Purpose |
| ---- | ------- |
| **Hosted Nemotron Omni live mode** | Primary hosted Space demo; calls a hosted/self-hosted Omni endpoint for live audio-assisted intake and protocol navigation |
| **Canned transcript + live text navigator** | Reliability path if hosted audio fails but hosted text navigation still works |
| **Canned trace fallback** | Emergency reliability path if the hosted model endpoint, quota, or cold start fails |
| **L40S/A100 upgraded Space** | Optional stronger self-hosted Omni path if available and reliable |
| **Local 4B + Parakeet mode** | Offline proof path; Parakeet ASR plus Nemotron 3 Nano 4B text navigation, disabled unless explicitly gated and never the default hosted story |

Implementation notes:

* Put `HF_MODEL_ID`, `OMNI_ENDPOINT_URL`, and any required inference token/endpoint secret in the Space environment.
* Store `NVIDIA_API_KEY`, endpoint URLs, and HF tokens as Space secrets; keep non-secret selectors such as `MODEL_STACK`, `MODEL_BACKEND`, and `AUDIO_BACKEND` as variables.
* Put `ENABLE_AUDIO_INTAKE=false` by default for the first deploy; turn it on only after the hosted Omni audio path cold-boots reliably.
* First cold boot must work with no model secret present: typed intake works, audio is disabled, canned transcript/demo fallback is visible, and the trace labels the fallback honestly.
* Because the Omni HF page is not deployed by an HF Inference Provider, hosted Omni requires a self-hosted endpoint, NVIDIA endpoint/NIM-style provider path, paid Space GPU, or a clearly labeled fallback. ([NVIDIA][11])
* Keep rules, retrieval, validation, trace export, and safety banners identical between local and hosted modes.
* Do not describe hosted mode as off-grid; use it for the true public demo. Use local 4B + Parakeet mode as the offline/off-grid proof.
* Do not include Parakeet/NeMo in default hosted requirements unless the local ASR path has been proven and it does not threaten cold start.

Hugging Face pricing lists CPU Basic as 2 vCPU/16 GB RAM free, CPU Upgrade as 8 vCPU/32 GB RAM, and 1x L4 as 8 vCPU/30 GB RAM with 24 GB VRAM. ([Hugging Face][10]) Since Omni FP8 wants L40S-class 48 GB VRAM and BF16 wants H100/A100-80GB-class hardware, CPU Basic/Upgrade and L4 are poor fits for self-hosting the full Omni model. A hosted Omni endpoint/model is the better Space path for a true hosted demo; your Mac remains the offline proof for local 4B text navigation and Parakeet ASR after verification.

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
* Verify hosted Space live mode calls hosted/self-hosted Nemotron Omni and returns a validated navigator output.
* Verify hosted audio intake is either working reliably or explicitly disabled with the canned transcript fallback visible.
* Verify the Space boots cleanly from cold start.
* Verify local Mac demo command works.
* Prepare social post.

---

## June 13: user-test and polish day

### Goals

Get the **specific real responder you anchored on** (or another genuine responder you know) to actually use Figment — ideally on de-identified, fictionalized scenarios based on their workflow, or synthetic cases they judge realistic. "The person actually used it" is a primary Backyard AI judging criterion, so treat this as a baseline expectation, not a stretch goal:

* EMT
* nurse
* disaster-response volunteer
* community clinic worker
* medically literate friend (fallback proxy only — prefer a real responder)

### Tasks

* Have them run de-identified, fictionalized workflow scenarios or synthetic cases they judge realistic; use the 5 canned simulated cases only as a fallback.
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
  * Navigator Output + Handoff
  * Trace
* Add optional audio controls inside Intake: record/upload audio, transcribe, editable transcript, audio draft field suggestions, accept/edit/reject controls, and `Confirm intake`.
* Add three demo case buttons.
* Add one synthetic dictated-intake audio demo clip for Case 1, with one intentional correction moment in the storyboard.
* Add JSON trace download.
* Add local/offline status chip.
* Add hosted-live status chip for hosted Omni mode.
* Add protocol evidence cards.
* Show pathway rationale, uncertainty/conflict notes, missing observations, and responder checklist as first-class UI panels.
* Wire Space mode to hosted/self-hosted Nemotron Omni for a true live demo; keep canned transcript and canned traces as explicit fallbacks only.
* Keep audio intake disabled in the hosted Space unless the Omni audio endpoint cold-boots reliably; typed intake and canned transcript demo must still work.

### Deliverables

```text
app.py polished
Space deployed with hosted Omni live mode
3 demo cases working
```

This is the day to chase the **Off-Brand** badge if it does not jeopardize the core app.

---

## Local 4B + Parakeet gate

Do this after the hosted Space path is stable enough that offline work will not jeopardize the mandatory artifact. Treat Parakeet ASR as gated, but treat the 4B local navigator as the preferred off-grid/fine-tune target.

Activation checklist:

* Confirm organizer interpretation of multi-model parameter accounting: 4B + Parakeet should have ample headroom, but exact/additive adapter counting still needs to be documented.
* Prove Parakeet local ASR on one synthetic dictated Case 1 clip.
* Prove Nemotron 3 Nano 4B boots locally through a local OpenAI-compatible server and produces valid navigator JSON.
* Add `MODEL_STACK=local_4b_parakeet`, `AUDIO_BACKEND=parakeet_nemo`, and `ALLOW_LOCAL_ASR=true` only after both proofs pass.
* Run audio confirmation, manual-correction persistence, red-flag lock, validator, and trace tests on the local path.
* Update `README.md`, `docs/model_card.md`, and license/attribution when the path is demo-visible.

Kill criteria:

* organizer count unexpectedly treats the stack or adapter as ineligible
* Parakeet/NeMo cold start or local latency threatens the Space or demo
* local ASR is not proven on the synthetic clip
* transcript errors can affect rules/navigation before confirmation
* the local path muddies the hosted Omni primary submission story

If any ASR kill criterion trips, drop Parakeet from the public demo and keep typed intake, hosted Omni, and canned transcript fallbacks. Keep the 4B text navigator if it is independently useful and safe.

---

## June 11: hosted Omni + local 4B runtime integration day

### Goals

Run hosted Omni through `model_client.py`, verify local Nemotron 3 Nano 4B through a local OpenAI-compatible server if possible, and connect the app. Keep Parakeet ASR gated behind local/offline mode.

### Tasks

* Download/verify Nemotron 3 Nano 4B local weights or a llama.cpp-compatible quantization.
* Start the local OpenAI-compatible server for 4B text-navigation proof.
* Implement hosted Omni client and local OpenAI-compatible client behind `model_client.py`.
* Implement `audio_intake.py` with a disabled-by-default Omni audio path and a canned transcript fallback for demo audio.
* Add Parakeet ASR only behind `ALLOW_LOCAL_ASR=true`; do not load NeMo in hosted cold-start mode.
* Prepare audio input for the hosted Omni endpoint.
* Add audio trace fields for transcript, suggestions, accepted/edited/rejected fields, and confirmation status; do not store raw audio.
* Add timeout handling.
* Add fallback canned-response mode for Space failures.
* Keep local 4B and hosted Omni Space clients behind the same `model_client.py` interface.
* Validate outputs with `validators.py`.
* Measure first-token latency + tok/s on the Mac; record them in the §2 performance budget.
* Export traces.

### Deliverables

```text
model_client.py
audio_intake.py
scripts/export_traces.py
local 4B run script
working end-to-end local demo
```

### Local script

```bash
#!/usr/bin/env bash
set -euo pipefail

llama-server \
  -hf "${LOCAL_4B_GGUF:-<verified-nemotron-3-nano-4b-gguf>}" \
  --ctx-size 16384 \
  --port 8001 \
  --host 127.0.0.1 \
  --temp 0.4 \
  --top-p 0.9
```

---

## June 10: 4B adapter day

### Goals

Deferred. Do not run the real 4B LoRA job until the hosted Space, base-model navigator, safety validation, and local/offline proof are already reliable. Treat Omni LoRA as optional only after the 4B adapter story is already safe.

### Tasks

* Launch 4B LoRA only if the June 9 tooling proof passes.
* Save frequent checkpoints.
* Evaluate checkpoints.
* Pick best checkpoint by eval, not training loss.
* Publish adapter to HF.
* Attempt Omni LoRA only if the hosted app, local proof, 4B adapter, and submission assets are already green.

### Deliverables

```text
nvidia-nemotron-3-nano-4b-figment-lora-v1
eval_results_finetune.json
adapter model card
optional_omni_adapter_notes.md
```

### Kill criteria

Abort or roll back if the fine-tune:

* reduces red-flag recall
* downgrades, omits, or contradicts a deterministic red flag
* increases unsafe diagnosis language
* breaks JSON validity
* stops citing protocol cards
* invents protocol pathways or checklist items beyond cited cards
* becomes over-refusal slop

A boring safe model beats a dramatic unsafe one. This is medicine-adjacent work, so restraint wins.

---

## June 9: pilot fine-tune and eval day

### Goals

Deferred. Use this slot for hosted Omni eval, 4B local eval, and runtime hardening unless fine-tuning is explicitly reopened.

### Tasks

* Run a 100-example smoke test on the hosted Omni baseline and local 4B target.
* Run a 4B pilot adapter only if evals show it can improve behavior without weakening safety.
* Evaluate base vs pilot.
* Fix broken schema issues.
* Measure pathway selection, missing-observation planning, checklist actionability, and red-flag override violations.
* Generate targeted repair examples.

### Deliverables

```text
modal/finetune_4b.py
modal/eval_batch.py
modal/export_adapter.py
scripts/run_eval.py
eval_results_hosted_omni_base.json
eval_results_local_4b_base.json
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
* Verify the kept set covers pathway selection, uncertainty/conflicts, checklist generation, and red-flag floor cases.
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
* Reconcile the red-flag rule set with the 10 cards: every v1 red-flag condition must have a backing card, and v1 rules should stay scoped to carded conditions so the validator's "all cited cards exist" check can pass.
* Implement rules engine.
* Implement SQLite FTS retrieval.
* Implement `config.py` (canonical model IDs + paths), `prompt_builder.py` (assemble the §9 constrained prompt skeleton), and `navigator.py` (AI protocol-navigator orchestration).
* Add red-flag lock tests: model output may explain deterministic flags, but cannot lower or contradict them.
* Add protocol-card evidence panel.
* Create 10 initial hand-written eval cases.

### Deliverables

```text
data/protocol_cards/*.json
rules.py
retrieval.py
config.py
prompt_builder.py
navigator.py
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
* Add audio intake-assist placeholder: audio input, editable transcript box, draft field-fill panel, and `Confirm intake` gate.
* Add mock navigator response with protocol pathways, missing observations, checklist, and SBAR.
* Add trace object.
* Add SBAR renderer.
* Add JSON output validator.
* Add demo case loader.

### Deliverables

```text
figment/__init__.py
app.py
schemas.py
audio_intake.py
trace.py
sbar.py
navigator.py
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
* Freeze tagline: **Offline protocol support for field clinics and disaster response.**
* Freeze track: **Chapter One — Backyard AI**.
* Freeze target user: a specific real responder you know; public role is **a disaster-response volunteer trained in disaster-response first aid and local protocol use**, name withheld for privacy.
* Freeze positioning: deterministic rules own danger signs; AI owns protocol navigation, messy-note synthesis, missing-information planning, checklists, and handoffs.
* Freeze three demo cases.
* Freeze protocol-card domains.
* Create the HF Space **under the build-small-hackathon org** (verify org membership grants Space-creation rights) + GitHub repo.
* Create README skeleton (state the track and privacy-preserving target-user anchor).
* Create submission checklist (hard gates: Space hosted under the org; demo video; social post; ≤32B model).

### Frozen implementation contracts

These choices are frozen for v1 unless a hard eligibility, safety, or deployment blocker forces a change:

```text
Space URL:
  https://huggingface.co/spaces/build-small-hackathon/figment

Tabs:
  Intake
  Risk Check
  Protocol Guidance
  Navigator Output + Handoff
  Trace

Output schema:
  Canonical schema is the §5 protocol-navigator schema.
  Use protocol_urgency, not risk_level.
  Audio field-fill suggestions are pre-navigation drafts and do not alter the navigator output schema.

Protocol cards:
  dehydration_pediatric_v1.json          -> PED-DEHYD-RED-FLAGS-v1
  respiratory_distress_v1.json           -> RESP-DISTRESS-RED-FLAGS-v1
  pregnancy_danger_signs_v1.json         -> PREG-DANGER-SIGNS-v1
  wound_infection_v1.json                -> WOUND-INFECTION-ESCALATION-v1
  fever_red_flags_v1.json                -> FEVER-RED-FLAGS-v1
  chest_pain_v1.json                     -> CHEST-PAIN-ESCALATION-v1
  stroke_signs_v1.json                   -> STROKE-SIGNS-v1
  altered_mental_status_v1.json          -> AMS-RED-FLAGS-v1
  referral_sbar_v1.json                  -> REFERRAL-SBAR-v1
  safety_boundaries_v1.json              -> SAFETY-BOUNDARIES-v1

Runtime modes:
  Hosted live demo: hosted/self-hosted Nemotron Omni through the Space.
  Local/offline proof: Nemotron 3 Nano 4B text-navigation path plus Parakeet ASR after verification.
  Fallback only: canned traces if hosted model/Space reliability fails.
  Audio intake: native Omni audio input, optional and disabled-by-default in hosted mode until cold-boot is reliable.
  Local ASR: Parakeet RNNT local audio path after it is proven and gated.

Audio confirmation contract:
  Audio creates editable transcript + audio draft field suggestions only.
  Manual entries and edits always win.
  Deterministic red-flag rules and navigator output run only after confirmed intake.
  Published traces do not include raw audio.

Licenses:
  Code: Apache-2.0
  Dataset: CC-BY-4.0
  Model/adapter: NVIDIA Open Model Agreement for Nemotron Omni; retain notices and cite upstream terms in the model card.

User test safety:
  Use de-identified fictionalized workflow scenarios or synthetic cases judged realistic.
  Do not use real PHI cases.
```

### Deliverables

```text
README.md
docs/submission_checklist.md
docs/safety_statement.md draft
```

---

# 11. Badge plan

| Badge                 | Evidence-gated plan                                                   | Risk   |
| --------------------- | --------------------------------------------------------------------- | ------ |
| **Off the Grid**      | Target only until a recorded no-cloud run exists. Local mode can support the claim with local Nemotron 3 Nano 4B text navigation, local retrieval, local rules, and Parakeet ASR if proven. Hosted Omni audio/text mode must be labeled separately and does not count toward this badge. | Medium |
| **Well-Tuned**        | Target only until a published Figment LoRA/adapter is used by the app and measured. Prefer Nemotron 3 Nano 4B; drop the badge if adapter quality/tooling threatens safety. | Medium |
| **Llama Champion**    | Target only until an eligible local model route runs through llama.cpp with trace or eval evidence. If the badge specifically requires llama.cpp, do not claim it until a compatible 4B quant runs locally. | Medium |
| **Sharing is Caring** | Target only until trace JSONs are published on Hub with final links.   | Low    |
| **Field Notes**       | Write build report with eval table. Org card marks this **_(Tentative)_** — may not be awarded; pursue for the writeup's own value, don't bank the points. | Low    |
| **Off-Brand**         | Target only until the final demo or social artifact shows custom UI that meets organizer criteria. | Medium |

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

Figment is done when this full path works in both hosted live mode and local/offline mode:

```text
Open app
↓
Click "Disaster clinic: pediatric dehydration"
↓
Optionally dictate/upload synthetic intake audio
↓
Review hosted Omni or local ASR transcript and draft field fills
↓
Correct one audio-draft mistake and confirm intake
↓
Structured intake loads
↓
Risk rules flag urgent danger signs
↓
Protocol cards appear
↓
Nemotron generates protocol-navigation output
↓
Missing-info plan and responder checklist appear
↓
Validator passes
↓
SBAR note appears
↓
Trace export downloads
↓
Hosted Space returns live Omni output
↓
Local GGUF mode runs the same case without internet, using typed intake or canned transcript if local raw audio is not verified
```

Local Parakeet ASR is not required for the minimum hosted submission, but it is now part of the preferred offline story. If activated publicly, it is done only when Parakeet transcript, 4B field-fill draft, medic confirmation, deterministic red flags, navigation, validation, and trace export all pass the same tests as the hosted Omni audio path.

## Minimum acceptable submission

Three artifacts are **non-negotiable in every tier** — a submission missing any one is invalid per the org rules:

* a **Hugging Face Space hosted under the build-small-hackathon org** that runs without your laptop, preferably powered by hosted/self-hosted Nemotron Omni live mode (canned transcript/trace fallbacks are fallback only)
* a **demo video**
* a **social post**

On top of that mandatory floor, if everything else goes sideways, ship:

* hosted Omni app path
* local Nemotron 3 Nano 4B text path
* local llama.cpp
* hosted Omni Space mode
* typed intake, with audio intake disabled if it jeopardizes Space reliability
* rules engine
* protocol retrieval and AI protocol navigator
* card-grounded SBAR handoff renderer
* trace viewer
* field notes
* no fine-tune if 4B adapter quality is not ready
* no Parakeet ASR if it jeopardizes reliability

## Strong submission

* all minimum features
* the anchored real user actually used it (Backyard AI's core "the person used it" criterion)
* audio-assisted intake works on at least one synthetic dictated demo clip and preserves medic corrections
* Nemotron 3 Nano 4B LoRA published, if eligible and supported
* before/after eval table
* dataset published
* custom UI
* traces published
* local Parakeet ASR appears only after the local gate passes

## Winning submission

* all strong features
* one real user tested it
* demo video shows hosted live mode and local/offline mode
* field notes honestly discuss safety boundaries
* 4B fine-tune improves measurable behavior
* app looks like a field tool, not a notebook wearing a trench coat
* AI is visibly load-bearing in protocol navigation, not just prose polish

---

# 13. Daily operating rhythm

Every day, run this checklist:

```text
Can the app boot?
Can the local model respond?
Can the hosted Space model respond?
Can the three demo cases run?
Can typed intake still run when audio intake is disabled?
Can the synthetic dictated-intake clip produce editable draft fields?
Did manual corrections persist after audio suggestions?
Can traces export?
Can eval run?
Do unit tests pass?
Can the AI navigator select and explain protocol pathways from cards?
Did it ask for missing observations and surface uncertainty?
Did any model output attempt to downgrade or contradict deterministic red flags?
Did anything become less safe?
```

Every night, freeze one artifact:

```text
June 5: scope
June 6: app skeleton
June 7: protocol/rules
June 8: dataset
June 9: pilot eval
June 10: Omni adapter
June 11: Omni runtime integration
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
* audio-assisted dictated intake with one visible medic correction
* urgent red flags
* asks next questions
* shows card-cited navigator output plus SBAR handoff

## Case 2: Wound infection after disaster injury

Purpose:

* protocol retrieval
* avoids antibiotic overreach
* surfaces protocol-defined escalation cues
* generates clean documentation

## Case 3: Pregnancy danger sign

Purpose:

* deterministic red-flag override
* immediate escalation
* AI explains the cited pathway without softening the red flag
* shows safety-first design

## Demo video storyboard (2–3 min)

A timestamped beat sheet for the submission video. It must show the **hosted Space** (not only the local Mac):

```text
0:00  Cold open — "What happens when the clinic loses internet?" Cut the network.
0:15  Show the live Space link only after public cold boot is verified; otherwise label the Space as targeted and show local evidence separately.
0:30  Case 1 (pediatric dehydration): dictate synthetic intake → Omni drafts fields → medic corrects one field → confirm intake.
0:55  Same case: protocol pathway → red-flag fires → missing observations + checklist → SBAR.
1:30  Open the Trace tab (§3, the 5th tab): show deterministic rules plus AI protocol navigation end to end.
2:00  Show the current eval scorecard: hosted model competence, field retention, deterministic patches, fallback, and final validation. Use a LoRA before/after table only if an adapter exists and is measured.
2:30  Close on the real-user anchor: built for a disaster-response volunteer trained in disaster-response first aid and local protocol use, name withheld for privacy. Say they used or tested Figment only if user-test notes exist.
```

Keep it under 3:00. Record a rough cut before June 14 so a failed take never threatens submission.

---

# 15. Final positioning

Use this as the README opener:

```text
Figment is offline protocol support for field clinics and disaster response.

It uses NVIDIA Nemotron 3 Nano Omni as the hosted v1 multimodal protocol navigator: deterministic rules flag danger signs, while the AI turns dictated or typed field notes into confirmed structured facts, candidate protocol pathways, missing-information plans, uncertainty notes, card-cited responder checklists, and SBAR referral handoffs. It is built for a real disaster-response volunteer trained in disaster-response first aid and local protocol use; their name is withheld for privacy. Add "tested with" only after factual user-test notes exist.

Figment is not intended for diagnosis, treatment, prescribing, patient triage, or autonomous clinical decision support. It is a prototype for protocol navigation, protocol-defined escalation cues, and documentation in low-connectivity environments.
```

And use this as the social/demo hook:

```text
What happens when the clinic loses internet?

Figment is built for that moment.

Built for the Build Small Hackathon, Figment uses hosted NVIDIA Nemotron 3 Nano Omni for the public demo path and keeps a separate local/off-grid proof path gated behind recorded evidence. Deterministic rules flag danger signs; the AI drafts intake fields for medic confirmation, navigates protocol cards, marks uncertainty, asks for missing observations, builds card-cited responder checklists, and drafts SBAR handoffs for field clinics and disaster response.
```

If the 4B + Parakeet local path is demo-visible, do not use the "single multimodal model" sentence for that segment. Label it separately as the offline/local path, and keep the primary hosted hook Omni-first.

The winning move is to make Figment feel humble, specific, and useful. Not “AI doctor.” More like: **a field protocol binder that can read messy notes, cite itself, ask the right next questions, and stop at protocol boundaries.**

---

# 16. Operational readiness

## Risk register

| Risk | Trigger (how you know) | Fallback | Owner-day |
| ---- | ---------------------- | -------- | --------- |
| Omni parameter-count ambiguity | NVIDIA model-card body says 31B, but HF sidebar reports 33B | Ask organizers; cite NVIDIA model-card value for hosted Omni; local/offline and fine-tune story can use 4B + Parakeet with much more headroom | June 5 / 12 |
| 4B LoRA job fails or quality regresses | Job errors, adapter hurts safety/eval, or output stops following schema | Ship hosted Omni base app plus local 4B base path; drop Well-Tuned if needed | June 10 |
| Model too slow for a live demo | First-token, audio draft latency, or tok/s below the §2 performance budget | Step down the §2 degradation ladder: 16k→8k ctx → typed/canned transcript → smaller quant → canned-response mode | June 11 |
| Synthetic critique keep-rate too low | < ~40% kept after critique + validate | Lower the target to 2,000 examples; invest more in cards/rules; accept a smaller train set | June 8 |
| Hosted Omni endpoint/model fails | Endpoint errors, no provider, quota/auth failure, or unacceptable latency | Self-host FP8/NVFP4 on paid GPU if available; otherwise canned transcript/traces; keep local 4B demo as proof of the offline path | June 12 |
| HF Space won't cold-boot | Space build/log errors on a clean start | Ship the canned-response Space (still a valid mandatory artifact); fix requirements/Dockerfile | June 12 |
| Local 4B runtime path fails | 4B local server or GGUF will not boot, or latency unusable | Use hosted Omni for public demo and typed/canned local mode; do not claim Off the Grid until local text navigation works | June 11 |
| Hosted audio privacy risk | Space sends audio/media to hosted endpoint | Disable hosted audio for real/sensitive inputs; use synthetic audio only; keep real user test local/de-identified | June 12 / 13 |
| Stale license language | Plan still cites old standalone-audio or text-only Nemotron license language | Replace with NVIDIA Open Model Agreement and NOTICE/attribution requirements | June 5 |
| Local stack parameter accounting surprise | Organizers count exact/additive model parameters, ASR, and adapters in an unexpected way | Document 4B + Parakeet + adapter count; drop Parakeet from public demo if needed | Local gate / June 14 |
| Local model story drift | README, social copy, or demo says "single multimodal model" while showing Parakeet + 4B | Keep Omni as hosted primary copy; label 4B + Parakeet as offline/local path | Local gate / June 14 |
| Parakeet/NeMo runtime fragility | Optional ASR import breaks cold boot, needs unavailable hardware, or latency is poor | Keep `ALLOW_LOCAL_ASR=false`; use hosted Omni native audio, typed intake, or canned transcript | Local gate |
| ASR transcript over-trust | Parakeet mistranscription changes red-flag meaning or prompt-injection text is treated as instruction | Treat transcript as untrusted draft only; require confirmation; fail closed in validators; never let transcript set final red flags | June 11 / Local gate |
| Parakeet license/attribution gap | Parakeet appears in demo/model card without CC-BY-4.0 attribution | Add Parakeet attribution or remove local ASR from public demo materials | Local gate / June 14 |
| Fine-tune regresses safety | Any June 10 kill criterion trips (recall down, unsafe diagnosis up, JSON breaks, stops citing, over-refusal) | Roll back to base or the prior checkpoint; publish base as the demo model | June 10 |
| AI invents or overstates a protocol pathway | Selected pathway is not in retrieved cards, or rationale implies diagnosis/treatment | Force card-only output, tighten validator, add targeted repair examples, or ship canned traces | June 9 / 12 |
| AI downgrades deterministic red flags | Output urgency is below the rules engine result, or language softens escalation cues | Fail validation; show deterministic warning; repair prompt/fine-tune or fall back to base/canned traces | June 7 / 10 |
| No real user available by June 13 | No responder confirmed | Use a medically literate proxy on simulated cases and say so honestly (honest-fit); keep recruiting | June 13 |

## Testing & CI

Unit-test the safety-critical deterministic components (they must be boringly correct):

* `rules.py` — each red-flag condition fires on a gold positive input and stays silent on a gold negative.
* `validators.py` — rejects invalid JSON, empty `source_cards`, citations to non-existent cards, forbidden phrases, `protocol_urgency` below the deterministic red-flag floor, uncited pathways, and ungrounded checklist/handoff facts.
* `schemas.py` — enum fields (`protocol_urgency`) reject out-of-vocabulary values.
* `audio_intake.py` — produces provisional suggestions only, never overwrites manual values, blocks navigation until confirmation, and records audio provenance without raw audio.
* `navigator.py` — mock and live outputs preserve red-flag locks, cite cards for every selected pathway, avoid diagnosis/prescribing, and keep checklist/SBAR facts grounded.
* `config.py` — rejects illegal `MODEL_STACK`, `MODEL_BACKEND`, `AUDIO_BACKEND`, `ENABLE_AUDIO_INTAKE`, `ALLOW_LOCAL_ASR`, and legacy `ALLOW_STRETCH_STACK` combinations.
* Space smoke test — with no model secrets, app boots, typed intake works, audio is disabled, canned fallback is visible, and trace labels fallback mode.
* Hosted timeout test — mock Omni endpoint timeout falls back without changing confirmed intake, red-flag results, or validator state.
* Optional Parakeet tests — mark `pytest.mark.optional_nemo`; skip unless NeMo is installed; assert transcript is provisional, manual corrections persist, and navigation remains blocked until confirmation.
* Local 4B smoke — health check plus one fixture returning valid navigator JSON for the configured local model.

Use small gold fixtures under `tests/`; run with `pytest`. "Do unit tests pass?" is part of the §13 daily checklist. Owner-days: June 6 (scaffold tests with the skeleton), June 7 (rules/validators tests once those modules exist).

[1]: https://huggingface.co/build-small-hackathon?utm_source=chatgpt.com "Build Small Hackathon"
[2]: https://www.who.int/news/item/18-01-2024-who-releases-ai-ethics-and-governance-guidance-for-large-multi-modal-models?utm_source=chatgpt.com "WHO releases AI ethics and governance guidance for ..."
[3]: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software?utm_source=chatgpt.com "Clinical Decision Support Software - Guidance"
[4]: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
[5]: https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Omni-report.pdf "NVIDIA Nemotron 3 Omni Technical Report"
[7]: https://modal.com/pricing?utm_source=chatgpt.com "Plan Pricing"
[8]: https://modal.com/docs/guide/gpu?utm_source=chatgpt.com "GPU acceleration | Modal Docs"
[10]: https://huggingface.co/pricing?utm_source=chatgpt.com "Pricing"
[11]: https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning "NVIDIA Nemotron 3 Nano Omni endpoint"
[13]: https://huggingface.co/nvidia/parakeet-rnnt-1.1b "nvidia/parakeet-rnnt-1.1b"
[16]: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
