# Figment Safety Statement

Status: submission-ready draft. Evidence links refreshed 2026-06-15; preserve the route and safety boundaries when citing submission claims.

## Intended Use

Figment is a prototype protocol-navigation aid for trained field responders working with synthetic or de-identified scenarios. It is designed to help structure field notes, surface retrieved protocol cards, preserve deterministic danger-sign rules, list missing observations, draft responder checklists, and prepare SBAR-style handoffs.

Figment is not a medical device. It is not intended for diagnosis, treatment, prescribing, medication dosing, triage automation, or use by untrained members of the public.

## Human Role

The trained responder remains responsible for judgment and action. Figment outputs are advisory, card-cited, and constrained by deterministic safety gates. A responder should ignore, correct, or escalate beyond Figment whenever local protocol, clinical supervision, or field conditions require it.

## Safety Controls

- Deterministic red-flag rules run before model navigation and set a minimum urgency floor.
- Retrieved protocol cards are treated as the source of truth for the navigator.
- The model is constrained to bounded fields: candidate pathways, uncertainty notes, missing observations, responder checklist, and handoff draft.
- Validators reject malformed JSON, unsupported citations, unsafe actions, and outputs that conflict with deterministic urgency floors.
- Canned fallback is labeled as fallback and must not be counted as live model proof.
- Audio intake drafts editable fields only. Typed or edited values must be confirmed before rules or navigation run.
- Traces expose the route, fallback reason, validation state, and source-card IDs so reviewers can audit behavior.

## Data Handling

The submission should use synthetic or de-identified scenarios only. Do not enter real PHI into the hosted demo.

Local mode is intended to keep runtime inputs on the local machine. Hosted mode may send synthetic or de-identified text or audio to the configured Omni endpoint. Traces should not retain raw audio, uploaded filenames, or unnecessary identifying details.

## Off-Grid Claim Boundary

Figment claims Off the Grid from its offline-capable local design: local protocol cards, deterministic rules, local model artifacts, and local ASR/text-navigation paths can run without cloud APIs. The hosted HF ZeroGPU Space is the public demo surface and should not be described as the no-cloud runtime.

The important safety boundary is separation, not demotion:

- hosted Space evidence proves the public demo surface and route labeling;
- local artifacts and architecture support the no-cloud design claim;
- local ASR remains draft-only until a responder confirms fields before rules or navigation run.

## Current Evidence Status

Achieved in-repo artifacts:

- deterministic safety contract and protocol-card navigation scaffold;
- hosted NVIDIA Omni client path and labeled canned fallback path;
- trace and validator surfaces for review;
- synthetic demo-audio asset path;
- public Hugging Face Space cold boot with app files present, verified 2026-06-15;
- embedded launch demo video at https://huggingface.co/spaces/build-small-hackathon/figment/resolve/main/assets/figment-live-space-launch-final.mp4;
- published Field Notes blog at https://huggingface.co/blog/build-small-hackathon/figment-build-blog;
- published social post at https://x.com/ThomsenDrake/status/2066630062649328098?s=20;
- published 4B LoRA model archive and eval/training dataset repo;
- measured v14p repair-union result on the corrected 150-case field-workflow holdout: 150/150 competence, 150/150 expected labels, 150/150 final validation, 0 deterministic patches, and 0 fallback;
- hosted Omni follow-up eval: 31/50 whole-output competence, 8/50 full fallback, 480/650 model-retained fields, 170/650 deterministic patches, and 50/50 final validation;
- this safety statement;
- Apache-2.0 license file.

Remaining external evidence gaps:

- recorded live model run whose trace is not canned fallback;
- real trained-responder user-test notes using synthetic or de-identified scenarios;
- local Parakeet ASR provider proof before claiming local audio.

## Non-Goals

Figment will not:

- diagnose a condition as fact;
- prescribe medication or provide doses beyond cited protocol-card content;
- replace clinician, supervisor, or responder judgment;
- hide when output came from fallback;
- store PHI or raw audio as a submission artifact;
- present the hosted ZeroGPU route itself as the no-cloud runtime.
