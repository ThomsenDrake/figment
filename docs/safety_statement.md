# Figment Safety Statement

Status: submission-ready draft. Update the evidence links before final submission; do not turn pending items into claims until the artifacts exist.

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

Nemotron 3 Nano Omni can technically support an Off the Grid claim if it is self-hosted on adequate local hardware and no runtime cloud APIs are used. The current gap is hardware and recorded evidence, not an architecture impossibility.

Until a no-cloud run is recorded, Figment should describe off-grid support as targeted or proof-needed. A valid proof can come from either:

- self-hosted Omni running locally with no cloud APIs; or
- the smaller verified local stack for text navigation, with Parakeet ASR only after its gate passes.

## Current Evidence Status

Achieved in-repo artifacts:

- deterministic safety contract and protocol-card navigation scaffold;
- hosted NVIDIA Omni client path and labeled canned fallback path;
- trace and validator surfaces for review;
- synthetic demo-audio asset path;
- this safety statement;
- Apache-2.0 license file.

Proof still needed before final submission claims:

- public Hugging Face Space cold boot with app files present;
- recorded live model run whose trace is not canned fallback;
- recorded no-cloud run before claiming Off the Grid as achieved;
- real trained-responder user-test notes using synthetic or de-identified scenarios;
- demo video link;
- social post link;
- measured eval results before claiming model quality thresholds.

## Non-Goals

Figment will not:

- diagnose a condition as fact;
- prescribe medication or provide doses beyond cited protocol-card content;
- replace clinician, supervisor, or responder judgment;
- hide when output came from fallback;
- store PHI or raw audio as a submission artifact;
- present local/off-grid claims without a recorded no-cloud run.
