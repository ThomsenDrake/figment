# Smaller LLM Load-Bearing Action Plan

Date: 2026-06-07

This note captures the adversarial review items from the Build Small Hackathon review. The goal is to make Figment more honest and more robust when smaller LLMs or companion small models are load-bearing, while preserving deterministic safety gates.

## Source Constraints

- Build Small requires a Gradio app hosted as a Hugging Face Space, a short demo video, and a social post.
- Total model parameters must be <= 32B.
- Backyard AI is judged on a specific real problem, real use by the target person, honest fit to small models, and Gradio polish.
- Off the Grid needs no cloud APIs at runtime.
- Well-Tuned needs a published fine-tuned model used by the app.
- Llama Champion needs a llama.cpp runtime path.

## Current Evidence Snapshot

- Local tests passed after the fanout/integration pass, evidence-manifest updates, claim-audit guard, and evidence-gate report: `python3 -m pytest -q` reported `133 passed`.
- Hosted Omni baseline eval: `28/50` whole-output competence, `22/50` full deterministic fallback, `50/50` final validation.
- Hosted Omni load-bearing follow-up eval: `31/50` whole-output competence, `8/50` full fallback, `480/650` model-retained fields, `170/650` deterministic patches, `50/50` final validation.
- Public Space cold boot is now verified in no-secret canned-fallback mode at Space commit `5dcfc5c830de7331eca9020b17e1c571a8619654`: Space API `runtime.stage=RUNNING`, `app.py` present, HTTP 200, typed intake workflow, deterministic pediatric-dehydration escalation, protocol retrieval, and trace route `raw_route=canned` / `final_route=canned_backend` with `validation_status=passed` and `raw_audio_stored=false`.
- The full-weight local 4B text artifact is now downloaded: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` at commit `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`, with `model.safetensors` size `7947142640` bytes and SHA-256 `55d4e2519456c4a9bddf596b0748d630e3b2ce6ff6f4c2b7ed3e07e2b00dad42`. Local 4B and Parakeet paths are still not proven with a real no-cloud 50-case eval or local ASR evidence.
- Local evidence capture is now scripted via `scripts/run_local_4b_evidence.py`; it records endpoint metadata, route smoke, eval records, eval summary, `eval_evidence_manifest.json`, and gated proof flags.
- The local Parakeet ASR artifact is now downloaded: `nvidia/parakeet-rnnt-1.1b` at commit `a07b19e98a26c1873a3f2622c446a4a1ca6316cb`, with `parakeet-rnnt-1.1b.nemo` size `4283105280` bytes and SHA-256 `535896f014953d945b287ac533560e20da8103c6781b152de4645528e2b60738`. ASR evidence capture is scripted via `scripts/run_local_asr_evidence.py`, including `asr_evidence_manifest.json`, but no real local ASR provider payload has passed yet.
- Submission-facing claim drift is now audited by `scripts/audit_submission_claims.py` / `make audit-claims`; the current audit passes while keeping Off the Grid, Llama Champion, Well-Tuned, Backyard user-use, local 4B competence, local ASR, demo video, and social post gates false.
- Evidence-gate status is now reportable via `scripts/evidence_gate_status.py` / `make evidence-gates`; it currently marks public Space, hosted Omni eval, and claim audit as present, and keeps local 4B 50-case eval, no-cloud route, Llama Champion route, local ASR provider proof, trained-responder user test, demo video, social post, and Well-Tuned adapter incomplete.

## Work Items

### 1. Make the public Hugging Face Space runnable

- [x] Push the full Gradio app, requirements, protocol cards, demo assets, and README to `build-small-hackathon/figment`.
- [x] Verify a public cold boot from the Space URL, not only local startup.
- [x] Record Space evidence in `docs/submission_checklist.md`.
- [x] Ensure no-secret mode cold-boots with typed intake, honest canned fallback, and trace labeling.

### 2. Add a parameter and evidence ledger

- [x] Create a compact ledger with model ID, route, total-parameter source, active-parameter note, adapter parameter count, ASR companion count, endpoint locality, and organizer-confirmation status.
- [x] Explicitly mark active parameters as not the compliance number.
- [x] Include the Omni 31B body-count versus 33B sidebar ambiguity.
- [x] Include the local 4B + Parakeet additive story and adapter headroom.

### 3. Prove the smaller local LLM path separately

- [x] Download the full-weight `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` artifact for the local route.
- [x] Add an evidence-bundle helper for the local endpoint.
- [ ] Run the 50-case eval through `MODEL_BACKEND=llama_cpp` against a real local OpenAI-compatible endpoint.
- [ ] Record model/server metadata, no-cloud evidence, raw success, repair success, full fallback, field provenance, latency, and trace hashes. The helper now generates `eval_evidence_manifest.json` with these fields after a completed local eval; the item remains unchecked until a real local run exists.
- [x] Add a run note that distinguishes local 4B evidence from hosted Omni evidence.
- [x] Ensure whole fallback cannot count as local model competence.

### 4. Publish model scorecards separate from app safety

- [x] Treat `final_validation_successes` as app safety, not model competence.
- [x] Publish raw-only, repaired, hybrid, full-fallback, and per-field scorecards.
- [x] Update stale README and checklist language that still says no full eval has run.
- [x] Highlight remaining weak areas, especially SBAR and latency.

### 5. Align live app validation with strict eval validation

- [x] Make runtime navigation use strict schema checks, retrieved-card constraints, and retrieved-card observation grounding.
- [x] If runtime and eval validation intentionally differ, trace the mode explicitly.
- [x] Prevent sparse raw model output from being labeled fully `model_raw` when it would fail strict eval.

### 6. Add hybrid route labeling and field provenance UI

- [x] Add a distinct route for outputs with deterministic patches, such as `model_with_deterministic_patches`.
- [x] Derive runtime labels from trace state rather than configured backend.
- [x] Show per-field or per-section provenance counts in Navigator Output and Trace.
- [x] Rename config-derived labels like "Hosted Omni (live)" to "Configured backend" unless the post-run trace proves live generation.

### 7. Cap or batch repair calls for smaller models

- [x] Cap focused repair attempts.
- [x] Prefer one compact JSON-only repair call when multiple scopes fail.
- [x] Add metrics for repair call count and repair latency.
- [x] Keep deterministic fallback for fields that still fail after the cap.

### 8. Score expected eval labels directly

- [x] Compare actual red flags to `expected_red_flag_rule_ids`.
- [x] Compare actual urgency to `expected_min_protocol_urgency`.
- [x] Compare source cards and candidate pathways to target or expected card IDs.
- [x] Compare missing observations to expected missing-observation cues.
- [x] Check case-level forbidden behavior explicitly.

### 9. Treat Parakeet as unproven until real local ASR exists

- [x] Add `transcript_source` or equivalent provenance to audio drafts.
- [x] Keep typed transcript heuristics labeled as typed or heuristic, not Parakeet ASR.
- [x] Only emit Parakeet provenance from a real gated ASR adapter or smoke.
- [x] Download the `nvidia/parakeet-rnnt-1.1b` artifact and add a gated evidence helper.
- [x] Add an ASR evidence manifest for artifact, provider-payload, draft-check, route, and raw-audio proof fields.
- [ ] Add a real local ASR proof note before making Parakeet demo-visible.

### 10. Harden audio confirmation UX

- [x] Prevent "Apply Audio Draft" from becoming quiet bulk acceptance.
- [x] Preserve applied draft fields as `applied_unreviewed` until the responder explicitly accepts, edits, or rejects each suggestion.
- [x] Block navigation while any applied audio-derived field is still unreviewed.
- [x] Keep manual edits winning over audio drafts.

### 11. Fix demo audio and hosted audio disclosure

- [x] Derive canned/demo transcript text from one source of truth so ages and case facts stay aligned.
- [x] Label committed Voxtral clips as synthetic demo assets, not local ASR proof.
- [x] Add hosted-mode UI copy that audio is sent to the configured hosted endpoint and must be synthetic or de-identified.
- [x] Add size and duration caps for hosted audio drafts.

### 12. Keep badge and submission copy evidence-gated

- [x] Keep Off the Grid, Llama Champion, Well-Tuned, Backyard AI user-use, demo video, and social post claims conditional until artifacts exist.
- [x] Replace overclaiming workback/social snippets with achieved-versus-targeted wording.
- [x] Add an automated submission-copy audit for premature achieved/proven/used/tested wording.
- [x] Add an evidence-gate status report that lists current proof paths and next actions.
- [ ] Fill user-test notes from a real trained-responder session before claiming the target user used or tested Figment.

## Parallel Work Map

### Worker A: Submission Docs And Badge Honesty

Owns work items 1, 2, 4, and 12 where they are documentation-only.

Primary files:

- `README.md`
- `docs/submission_checklist.md`
- `docs/hosted_omni_eval_results.md`
- `docs/model_load_bearing_work_items.md`
- `docs/figment-workback-plan.md`
- New ledger note if needed under `docs/`

### Worker B: Runtime Validation And Provenance

Owns work items 5, 6, and 7.

Primary files:

- `figment/navigator.py`
- `figment/trace.py`
- `figment/validators.py`
- `figment/field_provenance.py`
- `figment/focused_repair.py`
- `app.py` route/provenance display only
- Related tests under `tests/`

### Worker C: Eval Scoring And Local LLM Evidence

Owns work items 3 and 8.

Primary files:

- `scripts/run_eval.py`
- `scripts/smoke_model_route.py`
- `figment/eval_metrics.py`
- `tests/test_eval_runner.py`
- `tests/test_eval_metrics.py`
- `tests/test_model_route_smoke_script.py`
- Eval docs under `docs/`

### Worker D: Audio Provenance And Confirmation Safety

Owns work items 9, 10, and 11.

Primary files:

- `figment/audio_intake.py`
- `figment/model_client.py`
- `app.py` audio UI/functions only
- `data/demo_audio/manifest.json`
- `tests/test_audio_app_flow.py`
- `tests/test_runtime_honesty.py`

## Integration Rules

- Do not weaken deterministic red-flag rules, intake confirmation, validators, or raw-audio scrubbing.
- Do not count canned fallback or deterministic patches as pure model competence.
- Do not claim Off the Grid, Well-Tuned, Llama Champion, or target-user use without traceable evidence.
- Preserve typed-only operation when audio is disabled.
- Keep hosted demo behavior honest when secrets or model endpoints are absent.
