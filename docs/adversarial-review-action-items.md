# Figment adversarial review action items

Date: 2026-06-07

This note tracks the nine outstanding items from the adversarial review against the Build Small Hackathon outline, with emphasis on making model behavior visibly load-bearing while preserving Figment's deterministic safety contract.

Important nuance: the current Omni-first architecture can technically support an Off the Grid claim if Nemotron 3 Nano Omni is self-hosted on adequate local hardware with no runtime cloud APIs. The present gap is proof and available hardware, not that the architecture is inherently cloud-only. On the hardware available in this workspace, the practical local proof remains the smaller 4B text path, optionally with Parakeet after its gate passes.

## 1. Make the Hugging Face Space actually runnable

Current status: the Space now cold-boots in no-secret canned-fallback mode. On 2026-06-07 the public Space API reported `runtime.stage=RUNNING`, `sdk=gradio`, `sha=5dcfc5c830de7331eca9020b17e1c571a8619654`, 92 siblings, and `app.py` present; the Space URL served HTTP 200. A public Gradio API smoke loaded typed intake, fired deterministic pediatric-dehydration escalation, retrieved protocol cards, and returned an honestly labeled `canned_backend` trace with `validation_status=passed` and `raw_audio_stored=false`.

Fix direction:
- Push the full Gradio app, requirements, protocol cards, and demo assets to `build-small-hackathon/figment`.
- Verify cold boot from the Space URL, not only locally.
- Align Space metadata and dependency versions with the local repo.

Done when:
- [x] The public Space boots to the Figment Gradio UI.
- [x] The Space can run typed intake with canned fallback and trace labeling when secrets are absent.
- [ ] Hosted Omni mode works when secrets are present, or failure is clearly labeled in a final demo trace.

## 2. Make live model contribution visible

Current risk: checked-in demo traces are canned, and UI labels can imply configured hosted/local mode even after fallback.

Fix direction:
- Add trace and UI states that distinguish `live_model_generated`, `model_repaired`, `validation_fallback`, and `canned_backend`.
- Commit at least one validated non-canned trace from hosted Omni or a local OpenAI-compatible model.
- Ensure fallback output never looks like live model proof.

Done when:
- Trace JSON records raw route, final route, fallback reason, and validation status.
- The Navigator and Trace tabs show the actual post-run route, not only the configured backend.

## 3. Make the 4B or Omni model load-bearing in bounded fields

Current risk: deterministic rules, retrieval, validators, and canned fallback can make the app look useful without proving model reasoning.

Fix direction:
- Keep deterministic rules, urgency floors, retrieval, and validators as safety boundaries.
- Make the configured model responsible for bounded, visible fields: candidate pathways, missing observations, checklist, uncertainty notes, and SBAR draft.
- Preserve fallback for safety, but treat fallback as reliability support rather than model competence.

Done when:
- A non-canned trace shows model-generated values for the bounded navigator fields.
- The trace/UI makes clear which fields came from model output versus deterministic fallback.

## 4. Build the eval harness before fine-tuning

Current risk: target metrics are documented, but no model eval script or result artifact exists.

Fix direction:
- Add `scripts/run_eval.py`.
- Record raw model output, repaired output, fallback output, validation failures, fallback reason, latency, model ID, and trace hash.
- Separate raw model pass, repair pass, and canned fallback pass.

Done when:
- The initial eval cases can be run against `canned`, `hosted_omni`, and `llama_cpp` routes.
- Eval output cannot count fallback as small-model success.

## 5. Expand eval data from seed fixtures to proof

Current risk: 10 handwritten cases cover card shape and references but are not enough to prove small-model protocol navigation.

Fix direction:
- Expand toward at least 50 cases.
- Include negatives, negations, paraphrases, multi-card cases, no-relevant-card cases, noisy notes, prompt injection, routine/monitor cases, and ASR-like transcription errors.

Done when:
- The eval set exercises both safety boundaries and usefulness.
- The README can report measured results rather than targets only.

## 6. Tighten validators for small-model failure modes

Current risk: validators check important basics but miss some subtle failures: wrong known-but-not-retrieved cards, generic checklists, shallow SBAR grounding, unsafe synonyms, and fallback masking.

Fix direction:
- Validate source cards against retrieved cards, not all known cards.
- Require fired rule cards to be included.
- Enforce full schema shape.
- Compare missing observations to retrieved card requirements where possible.
- Broaden forbidden action detection.
- Track fallback separately in eval metrics.

Done when:
- A valid-looking but wrong, generic, or unsafe model output fails deterministically.
- Tests cover the new validator behavior.

## 7. Stop UI and trace overclaiming

Current risk: audio and model labels can imply Omni, local 4B, or SQLite FTS even when the runtime path was typed transcript, canned fallback, or JSON search.

Fix direction:
- Derive post-run badges from trace state.
- Conditionally relabel or disable audio UI when audio intake is off.
- Label typed transcript heuristics separately from real Omni audio or Parakeet ASR.
- Report retrieval source as `sqlite_fts` or `json_fallback`.

Done when:
- The UI remains honest under no-secret, hosted, local, audio-disabled, and fallback modes.

## 8. Prove or demote local/off-grid claims

Current risk: local Parakeet is metadata-only today, and local 4B is currently just an OpenAI-compatible client route. Omni can be off-grid in principle on adequate hardware, but this repo still needs evidence for any claimed off-grid runtime.

Fix direction:
- Add gated smoke tests for local OpenAI-compatible text navigation.
- Add optional Parakeet/ASR proof only behind its explicit gate.
- Add a hardware/evidence note that distinguishes theoretical Omni self-hosting from the currently verified local route.
- Claim Off the Grid only for a recorded no-cloud run, whether that run uses self-hosted Omni on suitable hardware or the smaller local stack.

Done when:
- A trace and short run note prove a no-cloud model path, or the claim is clearly marked unproven.

## 9. Complete submission evidence and docs

Current risk: the repo lacks final submission evidence such as user test notes, safety statement, license file, and demo/social links.

Fix direction:
- Add `docs/user_test_notes.md`.
- Add `docs/safety_statement.md`.
- Add `LICENSE`.
- Add demo video and social post placeholders or final links.
- Replace broad badge language with achieved-versus-targeted status.

Done when:
- Backyard AI evidence shows a real trained responder used it on synthetic or de-identified scenarios.
- The README/submission materials only claim badges backed by artifacts.
