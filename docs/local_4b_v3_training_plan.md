# Local 4B V3 Training Plan

Date: 2026-06-09

## Purpose

The v3 goal is not to make the local 4B model a broadly useful assistant. The goal is to make it better at the specific job Figment exists to support: helping rural clinic medics and disaster first-response medics move faster and more safely through patient intake, red-flag escalation, protocol navigation, missing-observation collection, and handoff drafting.

The v2 LoRA is a real improvement, but it was trained from the current evaluator's pressure points. V3 should keep those gains while reducing the risk that the model has learned "pass the 50-case exam" rather than "make the medic's workflow easier inside this harness."

## Current Evidence

Latest v2 eval trace:

- Trace: `traces/local_4b_finetuned_v2_evidence_20260609T103344Z/`
- Dataset: `data/finetune/figment_sft_v2.jsonl`
- Dataset manifest: `data/finetune/figment_sft_v2_manifest.json`
- Exact overlap between v2 training case ids and the 50 eval case ids: `0/50`

V2 eval results:

- `competence_successes`: `33/50`
- `raw_configured_model_successes`: `33/50`
- `fallback_uses`: `0`
- `repair_successes`: `0`
- `final_validation_successes`: `50/50`
- `model_retained_field_count`: `627/650`
- `model_field_pass_rate`: `0.9646`
- `expected_label_successes`: `15/50`
- Remaining expected-label failures: mostly missing-observation cue coverage (`29`), source-card coverage (`9`), red-flag match (`7`), and candidate pathway coverage (`4`).

V2 data shape:

- `1500` accepted rows.
- `1000` full navigator rows.
- `500` focused repair rows.
- Dominant categories were current-eval failure classes:
  - `missing_observation_cues`: `483`
  - `negation_safety_boundary`: `236`
  - `source_card_candidate_pathway`: `227`
  - `focused_repair:handoff_note_sbar`: `125`
  - `focused_repair:missing_observations`: `125`

Interpretation:

- V2 is not merely memorizing the named 50 eval cases.
- V2 is still at risk of semantic over-rotation because its category mix is tightly coupled to the current evaluator's visible failures.
- V3 should introduce a separate field-workflow target and a new held-out workflow suite before adding more training rows.

## V3 North Star

Train and evaluate the local 4B route as a bounded field-workflow model.

The model should be good at:

- Turning messy confirmed intake into a concise, card-cited navigator output.
- Preserving deterministic red flags and urgency floors.
- Avoiding diagnosis, treatment orders, dosing, discharge advice, or autonomous triage.
- Asking for the next observations that will actually help a responder move the case forward.
- Producing compact SBAR handoff language grounded in confirmed intake, deterministic rules, and retrieved protocol cards.
- Handling denied symptoms, uncertain reports, ASR-like noise, missing vitals, and low-resource context without hallucinating.
- Reducing responder cognitive load: fewer irrelevant fields, fewer generic checklists, clearer next steps, and faster handoff readiness.

The model does not need to be good at:

- General chat.
- Medical reasoning outside retrieved protocol cards.
- Open-ended clinical advice.
- Audio transcription itself.
- Replacing deterministic red-flag rules, validators, or local protocol.

## Anti-Overfitting Policy

V3 must use three distinct evaluation surfaces:

1. Locked 50-case regression eval.
   - Existing eval files stay locked.
   - Never train on these cases or close paraphrases.
   - Use this to ensure v3 does not regress from v2.

2. New field-workflow holdout eval.
   - Create before v3 training data generation.
   - Freeze case ids, row hashes, expected outcomes, and prompt hashes.
   - Never train on it.
   - Use it as the primary v3 success metric.

3. Synthetic development eval.
   - Regenerable and expandable.
   - Used for iteration, debugging, and per-category acceptance.
   - Safe to use for failure analysis, but not copied into training rows.

Every generated training row should pass near-neighbor rejection against both locked eval surfaces. Exact id exclusion is not enough. Reject rows with high similarity in:

- normalized confirmed intake text,
- target card,
- retrieved card set,
- red-flag set,
- missing-observation target set,
- SBAR situation/background wording,
- scenario template and patient presentation.

## New Field-Workflow Holdout

Create `data/eval/field_workflow_holdout_v1.jsonl` with 150 to 250 cases.

This holdout should measure whether Figment makes field work easier, not only whether it satisfies current evaluator slots.

Recommended categories:

- Rural clinic intake: limited equipment, missing vitals, one medic, delayed clinician callback.
- Disaster triage desk: noisy notes, multiple patients, scarce transport, incomplete identity details.
- Radio or runner handoff: fragmented observations, corrections, repeated facts, ambiguous timing.
- ASR-like confirmed text: homophones, dropped negations, punctuation-free fragments, but still confirmed by the responder before navigation.
- Escalation precision: clear red flags, near misses, denied symptoms, historical symptoms, contradictory witness reports.
- Missing-observation prioritization: ask for the few observations that change escalation or handoff quality first.
- SBAR usefulness: compact situation/background/assessment/request that a receiving clinician or transport coordinator can act on.
- Source-card discipline: relevant card, distractor card, missing card, and safety-boundary fallback cases.
- Low-resource constraints: no pulse ox, no BP cuff, no transport yet, paper protocol only, intermittent radio.
- Workflow recovery: previous output weak or overlong; focused repair should improve only the bad fields.

Holdout scoring should include both current harness metrics and workflow metrics:

- current schema and deterministic validation,
- red-flag match,
- urgency floor preservation,
- source-card and candidate-pathway correctness,
- observation cue coverage,
- forbidden clinical behavior absence,
- SBAR grounding,
- number of high-value next observations in the first five suggestions,
- generic/low-value checklist ratio,
- unsupported-fact count,
- output brevity and scanability,
- "handoff readiness" binary score,
- estimated responder time saved proxy.

For v3, the primary success claim should come from this holdout, not from the existing 50-case eval alone.

## V3 Dataset Target

Create `data/finetune/figment_sft_v3.jsonl` with 2500 to 3500 accepted rows after validation.

Recommended accepted-row mix:

- 900 to 1100 full navigator rows from rural clinic and disaster workflow scenarios.
- 400 to 600 escalation precision rows covering red flags, denied red flags, ambiguous reports, and contradiction handling.
- 350 to 500 missing-observation prioritization rows where the target is not "include every cue" but "surface the next useful observations in priority order."
- 300 to 450 SBAR usefulness rows focused on compact, grounded handoff language.
- 200 to 300 source-card discipline rows with distractors, missing relevant cards, and safety-boundary fallbacks.
- 150 to 250 low-resource workflow rows where missing equipment changes what the model should ask for.
- 300 to 500 focused-repair rows sampled from actual v2 failures and new workflow dev failures.

Do not simply add more `missing_observation_cues` rows in the v2 style. V3 should include required observations, but the target behavior is field usefulness: ask for observations that reduce uncertainty, support escalation, or improve handoff.

## Teacher Generation Strategy

Continue using `nvidia/nemotron-3-ultra-550b-a55b` as the teacher through the existing OpenAI-compatible hosted endpoint and existing secret wiring.

Teacher calls should generate three artifacts per candidate scenario:

1. Scenario spec.
   - Synthetic and de-identified.
   - Includes setting, responder constraints, confirmed intake, available supplies, missing equipment, and communication channel.

2. Workflow rubric.
   - Expected urgency floor.
   - Expected red flags.
   - Relevant and distractor cards.
   - High-value observations in priority order.
   - SBAR facts allowed and disallowed.
   - Safety-boundary constraints.

3. Gold assistant output.
   - The exact harness prompt shape as input.
   - Final assistant output as JSON only.
   - No visible reasoning, markdown, or teacher critique.

Teacher output is not trusted directly. Accept a row only after deterministic validators and workflow validators pass.

## V3 Validators

Keep all v2 validators:

- schema validation,
- harness prompt alignment,
- card-id validation,
- deterministic urgency floor validation,
- red-flag validation,
- source-card validation,
- candidate-pathway validation,
- forbidden behavior scanner,
- no teacher notes or reasoning leakage,
- synthetic/de-identified metadata,
- no locked-eval copy or near paraphrase.

Add v3 workflow validators:

- Priority validator: the first 3 to 5 `next_observations_to_collect` must include observations that would materially help escalation, monitoring, or handoff.
- Generic-output validator: reject rows dominated by vague suggestions such as "monitor closely", "repeat vitals", or "follow protocol" without case-specific observation targets.
- Low-resource validator: if equipment is unavailable, the output should not ask for that measurement as if it were immediately available; it can ask for alternatives or state unavailable status.
- Handoff-readiness validator: SBAR must include a concise situation, grounded background, observation-only assessment, and a specific request/pathway.
- Unsupported-fact validator: SBAR and checklist must not add facts absent from confirmed intake, deterministic rules, or retrieved cards.
- Cognitive-load validator: reject overlong lists unless the case has genuine multi-card complexity.
- Similarity validator: reject near-neighbors of locked 50-case eval and field-workflow holdout cases.

## Training Technique

Use the v2 LoRA as a baseline, but do not assume v3 should continue from it blindly.

Run two training variants on Modal:

1. Fresh v3 LoRA from the full BF16 base.
   - Best for checking whether v2 overfit is baked into the adapter.
   - Train on v3 only, with task-balanced sampling.

2. Continued LoRA from v2.
   - Best for preserving v2 schema and raw-output gains.
   - Train at lower learning rate with replay rows from v2.

Compare both variants on the locked 50-case eval and the new field-workflow holdout.

Recommended training recipe:

- Base model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`.
- Method: LoRA SFT, full-weight BF16 base, merged back to BF16 before GGUF conversion.
- Context: keep `16384` unless memory requires an explicitly labeled short-context ablation.
- Start with LoRA rank `16` or `32`; use rank `32` only if v3 underfits workflow diversity.
- Keep dropout around `0.05`.
- Use task-balanced sampling rather than raw row order.
- Use replay mixing: 15% to 25% high-quality v2 rows so schema discipline does not regress.
- Do not optimize checkpoint selection on training loss alone.

If SFT improves format but workflow usefulness remains brittle, add a small preference-tuning stage using deterministic pairs:

- preferred output: concise, grounded, high-value next observations, correct cards, safe SBAR;
- rejected output: schema-valid but generic, overlong, eval-cue-stuffed, unsupported, or not useful to a field medic.

Use preference tuning only after the holdout suite exists.

## Modal Job Shape

Reuse the existing Modal training structure:

- Stage `data/finetune/figment_sft_v3.jsonl` and manifests into the Modal data volume.
- Train adapter under a new dataset version such as `figment_sft_v3`.
- Save adapter and training manifest under a new output name, for example `figment-sft-v3-lora`.
- Run a merge-only job to produce merged BF16 Hugging Face weights.
- Pull merged weights locally.
- Convert to BF16 GGUF with the repo-local `tools/llama.cpp/convert_hf_to_gguf.py`.
- Serve with `llama-server` under the same local OpenAI-compatible route.
- Run locked 50-case eval and field-workflow holdout eval.

Do not publish or treat the checkpoint as accepted until both eval surfaces are complete and artifact-linked.

## Acceptance Targets

V3 should be judged against v2, not just against the original baseline.

Must not regress on the locked 50-case eval:

- `competence_successes`: at least `33/50`.
- `raw_configured_model_successes`: at least `33/50`.
- `fallback_uses`: `0`.
- `final_validation_successes`: `50/50`.
- `model_field_pass_rate`: at least `0.96`.
- `deterministic_patch_count`: at most `23`, or a documented reason if workflow improvements trade off with cue-stuffing.
- `forbidden_behavior_absent`: `50/50`.

Must improve on field-workflow holdout:

- Handoff readiness: at least `80%`.
- High-value first-five observation coverage: at least `80%`.
- Unsupported-fact rate: at most `5%`.
- Generic-output failure rate: at most `10%`.
- Low-resource mismatch rate: at most `5%`.
- Red-flag/urgency safety: no critical misses.
- No increase in forbidden clinical behavior.

Stretch target:

- Improve locked expected-label success above v2's `15/50`, but do not optimize v3 primarily for that number if it conflicts with field-workflow usefulness.

## Immediate Implementation Steps

1. Add `data/eval/field_workflow_holdout_v1.jsonl` and a manifest with frozen hashes.
2. Add a field-workflow eval runner or extend `scripts/run_eval.py` with workflow metrics that do not train on the holdout.
3. Add v3 scenario generators for rural clinic, disaster triage, radio handoff, ASR-like confirmed text, low-resource constraints, and workflow repair.
4. Add workflow validators for prioritization, generic output, low-resource mismatch, handoff readiness, unsupported facts, cognitive load, and near-neighbor rejection.
5. Generate a 100-row v3 smoke dataset and inspect category diversity.
6. Generate the full v3 dataset with the Ultra teacher.
7. Run harness alignment verification on v3.
8. Train fresh-v3 and continued-from-v2 LoRA variants on Modal.
9. Merge, convert, serve, and evaluate both variants locally.
10. Select the checkpoint that best improves field-workflow holdout without regressing locked 50-case safety and competence.

## Decision Rule

Accept v3 only if it improves the real product job: making rural clinic and disaster response intake/escalation/handoff faster, more grounded, and easier to act on.

If v3 only improves the locked 50-case eval while failing the field-workflow holdout, reject it as overfit.

If v3 improves the field-workflow holdout but slightly underperforms one non-safety cue-count metric from the locked eval, prefer the field-workflow result and update the next training/eval plan accordingly.
