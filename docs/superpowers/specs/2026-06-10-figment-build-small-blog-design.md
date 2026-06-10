# Figment Build Small Blog Post Design

Date: 2026-06-10

## Purpose

Write a rough blog post about what I learned while building Figment for the Hugging Face Build Small Hackathon.

The post should primarily serve judges and public project readers, while still being useful to other builders. It should be candid about failed evals, fallback, training iteration, and remaining weaknesses, but should not read like an internal debugging diary.

## Audience

Primary:

- Hugging Face Build Small judges.
- Public readers who want to understand what Figment proves and what it does not yet prove.

Secondary:

- Hackathon builders and AI engineers interested in small-model product design.

## Tone

Use earned candor:

- Be honest about the messy parts.
- Separate app safety from model competence.
- Keep the center of gravity on what was learned and why the project improved.
- Avoid overclaiming medical, local/off-grid, or fine-tuning success.

The voice should sound like a reflective builder: technical, grounded, and specific, but still readable as a public blog post.

## Recommended Angle

Use a hybrid of:

1. The field tool story.
   - Figment is a protocol navigator for rural clinics, mobile units, and disaster response contexts.
   - It is not an AI doctor.

2. The small models need systems story.
   - Small models become useful when surrounded by scope, contracts, validators, provenance, traces, and honest evals.

Use the evaluation/training evidence as proof points throughout rather than opening with a wall of metrics.

## Proposed Structure

1. Set the scene: why Figment exists and why the Build Small constraint mattered.
2. Lesson 1: audio should draft, not decide.
3. Lesson 2: deterministic safety rules are not a fallback; they are the floor.
4. Lesson 3: model competence and app safety are different numbers.
5. Lesson 4: field-level provenance changed how fallback felt.
6. Lesson 5: fine-tuning worked only after the eval measured the real workflow.
7. What still is not good enough: SBAR and radio handoff usefulness.
8. What comes next: focused v4 work on handoff, cue ownership, and workflow metrics.
9. Closing: Build Small taught me that useful small-model apps are systems of restraint.

## Evidence To Include

Use these numbers sparingly, as anchors:

- Hosted Omni baseline eval: `28/50` hosted model competence and `50/50` final validation.
- Hosted Omni load-bearing follow-up: `31/50` whole-output competence, `8/50` full fallback, `480/650` model-retained fields, `170/650` deterministic patches.
- Local 4B baseline after scaffolding: `26/50` competence, mostly via repair, with `50/50` final validation.
- Local 4B v1 pilot: proved the train, merge, convert, serve, and eval loop, but regressed competence to `11/50`.
- Local 4B v2: improved to `33/50` on the locked 50-case eval.
- Local 4B v3 field-workflow holdout: `107/150` competence, `93/150` raw successes, `14/150` repairs, `2/150` full fallbacks, `148/150` final validation.
- V3 weakness: `REFERRAL-SBAR-v1` at `0/27`, `radio_handoff` at `0/16`, `sbar_handoff_usefulness` at `0/10`.
- Modal v3 training: `700/700` steps, final `eval_loss=0.04357146`, final `train_loss=0.60960097`, artifacts present in `figment-checkpoints:/figment_sft_v3/figment-sft-v3-lora`.

## Claims To Avoid

- Do not claim Figment is clinically validated.
- Do not claim it diagnoses, prescribes, or makes autonomous triage decisions.
- Do not claim final validation is pure model competence.
- Do not claim local/off-grid proof is complete unless referring narrowly to recorded local text-model evidence.
- Do not imply the v3 fine-tune solved handoff usefulness.

## Drafting Notes

The rough draft should read as a blog post, not a project README. It can use section headings, short paragraphs, and a few concrete metrics. It should emphasize decisions and lessons, not implementation minutiae.

The strongest closing thought is that "small" was not just about model size. It was about narrowing the job until the model could be useful, measured, and corrected.
