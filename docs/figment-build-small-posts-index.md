# Figment Build Small Blog Drafts

Draft status: working index for the Build Small writing set.

## 1. Building Figment For Build Small: What I Learned About Making Small Models Useful

File: `docs/figment-build-small-lessons-draft.md`

Audience: judges, builders, and public readers who want the high-level story.

Core argument: useful small-model apps are systems of restraint, and that restraint turns failures into an iteration engine.

Best current evidence to preserve in revisions:

- hosted Omni split between app safety and model competence,
- local v3 field-workflow holdout jump to `107/150`,
- v5 false comfort: `150/150` final validation but only `2/150` model competence,
- corrected scoring view with six changed cases,
- v14p repair-union: `150/150` competence/final validation/expected labels, zero deterministic patches, zero fallback, with `146/150` raw success and `4/150` model-repair cases,
- Field Kit Workbench UI as a clearer product surface without changing the harness contract,
- public Hugging Face artifacts and trace receipts.

## 2. The Eval Loop That Made Figment Better

File: `docs/figment-build-small-eval-loop-draft.md`

Audience: builders who want the technical learning loop.

Core argument: final app validation is not model competence; the later Figment loop worked because raw success, repair success, deterministic patches, and fallback were measured separately.

Best current evidence to preserve in revisions:

- v5 as the run that prevented a misleading success story,
- v6/v7 targeted replay-and-delta curriculum,
- corrected benchmark hygiene instead of training around scorer bugs,
- prompt-policy probes that made results worse,
- v14p repair-union as a narrow but defensible system claim.

## Likely Follow-Up Angles

- A demo-focused post or submission section comparing Figment's evidence depth against clearer Backyard demos.
- A short "what the trace shows" post with screenshots or annotated examples.
- A public artifact guide that points readers to the model repo, dataset configs, and selected eval traces.
