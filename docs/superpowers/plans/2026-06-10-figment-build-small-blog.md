# Figment Build Small Blog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a rough judge-facing Build Small blog post about what Figment taught me.

**Architecture:** The deliverable is a single markdown draft under `docs/`, based on the approved design spec. It should use evidence from project docs, prior thread summaries, git history, Modal metadata, and eval traces while keeping the tone public and readable.

**Tech Stack:** Markdown documentation in the existing Figment repo.

---

### Task 1: Draft The Blog Post

**Files:**
- Create: `docs/figment-build-small-lessons-draft.md`
- Reference: `docs/superpowers/specs/2026-06-10-figment-build-small-blog-design.md`

- [x] **Step 1: Create the draft markdown file**

Write a rough post with these sections:

```markdown
# Building Figment For Build Small: What I Learned About Making Small Models Useful

Opening: Figment is a protocol navigator for field responders, not an AI doctor or autonomous clinical decision tool.

## Audio Should Draft, Not Decide

## Deterministic Safety Rules Are The Floor

## App Safety And Model Competence Are Different Numbers

## Field-Level Provenance Changed My Relationship With Fallback

## Fine-Tuning Only Helped After The Eval Got More Honest

## What Still Is Not Good Enough

## What I Would Build Next

## The Lesson I Am Taking From Build Small
```

- [x] **Step 2: Include evidence anchors**

Use these exact evidence points where they naturally fit:

- Hosted Omni baseline: `28/50` model competence, `50/50` final validation.
- Hosted load-bearing follow-up: `31/50` whole-output competence, `480/650` model-retained fields.
- Local v1 pilot: proved the train/merge/convert/serve/eval loop but regressed to `11/50`.
- Local v2: `33/50` on the locked 50-case eval.
- Local v3 holdout: `107/150` competence, `148/150` final validation.
- V3 weakness: `REFERRAL-SBAR-v1` `0/27`, `radio_handoff` `0/16`, `sbar_handoff_usefulness` `0/10`.
- Modal v3: `700/700` steps, final `eval_loss=0.04357146`, final `train_loss=0.60960097`.

- [x] **Step 3: Avoid unsafe or unsupported claims**

Check the draft does not claim:

- clinical validation,
- diagnosis or treatment,
- autonomous triage,
- complete local/off-grid proof,
- final validation as pure model competence,
- v3 as solved handoff usefulness.

- [x] **Step 4: Self-edit for public readability**

Read the full draft once and tighten:

- make section openings concrete,
- remove repeated metrics,
- keep caveats readable,
- preserve the "systems of restraint" closing.

- [x] **Step 5: Verify and commit**

Run:

```bash
rg -n "diagnose|prescribe|clinical validation|autonomous triage|off-grid proof|TODO|TBD" docs/figment-build-small-lessons-draft.md
git diff --check
```

Expected:

- Any medical-risk phrases appear only as disclaimers or non-goals.
- No placeholder text.
- `git diff --check` prints no output.
