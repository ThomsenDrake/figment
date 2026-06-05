# Figment workback plan — additions design

- **Date:** 2026-06-05
- **Status:** Draft for approval
- **Target file:** `docs/figment-workback-plan.md`
- **Author:** brainstorming session (Claude)

## 1. Context

`docs/figment-workback-plan.md` is an executable, day-by-day workback plan for shipping **Figment** (an offline field-clinic copilot) to the Build Small Hackathon by June 15, 2026. It has already been validated against the hackathon org card and corrected. It is strong on schedule, scope, data pipeline, fine-tuning, and eval targets.

This spec defines a set of **additions** to that plan to close gaps in four dimensions the plan currently under-specifies. It does **not** change any existing decision; it adds content.

## 2. Goal & scope

Add the highest-leverage missing content across four lenses, chosen by the author:

1. **Win the judging** (Backyard AI rubric + badges)
2. **De-risk the sprint** (solo, 10 days, zero buffer)
3. **Safety & credibility** (medicine-adjacent)
4. **Engineering rigor** (deterministic safety components, eval, prompting)

**Delivery style (chosen): Lean & integrated.** Additions are woven into existing sections as subsections wherever a natural home exists, and tied to an existing owner-day in §10. Only **one** new top-level section is created (§16) to avoid renumbering churn. Scope is **9 additions** total.

### Design principles / constraints

- Preserve the plan as a single, executable source of truth — additions must be skimmable and actionable, not a binder.
- Every new work item maps to an existing owner-day in the §10 schedule (June 5–15) and, where relevant, to the §13 daily rhythm.
- No change to existing section numbering except appending **§16**. Existing §1–§15 keep their numbers; additions are subsections within them.
- Additions reference, not duplicate, content already in the plan (e.g., the WHO/FDA citations, the output schema in §5, the kill criteria in §10 June 10, the minimum tier in §12).
- This spec describes *what to add and where*; the implementation plan (writing-plans) will sequence the edits.

## 3. The 9 additions

Each addition lists: **lens**, **placement** (host section + owner-day), **content outline**, and **acceptance criteria** (what must be true when done).

### W1 — "Non-goals / what Figment will not do" box
- **Lens:** Win the judging (+ Safety).
- **Placement:** New subsection in **§3 (Figment v1 scope)**, e.g. `## Non-goals`. Owner-day: June 5 (scope freeze).
- **Content:** A tight bullet list of explicit non-goals: will not diagnose; will not prescribe or dose medication; will not replace a clinician; is not for untrained users; does not store or transmit PHI; does not work as an autonomous decision-maker. One sentence framing it as deliberate scope, not a limitation to hide.
- **Acceptance criteria:** §3 contains a clearly headed non-goals list of ≥5 items; each non-goal is phrased as something Figment refuses/avoids; it cross-references the safety statement (S1).

### W2 — Demo-video storyboard (2–3 min beat sheet)
- **Lens:** Win the judging.
- **Placement:** New subsection in **§14 (canonical demo cases)**, e.g. `## Demo video storyboard`; add a pointer task to **§10 June 14**.
- **Content:** A timestamped beat sheet for the 2–3 min video: (0:00) cold open — clinic loses internet; (0:15) flip offline indicator, show Space link is live; (0:30) run Case 1 pediatric dehydration → red-flag fires, missing-info asked, SBAR generated; (1:30) open Trace tab to show the deterministic pipeline; (2:00) one line + table on base-vs-fine-tune; (2:30) close on the named real user / honest-fit statement. Note that the video must show the **hosted Space**, not only the local Mac.
- **Acceptance criteria:** §14 contains an ordered, timestamped beat list totaling ≤3:00; it references Case 1 from §14 and the Trace tab from §3.5; it explicitly requires showing the hosted Space.

### D1 — Risk register & fallbacks
- **Lens:** De-risk the sprint.
- **Placement:** New top-level **§16 (Operational readiness)**, subsection `## Risk register`.
- **Content:** A table with columns `Risk | Trigger (how you know) | Fallback | Owner-day`. Minimum rows: Modal 30B job fails/OOM; model too slow for live demo; synthetic critique keep-rate too low; HF Space won't cold-boot; fine-tune regresses safety (links to June 10 kill criteria); no real user available by June 13. Each fallback points to an existing mechanism (minimum tier in §12, canned-response mode, smaller quant, simulated-case fallback).
- **Acceptance criteria:** §16 contains a 4-column table with ≥6 rows; every fallback references an existing plan mechanism; every row names an owner-day.

### D2 — Performance budget
- **Lens:** De-risk the sprint.
- **Placement:** New subsection in **§2 (local hardware plan)**, e.g. `## Performance budget`; add a measurement task to **§10 June 11**.
- **Content:** Target ranges for first-token latency and tokens/sec for 30B-Q4_K_M on the M4 Pro (to be measured, with a placeholder to fill on June 11), and an explicit degradation ladder with thresholds: if latency/throughput is below target → drop 16k→8k context → drop to a smaller quant → fall back to canned-response mode for the live demo.
- **Acceptance criteria:** §2 states a measurable target (even if values are filled later) and a 3-step degradation ladder with named triggers; §10 June 11 has a task to measure and record the numbers.

### S1 — Safety statement contents
- **Lens:** Safety & credibility.
- **Placement:** New subsection in **§1 (final demo shape)** near §1.2 (clinical restraint), e.g. `## Safety statement (what safety_statement.md must contain)`; referenced by §10 June 5 (draft) and June 14 (final).
- **Content:** The required elements of `safety_statement.md`: intended use; intended user (a trained responder — the named anchor); explicit "not a medical device, not diagnostic, not prescribing"; known limitations (synthetic data, prototype cards, model can err); escalation-not-replacement framing; and the WHO/FDA citations already in §1.2.
- **Acceptance criteria:** §1 lists ≥6 required elements for `safety_statement.md`; it reuses the existing WHO/FDA references; June 5 and June 14 tasks reference it.

### S2 — Licensing & data handling
- **Lens:** Safety & credibility (+ badge requirements).
- **Placement:** New subsection in **§5 (data plan)**, e.g. `## Licensing & data handling`; tie notes to the `model_card.md` / `dataset_card.md` deliverables.
- **Content:** A small license matrix — base model (Nemotron license → governs the published adapter), synthetic dataset (e.g. CC-BY-4.0, decision to confirm), code (e.g. Apache-2.0/MIT, decision to confirm) — plus a PHI/data-handling stance: app is local-only, patient inputs are not logged or transmitted, training data is synthetic with no real PHI (reaffirming the §6 generator rule).
- **Acceptance criteria:** §5 contains a license line for model, dataset, and code (values may be marked "confirm"); a PHI stance of ≥3 statements; cross-reference to model_card/dataset_card deliverables.

### E1 — Testing & CI
- **Lens:** Engineering rigor.
- **Placement:** **§16 (Operational readiness)**, subsection `## Testing & CI`; add one line to **§13 (daily rhythm)** checklist.
- **Content:** A pytest plan for the safety-critical deterministic components: `rules.py` (red-flag triggers fire correctly on gold inputs), `validators.py` (rejects invalid JSON, missing citations, forbidden phrases, risk/red-flag inconsistency), and schema enum validation. Use small gold fixtures. Add "unit tests pass" to the daily checklist. Owner-days: June 6 (scaffold tests with skeleton), June 7 (rules/validators tests).
- **Acceptance criteria:** §16 names the three components under test with ≥1 concrete assertion each; §13 daily checklist gains a "tests pass" line; owner-days assigned.

### E2 — Eval methodology
- **Lens:** Engineering rigor.
- **Placement:** New subsection in **§8 (evaluation plan)**, e.g. `## How each metric is measured`.
- **Content:** For each metric already in the §8 targets table, state the measurement method: deterministic (valid JSON %, source-card citation present, forbidden-phrase scan, risk-level/red-flag consistency, missing-info question presence) vs judge-model-scored (SBAR factuality, unsupported-diagnosis rate, prompt-injection compliance) — including which model judges and the pass rule. Note deterministic metrics run in `run_eval.py`; judge metrics run in `modal/eval_batch.py`.
- **Acceptance criteria:** Every row in the §8 targets table has a stated measurement method; each method is labeled deterministic or judge-scored; the script that computes it is named.

### E3 — Constrained system-prompt / prompt_builder skeleton
- **Lens:** Engineering rigor.
- **Placement:** New subsection in **§9 (app architecture)**, e.g. `## Constrained prompt skeleton`; tie to **§10 June 7** (prompt_builder.py) and referenced by §6 (expected outputs).
- **Content:** The actual constrained prompt structure: a system message establishing role + restraint ("You are Figment… not a clinician; do not diagnose or prescribe"); injection slots for the structured intake, the 3–6 retrieved protocol cards (with `card_id`), and the deterministic red-flag results; behavioral rules (stay inside retrieved cards; cite `card_id`s in `source_cards`; no drug dose unless the card contains one; if critical info missing, populate `missing_info_to_collect` and ask; escalate red flags via `risk_level`; if no relevant card, say so and recommend escalation; refuse out-of-scope via `safety_boundary`); and a strict-JSON output contract matching the §5 output schema with thinking disabled in user-facing mode.
- **Acceptance criteria:** §9 contains a prompt skeleton showing the system message, the injection slots, ≥6 behavioral rules, and a reference to the §5 output schema; §10 June 7 references producing it in `prompt_builder.py`.

## 4. Placement summary

| ID | Lens | Host section | New top-level? | Owner-day(s) |
|----|------|--------------|----------------|--------------|
| W1 | Judging | §3 subsection | no | June 5 |
| W2 | Judging | §14 subsection | no | June 14 |
| D1 | De-risk | §16 | **yes (§16)** | per-row |
| D2 | De-risk | §2 subsection | no | June 11 |
| S1 | Safety | §1 subsection | no | June 5 / 14 |
| S2 | Safety | §5 subsection | no | June 14 (cards) |
| E1 | Rigor | §16 + §13 line | shares §16 | June 6 / 7 |
| E2 | Rigor | §8 subsection | no | June 9 |
| E3 | Rigor | §9 subsection | no | June 7 |

Net structural change: **one** new top-level section (§16 Operational readiness, hosting D1 + E1), plus subsection insertions and a few §10/§13 task lines.

## 5. Non-goals of this change

- No change to the existing scope, schedule dates, model choice, dataset sizes, fine-tune config, or badge plan.
- No renumbering of §1–§15.
- No new code is written; this only edits the planning document.
- No expansion into multi-language/accessibility, telemetry, or post-hackathon roadmap (out of scope for v1).

## 6. Rollout & verification

- Apply the 9 additions as surgical `Edit`s to `docs/figment-workback-plan.md`.
- Re-verify after editing: markdown still well-formed (balanced code fences, valid tables), no broken section references, every new owner-day task maps to a real day, and §16 is the only new top-level section.
- Confirm no contradiction with existing content (e.g., the new non-goals box must agree with §1.2 clinical restraint and the §12 tiers).

## 7. Success criteria

- All 9 additions present, each meeting its acceptance criteria.
- The plan remains a skimmable, executable single source of truth (no orphaned or owner-less work items).
- Every addition is traceable to one of the four lenses and to an owner-day.
- Document passes a structural re-check (fences/tables/references) and a consistency re-check against existing sections.
