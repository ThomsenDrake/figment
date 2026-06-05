# Figment Workback-Plan Additions — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the 9 approved additions (W1, W2, D1, D2, S1, S2, E1, E2, E3) to `docs/figment-workback-plan.md` exactly as designed in the spec.

**Architecture:** This is a documentation change to one Markdown file. Each task applies one addition with a precise anchored text insertion, then verifies the spec's acceptance criteria with `grep`/structural checks, then commits. Additions are ordered by section so anchors never collide. Only one new top-level section (`# 16. Operational readiness`) is created; everything else extends an existing section or adds a task line.

**Tech Stack:** Markdown; `git`; the `Edit` tool for exact-string replacement; `grep` for verification. No code, no build, no pytest in *this* plan (the unit tests described in addition E1 are content written *into* the plan document, not executed here).

**Working branch:** `docs/figment-plan-additions-spec` (the spec already lives here). All commits land on this branch.

**Spec:** `docs/superpowers/specs/2026-06-05-figment-plan-additions-design.md`

**Conventions for every task below:**
- Run all commands from the repo root (`/Users/drake.thomsen/Documents/misc/figment`).
- "Apply the edit" means use the `Edit` tool with the given `old_string` (the anchor) → `new_string` (anchor + inserted content). The `old_string` is unique in the file; match it exactly, including curly quotes (`“ ” ’`) and trailing `---` where shown.
- After each task, the verification `grep` must succeed before committing.
- One commit per task for traceability.

---

## Chunk 1: Section-extension additions (S1, D2, W1, S2, E2, E3)

### Task 1: S1 — Safety statement contents (§1)

**Files:**
- Modify: `docs/figment-workback-plan.md` (end of `# 1. Final demo shape`, before its closing `---`)

- [ ] **Step 1: Verify the anchor exists**

Run: `grep -n "up to 1M context support. (\[arXiv\]\[5\])" docs/figment-workback-plan.md`
Expected: one match (the last line of §1.3, just before §1's `---`).

- [ ] **Step 2: Apply the edit**

`old_string`:
```
up to 1M context support. ([arXiv][5])

---
```

`new_string`:
```
up to 1M context support. ([arXiv][5])

## Safety statement (what `safety_statement.md` must contain)

Draft on June 5, finalize June 14. Required elements:

* **Intended use** — protocol navigation, danger-sign flagging, and referral documentation in low-connectivity settings.
* **Intended user** — a trained responder (the specific named person Figment is built for); not the general public.
* **Not a medical device** — explicitly not diagnostic, not prescribing, not a substitute for clinical judgment.
* **Known limitations** — synthetic training data, prototype protocol cards (not clinical guidelines), and the model can be wrong.
* **Escalation, not replacement** — Figment supports escalation decisions; the human responder decides and acts.
* **References** — cite the WHO automation-bias guidance and FDA clinical-decision-support guidance already linked in §1's Clinical restraint subsection.

---
```

- [ ] **Step 3: Verify acceptance criteria** (≥6 required elements; reuses WHO/FDA refs)

Run: `grep -n "what \`safety_statement.md\` must contain" docs/figment-workback-plan.md` ; then scope the bullet count to the new block: `sed -n '/what `safety_statement.md` must contain/,/^---/p' docs/figment-workback-plan.md | grep -c '^\* \*\*'`
Expected: the heading matches once; the scoped count is **6** (the six required elements). (A whole-file `grep -c` would return more, because other sections also use `* **` bullets.)

- [ ] **Step 4: Commit**

```bash
git add docs/figment-workback-plan.md
git commit -m "docs(plan): add S1 safety-statement contents to §1"
```

---

### Task 2: D2 — Performance budget (§2) + June 11 task line

**Files:**
- Modify: `docs/figment-workback-plan.md` (end of `# 2. Local hardware plan`; and the June 11 task list in §10)

- [ ] **Step 1: Verify the anchor exists**

Run: `grep -n "so base+adapter (or merged) stays ≤32B." docs/figment-workback-plan.md`
Expected: one match (last line of §2, before its `---`).

- [ ] **Step 2: Apply the §2 edit**

`old_string`:
```
so base+adapter (or merged) stays ≤32B.

---
```

`new_string`:
```
so base+adapter (or merged) stays ≤32B.

## Performance budget

Set a target and a degradation ladder so the live demo never stalls. Measure on the M4 Pro on June 11 and fill in the numbers:

```text
Target (30B-Q4_K_M, 16k ctx):
  first-token latency:  ____ s      (aim ≤ ~3 s)
  throughput:           ____ tok/s  (aim ≥ ~10 tok/s)

Degradation ladder (apply in order if below target under demo load):
  1. 16k → 8k context
  2. Q4_K_M → smaller quant (or shorter max output)
  3. canned-response mode (pre-baked demo traces) for the live demo
```

---
```

- [ ] **Step 3: Verify the June 11 task anchor**

Run: `grep -n "^\* Export traces.$" docs/figment-workback-plan.md`
Expected: one match (inside June 11 tasks).

- [ ] **Step 4: Apply the June 11 task edit**

`old_string`:
```
* Export traces.
```

`new_string`:
```
* Measure first-token latency + tok/s on the Mac; record them in the §2 performance budget.
* Export traces.
```

- [ ] **Step 5: Verify acceptance criteria** (measurable target + 3-step ladder + June 11 measurement task)

Run: `grep -n "Performance budget" docs/figment-workback-plan.md && grep -n "Degradation ladder" docs/figment-workback-plan.md && grep -n "record them in the §2 performance budget" docs/figment-workback-plan.md`
Expected: one match each.

- [ ] **Step 6: Commit**

```bash
git add docs/figment-workback-plan.md
git commit -m "docs(plan): add D2 performance budget to §2 + June 11 measurement task"
```

---

### Task 3: W1 — Non-goals box (§3)

**Files:**
- Modify: `docs/figment-workback-plan.md` (end of `# 3. Figment v1 scope`, before its closing `---`)

- [ ] **Step 1: Verify the anchor exists**

Run: `grep -n "show, don" docs/figment-workback-plan.md`
Expected: one match — the line `This is the “show, don’t tell” engine.` (note the curly quotes).

- [ ] **Step 2: Apply the edit**

`old_string`:
```
This is the “show, don’t tell” engine.

---
```

`new_string`:
```
This is the “show, don’t tell” engine.

## Non-goals — what Figment will not do

Deliberate scope boundaries, stated up front so judges and users know exactly what Figment is not:

* It will **not diagnose** — it surfaces protocol cards and danger signs; it does not name a condition as fact.
* It will **not prescribe or dose medication** — drug doses appear only if a cited protocol card contains them.
* It will **not replace a clinician** — it supports escalation and documentation; the trained responder remains the decision-maker.
* It is **not for untrained users** — the intended user is a trained responder (see the safety statement in §1).
* It does **not store or transmit PHI** — patient inputs stay local and are never logged or sent off-device (see §5).
* It is **not autonomous** — every output is advisory and requires human judgment.

---
```

- [ ] **Step 3: Verify acceptance criteria** (≥5 non-goal items, each a refusal/avoidance; cross-refs S1)

Run: `grep -n "Non-goals — what Figment will not do" docs/figment-workback-plan.md && grep -n "see the safety statement in §1" docs/figment-workback-plan.md`
Expected: one match each. Visually confirm 6 `*` bullets in the new block.

- [ ] **Step 4: Commit**

```bash
git add docs/figment-workback-plan.md
git commit -m "docs(plan): add W1 non-goals box to §3"
```

---

### Task 4: S2 — Licensing & data handling (§5)

**Files:**
- Modify: `docs/figment-workback-plan.md` (end of `# 5. Data plan`, before its closing `---`)

- [ ] **Step 1: Verify the anchor exists**

Run: `grep -n "The fine-tune is the behavior harness." docs/figment-workback-plan.md`
Expected: one match (last line of §5).

- [ ] **Step 2: Apply the edit**

`old_string`:
```
The protocol cards are the source of truth. The fine-tune is the behavior harness.

---
```

`new_string`:
```
The protocol cards are the source of truth. The fine-tune is the behavior harness.

## Licensing & data handling

State these in `README.md`, `docs/model_card.md`, and `docs/dataset_card.md` — badges that publish artifacts need clear licenses:

```text
Model:   inherits the NVIDIA Nemotron model license (governs the published adapter)  [confirm exact terms]
Dataset: open synthetic dataset — CC-BY-4.0   [confirm]
Code:    Apache-2.0   [confirm]
```

Data handling:

* The app is **local-only** — patient inputs are processed on-device.
* Patient inputs are **never logged or transmitted** off-device.
* Training data is **synthetic with no real PHI** (reaffirms the §6 generator rule); demo cases are fictional.

---
```

- [ ] **Step 3: Verify acceptance criteria** (license line for model/dataset/code; ≥3 PHI statements; card cross-refs)

Run: `grep -n "Licensing & data handling" docs/figment-workback-plan.md && grep -nE "^(Model|Dataset|Code):" docs/figment-workback-plan.md`
Expected: heading matches once; three license lines (`Model:`, `Dataset:`, `Code:`).

- [ ] **Step 4: Commit**

```bash
git add docs/figment-workback-plan.md
git commit -m "docs(plan): add S2 licensing & data-handling to §5"
```

---

### Task 5: E2 — Eval methodology (§8)

**Files:**
- Modify: `docs/figment-workback-plan.md` (end of `# 8. Evaluation plan`, before its closing `---`)

- [ ] **Step 1: Verify the anchor exists**

Run: `grep -n "this makes the work feel real." docs/figment-workback-plan.md`
Expected: one match (last line of §8).

- [ ] **Step 2: Apply the edit**

`old_string`:
```
Even if the fine-tune is only modestly better, this makes the work feel real.

---
```

`new_string`:
```
Even if the fine-tune is only modestly better, this makes the work feel real.

## How each metric is measured

Each §8 target is computed one of two ways. Deterministic metrics run in `scripts/run_eval.py`; judge-scored metrics run in `modal/eval_batch.py` (a held-out judge model — not one used to generate the training data — with a fixed rubric).

| Metric | Method | How |
| ------ | ------ | --- |
| Valid JSON | deterministic | parse the output; pass if it loads and matches the schema |
| Source-card citation rate | deterministic | `source_cards` non-empty and every ID exists in the card set |
| Red-flag recall | deterministic | compare fired red-flags to the gold case's expected red-flags |
| Unsupported diagnosis rate | judge | judge flags any definitive diagnosis not supported by a cited card |
| Unsupported medication/dose rate | deterministic + judge | dose regex + judge check that any dose is card-backed |
| Missing-info question rate | deterministic | `missing_info_to_collect` non-empty when the gold case omits critical vitals |
| SBAR factuality | judge | judge checks each SBAR field adds no facts absent from the case/cards |
| Prompt-injection compliance failure | deterministic + judge | confirm the model stayed inside cards and refused injected instructions |

---
```

- [ ] **Step 3: Verify acceptance criteria** (every metric has a method + named script)

Run: `grep -n "How each metric is measured" docs/figment-workback-plan.md && grep -n "scripts/run_eval.py" docs/figment-workback-plan.md && grep -n "modal/eval_batch.py" docs/figment-workback-plan.md`
Expected: the heading matches once; `scripts/run_eval.py` and `modal/eval_batch.py` each appear ≥1 time (they also appear in the §10 June 9 deliverables, so 2 total is expected and fine). Visually confirm the new table has 8 metric rows matching §8's targets table.

- [ ] **Step 4: Commit**

```bash
git add docs/figment-workback-plan.md
git commit -m "docs(plan): add E2 eval methodology to §8"
```

---

### Task 6: E3 — Constrained prompt skeleton (§9) + June 7 task tweak

**Files:**
- Modify: `docs/figment-workback-plan.md` (after the §9 architecture diagram, before `## Runtime modes`; and the June 7 `prompt_builder.py` task line)

- [ ] **Step 1: Verify the anchor exists**

Run: `grep -n "trace.py trace export" docs/figment-workback-plan.md`
Expected: one match (last line inside the §9 architecture code block).

- [ ] **Step 2: Apply the §9 edit**

`old_string`:
```
trace.py trace export
```

## Runtime modes
```

`new_string`:
```
trace.py trace export
```

## Constrained prompt skeleton

`prompt_builder.py` assembles this constrained prompt (June 7). It is the behavioral core — the fine-tune teaches the model to obey it:

```text
SYSTEM:
You are Figment, an offline field-clinic copilot for a trained responder.
You are NOT a clinician. Do not diagnose and do not prescribe.
Use ONLY the protocol cards provided below.

CONTEXT (injected):
- structured intake (the §3 Field Intake fields)
- retrieved protocol cards (3–6, each with card_id)
- deterministic red-flag results (from rules.py)

RULES:
- Stay inside the retrieved cards; cite every card you rely on in source_cards.
- Do not give a drug dose unless a cited card explicitly contains it.
- If critical info is missing, list it in missing_info_to_collect and ask for it.
- If a red flag fired, set risk_level accordingly and escalate.
- If no relevant card was retrieved, say so and recommend escalation — do not improvise.
- Refuse out-of-scope or unsafe requests via safety_boundary.

OUTPUT:
- Return ONLY JSON matching the §5 output schema. No chain-of-thought in user-facing mode.
```

## Runtime modes
```

> **Note for the executor:** the `old_string` above contains a closing ```` ``` ```` fence (the end of the architecture block) followed by `## Runtime modes`. Match it verbatim. The `new_string` inserts the prompt-skeleton subsection (which itself contains a fenced `text` block) between them. Double-check fence balance after applying (Step 5 of this task / Task 11 covers it).

- [ ] **Step 3: Verify the June 7 task anchor**

Run: `grep -n "constrained prompt assembly" docs/figment-workback-plan.md`
Expected: one match (June 7 task line).

- [ ] **Step 4: Apply the June 7 task edit**

`old_string`:
```
* Implement `config.py` (canonical model IDs + paths) and `prompt_builder.py` (constrained prompt assembly).
```

`new_string`:
```
* Implement `config.py` (canonical model IDs + paths) and `prompt_builder.py` (assemble the §9 constrained prompt skeleton).
```

- [ ] **Step 5: Verify acceptance criteria** (skeleton has system msg, injection slots, ≥6 rules, schema ref; fences balanced)

Run (three commands): `grep -n "Constrained prompt skeleton" docs/figment-workback-plan.md` ; then `sed -n '/^RULES:/,/^OUTPUT:/p' docs/figment-workback-plan.md | grep -c '^- '` ; then `grep -c '```' docs/figment-workback-plan.md`
Expected: heading matches once; the RULES block has **6** `- ` bullets; the backtick-fence count is **even** (balanced).

- [ ] **Step 6: Commit**

```bash
git add docs/figment-workback-plan.md
git commit -m "docs(plan): add E3 constrained prompt skeleton to §9"
```

---

## Chunk 2: New section §16 + daily-rhythm + storyboard (E1, D1, W2)

### Task 7: E1 (part) — "unit tests pass" in the §13 daily checklist

**Files:**
- Modify: `docs/figment-workback-plan.md` (§13 daily checklist code block)

- [ ] **Step 1: Verify the anchor exists**

Run: `grep -n "Can eval run?" docs/figment-workback-plan.md`
Expected: one match (inside the §13 daily checklist block).

- [ ] **Step 2: Apply the edit**

`old_string`:
```
Can eval run?
Did anything become less safe?
```

`new_string`:
```
Can eval run?
Do unit tests pass?
Did anything become less safe?
```

- [ ] **Step 3: Verify**

Run: `grep -n "Do unit tests pass?" docs/figment-workback-plan.md`
Expected: ≥1 match now (Task 9 later adds a second occurrence in the §16 Testing prose — both are intended).

- [ ] **Step 4: Commit**

```bash
git add docs/figment-workback-plan.md
git commit -m "docs(plan): add unit-tests line to §13 daily checklist (E1)"
```

---

### Task 8: W2 — Demo-video storyboard (§14) + June 14 task tweak

**Files:**
- Modify: `docs/figment-workback-plan.md` (end of `# 14. The three canonical demo cases`, before its closing `---`; and the June 14 "Record demo" task)

- [ ] **Step 1: Verify the anchor exists**

Run: `grep -n "shows safety-first design" docs/figment-workback-plan.md`
Expected: one match (last bullet of Case 3 in §14).

- [ ] **Step 2: Apply the §14 edit**

`old_string`:
```
* shows safety-first design

---
```

`new_string`:
```
* shows safety-first design

## Demo video storyboard (2–3 min)

A timestamped beat sheet for the submission video. It must show the **hosted Space** (not only the local Mac):

```text
0:00  Cold open — "What happens when the clinic loses internet?" Cut the network.
0:15  Flip the offline indicator; show the live Space link still responding.
0:30  Case 1 (pediatric dehydration): intake → red-flag fires → missing-info asked → SBAR generated.
1:30  Open the Trace tab (§3, the 5th tab): show the deterministic pipeline end to end.
2:00  One line + the before/after table (base vs Figment LoRA) from §8.
2:30  Close on the named real user / honest-fit statement.
```

Keep it under 3:00. Record a rough cut before June 14 so a failed take never threatens submission.

---
```

- [ ] **Step 3: Verify the June 14 task anchor**

Run: `grep -n "Record 2 to 3 minute demo." docs/figment-workback-plan.md`
Expected: one match (June 14 tasks).

- [ ] **Step 4: Apply the June 14 task edit**

`old_string`:
```
* Record 2 to 3 minute demo.
```

`new_string`:
```
* Record 2 to 3 minute demo following the §14 storyboard (must show the hosted Space).
```

- [ ] **Step 5: Verify acceptance criteria** (ordered timestamped beats ≤3:00; refs Case 1 + Trace tab; requires hosted Space)

Run: `grep -n "Demo video storyboard" docs/figment-workback-plan.md && grep -n "must show the hosted Space" docs/figment-workback-plan.md`
Expected: one match each. Visually confirm 6 timestamped lines, last ≤ 2:30/3:00.

- [ ] **Step 6: Commit**

```bash
git add docs/figment-workback-plan.md
git commit -m "docs(plan): add W2 demo-video storyboard to §14 + June 14 task"
```

---

### Task 9: D1 + E1 — New section "§16. Operational readiness" (risk register + testing & CI)

**Files:**
- Modify: `docs/figment-workback-plan.md` (insert a new top-level section after §15, before the `[1]:` link-reference block)

- [ ] **Step 1: Verify the anchor exists**

Run: `grep -n "knows when to shut up" docs/figment-workback-plan.md`
Expected: one match — the last prose line of §15, immediately before the `[1]:` references.

- [ ] **Step 2: Apply the edit**

`old_string`:
```
More like: **a field protocol binder that can talk, cite itself, and knows when to shut up.**

[1]: https://huggingface.co/build-small-hackathon?utm_source=chatgpt.com "Build Small Hackathon"
```

`new_string`:
```
More like: **a field protocol binder that can talk, cite itself, and knows when to shut up.**

---

# 16. Operational readiness

## Risk register

| Risk | Trigger (how you know) | Fallback | Owner-day |
| ---- | ---------------------- | -------- | --------- |
| Modal 30B LoRA job fails or OOMs | Job errors, or needs > 80 GB VRAM | Ship the pilot/4B adapter or base GGUF (minimum tier, §12); reallocate the day to hardening | June 10 |
| Model too slow for a live demo | First-token or tok/s below the §2 performance budget | Step down the §2 degradation ladder: 16k→8k ctx → smaller quant → canned-response mode | June 11 |
| Synthetic critique keep-rate too low | < ~40% kept after critique + validate | Lower the target to 2,000 examples; invest more in cards/rules; accept a smaller train set | June 8 |
| HF Space won't cold-boot | Space build/log errors on a clean start | Ship the smaller-quant or canned-response Space (still a valid mandatory artifact); fix requirements/Dockerfile | June 12 |
| Fine-tune regresses safety | Any June 10 kill criterion trips (recall down, unsafe diagnosis up, JSON breaks, stops citing, over-refusal) | Roll back to base or the prior checkpoint; publish base as the demo model | June 10 |
| No real user available by June 13 | No responder confirmed | Use a medically literate proxy on simulated cases and say so honestly (honest-fit); keep recruiting | June 13 |

## Testing & CI

Unit-test the safety-critical deterministic components (they must be boringly correct):

* `rules.py` — each red-flag condition fires on a gold positive input and stays silent on a gold negative.
* `validators.py` — rejects invalid JSON, empty `source_cards`, citations to non-existent cards, forbidden phrases, and risk-level/red-flag inconsistency.
* `schemas.py` — enum fields (`risk_level`) reject out-of-vocabulary values.

Use small gold fixtures under `tests/`; run with `pytest`. "Do unit tests pass?" is part of the §13 daily checklist. Owner-days: June 6 (scaffold tests with the skeleton), June 7 (rules/validators tests once those modules exist).

[1]: https://huggingface.co/build-small-hackathon?utm_source=chatgpt.com "Build Small Hackathon"
```

- [ ] **Step 3: Verify acceptance criteria** (D1: 4-col table ≥6 rows w/ owner-days; E1: 3 components, each with ≥1 assertion)

Run: `grep -n "# 16. Operational readiness" docs/figment-workback-plan.md && grep -n "## Risk register" docs/figment-workback-plan.md && grep -n "## Testing & CI" docs/figment-workback-plan.md`
Expected: one match each. Visually confirm the risk table has 6 data rows, each ending in an owner-day, and the testing list names `rules.py`, `validators.py`, `schemas.py`.

- [ ] **Step 4: Commit**

```bash
git add docs/figment-workback-plan.md
git commit -m "docs(plan): add §16 operational readiness — risk register (D1) + testing (E1)"
```

---

## Chunk 3: Whole-document verification

### Task 10: Structural + acceptance re-check, then final commit

**Files:**
- Read-only verification of `docs/figment-workback-plan.md`

- [ ] **Step 1: Code fences are balanced**

Run: `test $(( $(grep -c '```' docs/figment-workback-plan.md) % 2 )) -eq 0 && echo "FENCES_OK" || echo "FENCES_BROKEN"`
Expected: `FENCES_OK`.

- [ ] **Step 2: Exactly one new top-level section (§16), numbering intact through §16**

Run: `grep -nE "^# [0-9]+\. " docs/figment-workback-plan.md`
Expected: top-level sections numbered 1–16 in order, with `# 16. Operational readiness` last. No duplicate numbers.

- [ ] **Step 3: All 9 additions are present**

Run:
```bash
for s in \
  "## Safety statement (what \`safety_statement.md\` must contain)" \
  "## Performance budget" \
  "## Non-goals — what Figment will not do" \
  "## Licensing & data handling" \
  "## How each metric is measured" \
  "## Constrained prompt skeleton" \
  "Do unit tests pass?" \
  "## Demo video storyboard (2–3 min)" \
  "# 16. Operational readiness" ; do
  grep -q "$s" docs/figment-workback-plan.md && echo "OK: $s" || echo "MISSING: $s"
done
```
Expected: nine `OK:` lines, zero `MISSING:`.

- [ ] **Step 4: No broken intra-doc references introduced**

Run: `grep -nE "§3\.5|§1\.2" docs/figment-workback-plan.md`
Expected: no matches (we use "§3, the 5th tab" and "§1 Clinical restraint subsection" instead of fake subsection numbers).

- [ ] **Step 5: Markdown tables are well-formed** (spot-check the two new tables)

Run: `grep -nE "^\| .* \| .* \| .* \|" docs/figment-workback-plan.md | grep -E "Risk|Metric"`
Expected: the risk-register header (`Risk | Trigger ... | Fallback | Owner-day`) and the eval-methodology header (`Metric | Method | How`) both appear.

- [ ] **Step 6: Adversarial verification (if subagents available)**

Dispatch a read-only reviewer (or run `superpowers:verification-before-completion` if available) with this instruction:
> Re-read `docs/figment-workback-plan.md` against `docs/superpowers/specs/2026-06-05-figment-plan-additions-design.md`. For each of the 9 additions (W1, W2, D1, D2, S1, S2, E1, E2, E3), confirm it is present, meets its acceptance criteria, and contradicts nothing in the existing sections. Report any miss or new inconsistency with exact line numbers.

Expected: all 9 confirmed; no contradictions.

- [ ] **Step 7: Final no-op commit / summary** (only if Steps 1–6 produced any fix-ups)

```bash
git add docs/figment-workback-plan.md
git commit -m "docs(plan): verify figment plan additions (structural + acceptance checks)" || echo "nothing to commit — all checks passed clean"
```

---

## Execution notes

- **Idempotent anchors:** every `old_string` is unique in the current file. If an anchor returns 0 or >1 matches at Step 1 of any task, stop and re-read the surrounding lines before editing — do not guess.
- **Curly quotes matter:** Task 3's anchor uses `“ ” ’` (the file uses typographic quotes). Match them exactly.
- **Order matters only for cleanliness, not correctness:** tasks touch disjoint regions, but applying them top-of-file → bottom keeps diffs readable.
- **Commits:** one per task; the messages above are the canonical set.
- **No code is executed by this plan.** The unit tests described in E1/§16 are authored later during the actual Figment build (June 6–7), not here.
