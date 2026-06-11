# Figment Local 4B V5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the v4 local 4B field-workflow result from `109/150` strict competence to `>=125/150` by fixing fired-rule citation invariants and training focused observation/SBAR ownership.

**Architecture:** Fix deterministic scaffolding first so fired rule cards cannot disappear from `source_cards`, then make the model explicitly own required-observation selection via structured `selected_required_observation_ids`. Train v5 as a focused continuation from v4 rather than a broad navigator refresh.

**Tech Stack:** Python, pytest, JSONL eval harness, Modal LoRA SFT, `llama.cpp` GGUF serving, Nemotron teacher-generated synthetic rows.

## Implementation Status

Updated 2026-06-11:

- Tasks 1-3 are implemented in the harness: fired-rule source cards are retained, selected required-observation IDs are traceable, visible observation text is required, and focused citation repairs preserve mandatory source cards.
- Task 4 is implemented: `figment_sft_v5` has focused corpus categories, v5 metadata, v5 row policy checks, a full-corpus wrapper, repair-scope scheduling, and harness-verifier rejection tests.
- Verification passed:
  - `PYTHONPATH=. .venv/bin/pytest tests/test_finetune_v5_data_plan.py tests/test_finetune_v2_data_plan.py tests/test_focused_repair.py tests/test_prompt_builder_contract.py tests/test_validators_strict.py tests/test_navigator_safety.py tests/test_eval_runner.py -q`
  - `PYTHONPATH=. .venv/bin/pytest tests/test_modal_finetune_prep.py tests/test_finetune_v5_data_plan.py -q`
  - `PYTHONPATH=. .venv/bin/pytest tests -q`
  - `PYTHONPATH=. .venv/bin/python scripts/generate_v5_full_corpus.py --navigator-count 2 --repair-count 0 --rows-per-shard 1 --parallelism 1 --base-start-index 61000 --shard-prefix /tmp/figment_v5_full_smoke_1781174291/shard --output /tmp/figment_v5_full_smoke_1781174291/figment_sft_v5.jsonl --case-specs /tmp/figment_v5_full_smoke_1781174291/figment_sft_v5_case_specs.jsonl --manifest /tmp/figment_v5_full_smoke_1781174291/figment_sft_v5_manifest.json --modal-output-dir /tmp/figment_v5_full_smoke_1781174291/modal --dry-run`
- Real-teacher smoke passed on OpenRouter:
  - `nvidia/nemotron-3-ultra-550b-a55b:free` generated 2/2 `sbar_observation_ownership` rows with `0` verifier issues.
  - The same teacher generated 1/1 accepted row for each v5 focus: `sbar_observation_ownership`, `required_observation_id_selection`, `source_card_invariant`, `noisy_field_audio_style`, and `general_regression`; combined verifier result was `5` rows, `0` issues.
  - Primary NVIDIA endpoint smoke with `nvidia/nemotron-3-ultra-550b-a55b` returned `429 Too Many Requests`, so `scripts/generate_v5_full_corpus.py` defaults to the working OpenRouter `:free` teacher id for now.
- Full-corpus generation checkpoint:
  - Started `PYTHONPATH=. .venv/bin/python scripts/generate_v5_full_corpus.py --parallelism 2 --teacher-error-retries 3 --teacher-error-sleep-seconds 10 --log-rejections`.
  - Paused after four complete shard manifests to avoid holding the local session for the entire OpenRouter run.
  - Completed shards: `data/finetune/shards/figment_sft_v5_full_shard0_manifest.json` through `figment_sft_v5_full_shard3_manifest.json`.
  - Verified partial merge: `/tmp/figment_sft_v5_partial_200.jsonl` plus `/tmp/figment_sft_v5_partial_200_case_specs.jsonl` contained `200` rows, all five v5 focus categories, and `0` verifier issues.
  - Completed-shard acceptance: `200` accepted rows over `212` attempts. Rejections were malformed teacher notes plus one v5-policy reward skip; accepted rows passed harness verification.
  - Resumable partial shards also exist: shard4 has `8` rows, shard5 has `1` row. Re-running the same wrapper command will use `--resume` for incomplete shards.
- Full-corpus generation checkpoint 2:
  - Resumed with `PYTHONPATH=. .venv/bin/python scripts/generate_v5_full_corpus.py --parallelism 4 --teacher-error-retries 3 --teacher-error-sleep-seconds 10 --log-rejections`.
  - Completed shards: `data/finetune/shards/figment_sft_v5_full_shard0_manifest.json` through `figment_sft_v5_full_shard7_manifest.json`.
  - Verified partial merge: `/tmp/figment_sft_v5_partial_400.jsonl` plus `/tmp/figment_sft_v5_partial_400_case_specs.jsonl` contained `400` rows, all five v5 focus categories, and `0` verifier issues.
  - Completed-shard acceptance: `400` accepted rows over `420` attempts. Rejections were transient teacher/backend-note issues plus one v5-policy reward skip; accepted rows passed harness verification.
  - Resumable partial shards also exist: shard8 has `11` rows, shard9 has `7` rows, shard10 has `4` rows, and shard11 has `1` row. Re-running the same wrapper command will use `--resume` for incomplete shards.
- Full-corpus generation checkpoint 3:
  - Resumed again with `PYTHONPATH=. .venv/bin/python scripts/generate_v5_full_corpus.py --parallelism 4 --teacher-error-retries 3 --teacher-error-sleep-seconds 10 --log-rejections`.
  - Completed shards: `data/finetune/shards/figment_sft_v5_full_shard0_manifest.json` through `figment_sft_v5_full_shard11_manifest.json`.
  - Verified partial merge: `/tmp/figment_sft_v5_partial_600.jsonl` plus `/tmp/figment_sft_v5_partial_600_case_specs.jsonl` contained `600` rows, all five v5 focus categories, and `0` verifier issues.
  - Completed-shard acceptance: `600` accepted rows over `628` attempts. Rejections were transient teacher/backend-note issues plus one v5-policy reward skip; accepted rows passed harness verification.
  - Resumable partial shards also exist: shard12 has `16` rows, shard13 has `5` rows, shard14 has `6` rows, and shard15 has `0` rows. Re-running the same wrapper command will use `--resume` for incomplete shards.
- Full-corpus generation checkpoint 4:
  - Resumed with `PYTHONPATH=. .venv/bin/python scripts/generate_v5_full_corpus.py --parallelism 4 --teacher-error-retries 3 --teacher-error-sleep-seconds 10 --log-rejections`.
  - Completed shards: `data/finetune/shards/figment_sft_v5_full_shard0_manifest.json` through `figment_sft_v5_full_shard15_manifest.json`.
  - Verified partial merge: `/tmp/figment_sft_v5_partial_800.jsonl` plus `/tmp/figment_sft_v5_partial_800_case_specs.jsonl` contained `800` rows, all five v5 focus categories, and `0` verifier issues.
  - Completed-shard acceptance: `800` accepted rows over `835` attempts. Rejections were transient teacher/backend-note issues plus one v5-policy reward skip; accepted rows passed harness verification.
  - Resumable partial shards also exist: shard16 has `15` rows, shard17 has `1` row, shard18 has `1` row, and shard19 has `0` rows. Re-running the same wrapper command will use `--resume` for incomplete shards.
- Full-corpus generation final checkpoint:
  - Resumed with `PYTHONPATH=. .venv/bin/python scripts/generate_v5_full_corpus.py --parallelism 4 --teacher-error-retries 3 --teacher-error-sleep-seconds 10 --log-rejections`.
  - Completed shards: `data/finetune/shards/figment_sft_v5_full_shard0_manifest.json` through `figment_sft_v5_full_shard21_manifest.json`.
  - Final dataset: `data/finetune/figment_sft_v5.jsonl` contains `1300` rows: `1100` navigator rows plus `200` focused-repair rows.
  - Final case specs: `data/finetune/figment_sft_v5_case_specs.jsonl` contains `1100` navigator case specs.
  - Explicit verifier passed: `PYTHONPATH=. .venv/bin/python scripts/verify_finetune_harness_alignment.py --dataset data/finetune/figment_sft_v5.jsonl --case-specs data/finetune/figment_sft_v5_case_specs.jsonl` reported `rows=1300`, `case_specs=1100`, and `issue_count=0`.
  - Modal split artifacts are staged in `data/finetune/modal/figment_sft_v5/`: `train.jsonl` has `1170` rows, `validation.jsonl` has `130` rows, and `manifest.json` records SHA256 hashes for both splits.
- Modal training checkpoint:
  - Fresh smoke passed on `figment_sft_v5` with finite loss and adapter artifacts at `/checkpoints/figment_sft_v5/figment-sft-v5-lora-smoke-fast-smoke`.
  - Resume-from-v4 smoke passed from `/checkpoints/figment_sft_v4/figment-sft-v4-lora` with finite loss and eval loss `1.0086122751235962`.
  - Full detached H100 continuation is running:
    - App ID: `ap-AtxsW6TXtHhD1hIrt6da8s`
    - Function call ID: `fc-01KTVEZF0JGJSFY7DVFMKVRFP1`
    - Dashboard: `https://modal.com/id/fc-01KTVEZF0JGJSFY7DVFMKVRFP1`
    - Output name: `figment-sft-v5-lora`
    - Expected artifact path: `/checkpoints/figment_sft_v5/figment-sft-v5-lora`
    - Runtime GPU proof from logs: `NVIDIA H100 80GB HBM3`.
- Next step: monitor `figment-sft-v5-lora`, verify adapter artifacts, then merge/convert/evaluate.

---

## Evidence From V4

Source run:

- `traces/local_4b_finetuned_v4_field_holdout_20260611T011930Z/eval_summary.json`
- `traces/local_4b_finetuned_v4_field_holdout_20260611T011930Z/local_4b_eval.jsonl`

Observed v4 scores:

- `109/150` raw model competence successes.
- `148/150` final validation successes.
- `149/150` expected-label successes.
- `2` canned fallback uses.
- `150/150` trace hashes present.
- `1846/1950` visible fields retained, or `94.67%`.

Failure shape:

- `39/41` strict competence misses were soft misses: the final output passed validation and expected labels, but deterministic scaffolding filled `missing_info_to_collect` and `next_observations_to_collect`.
- `2/41` were hard failures:
  - `field_workflow_holdout_v1-000054`: `STROKE-SIGNS-v1` fired but was not cited in `source_cards`.
  - `field_workflow_holdout_v1-000099`: `PREG-DANGER-SIGNS-v1` fired but was not cited in `source_cards`.
- `REFERRAL-SBAR-v1` was `0/27` strict competence but `26/27` final validation and `27/27` expected-label success, so the main weakness is not broad routing. It is model ownership of handoff-linked observation fields.

## V5 Acceptance Gates

- `>=125/150` strict competence on `data/eval/field_workflow_holdout_v1.jsonl`.
- `REFERRAL-SBAR-v1 >=20/27` strict competence.
- `0` final validation failures caused by missing fired-rule cards in `source_cards`.
- `missing_info_to_collect` model-owned in `>=140/150`.
- `next_observations_to_collect` model-owned in `>=140/150`.
- `0` or `1` fallback uses.
- No regression on `protocol_urgency`, red-flag matching, forbidden behavior, or SBAR handoff metrics.

## File Map

- Modify `figment/navigator.py`: enforce fired-rule source-card invariants in fallback/scaffold output.
- Modify `figment/prompt_builder.py`: expose required observation IDs and require `selected_required_observation_ids`.
- Modify `figment/observation_targets.py`: make required-observation display text stable and easy to inject into prompts/training rows.
- Modify `scripts/run_eval.py`: preserve and score model-owned selected observation IDs separately from deterministic fills.
- Modify `scripts/generate_finetune_data.py`: generate v5 rows that train observation-ID selection and fired-card citation invariants.
- Modify `scripts/verify_finetune_harness_alignment.py`: reject rows that omit fired-rule cards or observation IDs.
- Create `scripts/generate_v5_full_corpus.py`: v5 corpus wrapper with focused counts and metadata.
- Create tests in `tests/test_navigator_safety.py`, `tests/test_prompt_builder_contract.py`, `tests/test_eval_runner.py`, and `tests/test_finetune_v5_data_plan.py`.
- Create `docs/local_4b_v5_training_plan.md`: this plan.

## Task 1: Fix Fired-Rule Source-Card Invariants

**Files:**

- Modify: `figment/navigator.py`
- Modify: `scripts/run_eval.py`
- Test: `tests/test_navigator_safety.py`
- Test: `tests/test_eval_runner.py`

- [ ] **Step 1: Add a failing test for fired cards in fallback `source_cards`**

Add a regression test that mirrors the v4 hard failure:

```python
def test_fallback_source_cards_include_every_fired_rule_card(monkeypatch):
    # Use a case where STROKE-SIGNS-v1 fires even if retrieval did not rank it.
    # The final output must cite STROKE-SIGNS-v1 because deterministic rules used it.
    output, trace = run_navigation(
        intake={
            "chief_concern": "one-sided weakness",
            "symptoms": ["sudden one-sided weakness", "trouble speaking"],
            "age": 56,
            "pregnancy_status": "not_pregnant",
        },
        model_client=_model_that_omits_stroke_source_card(),
    )
    assert "STROKE-SIGNS-v1" in output["source_cards"]
    assert "STROKE-SIGNS-v1" in trace.to_dict()["harness_evidence"]["deterministic_rule_card_ids"]
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
.venv/bin/pytest tests/test_navigator_safety.py::test_fallback_source_cards_include_every_fired_rule_card -q
```

Expected before implementation: failure showing `STROKE-SIGNS-v1` is missing from `source_cards`.

- [ ] **Step 3: Add a single invariant helper**

Implement one helper near the existing fallback/scaffold helpers:

```python
def _mandatory_source_card_ids(rule_results: list[dict[str, Any]], candidate_card_ids: Iterable[str]) -> list[str]:
    required: list[str] = []
    for rule in rule_results:
        card_id = str(rule.get("card_id") or "").strip()
        if card_id and card_id not in required:
            required.append(card_id)
    for card_id in candidate_card_ids:
        card_id = str(card_id or "").strip()
        if card_id and card_id not in required:
            required.append(card_id)
    return required
```

Use it wherever fallback/scaffold source cards are built.

- [ ] **Step 4: Filter but do not drop fired known cards**

When the helper output is merged with retrieved cards:

```python
source_cards = []
for card_id in mandatory_source_card_ids + retrieved_ids:
    if card_id in known_cards and card_id not in source_cards:
        source_cards.append(card_id)
```

This ensures invalid card IDs are still rejected, but known fired cards are retained even if retrieval ranked them poorly.

- [ ] **Step 5: Add an eval-runner regression**

Add a test using a minimal record where `PREG-DANGER-SIGNS-v1` fires and the model omits it from `source_cards`. Assert:

```python
assert record["final_validation"]["passed"] is True
assert "PREG-DANGER-SIGNS-v1" in record["final_output"]["source_cards"]
assert record["competence_success"] is False
```

The final assertion keeps load-bearing honesty: deterministic repair can make the output safe without pretending the model authored it.

- [ ] **Step 6: Run tests**

Run:

```bash
.venv/bin/pytest tests/test_navigator_safety.py tests/test_eval_runner.py -q
```

Expected: all tests pass.

## Task 2: Make Observation Selection Structured

**Files:**

- Modify: `figment/prompt_builder.py`
- Modify: `figment/observation_targets.py`
- Modify: `scripts/run_eval.py`
- Test: `tests/test_prompt_builder_contract.py`
- Test: `tests/test_eval_runner.py`

- [ ] **Step 1: Add a failing prompt contract test**

Assert the prompt includes stable observation IDs:

```python
def test_prompt_includes_required_observation_id_table():
    prompt = build_navigation_prompt(...)
    assert "selected_required_observation_ids" in prompt.user_message
    assert "CHEST-PAIN-ESCALATION-v1::required_observation::1" in prompt.user_message
    assert "Choose required observation IDs before writing observation text" in prompt.user_message
```

- [ ] **Step 2: Add selected IDs to the model JSON schema**

In the prompt skeleton, include:

```json
"selected_required_observation_ids": []
```

Instruction text should say:

```text
Select the required observation IDs that matter for the cited source cards. Then write short responder-facing phrases for missing_info_to_collect and next_observations_to_collect using those IDs.
```

- [ ] **Step 3: Validate selected IDs before stripping**

In `scripts/run_eval.py`, keep the current trace-only behavior but make the scoring path explicit:

```python
model_selected_required_observation_ids = _string_list(raw_output.get("selected_required_observation_ids"))
invalid_selected_required_observation_ids = [
    observation_id
    for observation_id in model_selected_required_observation_ids
    if observation_id not in allowed_required_observation_ids
]
```

Strip the field from final user-facing output after validation/tracing.

- [ ] **Step 4: Score model-owned observation coverage**

Treat observation fields as model-owned when:

- The selected IDs are valid.
- Every cited non-exempt clinical card has at least one selected required-observation ID.
- The natural-language arrays contain the display text or accepted synonym for those selected IDs.

Keep deterministic fills as `deterministic_fallback`.

- [ ] **Step 5: Run tests**

Run:

```bash
.venv/bin/pytest tests/test_prompt_builder_contract.py tests/test_eval_runner.py -q
```

Expected: prompt contract and trace/scoring behavior pass.

## Task 3: Add Focused Source-Card Repair

**Files:**

- Modify: `scripts/run_eval.py`
- Modify: existing focused repair helper code used by `scripts/run_eval.py`
- Test: `tests/test_focused_repair.py`

- [ ] **Step 1: Add a failing repair-scope test**

Use failures like:

```python
failures = [
    "fired rule card STROKE-SIGNS-v1 is not cited in source_cards",
    "candidate pathway STROKE-SIGNS-v1 is not cited in source_cards",
]
```

Assert the selected scope is `citations_and_pathways` and the fields are:

```python
("source_cards", "candidate_protocol_pathways")
```

- [ ] **Step 2: Require focused repair to preserve mandatory source cards**

The repair prompt must include:

```text
Mandatory source cards: STROKE-SIGNS-v1, SAFETY-BOUNDARIES-v1, REFERRAL-SBAR-v1
Return exactly these top-level keys: source_cards, candidate_protocol_pathways.
Do not remove any mandatory source card.
```

- [ ] **Step 3: Reject repair outputs missing fired cards**

After repair:

```python
for card_id in mandatory_source_card_ids:
    if card_id not in _string_list(repair_output.get("source_cards")):
        repair_validation = {"passed": False, "failures": [f"repair omitted mandatory source card {card_id}"]}
```

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/pytest tests/test_focused_repair.py -q
```

Expected: all focused repair tests pass.

## Task 4: Generate V5 Focused Corpus

**Files:**

- Create: `scripts/generate_v5_full_corpus.py`
- Modify: `scripts/generate_finetune_data.py`
- Modify: `scripts/verify_finetune_harness_alignment.py`
- Test: `tests/test_finetune_v5_data_plan.py`

- [ ] **Step 1: Add corpus wrapper defaults**

Create `scripts/generate_v5_full_corpus.py` with these defaults:

```python
DEFAULT_OUTPUT_VERSION = "figment_sft_v5"
DEFAULT_COUNTS = {
    "sbar_observation_ownership": 350,
    "required_observation_id_selection": 250,
    "source_card_invariant": 150,
    "noisy_field_audio_style": 100,
    "general_regression": 250,
}
```

- [ ] **Step 2: Add v5 metadata to every row**

Each row must include metadata like:

```json
{
  "dataset_version": "figment_sft_v5",
  "training_focus": "sbar_observation_ownership",
  "excluded_eval_case_ids": ["field_workflow_holdout_v1-000054", "field_workflow_holdout_v1-000099"],
  "must_include_source_cards": ["REFERRAL-SBAR-v1", "SAFETY-BOUNDARIES-v1"],
  "must_include_selected_required_observation_ids": ["..."]
}
```

Use v4 failures as pattern templates only. Do not train on copied holdout case text.

- [ ] **Step 3: Add validator checks**

`verify_finetune_harness_alignment.py` must reject any v5 row when:

- A fired rule card is absent from `source_cards`.
- `selected_required_observation_ids` is empty for cited clinical cards with required observations.
- `REFERRAL-SBAR-v1` rows lack a complete SBAR object.
- Observation arrays are generic phrases like `repeat vitals`, `monitor closely`, or `ask anything else`.

- [ ] **Step 4: Run corpus tests**

Run:

```bash
.venv/bin/pytest tests/test_finetune_v5_data_plan.py -q
```

Expected: v5 row metadata and validator rejection tests pass.

## Task 5: Train V5 As A Focused Continuation

**Files:**

- Modify: `modal/finetune_figment_nemotron.py`
- Modify: `docs/local_4b_v5_training_plan.md`

- [ ] **Step 1: Verify v5 dataset locally**

Run:

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_v5_full_corpus.py
PYTHONPATH=. .venv/bin/python scripts/verify_finetune_harness_alignment.py \
  --dataset data/finetune/figment_sft_v5.jsonl \
  --case-specs data/finetune/figment_sft_v5_case_specs.jsonl
```

Expected:

- At least `1100` accepted rows.
- `0` copied holdout rows.
- `0` source-card invariant violations.
- `0` empty selected-observation-ID violations.

Current checkpoint:

- Full `1100/1100` navigator rows are complete in shards `0-21`.
- Final corpus is verified and staged for Modal:

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_finetune_harness_alignment.py \
  --dataset data/finetune/figment_sft_v5.jsonl \
  --case-specs data/finetune/figment_sft_v5_case_specs.jsonl
```

- [x] **Step 2: Smoke train on Modal**

Run the existing Modal trainer with:

```bash
.venv/bin/modal run modal/finetune_figment_nemotron.py \
  --dataset-version figment_sft_v5 \
  --output-name figment-sft-v5-lora-smoke \
  --max-steps 20
```

Expected:

- Training starts.
- Loss is finite.
- Adapter artifacts are written.

Actual:

- Fresh smoke and resume-from-v4 smoke both completed with finite losses.
- Resume smoke wrote `/checkpoints/figment_sft_v5/figment-sft-v5-lora-resume-smoke`.

- [x] **Step 3: Launch full detached training**

Run:

```bash
.venv/bin/modal run --detach modal/finetune_figment_nemotron.py \
  --dataset-version figment_sft_v5 \
  --output-name figment-sft-v5-lora
```

Record:

- Modal app ID.
- Function call ID.
- Dashboard URL.
- Expected artifact path in `figment-checkpoints`.

Actual:

- App ID: `ap-AtxsW6TXtHhD1hIrt6da8s`
- Function call ID: `fc-01KTVEZF0JGJSFY7DVFMKVRFP1`
- Dashboard URL: `https://modal.com/id/fc-01KTVEZF0JGJSFY7DVFMKVRFP1`
- Expected artifact path: `/checkpoints/figment_sft_v5/figment-sft-v5-lora`
- Runtime GPU: `NVIDIA H100 80GB HBM3`

- [ ] **Step 4: Merge and convert**

After completion:

```bash
.venv/bin/modal run modal/finetune_figment_nemotron.py \
  --dataset-version figment_sft_v5 \
  --output-name figment-sft-v5-lora \
  --merge-only

.venv/bin/python tools/llama.cpp/convert_hf_to_gguf.py \
  artifacts/modal_checkpoints/figment-sft-v5-lora-merged-bf16 \
  --outfile artifacts/modal_checkpoints/figment-sft-v5-lora-merged-bf16.gguf \
  --outtype bf16
```

- [ ] **Step 5: Run full local eval**

Run the full 150-case holdout:

```bash
/opt/homebrew/bin/llama-server \
  -m artifacts/modal_checkpoints/figment-sft-v5-lora-merged-bf16.gguf \
  --ctx-size 16384 \
  --host 127.0.0.1 \
  --port 8001 \
  --alias nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --parallel 1 \
  --temp 0 \
  --top-p 1 \
  --reasoning off
```

Then:

```bash
PYTHON_DOTENV_DISABLED=true \
FIGMENT_MODEL_TIMEOUT_SECONDS=180 \
LOCAL_GGUF_PATH=artifacts/modal_checkpoints/figment-sft-v5-lora-merged-bf16.gguf \
.venv/bin/python scripts/run_local_4b_evidence.py \
  --base-url http://127.0.0.1:8001/v1 \
  --timeout-seconds 180 \
  --force-eval \
  --cases data/eval/field_workflow_holdout_v1.jsonl \
  --output-dir traces/local_4b_finetuned_v5_field_holdout_$(date -u +%Y%m%dT%H%M%SZ)
```

Expected: v5 meets the acceptance gates above.

## Task 6: Decide Whether V5 Is Submission-Worthy

**Files:**

- Modify: `docs/local_4b_v5_training_plan.md`
- Optional modify: `docs/figment-build-small-lessons-draft.md`

- [ ] **Step 1: Compare v4 and v5**

Create a table with:

- strict competence,
- final validation,
- expected labels,
- fallback uses,
- `REFERRAL-SBAR-v1` strict competence,
- `missing_info_to_collect` model ownership,
- `next_observations_to_collect` model ownership,
- hard source-card failures.

- [ ] **Step 2: Make the call**

If v5 clears gates, use v5 as the local fine-tuned model for submission evidence.

If v5 improves observation ownership but misses `>=125/150`, keep v4 as the stable submission model and report v5 as a targeted experiment unless there is enough time for a v5.1 focused continuation.

If v5 regresses red flags, urgency, forbidden behavior, or final validation, discard it for demo use and keep v4.

## Implementation Order

1. Task 1: fired-rule source-card invariant.
2. Task 2: structured observation selection.
3. Task 3: focused citation repair.
4. Re-run the current v4 model on the 150-case holdout to measure scaffold-only improvement.
5. Task 4: v5 corpus generation.
6. Task 5: v5 training, merge, GGUF conversion, full eval.
7. Task 6: submission decision.

## Expected Outcome

The most likely improvement path is:

- Final validation: `148/150` -> `150/150`.
- Strict competence: `109/150` -> `125-135/150`.
- `REFERRAL-SBAR-v1`: `0/27` -> `20+/27`.
- Observation fields: `109/150` model-owned -> `140+/150` model-owned.
- Fallbacks: `2` -> `0-1`.

The project should still describe the model honestly: v5 is not meant to be a general medical assistant. It is meant to be better at the bounded Figment job: faster intake completion, visible escalation cues, grounded protocol-card citation, and concise SBAR handoff support for rural clinic or disaster first-response medics.
