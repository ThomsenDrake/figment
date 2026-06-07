# Hosted Omni eval results

Date: 2026-06-07

This note records the first full Figment eval run against the hosted Omni route after expanding the eval corpus to 50 synthetic cases. The run tests model load-bearing behavior for bounded protocol navigation fields; it is not a clinical validation study.

Evidence boundary: these runs prove hosted-model behavior through the eval harness, not a public Hugging Face Space cold boot. They also do not prove the local 4B + Parakeet no-cloud path, Parakeet ASR, Llama Champion eligibility, or a fine-tuned adapter. Keep those claims gated by the submission checklist and model parameter/evidence ledger.

## Corpus

- `data/eval/initial_handwritten_cases.jsonl`: 10 cases
- `data/eval/adversarial_strict_cases.jsonl`: 3 cases
- `data/eval/comprehensive_hosted_cases.jsonl`: 37 cases
- Total: 50 synthetic cases

Coverage includes all protocol cards, routine negatives, negated red flags, noisy/ASR-like phrasing, prompt injection, multi-card escalation, no-relevant-card safety cases, pediatric dehydration, pregnancy danger signs, SBAR handoff, and forbidden clinical-action probes.

## Hosted run

- Command backend: `hosted_omni`
- Model stack: `omni_native`
- Active model ID: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
- Trace artifact: `traces/hosted_omni_eval_20260607T194833Z.jsonl`

Summary:

- Total cases: 50
- Raw hosted-model validation successes: 17
- Hosted repair successes: 11
- Hosted model competence successes: 28
- Deterministic fallback uses: 22
- Deterministic fallback successes: 22
- Final validation successes: 50
- Average latency: 7686.8 ms
- Median latency: 7370.4 ms
- Max latency: 35357.3 ms

Interpretation: the hosted Omni path is load-bearing for 28/50 cases under the strict validator. The overall application remains safe on all 50 cases because deterministic fallback catches invalid hosted output, but fallback must not be counted as model competence.

## By target card

| Target card | Cases | Competence | Fallback |
| --- | ---: | ---: | ---: |
| `AMS-RED-FLAGS-v1` | 4 | 3 | 1 |
| `CHEST-PAIN-ESCALATION-v1` | 5 | 2 | 3 |
| `FEVER-RED-FLAGS-v1` | 4 | 3 | 1 |
| `PED-DEHYD-RED-FLAGS-v1` | 4 | 4 | 0 |
| `PREG-DANGER-SIGNS-v1` | 4 | 1 | 3 |
| `REFERRAL-SBAR-v1` | 4 | 0 | 4 |
| `RESP-DISTRESS-RED-FLAGS-v1` | 4 | 3 | 1 |
| `SAFETY-BOUNDARIES-v1` | 13 | 7 | 6 |
| `STROKE-SIGNS-v1` | 4 | 2 | 2 |
| `WOUND-INFECTION-ESCALATION-v1` | 4 | 3 | 1 |

## Main failure signatures

- Missing required SBAR `handoff_request` was the most common raw failure.
- Several outputs introduced unsupported high-risk handoff facts in SBAR fields, especially around pregnancy, pressure, numeric vitals, stroke, seizure, and headache.
- Some outputs omitted required observations tied to retrieved protocol cards.
- A small number of outputs used forbidden clinical language such as diagnosis, dosing, discharge, or instruction-like phrasing.
- Negation and routine cases remain challenging when retrieved emergency cards are nearby.

## Improvement targets

- Give the model an explicit JSON skeleton with every required key, especially the full SBAR object.
- Add prompt examples for routine negation cases where retrieved emergency cards are present but red flags are absent.
- Make required observations from retrieved cards more visible in the prompt so `missing_info_to_collect` is less generic.
- Add a focused SBAR repair prompt that rewrites only handoff fields after a missing or unsupported-fact failure.
- Keep fallback visible in trace/UI and continue reporting model competence separately from final validation.
- Re-run this eval after each prompt, retrieval, or validator change and compare against `traces/hosted_omni_eval_20260607T194833Z.jsonl`.

## Load-bearing follow-up run

After adding a constrained prompt contract, field-level provenance, focused repair helpers, and field-level eval metrics, the same 50-case hosted eval was re-run.

- Trace artifact: `traces/hosted_omni_eval_load_bearing_20260607T210047Z.jsonl`
- Raw hosted-model validation successes: 25
- Hosted repair successes: 6
- Whole-output hosted competence successes: 31
- Full deterministic fallback uses: 8
- Final validation successes: 50
- Model-retained fields: 480/650
- Deterministically patched fields: 170/650
- Model field pass rate: 73.8%
- Average latency: 11095.4 ms
- Median latency: 5251.7 ms
- Max latency: 83954.3 ms

Compared with the baseline, whole-output competence improved from 28/50 to 31/50, full fallback dropped from 22/50 to 8/50, and the eval can now show partial model contribution separately from deterministic patches. The main cost is latency: focused repair can add hosted calls, especially on cases with multiple validation failure scopes.

Interpretation: use this follow-up run as the current hosted Omni model-load-bearing score. The 50/50 final validation result is an application safety result after validation, repair, and fallback. It is not pure model competence, and deterministic patches must remain visible in scorecards, traces, and submission copy.

Remaining weak areas:

- `REFERRAL-SBAR-v1` still falls back heavily: 3/4 full fallback, 25.0% field-level model retention.
- Forbidden clinical language is intentionally conservative and can force full fallback.
- SBAR grounding remains the most common field-level patch.
- A repair-scope cap or batched repair call may be needed before using this path as the default hosted demo route.
