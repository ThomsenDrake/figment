# Figment Submission Checklist

Status: living checklist for evidence. Keep claims in README, submission copy, demo video, and social posts aligned with this file.

Primary tracker: [adversarial review action items](adversarial-review-action-items.md).

## Required Submission Artifacts

| Artifact | Status | Link / evidence |
| -------- | ------ | --------------- |
| Public Hugging Face Space | Runnable in no-secret canned-fallback mode | https://huggingface.co/spaces/build-small-hackathon/figment |
| Space cold boot with app files present | Verified 2026-06-07 | Space API `runtime.stage=RUNNING`, `sdk=gradio`, `sha=5dcfc5c830de7331eca9020b17e1c571a8619654`, 92 siblings, `app.py` present; Space URL served HTTP 200 |
| Demo video | Proof needed | Pending |
| Social post | Proof needed | Pending |
| Safety statement | Present | [safety_statement.md](safety_statement.md) |
| User test notes | Template present; results needed | [user_test_notes.md](user_test_notes.md) |
| License | Present | [../LICENSE](../LICENSE) |
| Live hosted Omni trace | Eval traces present; final demo trace still needed | Baseline: `traces/hosted_omni_eval_20260607T194833Z.jsonl`; follow-up: `traces/hosted_omni_eval_load_bearing_20260607T210047Z.jsonl` |
| No-cloud/off-grid trace | Proof needed before claiming Off the Grid achieved | Pending |
| Hosted Omni eval results | Measured | [hosted_omni_eval_results.md](hosted_omni_eval_results.md): 31/50 whole-output competence, 8/50 full fallback, 480/650 model-retained fields, 170/650 deterministic patches, 50/50 final validation in the follow-up run |
| Local 4B + Parakeet eval results | Artifacts and evidence helpers ready; proof needed | Full BF16 4B snapshot and Parakeet `.nemo` artifact are present locally; `scripts/run_local_4b_evidence.py` captures endpoint metadata, route smoke, eval records, eval summary, and `eval_evidence_manifest.json` once the local endpoint is live; `scripts/run_local_asr_evidence.py` captures Parakeet ASR provider evidence and `asr_evidence_manifest.json`. Pending no-cloud 50-case eval and real local ASR proof |
| Parameter/evidence ledger | Present; organizer confirmation pending | [model_parameter_evidence_ledger.md](model_parameter_evidence_ledger.md) |
| Submission claim audit | Present | `make audit-claims` scans submission-facing copy for premature Off the Grid, Llama Champion, Well-Tuned, Backyard user-use, local 4B, local ASR, demo-video, and social-post claims |
| Evidence gate status report | Present; incomplete by design until external proofs exist | `make evidence-gates` reports each evidence gate, paths found, and next actions. Current missing gates include local 4B 50-case eval, no-cloud route, Llama Champion route, local ASR provider proof, trained-responder user test, demo video, social post, and Well-Tuned adapter |

## Badge And Claim Status

| Claim / badge area | Submission wording allowed now | Evidence needed to upgrade |
| ------------------ | ------------------------------ | -------------------------- |
| Backyard AI | Targeted; built for a real trained responder, with identity withheld for privacy | Completed user-test notes from that responder on synthetic or de-identified scenarios. Do not claim use, testing, validation, approval, or endorsement before notes exist |
| Off the Grid | Targeted / proof-needed | Recorded no-cloud run using self-hosted Omni on adequate local hardware or a smaller verified local stack. Hosted API evidence does not count |
| Hosted Gradio Space | Runnable in no-secret canned-fallback mode; live hosted-model demo trace still proof-needed | Public Space app files present, cold boot, typed intake run, and route/fallback trace verified. Public workflow trace: `raw_route=canned`, `final_route=canned_backend`, `fallback_tier=canned`, `validation_status=passed`, `raw_audio_stored=false`, `model_retained_count=0`, `deterministic_patch_count=13` |
| Demo video | Targeted / proof-needed | Final video link showing only verified routes and labeling fallbacks honestly |
| Social post | Targeted / proof-needed | Final social post link with achieved-versus-targeted wording |
| Llama Champion | Targeted / proof-needed | Eligible local model route through llama.cpp with trace or eval evidence |
| Sharing is Caring | Targeted / proof-needed | Public Space, repo, demo video, social post, and open trace links |
| Well-Tuned | Stretch only / proof-needed | Published fine-tuned model or adapter used by the app, plus measured result. Fallback output cannot count |
| Field Notes | Tentative | Organizer confirmation and final field-note artifact |
| Off-Brand | Targeted / proof-needed | Final demo or social artifact meeting organizer criteria |

## Off-Grid Evidence Boundary

Omni is not architecturally disqualified from Off the Grid: it can support the claim if self-hosted on adequate local hardware with no runtime cloud APIs. The current repo does not yet include that recorded proof. Until a no-cloud run exists, use targeted or proof-needed language.

## User-Test Evidence Boundary

The README may say the project is built for a real trained responder. It should not say that responder used, approved, validated, or endorsed Figment until [user_test_notes.md](user_test_notes.md) contains factual session notes.

## Eval Evidence Boundary

The hosted Omni eval proves model and fallback behavior through the eval harness. The public Space proof currently proves deployment health and no-secret fallback behavior, not live hosted Omni generation. Use the follow-up hosted eval run as the current hosted eval score: 31/50 whole-output hosted competence, 8/50 full fallback, 480/650 model-retained fields, 170/650 deterministic patches, and 50/50 final validation.

Final validation is app safety. Whole-output model competence and field-level model retention are the model-load-bearing metrics. Deterministic fallback and deterministic patches must stay visible in traces, scorecards, submission copy, and the demo.

## Submission Copy Boundaries

- Space: may say the public Space is runnable in no-secret canned-fallback mode, with app files present, cold boot verified, typed intake working, and trace labeling verified. Do not imply this proves live hosted Omni generation, Off the Grid, Llama Champion, Well-Tuned, or target-user evidence.
- Demo video and social post: use pending placeholders until final links exist.
- Backyard AI: may say built for a real trained responder; do not say the target user used or tested Figment until factual notes exist.
- Off the Grid: claim only after a recorded no-cloud run.
- Llama Champion: claim only after an eligible llama.cpp route runs with trace or eval evidence.
- Well-Tuned: claim only after a published fine-tuned model or adapter is used by the app and measured.
- Parameter compliance: cite the [model parameter/evidence ledger](model_parameter_evidence_ledger.md), including the Omni 31B body-count versus 33B sidebar ambiguity and organizer-confirmation status.

Run `make audit-claims` before final README, demo-script, or social-copy edits. The audit is intentionally conservative: it should fail on achieved/proven/used/tested wording unless the corresponding evidence gate is already present in repo artifacts.

Run `make evidence-gates` before claiming a badge or submission artifact is complete. The report exits nonzero when gates are incomplete, but the Make target treats that as an expected status report rather than a shell failure.
