# Figment Submission Checklist

Status: living checklist for evidence. Last live Hub verification: 2026-06-15. Keep claims in README, submission copy, demo video, and social posts aligned with this file.

Primary tracker: [adversarial review action items](adversarial-review-action-items.md).

## Required Submission Artifacts

| Artifact | Status | Link / evidence |
| -------- | ------ | --------------- |
| Public Hugging Face Space | Runnable in no-secret canned-fallback mode | https://huggingface.co/spaces/build-small-hackathon/figment |
| Space cold boot with app files present | Verified 2026-06-15 | Space API `runtime.stage=RUNNING`, `hardware=cpu-basic`, 167 siblings, `app.py` and `figment/observation_targets.py` present; Space URL served HTTP 200 |
| GitHub source repo | Pushed | https://github.com/ThomsenDrake/figment on `main` |
| Public fine-tuned model archive | Published | https://huggingface.co/build-small-hackathon/figment-finetuned-model-archive at `7da772ec7c0de20011d42780ea8afa65af4aef70`; includes v1 pilot plus v5-v14p BF16/GGUF artifacts, merge manifests, and model card |
| Public eval/training dataset repo | Published | https://huggingface.co/datasets/build-small-hackathon/figment-eval-traces at `92d9564fe6c984c55d65c7ba35a4e04eddcdea01`; configs load for `default` plus `figment_sft_v1` through `figment_sft_v14p`; v14p viewer split is 4801 train / 534 validation rows with 47 columns |
| Demo video | Proof needed | Pending |
| Social post | Proof needed | Pending |
| Safety statement | Present | [safety_statement.md](safety_statement.md) |
| User test notes | Template present; results needed | [user_test_notes.md](user_test_notes.md) |
| License | Present | [../LICENSE](../LICENSE) |
| Live hosted Omni trace | Eval traces present; final demo trace still needed | Baseline: `traces/hosted_omni_eval_20260607T194833Z.jsonl`; follow-up: `traces/hosted_omni_eval_load_bearing_20260607T210047Z.jsonl` |
| No-cloud/off-grid trace | Proof needed before claiming Off the Grid achieved | Pending |
| Hosted Omni eval results | Measured | [hosted_omni_eval_results.md](hosted_omni_eval_results.md): 31/50 whole-output competence, 8/50 full fallback, 480/650 model-retained fields, 170/650 deterministic patches, 50/50 final validation in the follow-up run |
| 4B LoRA system eval results | Published and measured | v14p repair-union on the corrected 150-case field-workflow holdout: 150/150 competence, 150/150 expected labels, 150/150 final validation, 0 deterministic patches, 0 fallback; raw first-pass success is 146/150 and 4/150 close through focused model repair. This is model-system evidence, not no-cloud proof or clinical validation |
| Local 4B + Parakeet no-cloud evidence | Helpers ready; proof needed | Full BF16 4B snapshot and Parakeet `.nemo` artifact are present locally; `scripts/run_local_4b_evidence.py` captures endpoint metadata, route smoke, eval records, eval summary, and `eval_evidence_manifest.json` once the local endpoint is live; `scripts/run_local_asr_evidence.py` captures Parakeet ASR provider evidence and `asr_evidence_manifest.json`. Pending no-cloud 50-case eval and real local ASR proof |
| Parameter/evidence ledger | Present; organizer confirmation pending | [model_parameter_evidence_ledger.md](model_parameter_evidence_ledger.md) |
| Submission claim audit | Present | `make audit-claims` scans submission-facing copy for premature Off the Grid, Llama Champion, Well-Tuned, Backyard user-use, local 4B, local ASR, demo-video, and social-post claims |
| Evidence gate status report | Present; incomplete by design until external proofs exist | `make evidence-gates` reports each evidence gate, paths found, and next actions. Current missing gates are badge/demo/user-proof gates, not missing Hub repos: no-cloud route, Llama Champion route, local ASR provider proof, trained-responder user test, demo video, and social post |

## Badge And Claim Status

| Claim / badge area | Submission wording allowed now | Evidence needed to upgrade |
| ------------------ | ------------------------------ | -------------------------- |
| Backyard AI | Targeted; built for a real trained responder, with identity withheld for privacy | Completed user-test notes from that responder on synthetic or de-identified scenarios. Do not claim use, testing, validation, approval, or endorsement before notes exist |
| Off the Grid | Targeted / proof-needed | Recorded no-cloud run using self-hosted Omni on adequate local hardware or a smaller verified local stack. Hosted API evidence does not count |
| Hosted Gradio Space | Runnable in no-secret canned-fallback mode; live hosted-model demo trace still proof-needed | Public Space app files present, cold boot, typed intake run, and route/fallback trace verified. Public workflow trace: `raw_route=canned`, `final_route=canned_backend`, `fallback_tier=canned`, `validation_status=passed`, `raw_audio_stored=false`, `model_retained_count=0`, `deterministic_patch_count=13` |
| Demo video | Targeted / proof-needed | Final video link showing only verified routes and labeling fallbacks honestly |
| Social post | Targeted / proof-needed | Final social post link with achieved-versus-targeted wording |
| Llama Champion | Targeted / proof-needed | Eligible local model route through llama.cpp with trace or eval evidence |
| Sharing is Caring | Public Space, GitHub repo, model archive, dataset repo, and trace links are ready; demo video and social post still pending | Final demo video and social post links |
| Well-Tuned | Published measured tuned 4B artifacts exist; claim wording should stay tied to the model archive and local route support, not the no-secret hosted Space route | Organizer accepts the published merged-model archive plus measured v14p result as the Well-Tuned artifact. Fallback output cannot count |
| Field Notes | Tentative | Organizer confirmation and final field-note artifact |
| Off-Brand | Targeted / proof-needed | Final demo or social artifact meeting organizer criteria |

## Off-Grid Evidence Boundary

Omni is not architecturally disqualified from Off the Grid: it can support the claim if self-hosted on adequate local hardware with no runtime cloud APIs. The current repo does not yet include that recorded proof. Until a no-cloud run exists, use targeted or proof-needed language.

## User-Test Evidence Boundary

The README may say the project is built for a real trained responder. It should not say that responder used, approved, validated, or endorsed Figment until [user_test_notes.md](user_test_notes.md) contains factual session notes.

## Eval Evidence Boundary

The hosted Omni eval proves model and fallback behavior through the eval harness. The public Space proof currently proves deployment health and no-secret fallback behavior, not live hosted Omni generation. Use the follow-up hosted eval run as the current hosted eval score: 31/50 whole-output hosted competence, 8/50 full fallback, 480/650 model-retained fields, 170/650 deterministic patches, and 50/50 final validation.

The 4B LoRA model archive proves a published, measured tuned local-model artifact. Use the v14p repair-union result as the current small-model score: 150/150 competence, 150/150 expected labels, 150/150 final validation, 0 deterministic patches, and 0 fallback on the corrected 150-case field-workflow holdout. Do not convert that into Off the Grid, local ASR, target-user, or clinical-validation evidence.

Final validation is app safety. Whole-output model competence and field-level model retention are the model-load-bearing metrics. Deterministic fallback and deterministic patches must stay visible in traces, scorecards, submission copy, and the demo.

## Submission Copy Boundaries

- Space: may say the public Space is runnable in no-secret canned-fallback mode, with app files present, cold boot verified, typed intake working, and trace labeling verified. Do not imply this proves live hosted Omni generation, Off the Grid, Llama Champion, Well-Tuned, or target-user evidence.
- Demo video and social post: use pending placeholders until final links exist.
- Backyard AI: may say built for a real trained responder; do not say the target user used or tested Figment until factual notes exist.
- Off the Grid: claim only after a recorded no-cloud run.
- Llama Champion: claim only after an eligible llama.cpp route runs with trace or eval evidence.
- Well-Tuned: may cite the published measured v14p tuned artifacts, but do not imply the public no-secret Space is serving them.
- Parameter compliance: cite the [model parameter/evidence ledger](model_parameter_evidence_ledger.md), including the Omni 31B body-count versus 33B sidebar ambiguity and organizer-confirmation status.

Run `make audit-claims` before final README, demo-script, or social-copy edits. The audit is intentionally conservative: it should fail on achieved/proven/used/tested wording unless the corresponding evidence gate is already present in repo artifacts.

Run `make evidence-gates` before claiming a badge or submission artifact is complete. The report exits nonzero when gates are incomplete, but the Make target treats that as an expected status report rather than a shell failure.
