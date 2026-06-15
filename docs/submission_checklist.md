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
| Demo video | Published | https://huggingface.co/spaces/build-small-hackathon/figment/resolve/main/assets/figment-live-space-launch-final.mp4 |
| Social post | Proof needed | Pending |
| Safety statement | Present | [safety_statement.md](safety_statement.md) |
| User test notes | Template present; results needed | [user_test_notes.md](user_test_notes.md) |
| License | Present | [../LICENSE](../LICENSE) |
| Live hosted Omni trace | Eval traces present; final demo trace still needed | Baseline: `traces/hosted_omni_eval_20260607T194833Z.jsonl`; follow-up: `traces/hosted_omni_eval_load_bearing_20260607T210047Z.jsonl` |
| Off the Grid badge basis | Claimed from offline-capable design | Local protocol cards, deterministic rules, local model artifacts, and local ASR/text-navigation paths support the no-cloud design claim. The hosted HF ZeroGPU Space is the public demo surface, not the no-cloud runtime evidence. |
| Hosted Omni eval results | Measured | [hosted_omni_eval_results.md](hosted_omni_eval_results.md): 31/50 whole-output competence, 8/50 full fallback, 480/650 model-retained fields, 170/650 deterministic patches, 50/50 final validation in the follow-up run |
| 4B LoRA system eval results | Published and measured | v14p repair-union on the corrected 150-case field-workflow holdout: 150/150 competence, 150/150 expected labels, 150/150 final validation, 0 deterministic patches, 0 fallback; raw first-pass success is 146/150 and 4/150 close through focused model repair. This is model-system evidence, not no-cloud proof or clinical validation |
| HF ZeroGPU Parakeet audio draft | Live route check recorded | Public Space commit `79e487aebc8df11c084a2054152d447cc0838837` returned `audio_model_id=nvidia/parakeet-ctc-1.1b`, `audio_intake_path=parakeet_asr_plus_text_nemotron`, `confirmation_status=unconfirmed`, `raw_audio_stored=false`, and 5 draft fields from a committed demo WAV in 5.44 seconds |
| Local 4B + Parakeet no-cloud evidence | Helpers ready; proof needed | Full BF16 4B snapshot and Parakeet `.nemo` artifact are present locally; `scripts/run_local_4b_evidence.py` captures endpoint metadata, route smoke, eval records, eval summary, and `eval_evidence_manifest.json` once the local endpoint is live; `scripts/run_local_asr_evidence.py --audio <source.wav> --transcribe-audio` captures Parakeet ASR provider evidence and `asr_evidence_manifest.json`. Pending no-cloud 50-case eval and real local ASR proof |
| Parameter/evidence ledger | Present; organizer confirmation pending | [model_parameter_evidence_ledger.md](model_parameter_evidence_ledger.md) |
| Submission claim audit | Present | `make audit-claims` scans submission-facing copy for premature Off the Grid, Llama Champion, Well-Tuned, Backyard user-use, local 4B, local ASR, demo-video, and social-post claims |
| Evidence gate status report | Present; incomplete by design until external proofs exist | `make evidence-gates` reports badge-claim readiness separately from remaining non-badge gaps such as target-user notes, local endpoint proof, local ASR provider proof, and social post. |

## Badge And Claim Status

| Claim / badge area | Submission wording allowed now | Evidence needed to upgrade |
| ------------------ | ------------------------------ | -------------------------- |
| Backyard AI | Targeted; built for a real trained responder, with identity withheld for privacy | Completed user-test notes from that responder on synthetic or de-identified scenarios. Do not claim use, testing, validation, approval, or endorsement before notes exist |
| Off the Grid | Claimed | Offline-capable local design with local protocol cards, deterministic rules, local model artifacts, and local ASR/text-navigation paths. Hosted API evidence does not count as the no-cloud runtime itself |
| Hosted Gradio Space | Runnable in no-secret canned-fallback mode; live hosted-model demo trace still proof-needed | Public Space app files present, cold boot, typed intake run, and route/fallback trace verified. Public workflow trace: `raw_route=canned`, `final_route=canned_backend`, `fallback_tier=canned`, `validation_status=passed`, `raw_audio_stored=false`, `model_retained_count=0`, `deterministic_patch_count=13` |
| Demo video | Published | Launch video embedded in README and hosted at https://huggingface.co/spaces/build-small-hackathon/figment/resolve/main/assets/figment-live-space-launch-final.mp4 |
| Social post | Targeted / proof-needed | Final social post link with achieved-versus-targeted wording |
| Llama Champion | Claimed | The eval trace path used llama.cpp/GGUF, and the repo includes llama.cpp-compatible local serving instructions for the tuned 4B artifacts |
| Sharing is Caring | Claimed | Public Space, GitHub repo, model archive, dataset/eval traces, trace schema, embedded launch video, and field notes are published |
| Well-Tuned | Claimed | Published measured v14p tuned 4B artifacts in the model archive. Fallback output cannot count as model competence |
| Field Notes | Claimed | Published Hugging Face blog: https://huggingface.co/blog/build-small-hackathon/figment-build-blog |
| Off-Brand | Claimed | Custom Gradio Server UI with Field Kit Workbench workflow and trace surfaces, not the default Gradio Blocks look |

## Off-Grid Evidence Boundary

Figment claims Off the Grid from the offline-capable local design: local protocol cards, deterministic rules, local model artifacts, and local ASR/text-navigation paths can run without cloud APIs. The hosted HF ZeroGPU Space is the public demo surface and should not be described as the no-cloud runtime.

## User-Test Evidence Boundary

The README may say the project is built for a real trained responder. It should not say that responder used, approved, validated, or endorsed Figment until [user_test_notes.md](user_test_notes.md) contains factual session notes.

## Eval Evidence Boundary

The hosted Omni eval proves model and fallback behavior through the eval harness. The public Space proof currently proves live HF ZeroGPU v14p serving and deployment health, not hosted Omni generation. Use the follow-up hosted eval run as the current hosted eval score: 31/50 whole-output hosted competence, 8/50 full fallback, 480/650 model-retained fields, 170/650 deterministic patches, and 50/50 final validation.

The 4B LoRA model archive proves a published, measured tuned local-model artifact. Use the v14p repair-union result as the current small-model score: 150/150 competence, 150/150 expected labels, 150/150 final validation, 0 deterministic patches, and 0 fallback on the corrected 150-case field-workflow holdout. Keep that separate from local ASR, target-user, and clinical-validation evidence.

Final validation is app safety. Whole-output model competence and field-level model retention are the model-load-bearing metrics. Deterministic fallback and deterministic patches must stay visible in traces, scorecards, submission copy, and the demo.

## Submission Copy Boundaries

- Space: may say the public Space is runnable on HF ZeroGPU with app files present, typed intake, Parakeet draft audio, and trace labeling verified. Do not imply this proves target-user evidence or clinical validation.
- Demo video: use the published launch-video URL above.
- Social post: use a pending placeholder until the final link exists.
- Backyard AI: may say built for a real trained responder; do not say the target user used or tested Figment until factual notes exist.
- Off the Grid: claimed from the offline-capable local design; do not describe the hosted ZeroGPU route itself as no-cloud.
- Llama Champion: claimed from the llama.cpp/GGUF eval trace path and compatible local-serving instructions.
- Well-Tuned: claimed from the published measured v14p tuned artifacts.
- Parameter compliance: cite the [model parameter/evidence ledger](model_parameter_evidence_ledger.md), including the Omni 31B body-count versus 33B sidebar ambiguity and organizer-confirmation status.

Run `make audit-claims` before final README, demo-script, or social-copy edits. The audit is intentionally conservative: it should fail on achieved/proven/used/tested wording unless the corresponding evidence gate is already present in repo artifacts.

Run `make evidence-gates` before changing badge or submission copy. The report separates badge-claim readiness from remaining non-badge gaps, and the Make target treats incomplete external gates as an expected status report rather than a shell failure.
