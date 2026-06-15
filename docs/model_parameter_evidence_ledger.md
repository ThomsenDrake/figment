# Figment Model Parameter And Evidence Ledger

Date: 2026-06-15

Purpose: keep parameter, route, locality, adapter, ASR, and organizer-confirmation claims in one evidence-gated place. README, submission copy, demo video, and social posts should not upgrade a claim beyond this ledger.

## Current Claim Boundary

- Hosted Omni has measured eval evidence through the eval harness. The public Space proof is now live HF ZeroGPU v14p serving evidence, but it is not live hosted Omni generation evidence.
- The public Space target exists and runs on HF ZeroGPU. Current public API evidence on 2026-06-15: `runtime.stage=RUNNING`, `hardware=zero-a10g`, `MODEL_BACKEND=hf_zerogpu`, `MODEL_STACK=local_4b_parakeet`, and a synthetic `/run_navigator` route with `raw_route=hf_zerogpu`, `final_route=model_with_deterministic_patches`, and `validation_status=passed`. A live `/draft_audio` call returned Parakeet CTC ASR draft intake with `audio_model_id=nvidia/parakeet-ctc-1.1b`, `confirmation_status=unconfirmed`, `raw_audio_stored=false`, and 5 draft fields.
- The public model archive now records published measured Figment 4B LoRA merged artifacts through `figment_sft_v14p`. The strongest measured result is v14p repair-union on the corrected 150-case field-workflow holdout: 150/150 competence, 150/150 expected labels, 150/150 final validation, 0 deterministic patches, and 0 fallback. This is tuned model-system evidence, not no-cloud or clinical validation evidence.
- Figment claims Off the Grid from the offline-capable local design: local protocol cards, deterministic rules, local model artifacts, and local ASR/text-navigation paths can run without cloud APIs. The hosted HF ZeroGPU Space is the public demo surface, not the no-cloud runtime evidence.
- Figment claims Llama Champion from the llama.cpp/GGUF eval trace path and compatible local serving instructions for the tuned 4B artifacts.
- Figment claims Well-Tuned from the published measured v14p tuned artifacts. Keep deterministic patch use visible when citing public Space route checks.
- Organizer confirmation is still needed for the Omni 31B body-count versus 33B sidebar ambiguity and for any additive local stack or adapter-count interpretation.

## Parameter Ledger

| Route / artifact | Model ID(s) | Total-parameter source used for copy | Active-parameter note | Adapter parameter count | ASR companion count | Endpoint locality | Organizer-confirmation status | Current evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hosted Omni primary | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`; API route `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | NVIDIA model-card body: 31B total parameters; HF sidebar has been observed as 33B in planning docs | Roughly 3B active parameters per token is a runtime/MoE note, not the compliance number | None used in current evals | Native Omni speech encoder is part of the Omni model-card count; no separate ASR model is claimed for hosted Omni | Hosted NVIDIA API route in current evals; self-hosted no-cloud route not recorded | Pending: ask organizers whether model-card body count is acceptable if sidebar count differs | Baseline eval: 28/50 whole-output competence, 22/50 full fallback, 50/50 final validation. Follow-up eval: 31/50 competence, 8/50 full fallback, 480/650 model-retained fields, 170/650 deterministic patches, 50/50 final validation |
| Self-hosted Omni no-cloud target | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`, FP8, or NVFP4 variant if served locally | Same Omni 31B body-count claim, with same 33B sidebar ambiguity | Active parameters do not decide compliance | None recorded | Native Omni audio if used locally; included in Omni count if organizers accept the model-card count | Would be local/self-hosted only if served with no runtime cloud APIs | Pending for count ambiguity and hardware/runtime proof | No recorded self-hosted Omni no-cloud eval or public demo trace yet. The Off the Grid badge claim instead rests on the offline-capable local design and artifacts. |
| Local 4B + Parakeet proof path | Text base: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`; tuned archive: `build-small-hackathon/figment-finetuned-model-archive`; ASR: `nvidia/parakeet-rnnt-1.1b`; local route `MODEL_BACKEND=llama_cpp`, `MODEL_STACK=local_4b_parakeet` | Workback model-card notes: 3.97B text model plus about 1.1B Parakeet, roughly 5.1B nominal before adapters | No active-parameter substitution; use additive total-count story if organizers require stack accounting | Published merged LoRA artifacts exist through `figment_sft_v14p`; exact adapter-only parameter count is not used as the primary public claim | About 1.1B for Parakeet RNNT ASR, only if real local ASR is enabled and proven | Intended local OpenAI-compatible endpoint serving the tuned GGUF plus local ASR | Pending: confirm additive multi-model counting and adapter-count treatment | Full BF16 4B snapshot downloaded at repo revision `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`; `model.safetensors` is `7947142640` bytes with SHA-256 `55d4e2519456c4a9bddf596b0748d630e3b2ce6ff6f4c2b7ed3e07e2b00dad42`. Parakeet artifact downloaded at repo revision `a07b19e98a26c1873a3f2622c446a4a1ca6316cb`; `parakeet-rnnt-1.1b.nemo` is `4283105280` bytes with SHA-256 `535896f014953d945b287ac533560e20da8103c6781b152de4645528e2b60738`. The v14p repair-union measured result is 150/150 competence, 150/150 expected labels, 150/150 final validation, 0 deterministic patches, and 0 fallback on the corrected 150-case field-workflow holdout. The local design supports Off the Grid, and the llama.cpp/GGUF eval trace path supports Llama Champion; local ASR provider proof and target-user trace remain separate evidence gaps. |
| HF ZeroGPU v14p + Parakeet ASR Space route | Text model: `build-small-hackathon/figment-finetuned-model-archive` subfolder `figment_sft_v14p/figment-sft-v14p-lora-merged-bf16`; ASR model: `nvidia/parakeet-ctc-1.1b` when `PARAKEET_ASR_RUNTIME=transformers` | 4B tuned text route plus about 1.1B Parakeet CTC ASR | No active-parameter substitution | Published merged LoRA artifacts exist through `figment_sft_v14p` | About 1.1B for Parakeet CTC ASR on the Space route | Hosted HF ZeroGPU, not no-cloud/off-grid | Pending only for any organizer-specific additive stack interpretation | The text route is live on `zero-a10g`. A live `/draft_audio` call on the committed pediatric demo WAV returned `audio_intake_path=parakeet_asr_plus_text_nemotron`, `audio_model_id=nvidia/parakeet-ctc-1.1b`, `confirmation_status=unconfirmed`, `raw_audio_stored=false`, and 5 draft fields in 5.44 seconds. |
| 4B Figment tuned archive | `build-small-hackathon/figment-finetuned-model-archive`, especially `figment_sft_v14p/figment-sft-v14p-lora-merged-bf16/` and `figment_sft_v14p/figment-sft-v14p-lora-merged-bf16.bf16.gguf` | Base model is the 3.97B Nemotron 4B BF16 target; merged LoRA artifacts are derived from that base | Not applicable | Published as merged BF16 and GGUF artifacts with merge manifests; adapter-only count remains a documentation detail if needed by organizers | None for text-only eval | Public HF model repo plus local OpenAI-compatible app route support | Pending only for any organizer-specific adapter-count interpretation | Public model repo is `private=False`, current commit `7da772ec7c0de20011d42780ea8afa65af4aef70`, with v1 pilot plus v5-v14p artifacts. v14p model card records the repair-union result and GGUF SHA. |
| Canned fallback | No live model | Not a model-compliance artifact | Not applicable | Not applicable | Not applicable | Local deterministic fallback; public Space no-secret fallback verified | Not applicable | Useful for safety, deployment health, and cold-start fallback only. Cannot count as model competence, Off the Grid proof, Llama Champion proof, or Well-Tuned proof |

## Submission Gates

| Claim | Required upgrade evidence |
| --- | --- |
| Public Space runnable | Satisfied for HF ZeroGPU v14p: app files present, Space is `RUNNING` on `zero-a10g`, `/runtime` reports `MODEL_BACKEND=hf_zerogpu`, and a synthetic route check preserves fallback/patch labeling. |
| Hosted model load-bearing | Cite hosted eval metrics separately from final validation: 31/50 whole-output competence and 480/650 model-retained fields in the follow-up run |
| <=32B hosted Omni compliance | Organizer accepts the 31B model-card body count or the submission falls back to a clearly eligible smaller route |
| Off the Grid | Claimed from the offline-capable local design and artifacts; do not describe the hosted ZeroGPU route as the no-cloud runtime |
| Llama Champion | Claimed from the llama.cpp/GGUF eval trace path and compatible local-serving instructions |
| Well-Tuned | Claimed from the published measured v14p tuned archive |
| Backyard AI user use | Completed user-test notes from a real trained responder on synthetic or de-identified scenarios |
| Demo video | Published launch video: https://huggingface.co/spaces/build-small-hackathon/figment/resolve/main/assets/figment-live-space-launch-final.mp4 |
| Field Notes | Published Hugging Face blog: https://huggingface.co/blog/build-small-hackathon/figment-build-blog |
| Social post | Final link remains a separate submission artifact |

## Evaluation Score Boundary

Final validation success means the application returned a valid, safe output after validation, repair, or fallback. It is not the same as model competence.

Use these current hosted follow-up metrics when summarizing load-bearing behavior:

- Whole-output hosted competence: 31/50
- Full deterministic fallback: 8/50
- Model-retained fields: 480/650
- Deterministic patches: 170/650
- Final validation: 50/50

Do not count full fallback or deterministic patches as pure model output.
