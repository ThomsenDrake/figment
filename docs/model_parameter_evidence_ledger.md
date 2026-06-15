# Figment Model Parameter And Evidence Ledger

Date: 2026-06-15

Purpose: keep parameter, route, locality, adapter, ASR, and organizer-confirmation claims in one evidence-gated place. README, submission copy, demo video, and social posts should not upgrade a claim beyond this ledger.

## Current Claim Boundary

- Hosted Omni has measured eval evidence through the eval harness. The public Space is now verified runnable in no-secret canned-fallback mode, but that public Space proof is not live hosted Omni generation evidence.
- The public Space target exists and cold-boots from the Space URL. Current public API evidence: `runtime.stage=RUNNING`, `hardware=cpu-basic`, 167 siblings, `app.py` present, `figment/observation_targets.py` present, and Space URL HTTP 200. Public workflow evidence remains no-secret canned fallback unless a later non-canned public trace is recorded.
- The public model archive now records published measured Figment 4B LoRA merged artifacts through `figment_sft_v14p`. The strongest measured result is v14p repair-union on the corrected 150-case field-workflow holdout: 150/150 competence, 150/150 expected labels, 150/150 final validation, 0 deterministic patches, and 0 fallback. This is tuned model-system evidence, not no-cloud or clinical validation evidence.
- The local 4B + Parakeet route is still the preferred no-cloud/off-grid proof path. The full BF16 4B artifact and Parakeet artifact are present locally, and the tuned v14p GGUF is published, but no-cloud route proof and local ASR provider proof are still separate gates.
- Well-Tuned wording may cite the published measured v14p tuned artifacts if the submission/judging interpretation accepts the merged-model archive and local route support. Do not imply the public no-secret Space is serving the tuned model.
- Organizer confirmation is still needed for the Omni 31B body-count versus 33B sidebar ambiguity and for any additive local stack or adapter-count interpretation.

## Parameter Ledger

| Route / artifact | Model ID(s) | Total-parameter source used for copy | Active-parameter note | Adapter parameter count | ASR companion count | Endpoint locality | Organizer-confirmation status | Current evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hosted Omni primary | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`; API route `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | NVIDIA model-card body: 31B total parameters; HF sidebar has been observed as 33B in planning docs | Roughly 3B active parameters per token is a runtime/MoE note, not the compliance number | None used in current evals | Native Omni speech encoder is part of the Omni model-card count; no separate ASR model is claimed for hosted Omni | Hosted NVIDIA API route in current evals; self-hosted no-cloud route not recorded | Pending: ask organizers whether model-card body count is acceptable if sidebar count differs | Baseline eval: 28/50 whole-output competence, 22/50 full fallback, 50/50 final validation. Follow-up eval: 31/50 competence, 8/50 full fallback, 480/650 model-retained fields, 170/650 deterministic patches, 50/50 final validation |
| Self-hosted Omni no-cloud target | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`, FP8, or NVFP4 variant if served locally | Same Omni 31B body-count claim, with same 33B sidebar ambiguity | Active parameters do not decide compliance | None recorded | Native Omni audio if used locally; included in Omni count if organizers accept the model-card count | Would be local/self-hosted only if served with no runtime cloud APIs | Pending for count ambiguity and hardware/runtime proof | No recorded no-cloud eval or public demo trace yet. Do not claim Off the Grid achieved |
| Local 4B + Parakeet proof path | Text base: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`; tuned archive: `build-small-hackathon/figment-finetuned-model-archive`; ASR: `nvidia/parakeet-rnnt-1.1b`; local route `MODEL_BACKEND=llama_cpp`, `MODEL_STACK=local_4b_parakeet` | Workback model-card notes: 3.97B text model plus about 1.1B Parakeet, roughly 5.1B nominal before adapters | No active-parameter substitution; use additive total-count story if organizers require stack accounting | Published merged LoRA artifacts exist through `figment_sft_v14p`; exact adapter-only parameter count is not used as the primary public claim | About 1.1B for Parakeet RNNT ASR, only if real local ASR is enabled and proven | Intended local OpenAI-compatible endpoint serving the tuned GGUF plus local ASR; no-cloud only after recorded proof | Pending: confirm additive multi-model counting and adapter-count treatment | Full BF16 4B snapshot downloaded at repo revision `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`; `model.safetensors` is `7947142640` bytes with SHA-256 `55d4e2519456c4a9bddf596b0748d630e3b2ce6ff6f4c2b7ed3e07e2b00dad42`. Parakeet artifact downloaded at repo revision `a07b19e98a26c1873a3f2622c446a4a1ca6316cb`; `parakeet-rnnt-1.1b.nemo` is `4283105280` bytes with SHA-256 `535896f014953d945b287ac533560e20da8103c6781b152de4645528e2b60738`. The v14p repair-union measured result is 150/150 competence, 150/150 expected labels, 150/150 final validation, 0 deterministic patches, and 0 fallback on the corrected 150-case field-workflow holdout. No no-cloud local route proof, local ASR provider proof, or target-user trace is recorded yet |
| 4B Figment tuned archive | `build-small-hackathon/figment-finetuned-model-archive`, especially `figment_sft_v14p/figment-sft-v14p-lora-merged-bf16/` and `figment_sft_v14p/figment-sft-v14p-lora-merged-bf16.bf16.gguf` | Base model is the 3.97B Nemotron 4B BF16 target; merged LoRA artifacts are derived from that base | Not applicable | Published as merged BF16 and GGUF artifacts with merge manifests; adapter-only count remains a documentation detail if needed by organizers | None for text-only eval | Public HF model repo plus local OpenAI-compatible app route support | Pending only for any organizer-specific adapter-count interpretation | Public model repo is `private=False`, current commit `7da772ec7c0de20011d42780ea8afa65af4aef70`, with v1 pilot plus v5-v14p artifacts. v14p model card records the repair-union result and GGUF SHA. |
| Canned fallback | No live model | Not a model-compliance artifact | Not applicable | Not applicable | Not applicable | Local deterministic fallback; public Space no-secret fallback verified | Not applicable | Useful for safety, deployment health, and cold-start fallback only. Cannot count as model competence, Off the Grid proof, Llama Champion proof, or Well-Tuned proof |

## Submission Gates

| Claim | Required upgrade evidence |
| --- | --- |
| Public Space runnable | Satisfied for no-secret canned-fallback mode: app files present, clean cold boot from the Space URL, and route/fallback labeling preserved. Non-canned public model traces remain separate proof. |
| Hosted model load-bearing | Cite hosted eval metrics separately from final validation: 31/50 whole-output competence and 480/650 model-retained fields in the follow-up run |
| <=32B hosted Omni compliance | Organizer accepts the 31B model-card body count or the submission falls back to a clearly eligible smaller route |
| Off the Grid | Recorded no-cloud run with trace evidence, either self-hosted Omni or local 4B + Parakeet/typed intake |
| Llama Champion | Eligible model route runs through llama.cpp with trace or eval evidence |
| Well-Tuned | Cite the published measured v14p tuned archive only with wording that does not imply the no-secret public Space is serving it |
| Backyard AI user use | Completed user-test notes from a real trained responder on synthetic or de-identified scenarios |
| Demo video and social post | Final links exist and wording says achieved only for artifacts supported by this ledger |

## Evaluation Score Boundary

Final validation success means the application returned a valid, safe output after validation, repair, or fallback. It is not the same as model competence.

Use these current hosted follow-up metrics when summarizing load-bearing behavior:

- Whole-output hosted competence: 31/50
- Full deterministic fallback: 8/50
- Model-retained fields: 480/650
- Deterministic patches: 170/650
- Final validation: 50/50

Do not count full fallback or deterministic patches as pure model output.
