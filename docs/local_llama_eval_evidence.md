# Local llama.cpp eval evidence

Date: 2026-06-07

This note separates local llama.cpp evidence from hosted Omni evidence and from canned deterministic fallback.

## Full-weight local artifact

The canonical local text artifact is the full BF16 4B model, not a quantized GGUF:

- Model repo: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`
- Revision: `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`
- Local snapshot path: `/Users/drake.thomsen/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`
- Resolved snapshot size: `7.4G`
- Full weight file: `model.safetensors`
- Weight file size: `7947142640` bytes
- Weight file SHA-256: `55d4e2519456c4a9bddf596b0748d630e3b2ce6ff6f4c2b7ed3e07e2b00dad42`
- Required companion files present: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `modeling_nemotron_h.py`, `configuration_nemotron_h.py`

This download proves local artifact availability only. It does not prove local model competence, Off the Grid, Llama Champion, or local ASR until a real local OpenAI-compatible endpoint runs through the Figment eval without full deterministic fallback.

## Endpoint checks

Checked:

```bash
lsof -nP -iTCP:8001 -sTCP:LISTEN
curl -fsS --max-time 2 http://127.0.0.1:8001/v1/models
```

Observed: no listener on port `8001`; `curl` failed to connect to `127.0.0.1:8001`.

Rechecked after downloading the full BF16 artifact on 2026-06-07:

```bash
curl -fsS --max-time 2 http://127.0.0.1:8001/v1/models
```

Observed:

- `curl: (7) Failed to connect to 127.0.0.1 port 8001`

Local 4B, Off the Grid, and Llama Champion claims remain proof-needed. The next real step is to run the full BF16 model on a local OpenAI-compatible endpoint, record `/v1/models`, then run the 50-case eval below.

## Discarded quantized side check

A mistaken Q4 GGUF smoke was attempted before the full-weight requirement was clarified. It proved only that Figment could reach a local OpenAI-compatible endpoint; the model output failed navigator validation and fell back deterministically, so it does not count as route proof or model competence. The Q4 cache created by that attempt was removed.

## Evidence bundle command

Prefer the bundled helper once the local endpoint is live:

```bash
PYTHON_DOTENV_DISABLED=true \
python3 scripts/run_local_4b_evidence.py \
  --base-url <local-openai-compatible-endpoint>/v1
```

The helper writes a timestamped evidence directory under `traces/local_4b_evidence_*`:

- `endpoint_metadata.json`: `/v1/models` response or connection error
- `route_smoke.json`: one-case configured-route smoke result
- `local_4b_eval.jsonl`: 50-case eval records, only after the route smoke proves configured-model validation unless `--force-eval` is used
- `eval_summary.json`: model competence, fallback, provenance, and final validation counts
- `eval_evidence_manifest.json`: compact evidence manifest with model/server metadata, no-cloud route flags, raw/repair/full-fallback counts, field provenance, latency summary, eval-file hash, and per-case trace hashes
- `summary.json`: top-level status and whether the evidence counts as route proof or 50-case local competence

Exit codes are intentionally evidence-gated: `0` for a completed eval or passed smoke-only run, `2` for endpoint unavailable, and `1` when the route smoke fails and eval is skipped. Endpoint availability alone must not be counted as model competence.

## One-case route smoke

Use this only to prove the app can call the configured local route. A passed smoke is not a 50-case local competence run.

```bash
FIGMENT_MODE=local \
MODEL_STACK=local_4b_parakeet \
MODEL_BACKEND=llama_cpp \
LOCAL_MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
LLAMA_BASE_URL=<local-openai-compatible-endpoint>/v1 \
FIGMENT_SMOKE_ALLOW_NETWORK=true \
PYTHON_DOTENV_DISABLED=true \
python3 scripts/smoke_model_route.py
```

The smoke output now includes `local_llm_evidence.proof_status`, `counts_as_no_cloud_route_proof`, and `counts_as_50_case_local_llm_competence`. A skipped smoke must not be counted as local model competence.

## Real 50-case eval command

Run this directly only after starting a real local OpenAI-compatible server and recording `/v1/models` metadata. The bundled helper above runs the same eval after a passing route smoke.

```bash
FIGMENT_MODE=local \
MODEL_STACK=local_4b_parakeet \
MODEL_BACKEND=llama_cpp \
LOCAL_MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
LLAMA_BASE_URL=<local-openai-compatible-endpoint>/v1 \
PYTHON_DOTENV_DISABLED=true \
python3 scripts/run_eval.py \
  --backend llama_cpp \
  --model-stack local_4b_parakeet \
  --cases data/eval/initial_handwritten_cases.jsonl \
  --cases data/eval/adversarial_strict_cases.jsonl \
  --cases data/eval/comprehensive_hosted_cases.jsonl \
  --output traces/local_llama_cpp_eval_$(date -u +%Y%m%dT%H%M%SZ).jsonl
```

A real local evidence bundle should include the eval JSONL trace, `eval_evidence_manifest.json`, `/v1/models` response or equivalent server metadata, model file/hash or server launch command, and confirmation that no hosted model credentials were required for the run.
