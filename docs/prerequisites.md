# Figment Prerequisites

This page captures the setup contract for building and demoing Figment v1.

## Eligibility And Repos

Required for the Build Small Hackathon:

* Hugging Face account registered for the hackathon.
* Membership in the `build-small-hackathon` Hugging Face org.
* Gradio Space hosted under that org:
  `https://huggingface.co/spaces/build-small-hackathon/figment`
* Public repo for code and documentation.
* Final submission assets: Space link, demo video, and social post.
* Model total parameters at or below 32B.

## Accounts And Tokens

Required:

* Hugging Face token with write access for repo/Space pushes.
* Hugging Face token or endpoint access for hosted Nemotron 3 Nano live mode.
* Modal account with credits for fine-tuning and batch eval.

Build-time optional, depending on the synthetic-data path:

* Mistral API access for teacher generation or critique.
* MiniMax API access for teacher generation or critique.

## Local Machine

Reference local demo machine:

* macOS dev machine with 48 GB unified memory.
* At least 35 GB free disk/RAM headroom for the Q4/Q5 GGUF path.
* Internet access for initial model/tool downloads.

Local/offline proof target:

* `bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF` with `Q4_K_M` primary.
* `llama-server` on `http://127.0.0.1:8001`.
* 16k context by default, 8k fallback.

## CLI Tools

Install or verify:

```bash
git --version
python3 --version
uv --version
hf auth whoami
modal --version
docker --version
llama-server --help
```

Recommended install commands on macOS:

```bash
brew install llama.cpp
python3 -m pip install --upgrade huggingface_hub modal
```

`uvx --from huggingface_hub hf ...` is also acceptable when the `hf` executable is not installed globally.

## Python Dependencies

Runtime dependencies live in `requirements.txt`.

Development, testing, and training dependencies live in `requirements-dev.txt`.

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

## Environment Variables

Copy `.env.example` to `.env` locally and fill secrets there. Do not commit `.env`.

Required or expected variables:

* `FIGMENT_MODE` — `hosted`, `local`, or `canned`.
* `HF_MODEL_ID` — defaults to `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
* `HF_TOKEN` — Hugging Face token for hosted model access and Space secrets.
* `HF_ENDPOINT_URL` — optional dedicated HF Inference Endpoint URL.
* `LLAMA_BASE_URL` — local llama.cpp OpenAI-compatible endpoint.
* `FIGMENT_TRACE_DIR` — trace export directory.
* `MODAL_PROFILE` — optional Modal profile name.
* `MISTRAL_API_KEY` / `MINIMAX_API_KEY` — optional teacher-model keys.

## Runtime Modes

Hosted live demo:

* Gradio Space under `build-small-hackathon/figment`.
* HF-hosted Nemotron 3 Nano powers live navigator output.
* Rules, retrieval, validation, and trace rendering run in the Space.

Local/offline proof:

* Local Gradio app.
* Local protocol cards and SQLite retrieval.
* Local deterministic rules and validators.
* Local `llama-server` with Nemotron GGUF.

Fallback only:

* Canned traces if hosted model, quota, or Space cold-start reliability fails.

## Verification Checklist

Before implementation starts:

```bash
hf auth whoami
hf repos list --namespace build-small-hackathon --type space --search figment --limit 10
modal token info || modal setup
llama-server --help
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Before submission:

```text
Space boots cold under build-small-hackathon/figment.
Hosted live mode returns validated HF-hosted Nemotron output.
Local llama.cpp mode runs the same demo case without internet.
No patient PHI is used, logged, or committed.
```
