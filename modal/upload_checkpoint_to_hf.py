"""Upload a checkpoint folder from a Modal volume to Hugging Face.

This is intended for large merged model artifacts that should move directly
from Modal storage to the Hub without first pulling the full checkpoint local.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal


APP_NAME = "figment-checkpoint-hf-upload"
CHECKPOINT_VOLUME_NAME = "figment-checkpoints"
CHECKPOINT_DIR = "/checkpoints"
DEFAULT_REPO_ID = "build-small-hackathon/figment-finetuned-model-archive"


app = modal.App(APP_NAME)

checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=False)
huggingface_secret = modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])

upload_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub>=1.18,<2")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    image=upload_image,
    cpu=4,
    memory=16384,
    ephemeral_disk=524288,
    volumes={CHECKPOINT_DIR: checkpoint_volume},
    secrets=[huggingface_secret],
    timeout=6 * 60 * 60,
)
def upload_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import HfApi

    dataset_version = str(config["dataset_version"])
    checkpoint_name = str(config["checkpoint_name"])
    repo_id = str(config["repo_id"])
    repo_type = str(config.get("repo_type") or "model")
    path_in_repo = str(config.get("path_in_repo") or f"{dataset_version}/{checkpoint_name}").strip("/")
    commit_message = str(config.get("commit_message") or f"Upload {dataset_version} {checkpoint_name}")
    private = bool(config.get("private", False))
    required_files = list(config.get("required_files") or [])

    checkpoint_dir = Path(CHECKPOINT_DIR) / dataset_version / checkpoint_name
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"checkpoint path is not a directory: {checkpoint_dir}")

    missing = [name for name in required_files if not (checkpoint_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"checkpoint is missing required files: {missing}")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    commit_info = api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )

    files = sorted(path.name for path in checkpoint_dir.iterdir() if path.is_file())
    result = {
        "status": "uploaded",
        "checkpoint_volume": CHECKPOINT_VOLUME_NAME,
        "checkpoint_dir": str(checkpoint_dir),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "path_in_repo": path_in_repo,
        "commit_message": commit_message,
        "commit_url": getattr(commit_info, "commit_url", ""),
        "commit_oid": getattr(commit_info, "oid", ""),
        "files": files,
        "required_files": required_files,
    }
    print(json.dumps({"hf_checkpoint_upload": result}, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main(
    dataset_version: str,
    checkpoint_name: str,
    repo_id: str = DEFAULT_REPO_ID,
    path_in_repo: str = "",
    commit_message: str = "",
    private: bool = False,
) -> None:
    config = {
        "dataset_version": dataset_version,
        "checkpoint_name": checkpoint_name,
        "repo_id": repo_id,
        "repo_type": "model",
        "path_in_repo": path_in_repo or f"{dataset_version}/{checkpoint_name}",
        "commit_message": commit_message or f"Upload {dataset_version} {checkpoint_name}",
        "private": private,
        "required_files": [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "figment_merge_manifest.json",
        ],
    }
    result = upload_checkpoint.remote(config)
    print(json.dumps({"upload": result}, indent=2, sort_keys=True))
