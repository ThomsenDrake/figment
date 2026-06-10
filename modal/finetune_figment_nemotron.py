"""Modal LoRA fine-tuning job for Figment's local Nemotron 4B navigator.

Run a smoke job first:

    .venv/bin/modal run modal/finetune_figment_nemotron.py --smoke true

Then run the full pilot:

    .venv/bin/modal run modal/finetune_figment_nemotron.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal


APP_NAME = "figment-nemotron-4b-lora"
DEFAULT_DATASET_VERSION = "figment_sft_v1"
DEFAULT_BASE_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DEFAULT_DATASET_PATH = "data/finetune/figment_sft_v1.jsonl"
DATA_VOLUME_NAME = "figment-sft-data"
MODEL_CACHE_VOLUME_NAME = "figment-model-cache"
CHECKPOINT_VOLUME_NAME = "figment-checkpoints"
DATA_DIR = "/data"
MODEL_CACHE_DIR = "/model_cache"
CHECKPOINT_DIR = "/checkpoints"


app = modal.App(APP_NAME)

model_cache_volume = modal.Volume.from_name(MODEL_CACHE_VOLUME_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)
huggingface_secret = modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])

training_image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel")
    .apt_install("git", "build-essential", "ninja-build")
    .uv_pip_install(
        "accelerate>=1.8,<2",
        "datasets>=3.6,<5",
        "einops>=0.8,<1",
        "ninja>=1.11,<2",
        "peft>=0.15,<1",
        "protobuf>=5,<7",
        "safetensors>=0.5,<1",
        "sentencepiece>=0.2,<1",
        "setuptools>=70",
        "torch==2.7.1",
        "transformers>=4.52,<5",
        "wandb>=0.19,<1",
        "wheel>=0.45",
    )
    .run_commands(
        "python -m pip install --no-build-isolation --no-deps "
        "'causal-conv1d==1.6.2.post1' 'mamba-ssm==2.2.6.post3'"
    )
    .env(
        {
            "HF_HOME": MODEL_CACHE_DIR,
            "HF_HUB_CACHE": f"{MODEL_CACHE_DIR}/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)


def dataset_volume_paths(dataset_version: str) -> dict[str, str]:
    root = f"{DATA_DIR}/{dataset_version}"
    return {
        "root": root,
        "train": f"{root}/train.jsonl",
        "validation": f"{root}/validation.jsonl",
        "manifest": f"{root}/manifest.json",
    }


def build_train_config(
    *,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
    output_name: str = "pilot-lora",
    smoke: bool = False,
    max_steps: int = 40,
    max_seq_length: int = 16384,
    learning_rate: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    gradient_accumulation_steps: int = 8,
    validation_steps: int = 25,
    save_steps: int = 50,
) -> dict[str, Any]:
    if smoke:
        max_steps = max(1, min(max_steps, 5))
        max_seq_length = max(512, min(max_seq_length, 2048))
        if not output_name.endswith("-smoke"):
            output_name = f"{output_name}-smoke"

    return {
        "dataset_version": dataset_version,
        "base_model_id": base_model_id,
        "output_name": output_name,
        "output_dir": f"{CHECKPOINT_DIR}/{dataset_version}/{output_name}",
        "max_steps": max_steps,
        "max_seq_length": max_seq_length,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "validation_steps": max(1, min(validation_steps, max_steps)),
        "save_steps": max(1, min(save_steps, max_steps)),
        "warmup_ratio": 0.05,
        "weight_decay": 0.0,
        "seed": 42,
        "smoke": smoke,
    }


def build_merge_config(
    *,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
    adapter_name: str,
    output_name: str = "",
) -> dict[str, Any]:
    if not output_name:
        output_name = f"{adapter_name}-merged-bf16"
    return {
        "dataset_version": dataset_version,
        "base_model_id": base_model_id,
        "adapter_name": adapter_name,
        "adapter_dir": f"{CHECKPOINT_DIR}/{dataset_version}/{adapter_name}",
        "output_name": output_name,
        "output_dir": f"{CHECKPOINT_DIR}/{dataset_version}/{output_name}",
    }


@app.function(
    image=training_image,
    volumes={DATA_DIR: data_volume},
    timeout=30 * 60,
)
def stage_dataset(dataset_version: str, train_jsonl: str, validation_jsonl: str, manifest_json: str) -> dict[str, Any]:
    paths = dataset_volume_paths(dataset_version)
    root = Path(paths["root"])
    root.mkdir(parents=True, exist_ok=True)
    Path(paths["train"]).write_text(train_jsonl, encoding="utf-8")
    Path(paths["validation"]).write_text(validation_jsonl, encoding="utf-8")
    Path(paths["manifest"]).write_text(manifest_json, encoding="utf-8")
    data_volume.commit()
    return {
        "dataset_version": dataset_version,
        "train_path": paths["train"],
        "validation_path": paths["validation"],
        "manifest_path": paths["manifest"],
        "train_rows": _count_jsonl_text(train_jsonl),
        "validation_rows": _count_jsonl_text(validation_jsonl),
    }


@app.function(
    image=training_image,
    gpu="L40S",
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,
    volumes={
        DATA_DIR: data_volume,
        MODEL_CACHE_DIR: model_cache_volume,
        CHECKPOINT_DIR: checkpoint_volume,
    },
    secrets=[huggingface_secret],
    timeout=12 * 60 * 60,
)
def train(config: dict[str, Any]) -> dict[str, Any]:
    import inspect
    import os

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from peft import TaskType
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer
    from transformers import DataCollatorForSeq2Seq
    from transformers import Trainer
    from transformers import TrainingArguments

    paths = dataset_volume_paths(str(config["dataset_version"]))
    train_rows = _read_jsonl(Path(paths["train"]))
    validation_rows = _read_jsonl(Path(paths["validation"]))
    if not train_rows:
        raise ValueError(f"no training rows staged at {paths['train']}")
    if not validation_rows:
        raise ValueError(f"no validation rows staged at {paths['validation']}")

    token = os.environ["HF_TOKEN"]
    base_model_id = str(config["base_model_id"])
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        cache_dir=MODEL_CACHE_DIR,
        token=token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = _tokenized_dataset(train_rows, tokenizer, int(config["max_seq_length"]))
    validation_dataset = _tokenized_dataset(validation_rows, tokenizer, int(config["max_seq_length"]))
    if len(train_dataset) == 0:
        raise ValueError("all training rows lost supervised assistant tokens after tokenization")
    if len(validation_dataset) == 0:
        raise ValueError("all validation rows lost supervised assistant tokens after tokenization")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        cache_dir=MODEL_CACHE_DIR,
        token=token,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = str(config["output_dir"])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    args_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(config["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "learning_rate": float(config["learning_rate"]),
        "max_steps": int(config["max_steps"]),
        "warmup_ratio": float(config["warmup_ratio"]),
        "weight_decay": float(config["weight_decay"]),
        "logging_steps": 1,
        "save_steps": int(config["save_steps"]),
        "eval_steps": int(config["validation_steps"]),
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "report_to": [],
        "seed": int(config["seed"]),
    }
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        args_kwargs["eval_strategy"] = "steps"
    else:
        args_kwargs["evaluation_strategy"] = "steps"
    training_args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
        ),
    )
    train_result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = dict(train_result.metrics)
    manifest = {
        "config": _safe_config(config),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "tokenized_train_rows": len(train_dataset),
        "tokenized_validation_rows": len(validation_dataset),
        "metrics": metrics,
        "adapter_path": output_dir,
    }
    Path(output_dir, "figment_training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_volume.commit()
    model_cache_volume.commit()
    return manifest


@app.function(
    image=training_image,
    gpu="L40S",
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,
    volumes={
        MODEL_CACHE_DIR: model_cache_volume,
        CHECKPOINT_DIR: checkpoint_volume,
    },
    secrets=[huggingface_secret],
    timeout=4 * 60 * 60,
)
def merge_adapter(config: dict[str, Any]) -> dict[str, Any]:
    import os

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    token = os.environ["HF_TOKEN"]
    base_model_id = str(config["base_model_id"])
    adapter_dir = Path(str(config["adapter_dir"]))
    output_dir = Path(str(config["output_dir"]))
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"missing adapter_config.json in {adapter_dir}")
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"missing adapter_model.safetensors in {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        cache_dir=MODEL_CACHE_DIR,
        token=token,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        cache_dir=MODEL_CACHE_DIR,
        token=token,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    merged_model = peft_model.merge_and_unload(safe_merge=True)
    merged_model.config.use_cache = True
    if getattr(merged_model, "generation_config", None) is not None:
        merged_model.generation_config.do_sample = False
        merged_model.generation_config.top_p = None
        merged_model.generation_config.temperature = None
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(output_dir)

    files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    manifest = {
        "base_model_id": base_model_id,
        "adapter_dir": str(adapter_dir),
        "output_dir": str(output_dir),
        "files": files,
        "dtype": "bfloat16",
        "merge_method": "peft.merge_and_unload(safe_merge=True)",
    }
    (output_dir / "figment_merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_volume.commit()
    model_cache_volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    dataset_version: str = DEFAULT_DATASET_VERSION,
    dataset: str = DEFAULT_DATASET_PATH,
    prepared_dir: str = "",
    output_name: str = "pilot-lora",
    smoke: bool = False,
    skip_stage: bool = False,
    max_steps: int = 40,
    max_seq_length: int = 16384,
    gpu: str = "L40S",
    merge_only: bool = False,
    adapter_name: str = "pilot-20260608",
    merged_name: str = "",
    spawn_train: bool = False,
) -> None:
    if merge_only:
        merge_config = build_merge_config(
            dataset_version=dataset_version,
            adapter_name=adapter_name,
            output_name=merged_name,
        )
        merge_result = merge_adapter.with_options(gpu=gpu).remote(merge_config)
        print(json.dumps({"merge": merge_result}, indent=2, sort_keys=True))
        return

    config = build_train_config(
        dataset_version=dataset_version,
        output_name=output_name,
        smoke=smoke,
        max_steps=max_steps,
        max_seq_length=max_seq_length,
    )

    if not skip_stage:
        from scripts.prepare_modal_finetune_dataset import prepare_dataset

        output_dir = Path(prepared_dir) if prepared_dir else Path("data/finetune/modal") / dataset_version
        manifest = prepare_dataset(
            dataset_path=Path(dataset),
            output_dir=output_dir,
            dataset_version=dataset_version,
        )
        stage_result = stage_dataset.remote(
            dataset_version,
            Path(manifest["train_path"]).read_text(encoding="utf-8"),
            Path(manifest["validation_path"]).read_text(encoding="utf-8"),
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        print(json.dumps({"stage_dataset": stage_result}, indent=2, sort_keys=True))

    train_function = train.with_options(gpu=gpu)
    if spawn_train:
        train_call = train_function.spawn(config)
        print(
            json.dumps(
                {
                    "train_spawned": {
                        "function_call_id": train_call.object_id,
                        "dashboard_url": train_call.get_dashboard_url(),
                        "dataset_version": dataset_version,
                        "output_name": output_name,
                        "max_steps": max_steps,
                        "gpu": gpu,
                    }
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    train_result = train_function.remote(config)
    print(json.dumps({"train": train_result}, indent=2, sort_keys=True))


def _tokenized_dataset(rows: list[dict[str, Any]], tokenizer: Any, max_seq_length: int) -> Any:
    from datasets import Dataset

    tokenized = []
    skipped = 0
    for row in rows:
        item = _tokenize_row(row, tokenizer, max_seq_length)
        if item is None:
            skipped += 1
            continue
        tokenized.append(item)
    if skipped:
        print(f"Skipped {skipped} rows with no supervised assistant tokens after truncation")
    return Dataset.from_list(tokenized)


def _tokenize_row(row: dict[str, Any], tokenizer: Any, max_seq_length: int) -> dict[str, list[int]] | None:
    messages = row["messages"]
    prompt_messages = [messages[0]]
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        prompt_text = f"User:\n{messages[0]['content']}\n\nAssistant:\n"
        full_text = f"{prompt_text}{messages[1]['content']}{getattr(tokenizer, 'eos_token', '') or ''}"

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_seq_length:
        cut = len(full_ids) - max_seq_length
        input_ids = full_ids[cut:]
    else:
        cut = 0
        input_ids = full_ids
    labels = list(input_ids)
    mask_until = max(0, min(len(prompt_ids) - cut, len(labels)))
    for index in range(mask_until):
        labels[index] = -100
    if not any(label != -100 for label in labels):
        return None
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _count_jsonl_text(value: str) -> int:
    return sum(1 for line in value.splitlines() if line.strip())


def _safe_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if "token" not in key.lower() and "secret" not in key.lower()}
