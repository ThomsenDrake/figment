import importlib.util
import json
from pathlib import Path

import pytest


def _row(row_id: str, task_type: str, category: str, repair_scope: str | None = None) -> dict:
    metadata = {"task_type": task_type}
    if repair_scope:
        metadata["repair_scope"] = repair_scope
    return {
        "case_id": row_id,
        "uuid": row_id,
        "category": category,
        "messages": [
            {"role": "user", "content": f"prompt {row_id}"},
            {"role": "assistant", "content": '{"protocol_urgency":"routine"}'},
        ],
        "metadata": metadata,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _load_modal_module():
    module_path = Path(__file__).resolve().parents[1] / "modal" / "finetune_figment_nemotron.py"
    spec = importlib.util.spec_from_file_location("figment_modal_finetune", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_modal_dataset_stratifies_and_preserves_rows(tmp_path):
    from scripts.prepare_modal_finetune_dataset import prepare_dataset

    rows = []
    rows.extend(_row(f"full-{index}", "navigator_full", "missing_observation_cues") for index in range(8))
    rows.extend(_row(f"repair-schema-{index}", "focused_repair", "focused_repair:schema", "schema") for index in range(6))
    rows.extend(
        _row(
            f"repair-sbar-{index}",
            "focused_repair",
            "focused_repair:handoff_note_sbar",
            "handoff_note_sbar",
        )
        for index in range(6)
    )
    dataset = tmp_path / "input.jsonl"
    output_dir = tmp_path / "prepared"
    _write_jsonl(dataset, rows)

    manifest = prepare_dataset(
        dataset_path=dataset,
        output_dir=output_dir,
        dataset_version="figment_sft_test",
        validation_fraction=0.2,
        seed="unit-test",
        min_validation_group_size=5,
    )

    train_rows = [json.loads(line) for line in (output_dir / "train.jsonl").read_text().splitlines()]
    validation_rows = [json.loads(line) for line in (output_dir / "validation.jsonl").read_text().splitlines()]
    train_ids = {row["uuid"] for row in train_rows}
    validation_ids = {row["uuid"] for row in validation_rows}

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids | validation_ids == {row["uuid"] for row in rows}
    assert manifest["row_count"] == 20
    assert manifest["train_count"] + manifest["validation_count"] == 20
    assert manifest["validation_group_counts"]["navigator_full:missing_observation_cues"] >= 1
    assert manifest["validation_group_counts"]["focused_repair:schema"] >= 1
    assert manifest["validation_group_counts"]["focused_repair:handoff_note_sbar"] >= 1


def test_prepare_modal_dataset_rejects_rows_outside_chat_shape(tmp_path):
    from scripts.prepare_modal_finetune_dataset import DatasetPrepError
    from scripts.prepare_modal_finetune_dataset import prepare_dataset

    dataset = tmp_path / "bad.jsonl"
    bad_row = _row("bad-1", "navigator_full", "missing_observation_cues")
    bad_row["messages"] = [{"role": "user", "content": "prompt only"}]
    _write_jsonl(dataset, [bad_row])

    with pytest.raises(DatasetPrepError, match="expected user/assistant messages"):
        prepare_dataset(
            dataset_path=dataset,
            output_dir=tmp_path / "prepared",
            dataset_version="figment_sft_test",
        )


def test_modal_smoke_config_is_small_and_namespaced():
    module = _load_modal_module()

    config = module.build_train_config(
        dataset_version="figment_sft_v1",
        output_name="unit",
        smoke=True,
        max_steps=100,
        max_seq_length=12288,
    )
    paths = module.dataset_volume_paths("figment_sft_v1")

    assert config["max_steps"] == 5
    assert config["max_seq_length"] == 2048
    assert config["output_dir"].endswith("/figment_sft_v1/unit-smoke")
    assert paths["train"].endswith("/figment_sft_v1/train.jsonl")
    assert paths["validation"].endswith("/figment_sft_v1/validation.jsonl")


def test_modal_default_config_matches_first_run_plan():
    module = _load_modal_module()

    config = module.build_train_config(dataset_version="figment_sft_v1", output_name="pilot")

    assert config["learning_rate"] == 1e-4
    assert config["max_steps"] == 40
    assert config["max_seq_length"] == 16384
    assert config["gradient_accumulation_steps"] == 8
    assert config["validation_steps"] == 25
    assert config["save_steps"] == 40


def test_modal_v4_config_can_use_lower_lr_and_lora_controls():
    module = _load_modal_module()

    config = module.build_train_config(
        dataset_version="figment_sft_v4",
        output_name="figment-sft-v4-lora",
        max_steps=900,
        learning_rate=2e-5,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        gradient_accumulation_steps=8,
        validation_steps=50,
        save_steps=100,
    )

    assert config["dataset_version"] == "figment_sft_v4"
    assert config["learning_rate"] == 2e-5
    assert config["lora_r"] == 16
    assert config["lora_alpha"] == 32
    assert config["lora_dropout"] == 0.05
    assert config["gradient_accumulation_steps"] == 8
    assert config["validation_steps"] == 50
    assert config["save_steps"] == 100


def test_modal_v5_config_can_resume_from_v4_adapter():
    module = _load_modal_module()

    config = module.build_train_config(
        dataset_version="figment_sft_v5",
        output_name="figment-sft-v5-lora",
        resume_adapter_name="figment-sft-v4-lora",
        resume_adapter_dataset_version="figment_sft_v4",
    )

    assert config["resume_adapter_name"] == "figment-sft-v4-lora"
    assert config["resume_adapter_dataset_version"] == "figment_sft_v4"
    assert config["resume_adapter_dir"] == "/checkpoints/figment_sft_v4/figment-sft-v4-lora"
    assert config["output_dir"] == "/checkpoints/figment_sft_v5/figment-sft-v5-lora"


def test_modal_entrypoint_exposes_v4_training_knobs():
    source = (Path(__file__).resolve().parents[1] / "modal" / "finetune_figment_nemotron.py").read_text(
        encoding="utf-8"
    )

    for parameter in (
        "learning_rate",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "gradient_accumulation_steps",
        "validation_steps",
        "save_steps",
        "resume_adapter_name",
        "resume_adapter_dataset_version",
    ):
        assert f"{parameter}:" in source
        assert f"{parameter}={parameter}" in source or f'"{parameter}": {parameter}' in source


def test_modal_merge_config_points_at_checkpoint_volume():
    module = _load_modal_module()

    config = module.build_merge_config(
        dataset_version="figment_sft_v1",
        adapter_name="pilot-20260608",
    )

    assert config["adapter_dir"] == "/checkpoints/figment_sft_v1/pilot-20260608"
    assert config["output_dir"] == "/checkpoints/figment_sft_v1/pilot-20260608-merged-bf16"
    assert config["output_name"] == "pilot-20260608-merged-bf16"
