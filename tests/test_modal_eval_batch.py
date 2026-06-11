import importlib.util
from pathlib import Path


def _load_modal_eval_module():
    module_path = Path(__file__).resolve().parents[1] / "modal" / "eval_figment_nemotron.py"
    spec = importlib.util.spec_from_file_location("figment_modal_eval", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_modal_eval_config_points_at_v5_merged_checkpoint_and_result_volume():
    module = _load_modal_eval_module()

    config = module.build_eval_config(output_name="unit-v5-eval")

    assert config["checkpoint_model_dir"] == "/checkpoints/figment_sft_v5/figment-sft-v5-lora-merged-bf16"
    assert config["checkpoint_artifact"] == (
        "figment-checkpoints:/figment_sft_v5/figment-sft-v5-lora-merged-bf16"
    )
    assert config["output_dir"] == "/eval_results/unit-v5-eval"
    assert config["gguf_model_path"] == (
        "/eval_results/model_cache/figment_sft_v5/figment-sft-v5-lora-merged-bf16.bf16.gguf"
    )
    assert config["result_volume_name"] == "figment-eval-results"
    assert config["runtime"] == "llama_cpp_cuda"
    assert config["cuda_architectures"] == "90"
    assert config["case_paths"] == ["/tmp/figment_eval_cases/field_workflow_holdout_v1.jsonl"]
    assert config["expected_case_count"] == 150
    assert config["max_generation_tokens"] == 1536
