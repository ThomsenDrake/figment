"""Modal batch eval for Figment's merged Nemotron 4B checkpoint.

Run the full 150-case v5 eval as a terminating Modal job:

    .venv/bin/modal run modal/eval_figment_nemotron.py
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import modal


APP_NAME = "figment-nemotron-4b-batch-eval"
DEFAULT_DATASET_VERSION = "figment_sft_v5"
DEFAULT_MERGED_MODEL_NAME = "figment-sft-v5-lora-merged-bf16"
DEFAULT_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DEFAULT_CASE_PATH = "data/eval/field_workflow_holdout_v1.jsonl"
DEFAULT_CASE_COUNT = 150
DEFAULT_EVAL_GPU = "H100"
DEFAULT_CUDA_ARCHITECTURES = "90"
CHECKPOINT_VOLUME_NAME = "figment-checkpoints"
RESULT_VOLUME_NAME = "figment-eval-results"
CHECKPOINT_DIR = "/checkpoints"
RESULT_DIR = "/eval_results"
LLAMA_CPP_DIR = "/opt/llama.cpp"
LLAMA_SERVER_BIN = f"{LLAMA_CPP_DIR}/build/bin/llama-server"
LLAMA_CONVERT_SCRIPT = f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
MODEL_SERVER_BASE_URL = "http://127.0.0.1:8001/v1"
MODEL_SERVER_HOST = "127.0.0.1"
MODEL_SERVER_PORT = 8001


app = modal.App(APP_NAME)

checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)
result_volume = modal.Volume.from_name(RESULT_VOLUME_NAME, create_if_missing=True)

eval_image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel")
    .apt_install("git", "build-essential", "cmake", "curl", "ninja-build")
    .uv_pip_install(
        "accelerate>=1.8,<2",
        "einops>=0.8,<1",
        "ninja>=1.11,<2",
        "protobuf>=5,<7",
        "safetensors>=0.5,<1",
        "sentencepiece>=0.2,<1",
        "setuptools>=70",
        "torch==2.7.1",
        "transformers>=4.52,<5",
        "wheel>=0.45",
    )
    .add_local_dir("tools/llama.cpp", LLAMA_CPP_DIR, copy=True)
    .run_commands(
        f"python -m pip install -r {LLAMA_CPP_DIR}/requirements.txt",
        f"cmake -S {LLAMA_CPP_DIR} -B {LLAMA_CPP_DIR}/build "
        f"-DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release "
        f"-DCMAKE_CUDA_ARCHITECTURES={DEFAULT_CUDA_ARCHITECTURES}",
        f"cmake --build {LLAMA_CPP_DIR}/build --config Release --target llama-server -j $(nproc)",
    )
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHON_DOTENV_DISABLED": "true",
        }
    )
    .add_local_python_source("figment", "scripts")
)


def build_eval_config(
    *,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    merged_model_name: str = DEFAULT_MERGED_MODEL_NAME,
    output_name: str = "",
    model_id: str = DEFAULT_MODEL_ID,
    timeout_seconds: float = 300.0,
    expected_case_count: int = DEFAULT_CASE_COUNT,
    max_context_tokens: int = 16384,
    max_generation_tokens: int = 1536,
) -> dict[str, Any]:
    if not output_name:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_name = f"{dataset_version}_field_workflow_holdout_modal_gpu_{stamp}"
    checkpoint_model_dir = f"{CHECKPOINT_DIR}/{dataset_version}/{merged_model_name}"
    return {
        "dataset_version": dataset_version,
        "merged_model_name": merged_model_name,
        "checkpoint_model_dir": checkpoint_model_dir,
        "checkpoint_artifact": (
            f"{CHECKPOINT_VOLUME_NAME}:/{dataset_version}/{merged_model_name}"
        ),
        "result_volume_name": RESULT_VOLUME_NAME,
        "output_name": output_name,
        "output_dir": f"{RESULT_DIR}/{output_name}",
        "gguf_model_path": f"{RESULT_DIR}/model_cache/{dataset_version}/{merged_model_name}.bf16.gguf",
        "runtime": "llama_cpp_cuda",
        "cuda_architectures": DEFAULT_CUDA_ARCHITECTURES,
        "model_id": model_id,
        "base_url": MODEL_SERVER_BASE_URL,
        "case_paths": [f"/tmp/figment_eval_cases/{Path(DEFAULT_CASE_PATH).name}"],
        "timeout_seconds": timeout_seconds,
        "expected_case_count": expected_case_count,
        "max_context_tokens": max_context_tokens,
        "max_generation_tokens": max_generation_tokens,
    }


@app.function(
    image=eval_image,
    gpu=DEFAULT_EVAL_GPU,
    cpu=8,
    memory=131072,
    ephemeral_disk=524288,
    volumes={
        CHECKPOINT_DIR: checkpoint_volume,
        RESULT_DIR: result_volume,
    },
    timeout=6 * 60 * 60,
)
def run_batch_eval(
    config: dict[str, Any],
    case_files: list[dict[str, str]],
    protocol_cards: list[dict[str, str]],
) -> dict[str, Any]:
    import os
    import time

    import torch

    from scripts.run_local_4b_evidence import run_evidence

    started = datetime.now(UTC)
    model_dir = Path(str(config["checkpoint_model_dir"]))
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"missing merged model config at {model_dir}")
    if not (model_dir / "model.safetensors.index.json").exists():
        raise FileNotFoundError(f"missing merged model index at {model_dir}")

    staged_case_paths = _stage_case_files(case_files)
    _stage_protocol_cards(protocol_cards)

    runtime_gpu = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    print(json.dumps({"runtime_gpu": runtime_gpu}, sort_keys=True), flush=True)

    gguf_model_path = _ensure_gguf_model(model_dir=model_dir, config=config)
    result_volume.commit()

    server = _LlamaCppServer(
        gguf_model_path=gguf_model_path,
        model_id=str(config["model_id"]),
        max_context_tokens=int(config["max_context_tokens"]),
    )
    server.start()
    try:
        summary = run_evidence(
            base_url=str(config["base_url"]),
            model_id=str(config["model_id"]),
            output_dir=output_dir,
            case_paths=staged_case_paths,
            limit=None,
            timeout_seconds=float(config["timeout_seconds"]),
            force_eval=True,
        )
    finally:
        server.stop()

    records_path = output_dir / "local_4b_eval.jsonl"
    record_count = _count_jsonl(records_path)
    finished = datetime.now(UTC)
    manifest = {
        "status": "completed" if record_count == int(config["expected_case_count"]) else "count_mismatch",
        "config": config,
        "runtime_gpu": runtime_gpu,
        "checkpoint_artifact": config["checkpoint_artifact"],
        "output_volume": RESULT_VOLUME_NAME,
        "output_dir": str(output_dir),
        "records_path": str(records_path),
        "record_count": record_count,
        "expected_case_count": int(config["expected_case_count"]),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "runtime_seconds": round(time.monotonic() - server.started_monotonic, 3),
        "summary": summary,
        "files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "modal_eval_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result_volume.commit()

    if record_count != int(config["expected_case_count"]):
        raise ValueError(f"expected {config['expected_case_count']} records, found {record_count}")
    return manifest


def _ensure_gguf_model(*, model_dir: Path, config: dict[str, Any]) -> Path:
    import subprocess
    import sys

    gguf_path = Path(str(config["gguf_model_path"]))
    if gguf_path.exists() and gguf_path.stat().st_size > 0:
        print(
            json.dumps(
                {
                    "modal_eval_gguf": "reuse_existing",
                    "path": str(gguf_path),
                    "bytes": gguf_path.stat().st_size,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return gguf_path

    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        LLAMA_CONVERT_SCRIPT,
        str(model_dir),
        "--outfile",
        str(gguf_path),
        "--outtype",
        "bf16",
    ]
    print(json.dumps({"modal_eval_gguf": "convert_start", "command": command}, sort_keys=True), flush=True)
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed with exit code {completed.returncode}")
    print(
        json.dumps(
            {
                "modal_eval_gguf": "convert_finished",
                "path": str(gguf_path),
                "bytes": gguf_path.stat().st_size,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return gguf_path


class _LlamaCppServer:
    def __init__(self, *, gguf_model_path: Path, model_id: str, max_context_tokens: int) -> None:
        self.gguf_model_path = gguf_model_path
        self.model_id = model_id
        self.max_context_tokens = max_context_tokens
        self.process: Any = None
        self.started_monotonic = 0.0

    def start(self) -> None:
        import subprocess
        import threading
        import time
        import urllib.request

        command = [
            LLAMA_SERVER_BIN,
            "--model",
            str(self.gguf_model_path),
            "--ctx-size",
            str(self.max_context_tokens),
            "--host",
            MODEL_SERVER_HOST,
            "--port",
            str(MODEL_SERVER_PORT),
            "--alias",
            self.model_id,
            "--parallel",
            "1",
            "--temp",
            "0",
            "--top-p",
            "1",
            "--reasoning",
            "off",
            "--n-gpu-layers",
            "999",
        ]
        print(json.dumps({"modal_eval_llama_server": "start", "command": command}, sort_keys=True), flush=True)
        self.started_monotonic = time.monotonic()
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._stream_logs, daemon=True).start()

        deadline = time.monotonic() + 180
        last_error = ""
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited early with code {self.process.returncode}")
            try:
                with urllib.request.urlopen(f"{MODEL_SERVER_BASE_URL}/models", timeout=2) as response:
                    if response.status == 200:
                        print(json.dumps({"modal_eval_llama_server": "ready"}, sort_keys=True), flush=True)
                        return
            except OSError as exc:
                last_error = str(exc)
                time.sleep(1)
        raise RuntimeError(f"llama-server did not become ready: {last_error}")

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except Exception:
            self.process.kill()
            self.process.wait(timeout=10)

    def _stream_logs(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            print(f"[llama-server] {line.rstrip()}", flush=True)


class _OpenAICompatibleTransformersServer:
    def __init__(
        self,
        *,
        model_dir: Path,
        model_id: str,
        max_context_tokens: int,
        max_generation_tokens: int,
    ) -> None:
        self.model_dir = model_dir
        self.model_id = model_id
        self.max_context_tokens = max_context_tokens
        self.max_generation_tokens = max_generation_tokens
        self.httpd: Any = None
        self.thread: Any = None
        self.started_monotonic = 0.0

    def start(self) -> None:
        import threading
        import time
        import urllib.request
        from http.server import ThreadingHTTPServer

        model_runtime = _TransformersRuntime(
            model_dir=self.model_dir,
            model_id=self.model_id,
            max_context_tokens=self.max_context_tokens,
            max_generation_tokens=self.max_generation_tokens,
        )
        handler = _handler_factory(model_runtime)
        self.httpd = ThreadingHTTPServer((MODEL_SERVER_HOST, MODEL_SERVER_PORT), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.started_monotonic = time.monotonic()
        self.thread.start()
        deadline = time.monotonic() + 30
        last_error = ""
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{MODEL_SERVER_BASE_URL}/models", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError as exc:
                last_error = str(exc)
                time.sleep(0.25)
        raise RuntimeError(f"model server did not become ready: {last_error}")

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=10)


class _TransformersRuntime:
    def __init__(
        self,
        *,
        model_dir: Path,
        model_id: str,
        max_context_tokens: int,
        max_generation_tokens: int,
    ) -> None:
        import threading

        import torch
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.max_context_tokens = max_context_tokens
        self.max_generation_tokens = max_generation_tokens
        self.lock = threading.Lock()
        self.request_count = 0
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
        )
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.model.eval()
        self.model.config.use_cache = True

    def models_payload(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_id,
                    "object": "model",
                    "owned_by": "figment-modal-batch",
                }
            ],
        }

    def chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        import time
        import uuid

        messages = body.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        requested_max_tokens = int(body.get("max_tokens") or 1024)
        with self.lock:
            self.request_count += 1
            request_index = self.request_count
            started = time.monotonic()
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            input_len = int(input_ids.shape[-1])
            available_tokens = max(128, self.max_context_tokens - input_len - 8)
            max_new_tokens = max(1, min(requested_max_tokens, available_tokens, self.max_generation_tokens))
            print(
                json.dumps(
                    {
                        "modal_eval_request": request_index,
                        "status": "started",
                        "prompt_tokens": input_len,
                        "max_new_tokens": max_new_tokens,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            attention_mask = self.torch.ones_like(input_ids)
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated_ids = output_ids[0, input_len:]
            content = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            print(
                json.dumps(
                    {
                        "modal_eval_request": request_index,
                        "status": "finished",
                        "completion_tokens": int(generated_ids.numel()),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": int(generated_ids.numel()),
                "total_tokens": input_len + int(generated_ids.numel()),
            },
        }


def _handler_factory(runtime: _TransformersRuntime) -> Any:
    import json
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/v1/models":
                self._send_json(200, runtime.models_payload())
            else:
                self._send_json(404, {"error": f"unknown path: {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send_json(404, {"error": f"unknown path: {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                payload = runtime.chat_completion(body)
            except Exception as exc:  # pragma: no cover - exercised inside Modal runtime.
                self._send_json(500, {"error": str(exc)})
                return
            self._send_json(200, payload)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def _stage_case_files(case_files: list[dict[str, str]]) -> list[Path]:
    case_dir = Path("/tmp/figment_eval_cases")
    case_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for item in case_files:
        path = case_dir / Path(item["name"]).name
        path.write_text(item["text"], encoding="utf-8")
        paths.append(path)
    return paths


def _stage_protocol_cards(protocol_cards: list[dict[str, str]]) -> None:
    from figment import retrieval

    card_dir = Path(retrieval.DEFAULT_CARD_DIR)
    card_dir.mkdir(parents=True, exist_ok=True)
    for path in card_dir.glob("*.json"):
        path.unlink()
    for item in protocol_cards:
        (card_dir / Path(item["name"]).name).write_text(item["text"], encoding="utf-8")


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _read_case_files(paths: list[str]) -> list[dict[str, str]]:
    return [
        {"name": Path(path).name, "text": Path(path).read_text(encoding="utf-8")}
        for path in paths
    ]


def _read_protocol_cards() -> list[dict[str, str]]:
    return [
        {"name": path.name, "text": path.read_text(encoding="utf-8")}
        for path in sorted(Path("data/protocol_cards").glob("*.json"))
    ]


@app.local_entrypoint()
def main(
    output_name: str = "",
    dataset_version: str = DEFAULT_DATASET_VERSION,
    merged_model_name: str = DEFAULT_MERGED_MODEL_NAME,
    cases: str = DEFAULT_CASE_PATH,
    timeout_seconds: float = 300.0,
    gpu: str = DEFAULT_EVAL_GPU,
) -> None:
    case_paths = [item.strip() for item in cases.split(",") if item.strip()]
    config = build_eval_config(
        dataset_version=dataset_version,
        merged_model_name=merged_model_name,
        output_name=output_name,
        timeout_seconds=timeout_seconds,
    )
    if len(case_paths) != 1 or Path(case_paths[0]).name != Path(DEFAULT_CASE_PATH).name:
        config["case_paths"] = [f"/tmp/figment_eval_cases/{Path(path).name}" for path in case_paths]
        config["expected_case_count"] = sum(
            1 for path in case_paths for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
        )

    result = run_batch_eval.with_options(gpu=gpu).remote(
        config,
        _read_case_files(case_paths),
        _read_protocol_cards(),
    )
    print(json.dumps({"modal_batch_eval": result}, indent=2, sort_keys=True))
