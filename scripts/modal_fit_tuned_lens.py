"""Guarded Modal runner for the Qwen3.6-27B tuned-lens fit.

This file defines the remote job but intentionally requires an explicit
``--confirm-budget 5`` argument before submitting any GPU work.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import modal

GPU_RATE_PER_HOUR = 0.694
MEMORY_RATE_PER_GIB_HOUR = 0.00000222 * 3600
CPU_RATE_PER_CORE_HOUR = 0.0000131 * 3600
GPU = "A100-80GB"
GPU_COUNT = 1
CPU_CORES = 4
MEMORY_GIB = 96
STARTUP_SECONDS = 20 * 60
TIMEOUT_SECONDS = 90 * 60
MAX_BUDGET_USD = 5.0
SAFETY_MARGIN_USD = 0.50


def worst_case_cost(timeout_seconds: int = TIMEOUT_SECONDS, startup_seconds: int = STARTUP_SECONDS) -> float:
    """Conservative resource-request estimate, including startup time."""
    hours = (timeout_seconds + startup_seconds) / 3600
    return hours * (
        GPU_COUNT * GPU_RATE_PER_HOUR
        + MEMORY_GIB * MEMORY_RATE_PER_GIB_HOUR
        + CPU_CORES * CPU_RATE_PER_CORE_HOUR
    )


def validate_budget(confirmed: float, *, max_budget: float = MAX_BUDGET_USD) -> None:
    estimate = worst_case_cost()
    if abs(confirmed - max_budget) > 1e-6:
        raise ValueError(f"refusing to submit: pass --confirm-budget {max_budget:.0f} exactly")
    if estimate + SAFETY_MARGIN_USD >= max_budget:
        raise ValueError(f"configured worst-case estimate ${estimate:.2f} is too close to ${max_budget:.2f} cap")


def validate_artifact(output_path: Path, *, model_id: str, revision: str | None, expected_precision: str = "4bit") -> None:
    """Check the files and metadata needed for a portable tuned-lens artifact."""
    required = [output_path / "config.json", output_path / "params.pt", output_path / "fit_manifest.json"]
    missing = [str(path.name) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"fit produced an incomplete artifact; missing {missing}")
    config = json.loads((output_path / "config.json").read_text())
    manifest = json.loads((output_path / "fit_manifest.json").read_text())
    if config.get("base_model_name_or_path") != model_id or manifest.get("model") != model_id:
        raise RuntimeError("artifact model metadata does not match the requested canonical model ID")
    if revision and config.get("base_model_revision") != revision:
        raise RuntimeError("artifact config does not record the immutable base-model revision")
    if revision and manifest.get("model_revision") != revision:
        raise RuntimeError("artifact manifest does not record the immutable base-model revision")
    expected_quantization = {"4bit": "nf4", "8bit": "int8", "bf16": "none"}[expected_precision]
    actual_quantization = manifest.get("quantization")
    if actual_quantization is None and manifest.get("load_in_4bit") is True:
        actual_quantization = "nf4"  # compatibility with pre-precision manifests
    if actual_quantization != expected_quantization:
        raise RuntimeError("artifact manifest precision does not match the requested fit")
    checkpoint = __import__("torch").load(output_path / "params.pt", map_location="cpu", weights_only=True)
    tensors = [value for value in checkpoint.values() if hasattr(value, "numel")] if isinstance(checkpoint, dict) else []
    if not tensors or any(not __import__("torch").isfinite(value).all().item() for value in tensors):
        raise RuntimeError("artifact params.pt is empty or contains non-finite tensors")


def model_cache_complete(model_path: Path, *, revision: str | None = None) -> bool:
    """Return true only when metadata and at least one complete weight set exist."""
    if not (model_path / "config.json").is_file():
        return False
    if revision:
        try:
            if (model_path / ".jlens_revision").read_text().strip() != revision:
                return False
        except OSError:
            return False
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            weight_names = set(json.loads(index_path.read_text())["weight_map"].values())
        except (KeyError, json.JSONDecodeError, OSError):
            return False
        return bool(weight_names) and all(
            (model_path / name).is_file() and (model_path / name).stat().st_size > 0
            for name in weight_names
        )
    return any(
        path.is_file() and path.stat().st_size > 0
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for path in model_path.glob(pattern)
    )


def find_model_cache(model_root: Path, model_id: str, revision: str) -> Path | None:
    """Find a complete cache, including the directory-upload layout."""
    cache_root = model_root / model_id.replace("/", "--")
    # The local fallback was uploaded with `modal volume put`, whose directory
    # semantics created cache_root/qwen3.6-27b-hf/. Keep this compatibility
    # candidate until the Volume is deliberately normalized or replaced.
    candidates = [cache_root, cache_root / "qwen3.6-27b-hf"]
    return next(
        (path for path in candidates if model_cache_complete(path, revision=revision)),
        None,
    )


ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "jlens-qwen36-27b-tuned-fit"
MODEL_VOLUME_NAME = "jlens-qwen36-model-cache"
OUTPUT_VOLUME_NAME = "jlens-qwen36-tuned-artifacts"
HF_SECRET_NAME = "huggingface-secret"

# Keep Modal on the same maintained NVIDIA runtime that passed the local
# precision tests. The image supplies a coherent Torch/CUDA/compiler stack;
# do not recreate that stack with separately pinned Torch wheels.
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:25.11-py3",
    )
    .pip_install(
        "transformers",
        "datasets",
        "tuned-lens==0.2.0",
        "bitsandbytes==0.49.2",
        "accelerate",
        "huggingface-hub",
        "flash-linear-attention==0.5.1",
        "packaging",
        "ninja",
    )
    # causal-conv1d is a source package. Reuse the NGC image's installed
    # Torch/CUDA toolchain instead of creating a second isolated build env.
    .run_commands("pip install --no-build-isolation causal-conv1d==1.6.2.post1")
    .add_local_file(ROOT / "scripts" / "fit_tuned_lens_local.py", "/root/fit_tuned_lens_local.py")
)
app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)


@app.function(
    image=image,
    cpu=2,
    memory=16 * 1024,
    timeout=2 * 60 * 60,
    startup_timeout=STARTUP_SECONDS,
    retries=0,
    max_containers=1,
    secrets=[hf_secret],
    volumes={"/models": model_volume, "/outputs": output_volume},
)
def cache_model_remote(model_id: str = "Qwen/Qwen3.6-27B") -> str:
    """Download and commit the model without allocating the expensive GPU."""
    import os

    # Xet is the fast path for large Hub files. Keep high-performance range
    # requests enabled; the Hub version is pinned below to the locally tested
    # client version after 1.23.x produced a CAS 401 in Modal.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("refusing to download anonymously: Modal secret must provide HF_TOKEN")
    revision = HfApi(token=token).model_info(model_id).sha
    model_path = find_model_cache(Path("/models"), model_id, revision)
    if model_path is not None:
        print(f"[download] using cached model at {model_path}", flush=True)
        return str(model_path)
    model_path = Path("/models") / model_id.replace("/", "--")
    print(f"[download] caching {model_id} revision {revision} in {model_path}", flush=True)
    snapshot_download(
        model_id,
        revision=revision,
        local_dir=str(model_path),
        token=token,
        max_workers=16,
    )
    if not model_cache_complete(model_path):
        raise RuntimeError("Hugging Face download completed without a complete weight set")
    (model_path / ".jlens_revision").write_text(revision + "\n")
    model_volume.commit()
    print(f"[download] committed model cache at {model_path}", flush=True)
    return str(model_path)


@app.function(
    image=image,
    cpu=2,
    memory=4 * 1024,
    timeout=10 * 60,
    startup_timeout=STARTUP_SECONDS,
    retries=0,
    max_containers=1,
    secrets=[hf_secret],
    volumes={"/models": model_volume},
)
def cache_wikitext_remote() -> str:
    """Materialize Wikitext on the volume before any GPU function starts."""
    import os
    import time

    from datasets import load_dataset

    cache_dir = "/models/hf-datasets"
    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
    started = time.monotonic()
    print(f"[dataset] loading Wikitext into {cache_dir}", flush=True)
    data = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train",
        cache_dir=cache_dir,
    )
    print(f"[dataset] loaded {len(data)} rows in {time.monotonic() - started:.1f}s", flush=True)
    model_volume.commit()
    print("[dataset] committed persistent cache", flush=True)
    return cache_dir


@app.function(
    image=image,
    gpu=GPU,
    cpu=CPU_CORES,
    memory=MEMORY_GIB * 1024,
    timeout=TIMEOUT_SECONDS,
    startup_timeout=STARTUP_SECONDS,
    retries=0,
    max_containers=1,
    secrets=[hf_secret],
    volumes={"/models": model_volume, "/outputs": output_volume},
)
def fit_remote(
    model_id: str = "Qwen/Qwen3.6-27B",
    steps: int = 100,
    max_chunks: int = 512,
    max_length: int = 128,
    output_name: str = "qwen3.6-27b-tuned-wikitext-nf4",
    precision: str = "4bit",
) -> str:
    """Download/cache the model and fit the standard dense tuned translators."""
    import os

    import torch
    from huggingface_hub import HfApi

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in Modal container")
    if precision not in {"4bit", "8bit", "bf16"}:
        raise ValueError("precision must be one of: 4bit, 8bit, bf16")
    free, total = torch.cuda.mem_get_info()
    print(f"[preflight] GPU free={free / 2**30:.2f} GiB total={total / 2**30:.2f} GiB", flush=True)
    if free / 2**30 < 60:
        raise RuntimeError("refusing to fit: expected at least 60 GiB free before model load")
    revision = HfApi().model_info(model_id).sha
    model_path = find_model_cache(Path("/models"), model_id, revision)
    if model_path is None:
        raise RuntimeError("model cache is incomplete or revision-mismatched; cache step must succeed first")
    print(f"[download] using cached model at {model_path}", flush=True)
    os.environ.setdefault("HF_DATASETS_CACHE", "/models/hf-datasets")

    output_path = Path("/outputs") / output_name
    partial_path = Path("/outputs") / f".{output_name}.partial"
    if output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing output artifact: {output_path}")
    if partial_path.exists():
        print(f"[fit] removing stale partial output {partial_path}", flush=True)
        shutil.rmtree(partial_path)
    partial_path.mkdir(parents=True, exist_ok=True)
    precision_args = {
        "4bit": ["--load-in-4bit"],
        "8bit": ["--load-in-8bit"],
        "bf16": ["--dtype", "bf16"],
    }[precision]
    command = [
        "python",
        "/root/fit_tuned_lens_local.py",
        "--model",
        str(model_path),
        "--base-model-id",
        model_id,
        "--base-model-revision",
        revision,
        "--out",
        str(partial_path),
        "--steps",
        str(steps),
        "--max-chunks",
        str(max_chunks),
        "--max-length",
        str(max_length),
        *precision_args,
        "--min-free-gib",
        "6",
        "--require-fast-kernels",
        "--events-out",
        str(partial_path / "run-events.jsonl"),
    ]
    print("[fit] starting guarded trainer", flush=True)
    subprocess.run(command, check=True)
    validate_artifact(partial_path, model_id=model_id, revision=revision, expected_precision=precision)
    partial_path.rename(output_path)
    output_volume.commit()
    print(f"[complete] committed {output_path}", flush=True)
    return str(output_path)


@app.local_entrypoint()
def main(
    model_id: str = "Qwen/Qwen3.6-27B",
    confirm_budget: float = 0,
    cache_only: bool = False,
    image_only: bool = False,
    steps: int = 100,
    max_chunks: int = 512,
    max_length: int = 128,
    output_name: str = "qwen3.6-27b-tuned-wikitext-nf4",
    precision: str = "4bit",
) -> None:
    validate_budget(confirm_budget)
    estimate = worst_case_cost()
    print(f"[budget] worst-case configured estimate: ${estimate:.2f} (cap ${MAX_BUDGET_USD:.2f})")
    if image_only:
        print("[preflight] image-only mode: image build/import validation complete; no cache, GPU, or model work submitted")
        return
    if cache_only:
        print("[budget] cache-only resources: CPU=2, memory=16 GiB, no GPU, no fit allocation")
    else:
        print(f"[budget] GPU={GPU}, memory={MEMORY_GIB} GiB, precision={precision}, timeout={TIMEOUT_SECONDS // 3600}h, retries=0, max_containers=1")
    print("[submit] explicit budget confirmation accepted; caching model before allocating GPU")
    cache_result = cache_model_remote.remote(model_id=model_id)
    print(f"[cache] {cache_result}")
    dataset_result = cache_wikitext_remote.remote()
    print(f"[dataset-cache] {dataset_result}")
    if cache_only:
        print("[complete] cache-only run finished; no GPU fit submitted")
        return
    print("[submit] starting one remote GPU fit")
    result = fit_remote.remote(model_id=model_id, steps=steps, max_chunks=max_chunks, output_name=output_name, max_length=max_length, precision=precision)
    print(f"[result] {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check-budget", action="store_true")
    args, _ = parser.parse_known_args()
    if args.check_budget:
        print(f"worst_case_cost=${worst_case_cost():.2f}")
    else:
        print("Use `modal run scripts/modal_fit_tuned_lens.py --confirm-budget 5`; no job submitted by direct Python execution.")
