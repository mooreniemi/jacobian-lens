"""Evaluate the model-matched 27B tuned lens on the existing causal fixtures.

This deliberately runs only the missing tuned rows. The already-computed
J/logit/random rows remain the canonical baseline artifacts and are merged
locally after this job is validated.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL_VOLUME_NAME = "jlens-qwen36-model-cache"
OUTPUT_VOLUME_NAME = "jlens-qwen36-tuned-artifacts"
HF_SECRET_NAME = "huggingface-secret"
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_PATH = "/models/Qwen--Qwen3.6-27B"
TUNED_DIR = "/outputs/qwen3.6-27b-tuned-wikitext-nf4"
BASIS_PATH = "/outputs/qwen3.6-27b-tuned-wikitext-basis.pt"

base_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.11-py3")
    .pip_install(
        "transformers",
        "tuned-lens==0.2.0",
        "bitsandbytes==0.49.2",
        "accelerate",
        "huggingface-hub",
        "flash-linear-attention==0.5.1",
        "packaging",
        "ninja",
    )
    .run_commands("pip install --no-build-isolation causal-conv1d==1.6.2.post1")
    .add_local_dir(ROOT / "jlens", "/root/repo/jlens")
)
for name in (
    "causal_runtime.py",
    "eval_verbal_report_causal.py",
    "eval_multihop_causal.py",
    "eval_flexible_causal.py",
    "export_tuned_patch_basis.py",
):
    base_image = base_image.add_local_file(ROOT / "scripts" / name, f"/root/repo/scripts/{name}")
for name in (
    "verbal-report.json",
    "probe-swap.json",
    "flexible-generalization.json",
):
    base_image = base_image.add_local_file(ROOT / "data" / "experiments" / name, f"/root/repo/data/experiments/{name}")

app = modal.App("jlens-qwen36-27b-tuned-eval")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)


@app.function(
    image=base_image,
    gpu="A100-80GB",
    cpu=4,
    memory=96 * 1024,
    timeout=45 * 60,
    startup_timeout=20 * 60,
    retries=0,
    max_containers=1,
    secrets=[hf_secret],
    volumes={"/models": model_volume, "/outputs": output_volume},
)
def evaluate() -> dict[str, object]:
    import os

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    free, total = torch.cuda.mem_get_info()
    print(f"[preflight] GPU free={free / 2**30:.2f} GiB total={total / 2**30:.2f} GiB", flush=True)
    if free / 2**30 < 60:
        raise RuntimeError("refusing evaluation: insufficient free VRAM")
    # The first upload used `modal volume put` on a directory, which left a
    # wrapper directory alongside parent metadata. Prefer the nested path;
    # the parent may have a config file but incomplete/zero-byte shards.
    candidates = (Path(MODEL_PATH) / "qwen3.6-27b-hf", Path(MODEL_PATH))
    model_path = next(
        (p for p in candidates if (p / "config.json").exists() and any((p).glob("*.safetensors"))),
        None,
    )
    if model_path is None:
        raise RuntimeError("model cache not found")
    tuned_dir = Path(TUNED_DIR)
    if not (tuned_dir / "params.pt").exists():
        raise RuntimeError("tuned lens artifact not found")
    basis = Path(BASIS_PATH)
    subprocess_env = {**os.environ, "PYTHONPATH": "/root/repo"}
    if basis.exists():
        print(f"[basis] reusing {basis}", flush=True)
    else:
        subprocess.run([
            "python", "/root/repo/scripts/export_tuned_patch_basis.py",
            "--model", MODEL_ID, "--tuned-dir", str(tuned_dir), "--out", str(basis),
        ], check=True, env=subprocess_env)
        output_volume.commit()
    common = [
        "--model", str(model_path), "--lens-local", str(basis),
        "--layers", "16,32,60", "--methods", "tuned", "--load-in-4bit",
        "--qwen-kernels", "on", "--min-free-gib", "6",
    ]
    jobs = [
        ("verbal", ["python", "/root/repo/scripts/eval_verbal_report_causal.py", *common,
                    "--data", "/root/repo/data/experiments/verbal-report.json",
                    "--categories", "country", "color", "fruit", "sport", "instrument", "planet", "tree", "bird", "language", "profession", "beverage", "organ", "city", "river",
                    "--patch-positions", "all", "--out", "/outputs/verbal-report-causal-27b-tuned.json"]),
        ("multihop", ["python", "/root/repo/scripts/eval_multihop_causal.py", *common,
                      "--data", "/root/repo/data/experiments/probe-swap.json",
                      "--patch-positions", "all", "--out", "/outputs/multihop-causal-27b-tuned.json"]),
        ("flexible", ["python", "/root/repo/scripts/eval_flexible_causal.py", *common,
                      "--data", "/root/repo/data/experiments/flexible-generalization.json",
                      "--out", "/outputs/flexible-causal-27b-tuned.json"]),
    ]
    for name, command in jobs:
        print(f"[eval] starting {name}", flush=True)
        subprocess.run(command, check=True, cwd="/root/repo", env=subprocess_env)
        print(f"[eval] completed {name}", flush=True)
    output_volume.commit()
    return {"basis": str(basis), "outputs": [name for name, _ in jobs]}


@app.local_entrypoint()
def main() -> None:
    print(evaluate.remote())
