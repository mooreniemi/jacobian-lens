"""Fit a TunedLens affine translator without tuned-lens' obsolete DataLoader2.

This mirrors tuned-lens' reference objective: each layer gets a residual affine
translator, trained against the frozen model's final-token distribution with KL
loss. The corpus can be Wikitext (the original pilot) or local JSONL from the
Pile validation split (the reproduction path); the model and tokenizer are
always loaded from the same path passed on the command line.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import resource
import subprocess
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tuned_lens import TunedLens, model_surgery

EVENTS_OUT: Path | None = None
RUN_STARTED = time.monotonic()


def optional_package_version(package: str) -> str | None:
    """Return a package version without making optional dependencies mandatory."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(package)
    except PackageNotFoundError:
        return None


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.monotonic() - RUN_STARTED:8.1f}s] {message}", flush=True)


def event(name: str, **fields: object) -> None:
    payload = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.monotonic() - RUN_STARTED, 3),
        "event": name,
        **fields,
    }
    print("[event] " + json.dumps(payload, sort_keys=True), flush=True)
    if EVENTS_OUT is not None:
        with EVENTS_OUT.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def system_metrics() -> dict[str, float]:
    """Return cheap process/GPU telemetry for durable fit events."""
    metrics = {
        "process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "process_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
    }
    try:
        row = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=1).strip().split(",")
        metrics["gpu_utilization_pct"] = float(row[0])
        metrics["gpu_memory_used_gib"] = float(row[1]) / 1024
        metrics["gpu_memory_total_gib"] = float(row[2]) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return {key: round(value, 3) for key, value in metrics.items()}


def make_chunks(
    tokenizer,
    *,
    split: str,
    max_length: int,
    max_chunks: int,
    data_path: str | None,
    dataset_id: str,
    dataset_config: str,
    text_field: str,
) -> list[torch.Tensor]:
    if data_path:
        log(f"loading local JSONL corpus path={data_path}")
        data = load_dataset("json", data_files={split: data_path}, split=split)
    else:
        log(f"loading dataset={dataset_id} config={dataset_config} split={split}")
        data = load_dataset(dataset_id, dataset_config, split=split)
    texts = [x[text_field] for x in data if x[text_field].strip()]
    target_tokens = max_chunks * max_length
    eos_id = tokenizer.eos_token_id
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if eos_id is not None:
            ids.append(eos_id)
        if len(ids) >= target_tokens:
            break
    usable = (len(ids) // max_length) * max_length
    chunks = [torch.tensor(ids[i : i + max_length], dtype=torch.long) for i in range(0, usable, max_length)]
    if len(chunks) < max_chunks:
        raise RuntimeError(f"Wikitext supplied only {len(chunks)} chunks, need {max_chunks}")
    log(f"prepared {len(chunks)} chunks; using {max_chunks}")
    return chunks[:max_chunks]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-model-id", default=None, help="canonical model ID to record in the artifact")
    ap.add_argument("--base-model-revision", default=None, help="immutable model revision to record")
    ap.add_argument("--out", required=True, help="tuned-lens directory")
    ap.add_argument("--artifact-name", default=None, help="stable artifact label recorded in the manifest")
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-path", default=None, help="local JSONL corpus; overrides --dataset-id")
    ap.add_argument("--dataset-id", default="Salesforce/wikitext")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--dataset-label", default=None, help="stable label recorded in the manifest")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-chunks", type=int, default=512)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-free-gib", type=float, default=8.0)
    quant_group = ap.add_mutually_exclusive_group()
    quant_group.add_argument("--load-in-4bit", action="store_true", help="quantize the frozen base model with NF4")
    quant_group.add_argument("--load-in-8bit", action="store_true", help="quantize the frozen base model with 8-bit BitsAndBytes")
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16", help="dtype for the frozen unquantized model and lens compute")
    ap.add_argument("--require-fast-kernels", action="store_true", help="fail instead of using Qwen linear-attention fallbacks")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--events-out", default=None, help="optional JSONL path for durable phase/step events")
    args = ap.parse_args()
    global EVENTS_OUT
    EVENTS_OUT = Path(args.events_out) if args.events_out else None
    if EVENTS_OUT is not None:
        EVENTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.require_fast_kernels:
        try:
            import causal_conv1d
            import fla
        except ImportError as exc:
            raise SystemExit("fast Qwen linear-attention kernels unavailable; install flash-linear-attention and causal-conv1d") from exc
        log(f"fast linear-attention kernels enabled: fla={getattr(fla, '__version__', 'unknown')} causal_conv1d={getattr(causal_conv1d, '__version__', 'unknown')}")
        event("fast_kernels", fla_version=getattr(fla, "__version__", "unknown"), causal_conv1d_version=getattr(causal_conv1d, "__version__", "unknown"))
    free, total = torch.cuda.mem_get_info()
    log(f"preflight: {free / 2**30:.2f} GiB free / {total / 2**30:.2f} GiB total")
    event("preflight", free_gib=round(free / 2**30, 3), total_gib=round(total / 2**30, 3), model=args.base_model_id or args.model)
    if free / 2**30 < args.min_free_gib:
        raise SystemExit(f"refusing to start: need {args.min_free_gib:.1f} GiB free")

    log(f"loading matched model {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    load_kwargs = {"dtype": dtype, "low_cpu_mem_usage": True}
    if args.load_in_4bit:
        load_kwargs.update(
            {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ),
                "device_map": "auto",
            }
        )
    elif args.load_in_8bit:
        load_kwargs.update(
            {
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                "device_map": "auto",
            }
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    # BitsAndBytes 4-bit and 8-bit models are already placed by
    # ``device_map=auto``; calling ``.cuda()`` on either is unsupported.
    if not args.load_in_4bit and not args.load_in_8bit:
        model = model.cuda()
    free_after_load, _ = torch.cuda.mem_get_info()
    event("model_loaded", free_gib=round(free_after_load / 2**30, 3), device=torch.cuda.get_device_name())
    model.eval()
    model.requires_grad_(False)
    device = next(model.parameters()).device
    # tuned-lens 0.2.0 predates Qwen3. Its final norm has the same role as
    # LlamaModel.norm, so extend the package's model-surgery dispatch locally
    # and record this compatibility path in the fit manifest.
    base_type = type(model.base_model)
    if base_type.__name__ in ("Qwen3Model", "Qwen3_5TextModel"):
        original_get_final_norm = model_surgery.get_final_norm
        original_norm = model_surgery.Norm

        def qwen3_get_final_norm(candidate):
            if type(candidate.base_model).__name__ in ("Qwen3Model", "Qwen3_5TextModel"):
                return candidate.base_model.norm
            return original_get_final_norm(candidate)

        model_surgery.get_final_norm = qwen3_get_final_norm
        model_surgery.Norm = original_norm | type(model.base_model.norm)
    lens = TunedLens.from_model(model).to(device)
    lens.train()
    chunks = make_chunks(
        tokenizer,
        split=args.split,
        max_length=args.max_length,
        max_chunks=args.max_chunks,
        data_path=args.data_path,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        text_field=args.text_field,
    )
    opt = torch.optim.AdamW(lens.parameters(), lr=args.lr, weight_decay=1e-3)
    log(f"model ready: layers={model.config.num_hidden_layers} d_model={model.config.hidden_size}; lens params={sum(p.numel() for p in lens.parameters()):,}")
    event("fit_ready", layers=model.config.num_hidden_layers, d_model=model.config.hidden_size, lens_params=sum(p.numel() for p in lens.parameters()), chunks=len(chunks))

    for step in range(args.steps):
        step_started = time.monotonic()
        ids = chunks[step % len(chunks)].unsqueeze(0).to(device)
        opt.zero_grad(set_to_none=True)
        # ``no_grad`` (rather than inference_mode) is intentional: the cached
        # hidden states are consumed by the trainable affine translators.
        with torch.no_grad():
            output = model(input_ids=ids, use_cache=False, output_hidden_states=True)
            target = output.logits.detach().float().log_softmax(dim=-1)
            hidden_states = [h.detach() for h in output.hidden_states[:-1]]
        losses = []
        for idx, hidden in enumerate(hidden_states):
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=dtype)
                if args.dtype != "fp32"
                else contextlib.nullcontext()
            )
            with autocast_context:
                pred = lens(hidden, idx).float().log_softmax(dim=-1)
                loss = (target.exp() * (target - pred)).sum(dim=-1).mean()
            loss.backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
        opt.step()
        if step == 0 or (step + 1) % 10 == 0:
            free, _ = torch.cuda.mem_get_info()
            step_seconds = time.monotonic() - step_started
            log(f"step {step + 1}/{args.steps} mean_KL={sum(losses) / len(losses):.5f} step={step_seconds:.1f}s free={free / 2**30:.2f} GiB")
            event("step", step=step + 1, steps=args.steps, step_seconds=round(step_seconds, 3), mean_kl=round(sum(losses) / len(losses), 6), free_gib=round(free / 2**30, 3), allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 3), reserved_gib=round(torch.cuda.memory_reserved() / 2**30, 3), **system_metrics())

    out = Path(args.out)
    lens.eval()
    lens.save(out)
    (out / "fit_manifest.json").write_text(
        __import__("json").dumps({
            "artifact_name": args.artifact_name or Path(args.out).name,
            "model": args.base_model_id or args.model,
            "model_path_used": args.model,
            "model_revision": args.base_model_revision,
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "bitsandbytes_version": optional_package_version("bitsandbytes"),
            "tuned_lens_version": optional_package_version("tuned-lens"),
            "dataset": args.dataset_label or f"{args.dataset_id}/{args.dataset_config}",
            "data_path": args.data_path,
            "split": args.split, "max_length": args.max_length,
            "max_chunks": args.max_chunks, "steps": args.steps, "lr": args.lr,
            "seed": args.seed, "dtype": args.dtype,
            "quantization": "nf4" if args.load_in_4bit else "int8" if args.load_in_8bit else "none",
            "load_in_4bit": args.load_in_4bit, "load_in_8bit": args.load_in_8bit,
            "objective": "per-layer KL to frozen final logits",
            "tuned_lens_compat": "Qwen3-family final norm dispatch patched locally" if base_type.__name__ in ("Qwen3Model", "Qwen3_5TextModel") else "native tuned-lens model surgery",
            "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "cuda_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        }, indent=2) + "\n"
    )
    config_path = out / "config.json"
    if config_path.exists():
        config = __import__("json").loads(config_path.read_text())
        config["base_model_name_or_path"] = args.base_model_id or args.model
        config["base_model_revision"] = args.base_model_revision
        config_path.write_text(__import__("json").dumps(config, separators=(",", ":")) + "\n")
    del lens, model, tokenizer, chunks
    gc.collect()
    torch.cuda.empty_cache()
    event("complete", output=str(out), peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3), peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 3))
    log(f"saved tuned lens to {out}; CUDA cache released")


if __name__ == "__main__":
    main()
