"""Guarded one-prompt PyTorch/bitsandbytes smoke test for Qwen3.6-27B."""
from __future__ import annotations

import argparse
import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/alex/models/qwen3.6-27b-hf")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--min-free-gib", type=float, default=20.0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    free, total = torch.cuda.mem_get_info()
    log(f"preflight: {free / 2**30:.2f} GiB free / {total / 2**30:.2f} GiB total")
    if free / 2**30 < args.min_free_gib:
        raise SystemExit(f"refusing to start: need {args.min_free_gib:.1f} GiB free")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    log(f"loading 4-bit model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    free, _ = torch.cuda.mem_get_info()
    log(f"model loaded; {free / 2**30:.2f} GiB free")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    log("prompt: " + args.prompt)
    log("continuation: " + tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
    del model, tokenizer, inputs, output
    gc.collect()
    torch.cuda.empty_cache()
    log("smoke test complete; CUDA cache released")


if __name__ == "__main__":
    main()
