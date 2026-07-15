"""Fit a fresh Jacobian lens for a small Qwen3.5 checkpoint.

Example:
    uv run python scripts/fit_qwen35_small.py --n-prompts 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers

import jlens
from jlens.examples import load_wikitext_prompts
from jlens.qwen_runtime import configure_qwen_kernels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--qwen-kernels", choices=("auto", "on", "off"), default="on")
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--prompts", help="JSON file containing prompts or an items list")
    parser.add_argument("--dim-batch", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--out", default="data/lenses/qwen3.5-0.8b-lens.pt")
    parser.add_argument("--checkpoint", default="data/lenses/qwen3.5-0.8b-ckpt.pt")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    jlens.configure_logging()
    kernel_mode = configure_qwen_kernels(args.qwen_kernels)
    print(f"Qwen kernel mode: {kernel_mode}", flush=True)
    print(f"loading {args.model}", flush=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf_model, tokenizer)
    print(model, flush=True)

    if args.prompts:
        payload = json.loads(Path(args.prompts).read_text())
        rows = payload["items"] if isinstance(payload, dict) else payload
        prompts = [row["prompt"] if isinstance(row, dict) else row for row in rows]
    else:
        prompts = load_wikitext_prompts(args.n_prompts)
    out = Path(args.out)
    checkpoint = Path(args.checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    lens = jlens.fit(
        model,
        prompts,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=str(checkpoint),
        checkpoint_every=args.checkpoint_every,
    )
    lens.save(str(out))
    manifest = {
        "model": args.model,
        "lens": str(out),
        "n_prompts": lens.n_prompts,
        "source_layers": lens.source_layers,
        "d_model": lens.d_model,
        "prompts": args.prompts,
        "dim_batch": args.dim_batch,
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
        "checkpoint_every": args.checkpoint_every,
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved {lens} to {out}", flush=True)
    print(f"wrote manifest to {out.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
