#!/usr/bin/env python3
"""Fit a Jacobian lens over LoopLM prelude, recurrent, and coda states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers
from torch import nn

import jlens


class Stage(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class LoopLMResidualAdapter:
    """Expose LoopLM states as a sequential LensModel without changing weights."""

    def __init__(self, hf_model, tokenizer, recurrence: int):
        self.hf_model = hf_model
        self.core = hf_model.model
        self.tokenizer = tokenizer
        self.n_layers = self.core.cfg.n_prelude + recurrence + self.core.cfg.n_coda
        self.d_model = self.core.cfg.d_model
        self.recurrence = recurrence
        self.layers = nn.ModuleList()
        for block in self.core.prelude:
            self.layers.append(Stage(lambda x, block=block: block(x, self._cos, self._sin)))
        for _ in range(recurrence):
            self.layers.append(Stage(lambda h: self.core.loop(h, self._e, self._cos, self._sin)))
        for block in self.core.coda:
            self.layers.append(Stage(lambda x, block=block: block(x, self._cos, self._sin)))

    @property
    def input_device(self):
        return self.core.embed.weight.device

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        return encoded.input_ids.to(self.input_device)

    def forward(self, input_ids: torch.Tensor):
        x = self.core.embed(input_ids)
        self._cos, self._sin = self.core.rope(x.size(1), x.device, x.dtype)
        prelude_count = self.core.cfg.n_prelude
        for stage in self.layers[:prelude_count]:
            x = stage(x)
        self._e = self.core.prelude_norm(x)
        x = torch.zeros_like(self._e)
        for stage in self.layers[prelude_count:]:
            x = stage(x)
        return x

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.core.lm_head(self.core.final_norm(residual.to(self.core.final_norm.weight.dtype)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/alex/models/looplm-135m-naive-sft")
    parser.add_argument("--prompts", default="data/lens-prompts/fit-mix-120.json")
    parser.add_argument("--n-prompts", type=int, default=8)
    parser.add_argument("--dim-batch", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--skip-first", type=int, default=8)
    parser.add_argument("--recurrence", type=int, default=6)
    parser.add_argument("--out", default="data/lenses/looplm-135m-sft-fit-smoke-lens.pt")
    parser.add_argument("--checkpoint", default="data/lenses/looplm-135m-sft-fit-smoke-ckpt.pt")
    args = parser.parse_args()

    payload = json.loads(Path(args.prompts).read_text())
    rows = payload["items"] if isinstance(payload, dict) else payload
    prompts = [row["prompt"] if isinstance(row, dict) else row for row in rows[: args.n_prompts]]
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16
    ).cuda().eval()
    model = LoopLMResidualAdapter(hf_model, tokenizer, args.recurrence)
    lens = jlens.fit(
        model,
        prompts,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=args.checkpoint,
        checkpoint_every=1,
        resume=False,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lens.save(args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps({
        "model": args.model,
        "lens": args.out,
        "n_prompts": lens.n_prompts,
        "source_layers": lens.source_layers,
        "d_model": lens.d_model,
        "recurrence": args.recurrence,
        "dim_batch": args.dim_batch,
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
    }, indent=2) + "\n")
    print(f"saved {lens} to {args.out}", flush=True)


if __name__ == "__main__":
    main()
