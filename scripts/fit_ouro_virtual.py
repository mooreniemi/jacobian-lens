#!/usr/bin/env python3
"""Fit a Jacobian lens over Ouro's virtually unrolled 48x4 computation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers
from torch import nn
from transformers.masking_utils import create_causal_mask

import jlens


class Stage(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class OuroVirtualAdapter:
    """Expose every shared block in every Ouro recurrent pass as virtual depth."""

    def __init__(self, hf_model, tokenizer):
        self.hf_model = hf_model
        self.core = hf_model.model
        self.tokenizer = tokenizer
        self.blocks_per_pass = self.core.config.num_hidden_layers
        self.n_passes = self.core.total_ut_steps
        self.n_layers = self.blocks_per_pass * self.n_passes
        self.d_model = self.core.config.hidden_size
        self.layers = nn.ModuleList(
            Stage(self._make_stage(index)) for index in range(self.n_layers)
        )

    def _make_stage(self, index):
        pass_index, block_index = divmod(index, self.blocks_per_pass)
        decoder_layer = self.core.layers[block_index]

        def run(hidden_states):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=self._causal_mask[decoder_layer.attention_type],
                position_ids=self._position_ids,
                past_key_value=None,
                use_cache=False,
                cache_position=self._cache_position,
                position_embeddings=self._position_embeddings,
                current_ut=pass_index,
            )
            if block_index == self.blocks_per_pass - 1:
                hidden_states = self.core.norm(hidden_states)
            return hidden_states

        return run

    @property
    def input_device(self):
        return self.core.embed_tokens.weight.device

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        return encoded.input_ids.to(self.input_device)

    def forward(self, input_ids: torch.Tensor):
        hidden_states = self.core.embed_tokens(input_ids)
        self._cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        self._position_ids = self._cache_position.unsqueeze(0)
        self._position_embeddings = self.core.rotary_emb(hidden_states, self._position_ids)
        self._causal_mask = {
            "full_attention": create_causal_mask(
                config=self.core.config,
                inputs_embeds=hidden_states,
                attention_mask=None,
                past_key_values=None,
                position_ids=self._position_ids,
            )
        }
        for stage in self.layers:
            hidden_states = stage(hidden_states)
        return hidden_states

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.hf_model.lm_head(residual.to(self.hf_model.lm_head.weight.dtype))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/alex/models/ouro-2.6b-thinking")
    parser.add_argument("--prompts", default="data/lens-prompts/fit-mix-120.json")
    parser.add_argument("--n-prompts", type=int, default=1)
    parser.add_argument("--dim-batch", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--skip-first", type=int, default=4)
    parser.add_argument("--target-layer", type=int, default=191)
    parser.add_argument("--source-start", type=int, default=0)
    parser.add_argument("--out", default="data/lenses/ouro-2.6b-thinking-virtual-fit-smoke-lens.pt")
    parser.add_argument("--checkpoint", default="data/lenses/ouro-2.6b-thinking-virtual-fit-smoke-ckpt.pt")
    args = parser.parse_args()

    payload = json.loads(Path(args.prompts).read_text())
    rows = payload["items"] if isinstance(payload, dict) else payload
    prompts = [row["prompt"] if isinstance(row, dict) else row for row in rows[: args.n_prompts]]
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16
    ).cuda().eval()
    model = OuroVirtualAdapter(hf_model, tokenizer)
    target = args.target_layer if args.target_layer >= 0 else model.n_layers + args.target_layer
    source_layers = list(range(args.source_start, target))
    lens = jlens.fit(
        model,
        prompts,
        source_layers=source_layers,
        target_layer=target,
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
        "target_layer": target,
        "d_model": lens.d_model,
        "blocks_per_pass": model.blocks_per_pass,
        "total_ut_steps": model.n_passes,
        "dim_batch": args.dim_batch,
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
    }, indent=2) + "\n")
    print(f"saved {lens} to {args.out}", flush=True)


if __name__ == "__main__":
    main()
