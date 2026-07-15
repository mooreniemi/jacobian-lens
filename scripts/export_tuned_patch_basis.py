"""Export tuned-lens translator matrices as an explicit patch basis.

The tuned translator is h -> h + h W^T + b. The corresponding row-space
linear map is A = I + W^T, so a token direction is A^T U[token], matching the
coordinate convention used by the causal runners. The tuned lens' nonlinear
final RMSNorm is not folded into this basis and is recorded in metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jlens import JacobianLens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tuned-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    config = json.loads((Path(args.tuned_dir) / "config.json").read_text())
    state = torch.load(Path(args.tuned_dir) / "params.pt", map_location="cpu", weights_only=True)
    n_layers = int(config["num_hidden_layers"])
    d_model = int(config["d_model"])
    eye = torch.eye(d_model, dtype=torch.float32)
    jacobians = {}
    for i in range(n_layers):
        w = state[f"{i}.weight"].float()
        jacobians[i] = eye + w.T
    lens = JacobianLens(jacobians=jacobians, n_prompts=0, d_model=d_model)
    lens.save(args.out)
    manifest = {
        "model": args.model,
        "source_tuned_lens": str(Path(args.tuned_dir).resolve()),
        "basis": "I + translator.weight.T",
        "token_direction": "basis.T @ unembedding_row",
        "final_norm_folded": False,
        "purpose": "causal tuned-lens translator-basis comparison",
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved tuned patch basis to {args.out} ({n_layers} layers, d_model={d_model})")


if __name__ == "__main__":
    main()
