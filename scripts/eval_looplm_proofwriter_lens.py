#!/usr/bin/env python3
"""Run direct, logit-lens, and J-lens readouts on LoopLM ProofWriter items."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers
from fit_looplm_small import LoopLMResidualAdapter

import jlens

LABELS = ("True", "False", "Unknown")


def prompt(row: dict[str, object]) -> str:
    instruction = (
        "Decide whether the question is entailed, contradicted, or unknown "
        "given the facts and rules. True means the statement is provable. "
        "False means its opposite is provable. Unknown means neither is "
        "provable. Output exactly one label: True, False, or Unknown.\n\n"
        f"Facts and rules:\n{row['theory']}\n\nQuestion: {row['question']}\n"
    )
    return f"### Instruction:\n{instruction}\n### Response:\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/alex/models/looplm-135m-naive-sft")
    parser.add_argument("--lens", default="data/lenses/looplm-135m-sft-smoke-lens.pt")
    parser.add_argument("--manifest", default="data/benchmarks/proofwriter-selection-300.jsonl")
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--out", default="data/experiments/looplm-proofwriter-lens-smoke.json")
    parser.add_argument("--recurrence", type=int, default=6)
    parser.add_argument("--max-seq-len", type=int, default=128)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    rows = rows[: args.max_items]
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    label_ids = []
    for label in LABELS:
        ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise SystemExit(f"expected one-token label for {label!r}, got {ids}")
        label_ids.append(ids[0])
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16
    ).cuda().eval()
    model = LoopLMResidualAdapter(hf_model, tokenizer, args.recurrence)
    lens = jlens.JacobianLens.load(args.lens)
    layers = list(lens.source_layers)
    counts = {"final": [0, 0], "jlens": {layer: [0, 0] for layer in layers}, "logit": {layer: [0, 0] for layer in layers}}
    for row in rows:
        gold = LABELS.index(str(row["answer"]))
        j_logits, final_logits, _ = lens.apply(model, prompt(row), layers=layers, positions=[-1], max_seq_len=args.max_seq_len)
        l_logits, _, _ = lens.apply(model, prompt(row), layers=layers, positions=[-1], max_seq_len=args.max_seq_len, use_jacobian=False)
        final_pred = int(final_logits[0, label_ids].argmax())
        counts["final"][0] += final_pred == gold
        counts["final"][1] += 1
        for layer in layers:
            for method, logits in (("jlens", j_logits[layer]), ("logit", l_logits[layer])):
                predicted = int(logits[0, label_ids].argmax())
                counts[method][layer][0] += predicted == gold
                counts[method][layer][1] += 1
    result = {
        "kind": "looplm-proofwriter-lens-smoke",
        "model": args.model,
        "lens": args.lens,
        "manifest": args.manifest,
        "n_items": len(rows),
        "layers": layers,
        "accuracy": {
            "final": counts["final"][0] / counts["final"][1],
            "jlens": {str(k): v[0] / v[1] for k, v in counts["jlens"].items()},
            "logit": {str(k): v[0] / v[1] for k, v in counts["logit"].items()},
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
