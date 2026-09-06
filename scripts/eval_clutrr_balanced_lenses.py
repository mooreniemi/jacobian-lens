#!/usr/bin/env python3
"""Compare final, logit-lens, and J-lens CLUTRR readouts."""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from tabulate import tabulate

import jlens

LABELS = (
    "aunt", "brother", "daughter", "daughter-in-law", "father", "father-in-law",
    "granddaughter", "grandfather", "grandmother", "grandson", "mother",
    "mother-in-law", "nephew", "niece", "sister", "son", "son-in-law", "uncle",
)
CODES = tuple("ABCDEFGHIJKLMNOPQR")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def make_prompt(row: dict[str, object]) -> str:
    first, second = ast.literal_eval(str(row["sentence2"]))
    mapping = ", ".join(f"{code}={label}" for code, label in zip(CODES, LABELS, strict=True))
    return (
        "Infer the kinship relationship from the story. The query gives an ordered pair "
        "(first person, second person). Determine the relationship of the second person "
        "to the first person. Output exactly one code letter and no explanation.\n"
        f"Codes: {mapping}\n\n"
        f"Story: {row['sentence1']}\n"
        f"Query: ({first}, {second})\n"
        "Answer:"
    )


def metrics(gold: list[int], predicted: list[int]) -> dict[str, object]:
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for expected, actual in zip(gold, predicted, strict=True):
        matrix[expected][actual] += 1
    f1 = []
    for label in range(len(LABELS)):
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in range(len(LABELS)) if other != label)
        fn = sum(matrix[label][other] for other in range(len(LABELS)) if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": float(np.mean(np.asarray(gold) == np.asarray(predicted))), "macro_f1": float(np.mean(f1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()
    selection = [json.loads(line) for line in Path(args.selection).read_text().splitlines() if line.strip()]
    confirmation = [json.loads(line) for line in Path(args.confirmation).read_text().splitlines() if line.strip()]
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    code_ids = []
    for code in CODES:
        ids = tokenizer(" " + code, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise SystemExit(f"code {code} is not one token: {ids}")
        code_ids.append(ids[0])
    log(f"loading {len(selection)} selection and {len(confirmation)} confirmation rows")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(args.lens)
    layers = lens.source_layers
    log(f"model ready; evaluating {len(layers)} J-lens layers")
    scores = {"jlens": {layer: [] for layer in layers}, "logit": {layer: [] for layer in layers}, "final": []}
    for rows, name in ((selection, "selection"), (confirmation, "confirmation")):
        scores_local = {"jlens": {layer: [] for layer in layers}, "logit": {layer: [] for layer in layers}, "final": []}
        gold = []
        with torch.inference_mode():
            for index, row in enumerate(rows, 1):
                text = make_prompt(row)
                j_logits, final_logits, _ = lens.apply(model, text, layers=layers, positions=[-1], max_seq_len=args.max_seq_len)
                l_logits, _, _ = lens.apply(model, text, layers=layers, positions=[-1], max_seq_len=args.max_seq_len, use_jacobian=False)
                label = int(row["labels"])
                gold.append(label)
                scores_local["final"].append(int(final_logits[0, code_ids].argmax()))
                for layer in layers:
                    scores_local["jlens"][layer].append(int(j_logits[layer][0, code_ids].argmax()))
                    scores_local["logit"][layer].append(int(l_logits[layer][0, code_ids].argmax()))
                if index == 1 or index % 25 == 0 or index == len(rows):
                    log(f"{name}: scored {index}/{len(rows)}")
        if name == "selection":
            scores = scores_local
            selection_gold = gold
        else:
            confirmation_scores = scores_local
            confirmation_gold = gold
    selected = {}
    for method in ("jlens", "logit"):
        selected[method] = max(layers, key=lambda layer: metrics(selection_gold, scores[method][layer])["macro_f1"])
    selected["final"] = "final"
    rows = []
    for method, layer in selected.items():
        predictions = scores["final"] if method == "final" else scores[method][layer]
        confirm_predictions = confirmation_scores["final"] if method == "final" else confirmation_scores[method][layer]
        rows.append([method, layer, metrics(selection_gold, predictions), metrics(confirmation_gold, confirm_predictions)])
    table_rows = [[m, l, s["accuracy"], s["macro_f1"], c["accuracy"], c["macro_f1"]] for m, l, s, c in rows]
    print(tabulate(table_rows, headers=["method", "layer", "selection acc", "selection F1", "confirmation acc", "confirmation F1"], floatfmt=".3f", tablefmt="github"))
    output = {"schema_version": 1, "kind": "clutrr-balanced-lens-evaluation", "model": args.model, "lens": args.lens, "selection_manifest": args.selection, "confirmation_manifest": args.confirmation, "prompt_style": "explicit-codebook", "selected_layers": selected, "selection_metrics": {m: s for m, _, s, _ in rows}, "confirmation_metrics": {m: c for m, _, _, c in rows}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
