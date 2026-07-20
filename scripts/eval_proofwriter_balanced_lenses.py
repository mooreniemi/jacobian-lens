#!/usr/bin/env python3
"""Compare final, logit-lens, and J-lens readouts on a fixed ProofWriter set."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import jlens
import torch
import transformers
from tabulate import tabulate

LABELS = ("True", "False", "Unknown")


def make_prompt(row: dict[str, object], prompt_style: str) -> str:
    instruction = ""
    if prompt_style == "explicit":
        instruction = (
            "Decide whether the question is entailed, contradicted, or unknown "
            "given the facts and rules. True means the statement is provable. "
            "False means its opposite is provable. Unknown means neither is "
            "provable. Output exactly one label: True, False, or Unknown.\n\n"
        )
    return f"{instruction}Facts and rules:\n{row['theory']}\n\nQuestion: {row['question']}\nAnswer:"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def metrics(rows: list[tuple[str, str]]) -> dict[str, object]:
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for gold, predicted in rows:
        matrix[gold][predicted] += 1
    f1s = []
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "accuracy": sum(matrix[label][label] for label in LABELS) / len(rows),
        "macro_f1": sum(f1s) / len(f1s),
        "unknown_recall": per_class["Unknown"]["recall"],
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/alex/models/smollm2-135m-instruct")
    parser.add_argument("--lens", default="data/lenses/smollm2-135m-fit-204-lens.pt")
    parser.add_argument("--manifest", default="data/benchmarks/proofwriter-balanced-test-300.jsonl")
    parser.add_argument("--out", default="data/experiments/proofwriter-smollm2-balanced-lens-smoke.json")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--prompt-style", choices=("minimal", "explicit"), default="minimal")
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if args.max_items:
        rows = rows[: args.max_items]
    log(f"loading {len(rows)} fixed ProofWriter rows")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    label_ids = []
    for label in LABELS:
        ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise SystemExit(f"expected one-token label for {label!r}, got {ids}")
        label_ids.append(ids[0])
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(args.lens)
    layers = list(lens.source_layers)
    log(f"model ready; evaluating {len(layers)} J-lens layers")

    scores: dict[str, dict[int | str, list[tuple[str, str]]]] = {
        "jlens": {layer: [] for layer in layers},
        "logit": {layer: [] for layer in layers},
        "final": {"final": []},
    }
    all_results = []
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            text = make_prompt(row, args.prompt_style)
            j_logits, final_logits, input_ids = lens.apply(
                model, text, layers=layers, positions=[-1], max_seq_len=args.max_seq_len
            )
            l_logits, _final_again, _ = lens.apply(
                model, text, layers=layers, positions=[-1], max_seq_len=args.max_seq_len, use_jacobian=False
            )
            gold = str(row["answer"])
            record = {"id": row["id"], "gold": gold, "layers": {}}
            final_scores = final_logits[0, label_ids]
            final_pred = LABELS[int(final_scores.argmax())]
            scores["final"]["final"].append((gold, final_pred))
            for layer in layers:
                j_scores = j_logits[layer][0, label_ids]
                l_scores = l_logits[layer][0, label_ids]
                j_pred = LABELS[int(j_scores.argmax())]
                l_pred = LABELS[int(l_scores.argmax())]
                scores["jlens"][layer].append((gold, j_pred))
                scores["logit"][layer].append((gold, l_pred))
                record["layers"][str(layer)] = {
                    "jlens": {"predicted": j_pred, "probabilities": j_scores.softmax(dim=-1).tolist()},
                    "logit": {"predicted": l_pred, "probabilities": l_scores.softmax(dim=-1).tolist()},
                }
            record["final"] = {"predicted": final_pred, "probabilities": final_scores.softmax(dim=-1).tolist()}
            all_results.append(record)
            log(f"scored {index}/{len(rows)}")

    summary = []
    metric_output = {"final": metrics(scores["final"]["final"]), "jlens": {}, "logit": {}}
    for method in ("jlens", "logit"):
        for layer in layers:
            metric_output[method][str(layer)] = metrics(scores[method][layer])
    summary.append(["final", "final", metric_output["final"]["accuracy"], metric_output["final"]["macro_f1"], metric_output["final"]["unknown_recall"]])
    for method in ("jlens", "logit"):
        best = max(metric_output[method].items(), key=lambda pair: pair[1]["macro_f1"])
        summary.append([method, best[0], best[1]["accuracy"], best[1]["macro_f1"], best[1]["unknown_recall"]])
    print("\nBest readout summary (layer selected by macro-F1; exploratory only):")
    print(tabulate(summary, headers=["method", "layer", "accuracy", "macro-F1", "Unknown recall"], floatfmt=".3f", tablefmt="github"))

    output = {
        "schema_version": 1,
        "kind": "proofwriter-balanced-lens-smoke",
        "model": args.model,
        "lens": args.lens,
        "manifest": args.manifest,
        "prompt_style": args.prompt_style,
        "layers": layers,
        "metrics": metric_output,
        "results": all_results,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    log(f"wrote {output_path}")


if __name__ == "__main__":
    main()
