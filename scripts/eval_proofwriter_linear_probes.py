#!/usr/bin/env python3
"""Train leakage-safe linear probes on ProofWriter hidden states.

Layers are selected and probes are fit on the selection set, then the selected
probe is evaluated once on a source-disjoint confirmation set.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate

LABELS = ("True", "False", "Unknown")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def make_prompt(row: dict[str, object]) -> str:
    instruction = (
        "Decide whether the question is entailed, contradicted, or unknown "
        "given the facts and rules. True means the statement is provable. "
        "False means its opposite is provable. Unknown means neither is "
        "provable. Output exactly one label: True, False, or Unknown.\n\n"
    )
    return f"{instruction}Facts and rules:\n{row['theory']}\n\nQuestion: {row['question']}\nAnswer:"


def metrics(gold: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    matrix = {gold_label: {pred: 0 for pred in LABELS} for gold_label in LABELS}
    for gold_label, pred in zip(gold, predicted, strict=True):
        matrix[str(gold_label)][str(pred)] += 1
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
        "accuracy": float(np.mean(gold == predicted)),
        "macro_f1": float(np.mean(f1s)),
        "unknown_recall": per_class["Unknown"]["recall"],
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def extract_features(model, tokenizer, rows: list[dict[str, object]], layers: list[int], max_length: int) -> dict[int, np.ndarray]:
    features = {layer: [] for layer in layers}
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            encoded = tokenizer(
                make_prompt(row), return_tensors="pt", truncation=True, max_length=max_length
            ).to(model.device)
            output = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
            for layer in layers:
                # hidden_states[0] is the embedding output; layer L is L+1.
                value = output.hidden_states[layer + 1][0, -1, :].float().cpu().numpy()
                features[layer].append(value)
            if index == 1 or index % 25 == 0 or index == len(rows):
                log(f"extracted {index}/{len(rows)} prompts")
    return {layer: np.stack(values) for layer, values in features.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", required=True, help="comma-separated transformer layer numbers")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()

    selection = [json.loads(line) for line in Path(args.selection).read_text().splitlines() if line.strip()]
    confirmation = [json.loads(line) for line in Path(args.confirmation).read_text().splitlines() if line.strip()]
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    log(f"loading {len(selection)} selection and {len(confirmation)} confirmation rows")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    log(f"model ready; fitting probes for layers {layers}")
    selection_features = extract_features(model, tokenizer, selection, layers, args.max_length)
    selection_gold = np.asarray([str(row["answer"]) for row in selection])
    selection_ids = np.asarray([LABEL_TO_ID[label] for label in selection_gold])

    probes = {}
    selection_metrics = {}
    for layer in layers:
        probe = LogisticRegression(C=args.c, max_iter=2000, solver="lbfgs")
        probe.fit(selection_features[layer], selection_ids)
        predicted = np.asarray([LABELS[index] for index in probe.predict(selection_features[layer])])
        probes[layer] = probe
        selection_metrics[str(layer)] = metrics(selection_gold, predicted)
    best_layer = max(layers, key=lambda layer: selection_metrics[str(layer)]["macro_f1"])
    log(f"selected probe layer {best_layer} by selection macro-F1={selection_metrics[str(best_layer)]['macro_f1']:.3f}")

    confirmation_features = extract_features(model, tokenizer, confirmation, [best_layer], args.max_length)[best_layer]
    confirmation_gold = np.asarray([str(row["answer"]) for row in confirmation])
    probe = probes[best_layer]
    confirmation_predicted = np.asarray([LABELS[index] for index in probe.predict(confirmation_features)])
    output = {
        "schema_version": 1,
        "kind": "proofwriter-linear-probe-confirmation",
        "model": args.model,
        "selection_manifest": args.selection,
        "confirmation_manifest": args.confirmation,
        "prompt_style": "explicit",
        "layers": layers,
        "selected_layer": best_layer,
        "classifier": "multinomial logistic regression on final-position hidden state",
        "regularization_C": args.c,
        "selection_metrics": selection_metrics,
        "confirmation_metrics": metrics(confirmation_gold, confirmation_predicted),
        "confirmation_results": [
            {"id": row["id"], "source_group": row.get("source_group"), "gold": str(row["answer"]), "predicted": str(pred)}
            for row, pred in zip(confirmation, confirmation_predicted, strict=True)
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n")
    print(tabulate(
        [["selection", best_layer, selection_metrics[str(best_layer)]["accuracy"], selection_metrics[str(best_layer)]["macro_f1"], selection_metrics[str(best_layer)]["unknown_recall"]],
         ["confirmation", best_layer, output["confirmation_metrics"]["accuracy"], output["confirmation_metrics"]["macro_f1"], output["confirmation_metrics"]["unknown_recall"]]],
        headers=["split", "layer", "accuracy", "macro-F1", "Unknown recall"], floatfmt=".3f", tablefmt="github"))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
