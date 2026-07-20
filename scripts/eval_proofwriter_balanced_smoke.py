#!/usr/bin/env python3
"""Run a stratified ProofWriter True/False/Unknown smoke evaluation."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from tabulate import tabulate

LABELS = ("True", "False", "Unknown")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


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


def sample_manifest(split: str, per_class: int, seed: int) -> list[dict[str, object]]:
    dataset = load_dataset("tasksource/proofwriter", split=split)
    groups = {label: [] for label in LABELS}
    for row in dataset:
        answer = str(row["answer"])
        if answer in groups:
            groups[answer].append(dict(row))
    rng = random.Random(seed)
    selected = []
    for label in LABELS:
        if len(groups[label]) < per_class:
            raise SystemExit(f"split {split} has only {len(groups[label])} {label} rows")
        rng.shuffle(groups[label])
        selected.extend(groups[label][:per_class])
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/alex/models/smollm2-135m-instruct")
    parser.add_argument("--split", default="test")
    parser.add_argument("--prompt-style", choices=("minimal", "explicit"), default="minimal")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--manifest", default="data/benchmarks/proofwriter-balanced-test-300.jsonl")
    parser.add_argument("--out", default="data/experiments/proofwriter-smollm2-balanced-smoke.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        log(f"reusing fixed manifest {manifest_path} ({len(rows)} rows)")
    else:
        log(f"sampling {args.per_class} rows per label from ProofWriter/{args.split} with seed {args.seed}")
        rows = sample_manifest(args.split, args.per_class, args.seed)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        log(f"wrote fixed manifest {manifest_path}")

    log(f"label distribution: {dict(Counter(str(row['answer']) for row in rows))}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    label_ids = []
    for label in LABELS:
        ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise SystemExit(f"expected one-token label for {label!r}, got {ids}")
        label_ids.append(ids[0])

    log(f"loading {args.model}")
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    log(f"model ready; CUDA free={torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")

    results = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            encoded = tokenizer(
                [make_prompt(row, args.prompt_style) for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(model.device)
            logits = model(**encoded).logits[:, -1, :]
            probabilities = logits[:, label_ids].softmax(dim=-1).float().cpu()
            for row, probs in zip(batch, probabilities):
                predicted = LABELS[int(probs.argmax())]
                results.append({
                    "id": row["id"],
                    "gold": row["answer"],
                    "predicted": predicted,
                    "config": row["config"],
                    "maxD": row["maxD"],
                    "probabilities": {label: round(float(probs[i]), 6) for i, label in enumerate(LABELS)},
                })
            log(f"scored {min(start + len(batch), len(rows))}/{len(rows)} prompts")

    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for result in results:
        matrix[str(result["gold"])][result["predicted"]] += 1
    metric_rows = []
    f1_values = []
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        metric_rows.append([label, tp, precision, recall, f1])
    accuracy = sum(matrix[label][label] for label in LABELS) / len(results)
    print("\nProofWriter balanced Smol smoke:")
    print(tabulate(metric_rows, headers=["class", "correct", "precision", "recall", "F1"], floatfmt=".3f", tablefmt="github"))
    print(f"accuracy={accuracy:.3f}  macro-F1={sum(f1_values) / len(f1_values):.3f}")
    print("\nConfusion matrix (rows=gold, columns=prediction):")
    print(tabulate([[label, *[matrix[label][pred] for pred in LABELS]] for label in LABELS], headers=["gold", *LABELS], tablefmt="github"))

    output = {
        "schema_version": 1,
        "kind": "proofwriter-balanced-observational-smoke",
        "model": args.model,
        "split": args.split,
        "prompt_style": args.prompt_style,
        "seed": args.seed,
        "manifest": str(manifest_path),
        "n_items": len(rows),
        "metrics": {"accuracy": accuracy, "macro_f1": sum(f1_values) / len(f1_values)},
        "confusion_matrix": matrix,
        "results": results,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    log(f"wrote {output_path}")


if __name__ == "__main__":
    main()
