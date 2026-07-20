#!/usr/bin/env python3
"""Train a label-aware, low-rank adaptation of a Jacobian lens.

This is deliberately a separate research variant.  The released J-lens is a
task-agnostic average Jacobian.  Here we initialize from that J matrix and
learn a low-rank delta using ProofWriter labels, while keeping the base model
and its unembedding frozen.  The held-out confirmation set is never used for
optimization or layer selection.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from sklearn.linear_model import LogisticRegression

import jlens

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
    matrix = {label: {other: 0 for other in LABELS} for label in LABELS}
    for expected, actual in zip(gold, predicted, strict=True):
        matrix[str(expected)][str(actual)] += 1
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "accuracy": float(np.mean(gold == predicted)),
        "macro_f1": float(np.mean([per_class[label]["f1"] for label in LABELS])),
        "unknown_recall": per_class["Unknown"]["recall"],
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def extract_features(model, tokenizer, rows, layer: int, max_length: int) -> torch.Tensor:
    values = []
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            encoded = tokenizer(
                make_prompt(row), return_tensors="pt", truncation=True, max_length=max_length
            ).to(model.device)
            output = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
            values.append(output.hidden_states[layer + 1][0, -1, :].float().cpu())
            if index == 1 or index % 25 == 0 or index == len(rows):
                log(f"extracted layer {layer} features {index}/{len(rows)}")
    return torch.stack(values)


def readout_logits(wrapper, features: torch.Tensor, jacobian: torch.Tensor, label_ids: list[int]) -> torch.Tensor:
    device = wrapper.input_device
    transported = features.to(device) @ jacobian.to(device).T
    return wrapper.unembed(transported)[:, label_ids].float()


def predicted_labels(logits: torch.Tensor) -> np.ndarray:
    return np.asarray([LABELS[index] for index in logits.argmax(dim=-1).cpu().tolist()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2718)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    selection = [json.loads(line) for line in Path(args.selection).read_text().splitlines() if line.strip()]
    confirmation = [json.loads(line) for line in Path(args.confirmation).read_text().splitlines() if line.strip()]
    log(f"loading {len(selection)} selection and {len(confirmation)} confirmation rows")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    label_ids = []
    for label in LABELS:
        ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise SystemExit(f"expected one-token label for {label!r}, got {ids}")
        label_ids.append(ids[0])
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    wrapper = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(args.lens)
    if args.layer not in lens.source_layers:
        raise SystemExit(f"layer {args.layer} is not in lens source layers {lens.source_layers}")
    log(f"model ready; adapting generic J-lens at layer {args.layer} with rank {args.rank}")

    selection_features = extract_features(hf_model, tokenizer, selection, args.layer, args.max_length)
    confirmation_features = extract_features(hf_model, tokenizer, confirmation, args.layer, args.max_length)
    selection_gold = np.asarray([str(row["answer"]) for row in selection])
    confirmation_gold = np.asarray([str(row["answer"]) for row in confirmation])
    selection_ids = torch.tensor([LABEL_TO_ID[label] for label in selection_gold], device=wrapper.input_device)

    base_j = lens.jacobians[args.layer].float().to(wrapper.input_device)
    with torch.no_grad():
        selection_base_logits = readout_logits(wrapper, selection_features, base_j, label_ids)
        confirmation_base_logits = readout_logits(wrapper, confirmation_features, base_j, label_ids)

    # A zero-initialized update preserves the generic J-lens at step zero.
    d_model = lens.d_model
    update_left = torch.nn.Parameter(torch.zeros(d_model, args.rank, device=wrapper.input_device))
    update_right = torch.nn.Parameter(0.01 * torch.randn(d_model, args.rank, device=wrapper.input_device))
    optimizer = torch.optim.AdamW([update_left, update_right], lr=args.lr, weight_decay=1e-4)
    order = torch.arange(len(selection), device=wrapper.input_device)
    for step in range(1, args.steps + 1):
        if (step - 1) % max(1, len(selection) // args.batch_size) == 0:
            order = order[torch.randperm(len(order), device=order.device)]
        start = ((step - 1) * args.batch_size) % len(selection)
        indices = order[start : start + args.batch_size]
        if len(indices) == 0:
            continue
        adapted_j = base_j + update_left @ update_right.T
        logits = readout_logits(wrapper, selection_features[indices.cpu()], adapted_j, label_ids)
        loss = F.cross_entropy(logits, selection_ids[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            with torch.no_grad():
                train_pred = predicted_labels(logits)
                train_gold = selection_gold[indices.cpu().numpy()]
                log(f"step {step}/{args.steps} loss={loss.item():.4f} batch_acc={np.mean(train_pred == train_gold):.3f}")

    with torch.no_grad():
        adapted_j = base_j + update_left @ update_right.T
        selection_adapted_logits = readout_logits(wrapper, selection_features, adapted_j, label_ids)
        confirmation_adapted_logits = readout_logits(wrapper, confirmation_features, adapted_j, label_ids)
    selection_probe = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    selection_probe.fit(selection_features.numpy(), np.asarray([LABEL_TO_ID[label] for label in selection_gold]))
    confirmation_probe_pred = np.asarray([LABELS[index] for index in selection_probe.predict(confirmation_features.numpy())])
    output = {
        "schema_version": 1,
        "kind": "proofwriter-label-adapted-low-rank-jlens",
        "model": args.model,
        "base_lens": args.lens,
        "selection_manifest": args.selection,
        "confirmation_manifest": args.confirmation,
        "prompt_style": "explicit",
        "layer": args.layer,
        "rank": args.rank,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "selection_metrics": {
            "generic_jlens": metrics(selection_gold, predicted_labels(selection_base_logits)),
            "label_adapted_jlens": metrics(selection_gold, predicted_labels(selection_adapted_logits)),
        },
        "confirmation_metrics": {
            "generic_jlens": metrics(confirmation_gold, predicted_labels(confirmation_base_logits)),
            "label_adapted_jlens": metrics(confirmation_gold, predicted_labels(confirmation_adapted_logits)),
            "same_layer_linear_probe": metrics(confirmation_gold, confirmation_probe_pred),
        },
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    log(f"wrote {output_path}")


if __name__ == "__main__":
    main()
