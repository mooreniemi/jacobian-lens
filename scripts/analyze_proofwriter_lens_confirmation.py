#!/usr/bin/env python3
"""Uncertainty analysis for a locked ProofWriter lens confirmation run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tabulate import tabulate

LABELS = ("True", "False", "Unknown")


def metric(gold: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    accuracy = float(np.mean(gold == pred))
    f1s = []
    unknown_recall = 0.0
    for label in LABELS:
        tp = np.sum((gold == label) & (pred == label))
        fp = np.sum((gold != label) & (pred == label))
        fn = np.sum((gold == label) & (pred != label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        if label == "Unknown":
            unknown_recall = float(recall)
    return accuracy, float(np.mean(f1s)), unknown_recall


def ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/benchmarks/proofwriter-confirmation-1500.jsonl")
    parser.add_argument("--results", default="data/experiments/proofwriter-smollm2-balanced-source-disjoint-confirmation.json")
    parser.add_argument("--out", default="data/experiments/proofwriter-smollm2-balanced-source-disjoint-confirmation-analysis.json")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--jlens-layer", type=int, default=25)
    parser.add_argument("--logit-layer", type=int, default=6)
    args = parser.parse_args()

    manifest = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    output = json.loads(Path(args.results).read_text())
    results = output["results"]
    if len(manifest) != len(results):
        raise SystemExit(f"manifest/results length mismatch: {len(manifest)} vs {len(results)}")
    gold = np.asarray([str(row["answer"]) for row in manifest], dtype=object)
    groups = np.asarray([str(row["source_group"]) for row in manifest], dtype=object)
    predictions = {
        "final": np.asarray([row["final"]["predicted"] for row in results], dtype=object),
        "jlens": np.asarray([row["layers"][str(args.jlens_layer)]["jlens"]["predicted"] for row in results], dtype=object),
        "logit": np.asarray([row["layers"][str(args.logit_layer)]["logit"]["predicted"] for row in results], dtype=object),
    }
    unique_groups = np.unique(groups)
    group_indices = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(args.seed)
    boot = {method: [] for method in predictions}
    contrasts = {"jlens_minus_final": [], "jlens_minus_logit": []}
    for _ in range(args.draws):
        sampled = rng.integers(0, len(group_indices), size=len(group_indices))
        indices = np.concatenate([group_indices[index] for index in sampled])
        values = {method: metric(gold[indices], pred[indices]) for method, pred in predictions.items()}
        for method, value in values.items():
            boot[method].append(value)
        contrasts["jlens_minus_final"].append(values["jlens"][0] - values["final"][0])
        contrasts["jlens_minus_logit"].append(values["jlens"][0] - values["logit"][0])

    summary = {}
    for method, pred in predictions.items():
        point = metric(gold, pred)
        summary[method] = {
            "accuracy": point[0], "accuracy_ci95": ci(np.asarray([x[0] for x in boot[method]])),
            "macro_f1": point[1], "macro_f1_ci95": ci(np.asarray([x[1] for x in boot[method]])),
            "unknown_recall": point[2], "unknown_recall_ci95": ci(np.asarray([x[2] for x in boot[method]])),
        }
    summary["contrasts"] = {
        name: {"point": float((predictions["jlens"] == gold).mean() - (predictions[other] == gold).mean()),
               "ci95": ci(np.asarray(values))}
        for name, values, other in (("jlens_minus_final", contrasts["jlens_minus_final"], "final"),
                                    ("jlens_minus_logit", contrasts["jlens_minus_logit"], "logit"))
    }
    jlens_correct = predictions["jlens"] == gold
    final_correct = predictions["final"] == gold
    logit_correct = predictions["logit"] == gold
    j_only_final = int(np.sum(jlens_correct & ~final_correct))
    final_only_j = int(np.sum(final_correct & ~jlens_correct))
    j_only_logit = int(np.sum(jlens_correct & ~logit_correct))
    logit_only_j = int(np.sum(logit_correct & ~jlens_correct))
    def sign_p(a: int, b: int) -> float:
        n = a + b
        if not n:
            return 1.0
        tail = sum(math.comb(n, k) for k in range(a, n + 1)) / (2 ** n)
        return min(1.0, 2 * min(tail, 1 - tail + math.comb(n, a) / (2 ** n)))
    summary["paired"] = {
        "jlens_vs_final": {"jlens_only": j_only_final, "final_only": final_only_j, "same": int(np.sum(jlens_correct == final_correct)), "sign_test_p": sign_p(j_only_final, final_only_j)},
        "jlens_vs_logit": {"jlens_only": j_only_logit, "logit_only": logit_only_j, "same": int(np.sum(jlens_correct == logit_correct)), "sign_test_p": sign_p(j_only_logit, logit_only_j)},
    }
    summary["n_items"] = len(gold)
    summary["n_source_groups"] = len(unique_groups)
    summary["seed"] = args.seed
    summary["draws"] = args.draws
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    rows = []
    for method in ("final", "jlens", "logit"):
        item = summary[method]
        rows.append([method, item["accuracy"], item["accuracy_ci95"], item["macro_f1"], item["unknown_recall"]])
    print(tabulate(rows, headers=["method", "accuracy", "cluster CI95", "macro-F1", "Unknown recall"], floatfmt=".3f", tablefmt="github"))
    for name, item in summary["contrasts"].items():
        print(f"{name}: {item['point']:+.3f}, cluster CI95 {item['ci95']}, paired sign-test p={summary['paired'][name.replace('jlens_minus_', 'jlens_vs_')]['sign_test_p']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
