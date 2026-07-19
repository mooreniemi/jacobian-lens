#!/usr/bin/env python3
"""Compute paired per-item causal effects from saved rank outputs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/experiments"


def rank_fields(item: dict) -> tuple[str, str]:
    if "target_rank_before" in item:
        return "target_rank_before", "target_rank_after"
    if "swap_answer_rank_before" in item:
        return "swap_answer_rank_before", "swap_answer_rank_after"
    if "target_answer_rank_before" in item:
        return "target_answer_rank_before", "target_answer_rank_after"
    raise ValueError("unrecognized causal item schema")


def item_values(item: dict, method: str) -> tuple[float, float]:
    before_key, after_key = rank_fields(item)
    deltas = []
    successes = []
    for layer in item["layers"].values():
        record = layer[method]
        delta = float(item[before_key] - record[after_key])
        deltas.append(delta)
        successes.append(float(delta > 0))
    return sum(deltas) / len(deltas), sum(successes) / len(successes)


def bootstrap_difference(a: list[float], b: list[float], seed: int, draws: int = 2000) -> tuple[float, float, float, float]:
    rng = random.Random(seed)
    differences = [x - y for x, y in zip(a, b)]
    observed = sum(differences) / len(differences)
    boot = []
    for _ in range(draws):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    low, high = boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1]
    # A paired sign-flip permutation p-value for the null of zero mean effect.
    extreme = 0
    for _ in range(draws):
        signed = [value if rng.random() < 0.5 else -value for value in differences]
        if abs(sum(signed) / len(signed)) >= abs(observed):
            extreme += 1
    p = (extreme + 1) / (draws + 1)
    return observed, low, high, p


def analyze(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    methods = data.get("methods", [])
    if not data.get("items") or not {"jlens", "logit"}.issubset(methods):
        return []
    values = {method: [] for method in methods}
    for item in data["items"]:
        for method in methods:
            delta, success = item_values(item, method)
            values[method].append({"delta": delta, "success": success})
    pairs = [("jlens", "logit")]
    for method in ("tuned", "tuned_pile", "random"):
        if method in methods:
            pairs.append((method, "logit"))
    rows = []
    for left, right in pairs:
        left_delta = [row["delta"] for row in values[left]]
        right_delta = [row["delta"] for row in values[right]]
        left_success = [row["success"] for row in values[left]]
        right_success = [row["success"] for row in values[right]]
        delta = bootstrap_difference(left_delta, right_delta, seed=1000 + len(rows))
        success = bootstrap_difference(left_success, right_success, seed=2000 + len(rows))
        rows.append({
            "file": str(path.relative_to(ROOT)),
            "model": data.get("model"),
            "task": "verbal" if "target_rank_before" in data["items"][0] else ("multihop" if "swap_answer_rank_before" in data["items"][0] else "flexible"),
            "left": left,
            "right": right,
            "n_items": len(data["items"]),
            "n_layers": len(data.get("layers", [])),
            "left_mean_delta": sum(left_delta) / len(left_delta),
            "right_mean_delta": sum(right_delta) / len(right_delta),
            "paired_delta_mean": delta[0],
            "paired_delta_ci_low": delta[1],
            "paired_delta_ci_high": delta[2],
            "paired_delta_p": delta[3],
            "left_success_rate": sum(left_success) / len(left_success),
            "right_success_rate": sum(right_success) / len(right_success),
            "paired_success_diff": success[0],
            "paired_success_ci_low": success[1],
            "paired_success_ci_high": success[2],
            "paired_success_p": success[3],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*-causal-*.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/analysis/paired_causal_effects.json")
    args = parser.parse_args()
    rows = []
    for path in sorted(RESULTS.glob(args.glob)):
        if any(tag in path.name for tag in ("smoke", "short-2m")):
            continue
        try:
            rows.extend(analyze(path))
        except (KeyError, ValueError, TypeError):
            continue
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"analyzed {len(rows)} paired comparisons across saved artifacts")
    print("file | left vs right | paired success diff | 95% CI | p")
    for row in rows:
        print(f"{row['file']} | {row['left']} vs {row['right']} | {row['paired_success_diff']:+.3f} | [{row['paired_success_ci_low']:+.3f}, {row['paired_success_ci_high']:+.3f}] | {row['paired_success_p']:.3f}")


if __name__ == "__main__":
    main()
