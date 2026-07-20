#!/usr/bin/env python3
"""Create source-disjoint, label-balanced ProofWriter selection/confirmation sets."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

LABELS = ("True", "False", "Unknown")


def source_group(row: dict[str, object]) -> str:
    # A ProofWriter theory can yield several questions/labels. Keep the whole
    # theory in one split so related questions cannot leak across the split.
    theory = str(row["theory"])
    return hashlib.sha1(theory.encode("utf-8")).hexdigest()[:16]


def strata(row: dict[str, object]) -> tuple[str, int, str]:
    return (str(row["config"]), int(row["maxD"]), str(row["answer"]))


def choose(rows: list[dict[str, object]], per_class: int, used: set[str], rng: random.Random) -> tuple[list[dict[str, object]], set[str]]:
    by_label_and_group: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = source_group(row)
        if group not in used and str(row["answer"]) in LABELS:
            by_label_and_group[str(row["answer"])][group].append(row)

    chosen: list[dict[str, object]] = []
    newly_used: set[str] = set()
    occupied = set(used)
    for label in LABELS:
        groups = [group for group in by_label_and_group[label] if group not in occupied]
        rng.shuffle(groups)
        # Shuffle within strata, then round-robin strata so depth/configuration
        # are represented rather than taking one arbitrary slice of the corpus.
        buckets: dict[tuple[str, int, str], list[str]] = defaultdict(list)
        for group in groups:
            buckets[strata(by_label_and_group[label][group][0])].append(group)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        ordered: list[str] = []
        while buckets:
            for key in list(buckets):
                bucket = buckets[key]
                if bucket:
                    ordered.append(bucket.pop())
                if not bucket:
                    del buckets[key]
        if len(ordered) < per_class:
            raise RuntimeError(f"only {len(ordered)} source-disjoint {label} groups available; need {per_class}")
        for group in ordered[:per_class]:
            # Select one question per theory group. The group itself is the
            # resampling unit for uncertainty analysis.
            row = dict(rng.choice(by_label_and_group[label][group]))
            row["source_group"] = group
            row["source_split"] = "selection" if not used else "confirmation"
            chosen.append(row)
            newly_used.add(group)
            occupied.add(group)
    rng.shuffle(chosen)
    return chosen, newly_used


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--selection-per-class", type=int, default=100)
    parser.add_argument("--confirmation-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--selection-out", default="data/benchmarks/proofwriter-selection-300.jsonl")
    parser.add_argument("--confirmation-out", default="data/benchmarks/proofwriter-confirmation-1500.jsonl")
    parser.add_argument("--metadata-out", default="data/benchmarks/proofwriter-confirmation-manifests.json")
    args = parser.parse_args()

    dataset = load_dataset("tasksource/proofwriter", split=args.split)
    rows = [dict(row) for row in dataset if str(row["answer"]) in LABELS]
    rng = random.Random(args.seed)
    selection, selection_groups = choose(rows, args.selection_per_class, set(), rng)
    confirmation, confirmation_groups = choose(rows, args.confirmation_per_class, selection_groups, rng)
    if selection_groups & confirmation_groups:
        raise AssertionError("source-group leakage between selection and confirmation")
    for name, subset, expected in (("selection", selection, args.selection_per_class), ("confirmation", confirmation, args.confirmation_per_class)):
        counts = Counter(str(row["answer"]) for row in subset)
        if counts != Counter({label: expected for label in LABELS}):
            raise AssertionError(f"{name} label imbalance: {counts}")
        if len({row["source_group"] for row in subset}) != len(subset):
            raise AssertionError(f"{name} contains multiple questions from a source group")
    write(Path(args.selection_out), selection)
    write(Path(args.confirmation_out), confirmation)
    metadata = {
        "schema_version": 1,
        "dataset": "tasksource/proofwriter",
        "split": args.split,
        "seed": args.seed,
        "source_group_definition": "sha1(theory)[:16]; one question per theory group",
        "selection": {"path": args.selection_out, "items": len(selection), "source_groups": len(selection_groups)},
        "confirmation": {"path": args.confirmation_out, "items": len(confirmation), "source_groups": len(confirmation_groups)},
        "source_group_overlap": len(selection_groups & confirmation_groups),
        "label_counts": {
            "selection": dict(Counter(str(row["answer"]) for row in selection)),
            "confirmation": dict(Counter(str(row["answer"]) for row in confirmation)),
        },
        "config_counts": {
            "selection": dict(Counter(str(row["config"]) for row in selection)),
            "confirmation": dict(Counter(str(row["config"]) for row in confirmation)),
        },
    }
    Path(args.metadata_out).write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
