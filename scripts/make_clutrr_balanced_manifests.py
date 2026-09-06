#!/usr/bin/env python3
"""Make fixed, class-balanced CLUTRR selection and confirmation manifests."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

LABELS = (
    "aunt", "brother", "daughter", "daughter-in-law", "father", "father-in-law",
    "granddaughter", "grandfather", "grandmother", "grandson", "mother",
    "mother-in-law", "nephew", "niece", "sister", "son", "son-in-law", "uncle",
)


def load(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/benchmarks/clutrr-train.jsonl")
    parser.add_argument("--validation", default="data/benchmarks/clutrr-validation.jsonl")
    parser.add_argument("--selection-per-class", type=int, default=30)
    parser.add_argument("--confirmation-per-class", type=int, default=35)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--selection-out", default="data/benchmarks/clutrr-selection-540.jsonl")
    parser.add_argument("--confirmation-out", default="data/benchmarks/clutrr-confirmation-630.jsonl")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    def sample(path: str, n: int, split: str) -> list[dict[str, object]]:
        groups: dict[int, list[dict[str, object]]] = defaultdict(list)
        for index, row in enumerate(load(Path(path))):
            item = {"id": f"clutrr-{split}-{index:06d}", **row}
            groups[int(row["labels"])].append(item)
        missing = [label for label in range(len(LABELS)) if len(groups[label]) < n]
        if missing:
            raise SystemExit(f"not enough rows for labels {missing} in {path}")
        output = []
        for label in range(len(LABELS)):
            chosen = groups[label]
            rng.shuffle(chosen)
            output.extend(chosen[:n])
        rng.shuffle(output)
        return output

    selection = sample(args.train, args.selection_per_class, "train")
    confirmation = sample(args.validation, args.confirmation_per_class, "validation")
    for path, rows in ((Path(args.selection_out), selection), (Path(args.confirmation_out), confirmation)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    metadata = {
        "dataset": "tasksource/clutrr",
        "labels": list(enumerate(LABELS)),
        "selection": {"path": args.selection_out, "n": len(selection), "per_class": args.selection_per_class, "source_split": "train"},
        "confirmation": {"path": args.confirmation_out, "n": len(confirmation), "per_class": args.confirmation_per_class, "source_split": "validation"},
        "seed": args.seed,
    }
    Path(args.selection_out).with_name("clutrr-manifests.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
