#!/usr/bin/env python3
"""Run a small observational ProofWriter factorial smoke test.

This is deliberately a pre-causal benchmark check.  It scores the model's
next-token evidence for True/False/Unknown on the original theory and on
versions with the target entity's A fact, B fact, or both facts removed.  A
useful conjunction item should lose evidence for the positive conclusion when
either necessary fact is removed; this does not yet establish a J-lens claim.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import transformers
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))
from extract_proofwriter_strong_conjunctions import parse_theory


LABELS = ("True", "False", "Unknown")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def remove_fact(theory: str, entity: str, predicate: str) -> str:
    """Remove explicit target facts, leaving rules and other entities intact."""
    pattern = re.compile(
        rf"(?i)(?<![A-Za-z0-9]){re.escape(entity)}\s+is\s+{re.escape(predicate)}\.(?:\s*)"
    )
    return pattern.sub("", theory)


def make_variants(row: dict[str, object]) -> dict[str, str]:
    theory = str(row["theory"])
    entity = str(row["entity"])
    a = str(row["conjunct_a"])
    b = str(row["conjunct_b"])
    return {
        "full": theory,
        "minus_a": remove_fact(theory, entity, a),
        "minus_b": remove_fact(theory, entity, b),
        "minus_ab": remove_fact(remove_fact(theory, entity, a), entity, b),
    }


def prompt(theory: str, question: str) -> str:
    return f"Facts and rules:\n{theory}\n\nQuestion: {question}\nAnswer:"


def exact_positive_label(theory: str, entity: str, conclusion: str) -> str:
    """Evaluate the positive, non-negated ProofWriter fragment under OWA."""
    facts, rules = parse_theory(theory)
    closure = {name: set(predicates) for name, predicates in facts.items()}
    changed = True
    while changed:
        changed = False
        for name in list(closure):
            for first, second, result in rules:
                if first in closure[name] and (second is None or second in closure[name]) and result not in closure[name]:
                    closure[name].add(result)
                    changed = True
    return "True" if conclusion in closure.get(entity.lower(), set()) else "Unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/alex/models/smollm2-135m-instruct")
    parser.add_argument("--data", default="data/benchmarks/proofwriter-strong-conjunctions.jsonl")
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--out", default="data/experiments/proofwriter-smollm2-factorial-smoke.json")
    args = parser.parse_args()

    log(f"loading {args.max_items} ProofWriter factorial items")
    available = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    rows = [
        row for row in available
        if (args.split == "all" or row.get("source_split") == args.split)
        and "not " not in str(row["theory"]).lower()
        and "not " not in str(row["question"]).lower()
    ][: args.max_items]
    if not rows:
        raise SystemExit("no benchmark rows found")

    log("loading tokenizer and model")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    log(f"model ready; CUDA free={torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")

    cases = []
    for index, row in enumerate(rows):
        for condition, theory in make_variants(row).items():
            cases.append({
                "item_index": index,
                "source_id": row.get("source_id"),
                "entity": row["entity"],
                "conjunct_a": row["conjunct_a"],
                "conjunct_b": row["conjunct_b"],
                "conclusion": row["conclusion"],
                "gold": row.get("source_answer"),
                "condition": condition,
                "exact_label": exact_positive_label(theory, str(row["entity"]), str(row["conclusion"])),
                "text": prompt(theory, str(row["question"])),
            })

    label_ids = []
    for label in LABELS:
        ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise SystemExit(f"expected one-token label for {label!r}, got {ids}")
        label_ids.append(ids[0])

    log(f"scoring {len(cases)} prompts in batches of {args.batch_size}")
    results = []
    with torch.inference_mode():
        for start in range(0, len(cases), args.batch_size):
            batch = cases[start : start + args.batch_size]
            encoded = tokenizer(
                [case["text"] for case in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(model.device)
            logits = model(**encoded).logits[:, -1, :]
            label_logits = logits[:, label_ids]
            probabilities = label_logits.softmax(dim=-1).float().cpu()
            for case, probs in zip(batch, probabilities):
                best = int(probs.argmax())
                results.append({
                    key: case[key]
                    for key in ("item_index", "source_id", "entity", "conjunct_a", "conjunct_b", "conclusion", "gold", "condition", "exact_label")
                } | {
                    "predicted": LABELS[best],
                    "probabilities": {label: round(float(probs[i]), 6) for i, label in enumerate(LABELS)},
                })
            log(f"scored {min(start + len(batch), len(cases))}/{len(cases)} prompts")

    by_item = {}
    for result in results:
        by_item.setdefault(result["item_index"], {})[result["condition"]] = result

    summary = []
    for index, item_results in sorted(by_item.items()):
        full = item_results["full"]
        full_true = full["probabilities"]["True"]
        summary.append([
            index,
            full["gold"],
            f"{full['predicted']} / {full['exact_label']}",
            f"{full_true:.3f}",
            f"{item_results['minus_a']['predicted']} / {item_results['minus_a']['exact_label']}",
            f"{item_results['minus_b']['predicted']} / {item_results['minus_b']['exact_label']}",
            f"{item_results['minus_ab']['predicted']} / {item_results['minus_ab']['exact_label']}",
            f"{full_true - item_results['minus_a']['probabilities']['True']:+.3f}",
            f"{full_true - item_results['minus_b']['probabilities']['True']:+.3f}",
        ])
    print("\nProofWriter factorial smoke summary:")
    print(tabulate(summary, headers=["item", "gold", "full model / exact", "P(True)", "−A model / exact", "−B model / exact", "−A−B model / exact", "ΔP(True) A", "ΔP(True) B"], tablefmt="github"))

    output = {
        "schema_version": 1,
        "kind": "proofwriter-factorial-observational-smoke",
        "model": args.model,
        "data": args.data,
        "split": args.split,
        "n_items": len(rows),
        "conditions": ["full", "minus_a", "minus_b", "minus_ab"],
        "label_space": LABELS,
        "results": results,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    log(f"wrote {output_path}")


if __name__ == "__main__":
    main()
