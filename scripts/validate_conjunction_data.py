"""Validate conjunction experiment fixtures before running model evaluations.

This checks structural validity and tokenizer span alignment. It deliberately
reports warnings for scientifically questionable items instead of silently
excluding them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabulate import tabulate
from transformers import AutoTokenizer


def token_positions(tokenizer, text: str, fragment: str) -> list[int]:
    start = text.find(fragment)
    if start < 0:
        return []
    end = start + len(fragment)
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    return [
        i for i, (left, right) in enumerate(encoded["offset_mapping"])
        if right > start and left < end and right > left
    ]


def variants(item: dict[str, object]) -> dict[str, str]:
    prompt = str(item["prompt"])
    return {
        "single_a": str(item["single_a"]),
        "single_b": str(item["single_b"]),
        "both": prompt,
        "swap_a": prompt.replace(str(item["conjunct_a"]), str(item["swap_a"]), 1),
        "swap_b": prompt.replace(str(item["conjunct_b"]), str(item["swap_b"]), 1),
        "swap_both": prompt.replace(str(item["conjunct_a"]), str(item["swap_a"]), 1).replace(
            str(item["conjunct_b"]), str(item["swap_b"]), 1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/experiments/conjunction-swap.json")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--strict", action="store_true", help="exit nonzero on warnings")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    items = json.loads(Path(args.data).read_text())["items"]
    rows = []
    errors = []
    warnings = []
    for item in items:
        name = str(item["name"])
        item_errors = []
        item_warnings = []
        a = str(item["conjunct_a"])
        b = str(item["conjunct_b"])
        a2 = str(item["swap_a"])
        b2 = str(item["swap_b"])
        if a2 == a:
            item_warnings.append("A replacement is unchanged")
        if b2 == b:
            item_warnings.append("B replacement is unchanged")
        if a not in str(item["prompt"]):
            item_errors.append("A fragment absent from conjunction prompt")
        if b not in str(item["prompt"]):
            item_errors.append("B fragment absent from conjunction prompt")
        if a not in str(item["single_a"]):
            item_warnings.append("single-A prompt does not contain A text")
        if b not in str(item["single_b"]):
            item_warnings.append("single-B prompt does not contain B text")

        condition_rows = []
        for condition, prompt in variants(item).items():
            a_fragment = a2 if condition in {"swap_a", "swap_both"} else a
            b_fragment = b2 if condition in {"swap_b", "swap_both"} else b
            a_pos = token_positions(tokenizer, prompt, a_fragment)
            b_pos = token_positions(tokenizer, prompt, b_fragment)
            if condition not in {"single_a", "single_b"} and not a_pos:
                item_errors.append(f"{condition}: A span not tokenized")
            if condition not in {"single_a", "single_b"} and not b_pos:
                item_errors.append(f"{condition}: B span not tokenized")
            condition_rows.append((condition, len(tokenizer(prompt)["input_ids"]), len(a_pos), len(b_pos)))
        if item_errors:
            errors.append((name, item_errors))
        if item_warnings:
            warnings.append((name, item_warnings))
        rows.append([name, "OK" if not item_errors else "ERROR", "WARN" if item_warnings else "OK", "; ".join(item_warnings)])

    print(tabulate(rows, headers=["item", "structure", "scientific checks", "warnings"], tablefmt="github"))
    print(f"\\nValidated {len(items)} items: {len(errors)} errors, {len(warnings)} warnings")
    for name, messages in errors:
        print(f"ERROR {name}: {'; '.join(messages)}")
    for name, messages in warnings:
        print(f"WARN  {name}: {'; '.join(messages)}")
    if errors or (args.strict and warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
