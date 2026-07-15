"""Build a balanced fitting corpus from the repository evaluation prompts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

CATEGORIES = {
    "multihop": "lens-eval-multihop.json",
    "multilingual": "lens-eval-multilingual.json",
    "order_ops": "lens-eval-order-ops.json",
    "association": "lens-eval-association.json",
    "poetry": "lens-eval-poetry.json",
    "typo": "lens-eval-typo.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--min-tokens", type=int, default=18)
    parser.add_argument("--out", default="data/lens-prompts/eval-mix.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "data" / "evaluations"
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prefix = "Please carefully consider the following task and infer the relevant answer from the context before completing it. "
    records = []
    for category, filename in CATEGORIES.items():
        items = json.loads((root / filename).read_text())["items"]
        if len(items) < args.per_category:
            raise SystemExit(f"{category} has only {len(items)} items, need {args.per_category}")
        for item in rng.sample(items, args.per_category):
            original = item["prompt"]
            prompt = original
            if len(tokenizer(prompt, add_special_tokens=True)["input_ids"]) < args.min_tokens:
                prompt = prefix + prompt
            token_count = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            if token_count < args.min_tokens:
                raise SystemExit(f"could not make {item['name']} long enough: {token_count} tokens")
            records.append({"category": category, "name": item["name"], "prompt": prompt, "evaluation_prompt": original, "token_count": token_count})
    rng.shuffle(records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": args.seed, "per_category": args.per_category, "model": args.model, "min_tokens": args.min_tokens, "items": records}, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(records)} prompts to {out}")


if __name__ == "__main__":
    main()
