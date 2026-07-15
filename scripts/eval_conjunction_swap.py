"""Evaluate conjunction prompts and record J-lens readouts.

This is the observational stage of the conjunction experiment. It evaluates
single-A, single-B, A-and-B, A-swapped, B-swapped, and both-swapped prompts,
then records top-k tokens at each conjunct span and at the answer position.

Example:
    uv run python scripts/eval_conjunction_swap.py \\
      --lens data/lenses/qwen3.5-0.8b-lens.pt --max-items 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import transformers
from tabulate import tabulate

import jlens
from jlens.qwen_runtime import configure_qwen_kernels


def variants(item: dict[str, object], items: list[dict[str, object]] | None = None) -> dict[str, str]:
    prompt = str(item["prompt"])
    a = str(item["conjunct_a"])
    b = str(item["conjunct_b"])
    a2 = str(item["swap_a"])
    b2 = str(item["swap_b"])
    result = {
        "single_a": str(item["single_a"]),
        "single_b": str(item["single_b"]),
        "both": prompt,
        "swap_a": prompt.replace(a, a2, 1),
        "swap_b": prompt.replace(b, b2, 1),
        "swap_both": prompt.replace(a, a2, 1).replace(b, b2, 1),
        "control_order": prompt.replace(f"{a} and {b}", f"{b} and {a}", 1),
        "control_or": prompt.replace(" and ", " or ", 1),
    }
    if items:
        same_category = [other for other in items if other["category"] == item["category"] and other["name"] != item["name"]]
        other_category = [other for other in items if other["category"] != item["category"]]
        if same_category:
            other = same_category[0]
            result["control_same_type_scramble"] = prompt.replace(a, str(other["conjunct_a"]), 1).replace(b, str(other["conjunct_b"]), 1)
        if other_category:
            other = other_category[0]
            result["control_cross_type_scramble"] = prompt.replace(a, str(other["conjunct_a"]), 1).replace(b, str(other["conjunct_b"]), 1)
    return result


def token_positions(tokenizer, text: str, fragment: str) -> list[int]:
    start = text.find(fragment)
    if start < 0:
        return []
    end = start + len(fragment)
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return []
    return [
        i for i, (left, right) in enumerate(offsets)
        if right > start and left < end and right > left
    ]


def top_tokens(logits: torch.Tensor, tokenizer, k: int = 5) -> list[str]:
    ids = logits.topk(k).indices.tolist()
    return [tokenizer.decode([i]).replace("\n", "\\n") for i in ids]


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item()) + 1


def continuation_target(tokenizer, expected: object) -> tuple[str, int] | None:
    if expected is None:
        return None
    text = str(expected)
    target_text = text if text.startswith(" ") else " " + text
    ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    return (target_text, ids[0]) if ids else None


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)



def print_summary(results: list[dict[str, object]]) -> None:
    """Print compact tables so a run is useful without opening the JSON."""
    rows = []
    for item in results:
        conditions = item["conditions"]
        tops = {
            name: (conditions[name]["model_top"][0] if conditions[name]["model_top"] else "<none>")
            for name in ("single_a", "single_b", "both", "swap_a", "swap_b", "swap_both")
        }
        base = tops["both"]
        both_result = conditions["both"]
        best_j = both_result.get("best_j_lens")
        rows.append([
            item["name"], both_result.get("expected_answer") or "—", tops["both"],
            both_result.get("model_expected_rank") or "—",
            f"{best_j['rank']} @ L{best_j['layer']} P{best_j['position']}" if best_j else "—",
            tops["swap_a"], tops["swap_b"], tops["swap_both"],
            "yes" if tops["swap_a"] != base else "no",
            "yes" if tops["swap_b"] != base else "no",
        ])
    print("\nFindings (top-1 model prediction at the answer position):")
    print(tabulate(rows, headers=["item", "expected", "both", "model rank", "best J-lens", "swap A", "swap B", "swap A+B", "A changed?", "B changed?"], tablefmt="github"))

    control_rows = []
    for item in results:
        base = item["conditions"]["both"]["model_top"][0]
        for condition in ("control_order", "control_or", "control_same_type_scramble", "control_cross_type_scramble"):
            result = item["conditions"].get(condition)
            if result is not None:
                top = result["model_top"][0] if result["model_top"] else "<none>"
                control_rows.append([item["name"], condition, top, "yes" if top != base else "no"])
    print("\nObservational control arms (not pooled with primary swaps):")
    print(tabulate(control_rows, headers=["item", "control", "top-1", "changed vs both?"], tablefmt="github"))

    control_summary = []
    for condition in ("control_order", "control_or", "control_same_type_scramble", "control_cross_type_scramble"):
        comparisons = []
        for item in results:
            result = item["conditions"].get(condition)
            if result is not None and result["model_top"]:
                comparisons.append(result["model_top"][0] != item["conditions"]["both"]["model_top"][0])
        changed = sum(comparisons)
        control_summary.append([condition, changed, len(comparisons), f"{changed / len(comparisons):.1%}" if comparisons else "n/a"])
    print("\nAggregate observational control effects:")
    print(tabulate(control_summary, headers=["control", "changed", "scored", "fraction"], tablefmt="github"))

    n = len(results)
    change_rows = []
    for label, condition in (("A swap", "swap_a"), ("B swap", "swap_b"), ("A+B swap", "swap_both")):
        changed = sum(
            item["conditions"][condition]["model_top"][0]
            != item["conditions"]["both"]["model_top"][0]
            for item in results
        )
        change_rows.append([label, changed, n, f"{changed / n:.1%}" if n else "n/a"])
    print("\nAggregate top-1 change relative to the conjunction baseline:")
    print(tabulate(change_rows, headers=["contrast", "changed", "items", "fraction"], tablefmt="github"))

    accuracy_rows = []
    for condition in ("single_a", "single_b", "both", "swap_a", "swap_b", "swap_both"):
        scored = [
            item["conditions"][condition]
            for item in results
            if item["conditions"][condition].get("expected_answer") is not None
        ]
        hits = sum(
            result["model_top"]
            and result["model_top"][0].strip() == result["expected_answer"].strip()
            for result in scored
        )
        accuracy_rows.append([condition, hits, len(scored), f"{hits / len(scored):.1%}" if scored else "n/a"])
    if any(row[2] for row in accuracy_rows):
        print("\nExpected-answer top-1 agreement (only declared answers):")
        print(tabulate(accuracy_rows, headers=["condition", "hits", "scored", "fraction"], tablefmt="github"))

def free_vram_gib() -> float:
    if not torch.cuda.is_available():
        return float("inf")
    free, _total = torch.cuda.mem_get_info()
    return free / 2**30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--qwen-kernels", choices=("auto", "on", "off"), default="on")
    parser.add_argument("--lens", required=True)
    parser.add_argument("--data", default="data/experiments/conjunction-swap.json")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--out", default="data/experiments/conjunction-swap-results.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-summary", action="store_true",
                        help="only write JSON and progress logs")
    parser.add_argument("--min-free-gib", type=float, default=8.0,
                        help="refuse to start below this free VRAM (default: 8)")
    args = parser.parse_args()

    log("preflight: checking CUDA and free VRAM")
    free = free_vram_gib()
    log(f"preflight: {free:.2f} GiB free")
    if free < args.min_free_gib:
        raise SystemExit(
            f"refusing to start: only {free:.2f} GiB free, need "
            f"at least {args.min_free_gib:.2f} GiB; release another model first"
        )

    jlens.configure_logging()
    kernel_mode = configure_qwen_kernels(args.qwen_kernels)
    print(f"Qwen kernel mode: {kernel_mode}", flush=True)
    log(f"loading model {args.model}")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf_model, tokenizer)
    log(f"model ready: {model}; {free_vram_gib():.2f} GiB free")
    log(f"loading lens {args.lens}")
    lens = jlens.JacobianLens.load(args.lens)
    data = json.loads(Path(args.data).read_text())
    items = data["items"][: args.max_items]
    log(f"loaded {len(items)} conjunction items; {len(lens.source_layers)} lens layers")

    results = []
    for n, item in enumerate(items, 1):
        record = {"name": item["name"], "category": item["category"], "conditions": {}}
        log(f"item {n}/{len(items)}: {item['name']}")
        for condition, prompt in variants(item, items).items():
            log(f"  condition {condition}")
            a_fragment = str(item["swap_a"]) if condition in {"swap_a", "swap_both"} else str(item["conjunct_a"])
            b_fragment = str(item["swap_b"]) if condition in {"swap_b", "swap_both"} else str(item["conjunct_b"])
            a_positions = token_positions(tokenizer, prompt, a_fragment)
            b_positions = token_positions(tokenizer, prompt, b_fragment)
            input_ids = model.encode(prompt, max_length=128)
            answer_position = input_ids.shape[1] - 1
            positions = sorted(set(a_positions + b_positions + [answer_position]))
            layers = lens.source_layers
            lens_logits, model_logits, _ = lens.apply(
                model, prompt, layers=layers, positions=positions
            )
            logit_lens, _, _ = lens.apply(
                model, prompt, layers=layers, positions=positions, use_jacobian=False
            )
            expected_by_condition = {
                "single_a": item.get("answer"),
                "single_b": item.get("answer"),
                "both": item.get("answer"),
                "swap_a": item.get("swap_a_answer"),
                "swap_b": item.get("swap_b_answer"),
                "swap_both": item.get("swap_both_answer"),
            }
            expected_answer = expected_by_condition.get(condition)
            target = continuation_target(tokenizer, expected_answer)
            target_text = target[0] if target else None
            target_id = target[1] if target else None
            answer_index = positions.index(answer_position)
            best_j = None
            best_ll = None
            j_answer_ranks = {}
            ll_answer_ranks = {}
            if target_id is not None:
                for layer in layers:
                    j_ranks = [token_rank(logits, target_id) for logits in lens_logits[layer]]
                    ll_ranks = [token_rank(logits, target_id) for logits in logit_lens[layer]]
                    j_answer_ranks[str(layer)] = j_ranks[answer_index]
                    ll_answer_ranks[str(layer)] = ll_ranks[answer_index]
                    layer_best_j = min(j_ranks)
                    layer_best_ll = min(ll_ranks)
                    layer_j_pos = positions[j_ranks.index(layer_best_j)]
                    layer_ll_pos = positions[ll_ranks.index(layer_best_ll)]
                    if best_j is None or layer_best_j < best_j["rank"]:
                        best_j = {"rank": layer_best_j, "layer": layer, "position": layer_j_pos}
                    if best_ll is None or layer_best_ll < best_ll["rank"]:
                        best_ll = {"rank": layer_best_ll, "layer": layer, "position": layer_ll_pos}
            condition_result = {
                "prompt": prompt,
                "positions": {"a": a_positions, "b": b_positions, "answer": answer_position},
                "expected_answer": expected_answer,
                "target_text": target_text,
                "target_token": tokenizer.decode([target_id]) if target_id is not None else None,
                "model_top": top_tokens(model_logits[answer_index], tokenizer, args.top_k),
                "model_expected_rank": token_rank(model_logits[answer_index], target_id) if target_id is not None else None,
                "best_j_lens": best_j,
                "best_logit_lens": best_ll,
                "j_lens_answer_ranks": j_answer_ranks,
                "logit_lens_answer_ranks": ll_answer_ranks,
                "lens_top": {
                    str(layer): {
                        str(position): top_tokens(lens_logits[layer][index], tokenizer, args.top_k)
                        for index, position in enumerate(positions)
                    }
                    for layer in layers
                },
            }
            record["conditions"][condition] = condition_result
        results.append(record)
        log(f"completed item {n}/{len(items)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "lens": args.lens, "precision": {"compute_dtype": "bf16", "base_quantization": "none"}, "items": results}, ensure_ascii=False, indent=2) + "\n")
    log(f"wrote {len(results)} results to {out}")
    if not args.no_summary:
        print_summary(results)


if __name__ == "__main__":
    main()
