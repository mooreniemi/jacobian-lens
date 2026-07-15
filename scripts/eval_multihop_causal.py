"""Causal J-lens swaps on two-hop reasoning prompts.

Each item supplies an implicit intermediate concept and the answer expected
after replacing that intermediate with a same-type alternative. This follows
the paper's coordinate-swap intervention while measuring the downstream
answer, rather than only the immediate verbal-report token.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import transformers
from tabulate import tabulate

import jlens
from jlens.qwen_runtime import configure_qwen_kernels

try:
    from causal_runtime import load_causal_model
except ModuleNotFoundError:
    from scripts.causal_runtime import load_causal_model
sys.path.insert(0, str(Path(__file__).parent))
from eval_verbal_report_causal import (
    candidate_id,
    candidate_ranks,
    final_logits,
    patch_hook,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    ap.add_argument("--lens-file", default="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt")
    ap.add_argument("--lens-revision", default="qwen-n1000")
    ap.add_argument("--lens-local", default=None, help="load a local JacobianLens checkpoint instead of Hub")
    ap.add_argument("--data", default="data/experiments/probe-swap.json")
    ap.add_argument("--categories", nargs="*", default=[])
    ap.add_argument("--layers", default="24,28,30")
    ap.add_argument("--patch-positions", choices=("all", "answer"), default="all")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--methods", default="jlens,logit,random", help="comma-separated: jlens, tuned, logit, random")
    ap.add_argument("--out", default="data/experiments/multihop-causal-results.json")
    ap.add_argument("--qwen-kernels", choices=("auto", "on", "off"), default="on")
    ap.add_argument("--min-free-gib", type=float, default=8.0)
    ap.add_argument("--load-in-4bit", action="store_true", help="load with bitsandbytes NF4 and device_map=auto")
    args = ap.parse_args()

    kernel_mode = configure_qwen_kernels(args.qwen_kernels)
    log(f"Qwen kernel mode: {kernel_mode}")
    free, total = torch.cuda.mem_get_info()
    log(f"preflight: {free / 2**30:.2f} GiB free / {total / 2**30:.2f} GiB total")
    if free / 2**30 < args.min_free_gib:
        raise SystemExit("refusing to start: insufficient free VRAM")

    log(f"loading model {args.model}")
    hf_model = load_causal_model(args.model, load_in_4bit=args.load_in_4bit)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf_model, tokenizer)
    log(f"model ready: {model}; {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB free")
    lens = jlens.JacobianLens.load(args.lens_local) if args.lens_local else jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file, revision=args.lens_revision)
    data = json.loads(Path(args.data).read_text())["items"]
    if args.categories:
        data = [x for x in data if x.get("category") in args.categories]
    if args.max_items:
        data = data[:args.max_items]
    layers = [int(x) for x in args.layers.split(",") if x]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown_methods = set(methods) - {"jlens", "tuned", "logit", "random"}
    if unknown_methods:
        raise SystemExit(f"unknown methods: {sorted(unknown_methods)}")
    if any(layer not in lens.source_layers for layer in layers):
        raise SystemExit(f"requested layers {layers} are not in lens source layers")

    results = []
    table = []
    skipped = []
    for n, item in enumerate(data, 1):
        source_id = candidate_id(tokenizer, item["intermediate"])
        target_id = candidate_id(tokenizer, item["swap_to"])
        answer_id = candidate_id(tokenizer, item["answer"])
        swap_answer_id = candidate_id(tokenizer, item["swap_answer"])
        if None in (source_id, target_id, answer_id, swap_answer_id):
            skipped.append({"name": item["name"], "reason": "non_single_token"})
            continue
        input_ids = model.encode(item["prompt"], max_length=256)
        answer_position = input_ids.shape[1] - 1
        base_logits = final_logits(model, input_ids)
        all_candidates = {item["answer"]: answer_id, item["swap_answer"]: swap_answer_id}
        base_ranks = candidate_ranks(base_logits, all_candidates)
        positions = list(range(answer_position + 1)) if args.patch_positions == "all" else [answer_position]
        log(f"item {n}/{len(data)} {item['name']}: {item['intermediate']} -> {item['swap_to']}; answer {item['answer']} -> {item['swap_answer']}")
        rec = {
            **item,
            "answer_position": answer_position,
            "source_token": source_id,
            "target_token": target_id,
            "answer_token": answer_id,
            "swap_answer_token": swap_answer_id,
            "answer_rank_before": base_ranks[item["answer"]],
            "swap_answer_rank_before": base_ranks[item["swap_answer"]],
            "layers": {},
        }
        for layer in layers:
            lr = {}
            for method in methods:
                mode = "random_matched" if method == "random" else "coordinate_swap"
                vector_kind = "jlens" if method in ("jlens", "random") else ("tuned" if method == "tuned" else "logit")
                handle = patch_hook(model, lens, layer, source_id, target_id, positions=positions, mode=mode, alpha=1.0, vector_kind=vector_kind)
                try:
                    logits = final_logits(model, input_ids)
                finally:
                    handle.remove()
                ranks = candidate_ranks(logits, all_candidates)
                lr[method] = {
                    "answer_rank_after": ranks[item["answer"]],
                    "swap_answer_rank_after": ranks[item["swap_answer"]],
                    "swap_answer_top_k": item["swap_answer"] in [tokenizer.decode([i]).strip() for i in logits.topk(args.top_k).indices.tolist()],
                }
                table.append([item["name"], layer, method, base_ranks[item["answer"]], base_ranks[item["swap_answer"]], ranks[item["answer"]], ranks[item["swap_answer"]]])
            rec["layers"][str(layer)] = lr
        results.append(rec)

    aggregate = []
    for mode in methods:
        changes = []
        target_top1 = target_top5 = 0
        for item in results:
            for lr in item["layers"].values():
                z = lr[mode]
                changes.append(item["swap_answer_rank_before"] - z["swap_answer_rank_after"])
                target_top1 += z["swap_answer_rank_after"] == 1
                target_top5 += z["swap_answer_rank_after"] <= 5
        aggregate.append([mode, len(changes), sum(x > 0 for x in changes), round(statistics.median(changes), 1) if changes else None, target_top1, target_top5])

    print("\nMultihop causal summary:")
    print(tabulate(table, headers=["item", "layer", "method", "answer before", "swap-answer before", "answer after", "swap-answer after"], tablefmt="github"))
    print("\nAggregate swap-answer movement:")
    print(tabulate(aggregate, headers=["intervention", "n", "improved", "median Δrank", "top-1", "top-5"], tablefmt="github"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "lens_file": args.lens_file, "lens_local": args.lens_local, "precision": {"compute_dtype": "bf16", "base_quantization": "nf4" if args.load_in_4bit else "none"}, "kernel_mode": kernel_mode, "patch_positions": args.patch_positions, "layers": layers, "methods": methods, "n_input": len(data), "n_used": len(results), "skipped": skipped, "items": results, "aggregate": aggregate}, ensure_ascii=False, indent=2) + "\n")
    log(f"wrote {len(results)} results ({len(skipped)} skipped) to {out}")


if __name__ == "__main__":
    main()
