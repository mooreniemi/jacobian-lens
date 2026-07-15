"""Causal J-lens swaps for Anthropic's flexible-generalization protocol."""
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
    ap.add_argument("--data", default="data/experiments/flexible-generalization.json")
    ap.add_argument("--layers", default="24,28,30")
    ap.add_argument("--methods", default="jlens,logit,random")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--qwen-kernels", choices=("auto", "on", "off"), default="on")
    ap.add_argument("--out", default="data/experiments/flexible-causal-results.json")
    ap.add_argument("--min-free-gib", type=float, default=8.0)
    ap.add_argument("--load-in-4bit", action="store_true", help="load with bitsandbytes NF4 and device_map=auto")
    args = ap.parse_args()

    kernel_mode = configure_qwen_kernels(args.qwen_kernels)
    free, total = torch.cuda.mem_get_info()
    log(f"Qwen kernel mode: {kernel_mode}")
    log(f"preflight: {free / 2**30:.2f} GiB free / {total / 2**30:.2f} GiB total")
    if free / 2**30 < args.min_free_gib:
        raise SystemExit("refusing to start: insufficient free VRAM")

    log(f"loading model {args.model}")
    hf_model = load_causal_model(args.model, load_in_4bit=args.load_in_4bit)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(args.lens_local) if args.lens_local else jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file, revision=args.lens_revision)
    log(f"model ready: {model}; {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB free")

    raw = json.loads(Path(args.data).read_text())
    trials = []
    for category in raw["categories"]:
        for func in category["funcs"]:
            for source in category["args"]:
                for target in category["args"]:
                    if source == target:
                        continue
                    trials.append({
                        "category": category["name"],
                        "function": func["name"],
                        "template": func["template"],
                        "source": source,
                        "target": target,
                        "source_answer": func["answers"][source],
                        "target_answer": func["answers"][target],
                    })
    if args.max_items:
        trials = trials[:args.max_items]
    layers = [int(x) for x in args.layers.split(",") if x]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    if any(layer not in lens.source_layers for layer in layers):
        raise SystemExit(f"requested layers {layers} are not in lens source layers")
    if set(methods) - {"jlens", "tuned", "logit", "random"}:
        raise SystemExit(f"unknown methods: {sorted(set(methods) - {'jlens', 'tuned', 'logit', 'random'})}")

    results, skipped, table = [], [], []
    for i, trial in enumerate(trials, 1):
        ids = {key: candidate_id(tokenizer, trial[key]) for key in ("source", "target", "source_answer", "target_answer")}
        if any(value is None for value in ids.values()):
            skipped.append({**trial, "reason": "non_single_token"})
            continue
        prompt = trial["template"].format(arg=trial["source"])
        input_ids = model.encode(prompt, max_length=256)
        positions = list(range(input_ids.shape[1]))
        base_logits = final_logits(model, input_ids)
        candidates = {trial["source_answer"]: ids["source_answer"], trial["target_answer"]: ids["target_answer"]}
        base_ranks = candidate_ranks(base_logits, candidates)
        rec = {**trial, "prompt": prompt, "source_answer_rank_before": base_ranks[trial["source_answer"]], "target_answer_rank_before": base_ranks[trial["target_answer"]], "layers": {}}
        log(f"trial {i}/{len(trials)} {trial['category']}/{trial['function']}: {trial['source']} -> {trial['target']}")
        for layer in layers:
            lr = {}
            for method in methods:
                mode = "random_matched" if method == "random" else "coordinate_swap"
                vector_kind = "jlens" if method in ("jlens", "random") else ("tuned" if method == "tuned" else "logit")
                handle = patch_hook(model, lens, layer, ids["source"], ids["target"], positions=positions, mode=mode, alpha=1.0, vector_kind=vector_kind)
                try:
                    logits = final_logits(model, input_ids)
                finally:
                    handle.remove()
                ranks = candidate_ranks(logits, candidates)
                lr[method] = {"source_answer_rank_after": ranks[trial["source_answer"]], "target_answer_rank_after": ranks[trial["target_answer"]]}
                table.append([trial["category"], trial["function"], trial["source"], trial["target"], layer, method, base_ranks[trial["target_answer"]], ranks[trial["target_answer"]]])
            rec["layers"][str(layer)] = lr
        results.append(rec)

    aggregate = []
    for method in methods:
        changes = []
        top1 = top5 = 0
        for item in results:
            for lr in item["layers"].values():
                after = lr[method]["target_answer_rank_after"]
                changes.append(item["target_answer_rank_before"] - after)
                top1 += after == 1
                top5 += after <= 5
        aggregate.append([method, len(changes), sum(x > 0 for x in changes), round(statistics.median(changes), 1) if changes else None, top1, top5])

    print("\nFlexible-generalization causal summary:")
    print(tabulate(table, headers=["category", "function", "source", "target", "layer", "method", "target before", "target after"], tablefmt="github"))
    print("\nAggregate target-answer movement:")
    print(tabulate(aggregate, headers=["method", "n", "improved", "median Δrank", "top-1", "top-5"], tablefmt="github"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "lens_file": args.lens_file, "lens_local": args.lens_local, "precision": {"compute_dtype": "bf16", "base_quantization": "nf4" if args.load_in_4bit else "none"}, "kernel_mode": kernel_mode, "layers": layers, "methods": methods, "n_input": len(trials), "n_used": len(results), "skipped": skipped, "items": results, "aggregate": aggregate}, ensure_ascii=False, indent=2) + "\n")
    log(f"wrote {len(results)} results ({len(skipped)} skipped) to {out}")


if __name__ == "__main__":
    main()
