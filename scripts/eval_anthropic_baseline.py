"""Systematic baseline for the original Anthropic J-lens walkthrough.

This keeps the original multihop prompt as a named item, compares J-lens with
the vanilla logit lens, and prints tables plus detailed layer/position JSON.
It performs no interventions.
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


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def top_tokens(logits: torch.Tensor, tokenizer, k: int) -> list[str]:
    return [tokenizer.decode([i]).replace("\n", "\\n") for i in logits.topk(k).indices.tolist()]


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item()) + 1


def free_vram_gib() -> float:
    if not torch.cuda.is_available():
        return float("inf")
    free, _ = torch.cuda.mem_get_info()
    return free / 2**30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--qwen-kernels", choices=("auto", "on", "off"), default="on")
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument("--lens-file", default="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt")
    parser.add_argument("--lens-revision", default="qwen-n1000")
    parser.add_argument("--data", default="data/experiments/anthropic-baseline.json")
    parser.add_argument("--out", default="data/experiments/anthropic-baseline-results.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    args = parser.parse_args()

    kernel_mode = configure_qwen_kernels(args.qwen_kernels)
    print(f"Qwen kernel mode: {kernel_mode}", flush=True)

    log("preflight: checking CUDA and free VRAM")
    free = free_vram_gib()
    log(f"preflight: {free:.2f} GiB free")
    if free < args.min_free_gib:
        raise SystemExit(f"refusing to start: only {free:.2f} GiB free")

    log(f"loading model {args.model}")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf_model, tokenizer)
    log(f"model ready: {model}; {free_vram_gib():.2f} GiB free")
    log(f"loading pretrained lens {args.lens_repo}:{args.lens_revision}")
    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file, revision=args.lens_revision)
    data = json.loads(Path(args.data).read_text())
    results = []
    summary_rows = []

    for n, item in enumerate(data["items"], 1):
        log(f"item {n}/{len(data['items'])}: {item['name']}")
        prompt = item["prompt"]
        target_text = item["expected"] if item["expected"].startswith(" ") else " " + item["expected"]
        encoded_target = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        target_id = encoded_target[0]
        input_ids = model.encode(prompt, max_length=512)
        answer_position = input_ids.shape[1] - 1
        layers = lens.source_layers
        positions = list(range(input_ids.shape[1]))
        jlens_logits, model_logits, _ = lens.apply(model, prompt, layers=layers, positions=positions)
        logit_lens, _, _ = lens.apply(model, prompt, layers=layers, positions=positions, use_jacobian=False)

        layer_results = {}
        best_j = (10**9, None, None)
        best_ll = (10**9, None, None)
        for layer in layers:
            j = jlens_logits[layer]
            ll = logit_lens[layer]
            j_ranks = [token_rank(j[pos], target_id) for pos in range(j.shape[0])]
            ll_ranks = [token_rank(ll[pos], target_id) for pos in range(ll.shape[0])]
            for pos, rank in enumerate(j_ranks):
                if rank < best_j[0]:
                    best_j = (rank, layer, pos)
            for pos, rank in enumerate(ll_ranks):
                if rank < best_ll[0]:
                    best_ll = (rank, layer, pos)
            layer_results[str(layer)] = {
                "j_lens_answer_top": top_tokens(j[answer_position], tokenizer, args.top_k),
                "logit_lens_answer_top": top_tokens(ll[answer_position], tokenizer, args.top_k),
                "j_lens_answer_rank": j_ranks[answer_position],
                "logit_lens_answer_rank": ll_ranks[answer_position],
                "j_lens_best_position_rank": min(j_ranks),
                "logit_lens_best_position_rank": min(ll_ranks),
            }

        model_top = top_tokens(model_logits[-1], tokenizer, args.top_k)
        record = {
            "name": item["name"], "source": item["source"], "category": item["category"],
            "prompt": prompt, "expected": item["expected"], "target_text": target_text, "target_token": tokenizer.decode([target_id]),
            "answer_position": answer_position,
            "model_answer_top": model_top,
            "model_answer_rank": token_rank(model_logits[-1], target_id),
            "best_j_lens": {"rank": best_j[0], "layer": best_j[1], "position": best_j[2]},
            "best_logit_lens": {"rank": best_ll[0], "layer": best_ll[1], "position": best_ll[2]},
            "layers": layer_results,
        }
        results.append(record)
        summary_rows.append([
            item["name"], item["source"], item["expected"], model_top[0] if model_top else "<none>",
            record["model_answer_rank"], f"{best_j[0]} @ L{best_j[1]} P{best_j[2]}",
            f"{best_ll[0]} @ L{best_ll[1]} P{best_ll[2]}",
        ])

    print("\\nBaseline summary:")
    print(tabulate(summary_rows, headers=["item", "source", "expected", "model top-1", "model rank", "best J-lens", "best logit lens"], tablefmt="github"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "lens_repo": args.lens_repo, "lens_file": args.lens_file, "precision": {"compute_dtype": "bf16", "base_quantization": "none"}, "items": results}, ensure_ascii=False, indent=2)+"\\n")
    log(f"wrote {len(results)} baseline results to {out}")


if __name__ == "__main__":
    main()
