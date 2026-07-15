"""Reproduce the J-space verbal-report coordinate-swap experiment.

This implements the paper's coordinate patch:
    h' = h + V (swap(V^+ h) - V^+ h)
where V contains two J-lens token directions. It compares the spontaneous
candidate with a same-category target that was not initially in the candidate
top-10. This is a small, auditable causal smoke runner, not a full paper-scale
replication.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
import transformers
from tabulate import tabulate

import jlens
from jlens.qwen_runtime import configure_qwen_kernels

try:
    from causal_runtime import load_causal_model
except ModuleNotFoundError:  # imported as scripts.eval_* under pytest
    from scripts.causal_runtime import load_causal_model

CATEGORIES = ("country", "color", "fruit", "sport", "instrument", "planet", "tree", "bird", "language", "profession", "beverage", "organ", "city", "river")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def top_ids(logits: torch.Tensor, k: int) -> list[int]:
    return logits.topk(k).indices.tolist()


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item()) + 1


def candidate_id(tokenizer, text: str) -> int | None:
    ids = tokenizer(" " + text, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def candidate_ranks(logits: torch.Tensor, candidates: dict[str, int]) -> dict[str, int]:
    return {name: token_rank(logits, tid) for name, tid in candidates.items()}


def final_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model._hf_model(input_ids=input_ids, use_cache=False).logits[0, -1].float()


def lens_vector(model, lens, layer: int, token_id: int, kind: str = "jlens") -> torch.Tensor:
    """Return the residual-space J-lens direction for one output token.

    HFLensModel uses row residuals h @ J.T before unembedding, so the
    corresponding column direction is J.T @ W_U[token]. The final norm is not
    included in this first-order coordinate construction; the exact patch
    definition and this convention are recorded in the output metadata.
    """
    W = model._lm_head.weight[token_id].detach().to(model.input_device, dtype=torch.float32)
    if kind == "logit":
        return W
    if kind not in ("jlens", "tuned"):
        raise ValueError(f"unknown vector kind: {kind}")
    J = lens.jacobians[layer].to(model.input_device, dtype=torch.float32)
    return J.T @ W


def swap_lens_coordinates(hidden: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Swap the two pseudoinverse coordinates while preserving the residual.

    ``hidden`` is ``[n_positions, d_model]`` and ``V`` is ``[d_model, 2]``.
    The columns need not be orthogonal; the pseudoinverse handles the
    non-orthogonal two-vector frame used by the J-lens.
    """
    pinv = torch.linalg.pinv(V)
    coords = hidden @ pinv.T
    return hidden + (coords[:, [1, 0]] - coords) @ V.T


def patch_hook(model, lens, layer: int, source_id: int, target_id: int, *, positions: list[int], mode: str, alpha: float, vector_kind: str = "jlens"):
    vs = lens_vector(model, lens, layer, source_id, vector_kind)
    vt = lens_vector(model, lens, layer, target_id, vector_kind)
    V = torch.stack((vs, vt), dim=1)
    generator = torch.Generator(device=model.input_device).manual_seed(0)

    def hook(_module, _inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        patched = hidden.clone()
        pos = torch.tensor(positions, device=hidden.device, dtype=torch.long)
        h = hidden[0, pos].float()
        coordinate_delta = swap_lens_coordinates(h, V) - h
        if mode == "coordinate_swap":
            delta = coordinate_delta
        elif mode == "random_matched":
            random_delta = torch.randn(h.shape, device=h.device, generator=generator)
            random_delta *= (coordinate_delta.norm(dim=1, keepdim=True) /
                             random_delta.norm(dim=1, keepdim=True).clamp_min(1e-8))
            delta = random_delta
        else:
            raise ValueError(mode)
        patched[0, pos] = (h + alpha * delta).to(hidden.dtype)
        if torch.is_tensor(output):
            return patched
        return (patched, *output[1:])

    return model.layers[layer].register_forward_hook(hook)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument("--lens-file", default="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt")
    parser.add_argument("--lens-revision", default="qwen-n1000")
    parser.add_argument("--lens-local", default=None, help="load a local JacobianLens checkpoint instead of Hub")
    parser.add_argument("--data", default="data/experiments/verbal-report.json")
    parser.add_argument("--categories", nargs="*", default=["sport"])
    parser.add_argument("--layers", default="24,28,30")
    parser.add_argument("--patch-positions", choices=("all", "answer"), default="all")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--methods", default="jlens,random", help="comma-separated: jlens, tuned, logit, random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/experiments/verbal-report-causal-results.json")
    parser.add_argument("--qwen-kernels", choices=("auto", "on", "off"), default="on")
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--load-in-4bit", action="store_true", help="load with bitsandbytes NF4 and device_map=auto")
    args = parser.parse_args()

    kernel_mode = configure_qwen_kernels(args.qwen_kernels)
    log(f"Qwen kernel mode: {kernel_mode}")
    free, total = torch.cuda.mem_get_info()
    free_gib = free / 2**30
    log(f"preflight: {free_gib:.2f} GiB free / {total / 2**30:.2f} GiB total")
    if free_gib < args.min_free_gib:
        raise SystemExit(f"refusing to start: only {free_gib:.2f} GiB free")

    log(f"loading model {args.model}")
    hf_model = load_causal_model(args.model, load_in_4bit=args.load_in_4bit)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf_model, tokenizer)
    log(f"model ready: {model}; {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB free")
    lens = jlens.JacobianLens.load(args.lens_local) if args.lens_local else jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file, revision=args.lens_revision)
    data = json.loads(Path(args.data).read_text())
    rng = random.Random(args.seed)
    layers = [int(x) for x in args.layers.split(",") if x]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown_methods = set(methods) - {"jlens", "tuned", "logit", "random"}
    if unknown_methods:
        raise SystemExit(f"unknown methods: {sorted(unknown_methods)}")
    if any(layer not in lens.source_layers for layer in layers):
        raise SystemExit(f"requested layers {layers} are not all in lens source layers {lens.source_layers}")

    results = []
    summary = []
    for category in args.categories:
        candidates = {name: candidate_id(tokenizer, name) for name in data["candidates"][category]}
        candidates = {name: tid for name, tid in candidates.items() if tid is not None}
        prompt = f"Think of a {category}. Answer in one word."
        input_ids = model.encode(prompt, max_length=128)
        answer_position = input_ids.shape[1] - 1
        base_logits = final_logits(model, input_ids)
        ranks = candidate_ranks(base_logits, candidates)
        source = min(ranks, key=ranks.get)
        top10 = set(top_ids(base_logits, args.top_k))
        eligible = [name for name, tid in candidates.items() if tid not in top10 and name != source]
        if not eligible:
            raise SystemExit(f"no eligible target outside top-{args.top_k} for {category}")
        target = rng.choice(eligible)
        source_id, target_id = candidates[source], candidates[target]
        positions = list(range(answer_position + 1)) if args.patch_positions == "all" else [answer_position]
        log(f"category {category}: source={source} rank={ranks[source]}, target={target} rank={ranks[target]}, positions={len(positions)}")
        category_result = {
            "category": category, "prompt": prompt, "source": source, "target": target,
            "source_rank_before": ranks[source], "target_rank_before": ranks[target],
            "candidate_ranks_before": ranks, "answer_position": answer_position,
            "layers": {},
        }
        for layer in layers:
            layer_result = {}
            for method in methods:
                mode = "random_matched" if method == "random" else "coordinate_swap"
                vector_kind = "jlens" if method in ("jlens", "random") else ("tuned" if method == "tuned" else "logit")
                handle = patch_hook(model, lens, layer, source_id, target_id, positions=positions, mode=mode, alpha=1.0, vector_kind=vector_kind)
                try:
                    patched_logits = final_logits(model, input_ids)
                finally:
                    handle.remove()
                patched_ranks = candidate_ranks(patched_logits, candidates)
                layer_result[method] = {
                    "source_rank_after": patched_ranks[source],
                    "target_rank_after": patched_ranks[target],
                    "candidate_ranks_after": patched_ranks,
                    "top": [tokenizer.decode([i]).replace("\n", "\\n") for i in top_ids(patched_logits, args.top_k)],
                }
                summary.append([category, layer, method, ranks[source], ranks[target], patched_ranks[source], patched_ranks[target]])
            category_result["layers"][str(layer)] = layer_result
        results.append(category_result)

    print("\nCausal verbal-report summary:")
    print(tabulate(summary, headers=["category", "layer", "method", "src before", "tgt before", "src after", "tgt after"], tablefmt="github"))
    aggregate = []
    for mode in methods:
        changes = []
        top1 = 0
        top10 = 0
        for item in results:
            for layer_result in item["layers"].values():
                after = layer_result[mode]["target_rank_after"]
                changes.append(item["target_rank_before"] - after)
                top1 += after == 1
                top10 += after <= 10
        aggregate.append([
            mode,
            len(changes),
            sum(x > 0 for x in changes),
            round(statistics.median(changes), 1),
            round(statistics.mean(changes), 1),
            top1,
            top10,
        ])
    print("\nAggregate target-rank changes (positive means target improved):")
    print(tabulate(aggregate, headers=["method", "n", "improved", "median Δrank", "mean Δrank", "target top-1", "target top-10"], tablefmt="github"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "lens_repo": args.lens_repo, "lens_file": args.lens_file, "lens_revision": args.lens_revision, "lens_local": args.lens_local, "precision": {"compute_dtype": "bf16", "base_quantization": "nf4" if args.load_in_4bit else "none"}, "kernel_mode": kernel_mode, "patch_positions": args.patch_positions, "layers": layers, "methods": methods, "items": results, "aggregate": aggregate}, ensure_ascii=False, indent=2) + "\n")
    log(f"wrote {len(results)} causal results to {out}")


if __name__ == "__main__":
    main()
