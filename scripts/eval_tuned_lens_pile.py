"""Evaluate a model-matched tuned lens on held-out Pile JSONL.

This follows the public tuned-lens evaluation convention: score a fixed token
budget from Pile test, layer by layer, against the frozen model's final output
distribution. It also reports next-token metrics for the tuned lens and the
ordinary logit lens. The official CLI cannot currently import with our modern
torchdata, so this small evaluator keeps the same measurement without its
obsolete DataLoader2 dependency.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens import TunedLens, model_surgery


def patch_qwen_norm(model) -> None:
    """Teach tuned-lens 0.2.0 the Qwen-family final norm convention."""
    base_type = type(model.base_model)
    if base_type.__name__ not in ("Qwen3Model", "Qwen3_5TextModel"):
        return
    original = model_surgery.get_final_norm

    def get_final_norm(candidate):
        if type(candidate.base_model).__name__ in ("Qwen3Model", "Qwen3_5TextModel"):
            return candidate.base_model.norm
        return original(candidate)

    model_surgery.get_final_norm = get_final_norm
    model_surgery.Norm = model_surgery.Norm | type(model.base_model.norm)


def token_chunks(tokenizer, path: Path, max_tokens: int, length: int):
    ids: list[int] = []
    emitted = 0
    with path.open() as handle:
        for line in handle:
            text = json.loads(line).get("text", "")
            if not text.strip():
                continue
            ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            while len(ids) >= length and emitted < max_tokens:
                yield torch.tensor(ids[:length], dtype=torch.long)
                ids = ids[length:]
                emitted += length
                if emitted >= max_tokens:
                    return


def write_event(handle, event: str, **fields) -> None:
    """Write one immediately flushed progress/provenance event."""
    record = {"event": event, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}
    handle.write(json.dumps(record) + "\n")
    handle.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=16_400_000)
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--progress-every-batches", type=int, default=100)
    ap.add_argument("--events-out", help="JSONL progress/provenance path")
    ap.add_argument("--checkpoint-out", help="resumable aggregate checkpoint")
    ap.add_argument("--resume", action="store_true", help="resume from --checkpoint-out")
    args = ap.parse_args()
    started = time.monotonic()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    events_path = Path(args.events_out or f"{args.out}.events.jsonl")
    checkpoint_path = Path(args.checkpoint_out or f"{args.out}.checkpoint.pt")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events = events_path.open("w")
    target_chunks = math.ceil(args.tokens / args.length)
    target_scored_tokens = target_chunks * (args.length - 1)
    print(
        f"[preflight] Pile test budget={args.tokens:,} input tokens "
        f"batch_size={args.batch_size} seq_len={args.length} "
        f"events={events_path}",
        flush=True,
    )
    write_event(
        events,
        "preflight",
        target_input_tokens=args.tokens,
        target_scored_tokens=target_scored_tokens,
        batch_size=args.batch_size,
        sequence_length=args.length,
        dtype=args.dtype,
    )
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    write_event(events, "tokenizer_loaded", elapsed_s=time.monotonic() - started)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, low_cpu_mem_usage=True).cuda().eval()
    model.requires_grad_(False)
    patch_qwen_norm(model)
    lens = TunedLens.from_model(model)
    state = torch.load(Path(args.lens) / "params.pt", map_location="cpu", weights_only=True)
    lens.layer_translators.load_state_dict(state)
    lens = lens.cuda().eval()
    layers = model.config.num_hidden_layers
    write_event(
        events,
        "model_loaded",
        elapsed_s=time.monotonic() - started,
        layers=layers,
        hidden_size=model.config.hidden_size,
        peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
    )
    sums = {name: torch.zeros(layers, dtype=torch.float64) for name in ("kl", "nll", "top1", "top5", "logit_kl", "logit_nll", "logit_top1", "logit_top5")}
    count = 0
    input_count = 0
    batches_done = 0
    chunks_to_skip = 0
    chunks_done = 0
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = {
            "model": args.model,
            "lens": args.lens,
            "data": args.data,
            "tokens": args.tokens,
            "length": args.length,
            "batch_size": args.batch_size,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise SystemExit(f"checkpoint mismatch for {key}: {checkpoint.get(key)!r} != {value!r}")
        sums = {key: value.to(dtype=torch.float64) for key, value in checkpoint["sums"].items()}
        count = checkpoint["scored_tokens"]
        input_count = checkpoint["input_tokens"]
        batches_done = checkpoint["batches_done"]
        chunks_to_skip = checkpoint["chunks_done"]
        chunks_done = chunks_to_skip
        print(
            f"[resume] checkpoint={checkpoint_path} batches={batches_done} "
            f"scored_tokens={count:,} chunks_to_skip={chunks_to_skip:,}",
            flush=True,
        )
        write_event(events, "resume", checkpoint=str(checkpoint_path), batches=batches_done, scored_tokens=count)
    chunks = token_chunks(tokenizer, Path(args.data), args.tokens, args.length)

    def save_checkpoint() -> None:
        payload = {
            "model": args.model,
            "lens": args.lens,
            "data": args.data,
            "tokens": args.tokens,
            "length": args.length,
            "batch_size": args.batch_size,
            "sums": {key: value.cpu() for key, value in sums.items()},
            "scored_tokens": count,
            "input_tokens": input_count,
            "batches_done": batches_done,
            "chunks_done": chunks_done,
            "updated_s": time.monotonic() - started,
        }
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint_path)

    def report_progress() -> None:
        elapsed = time.monotonic() - started
        rate = count / elapsed if elapsed else 0.0
        remaining = max(target_scored_tokens - count, 0)
        eta = remaining / rate if rate else None
        fields = {
            "batches": batches_done,
            "input_tokens": input_count,
            "scored_tokens": count,
            "elapsed_s": round(elapsed, 1),
            "scored_tokens_per_s": round(rate, 2),
            "eta_s": round(eta, 1) if eta is not None else None,
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        }
        write_event(events, "progress", **fields)
        eta_text = f" eta={eta / 3600:.2f}h" if eta is not None else ""
        print(
            f"[progress] batch={batches_done}/{target_chunks} "
            f"input_tokens={input_count:,}/{args.tokens:,} "
            f"scored_tokens={count:,} rate={rate:,.0f}/s "
            f"elapsed={elapsed / 3600:.2f}h{eta_text} "
            f"peak_vram={fields['peak_allocated_gib']:.2f}GiB",
            flush=True,
        )

    with torch.inference_mode():
        batch = []
        for chunk_index, chunk in enumerate(tqdm(chunks, total=args.tokens // args.length, desc="Pile test chunks")):
            if chunk_index < chunks_to_skip:
                continue
            batch.append(chunk)
            if len(batch) < args.batch_size:
                continue
            ids = torch.stack(batch).cuda()
            batch.clear()
            batches_done += 1
            input_count += ids.numel()
            chunks_done += ids.shape[0]
            output = model(input_ids=ids, use_cache=False, output_hidden_states=True)
            final_logits = output.logits[:, :-1].float()
            labels = ids[:, 1:]
            final_logp = final_logits.log_softmax(dim=-1)
            final_prob = final_logp.exp()
            del final_logits
            count += labels.numel()
            for layer, hidden in enumerate(output.hidden_states[:-1]):
                hidden = hidden[:, :-1]
                with contextlib.nullcontext():
                    tuned_logits = lens(hidden, layer).float()
                    norm = model.base_model.norm(hidden.float())
                    logit_logits = model.lm_head(norm.to(model.lm_head.weight.dtype)).float()
                for prefix, logits in (("", tuned_logits), ("logit_", logit_logits)):
                    logp = logits.log_softmax(dim=-1)
                    sums[f"{prefix}kl"][layer] += (final_prob * (final_logp - logp)).sum().item()
                    sums[f"{prefix}nll"][layer] += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum").item()
                    top = logits.topk(5, dim=-1).indices
                    sums[f"{prefix}top1"][layer] += (top[..., :1] == labels.unsqueeze(-1)).any(dim=-1).sum().item()
                    sums[f"{prefix}top5"][layer] += (top == labels.unsqueeze(-1)).any(dim=-1).sum().item()
                    del logits, logp, top
            if batches_done % args.progress_every_batches == 0:
                save_checkpoint()
                report_progress()
        if batch:
            ids = torch.stack(batch).cuda()
            batches_done += 1
            input_count += ids.numel()
            chunks_done += ids.shape[0]
            with torch.inference_mode():
                output = model(input_ids=ids, use_cache=False, output_hidden_states=True)
                final_logits = output.logits[:, :-1].float()
                labels = ids[:, 1:]
                final_logp = final_logits.log_softmax(dim=-1)
                final_prob = final_logp.exp()
                del final_logits
                count += labels.numel()
                for layer, hidden in enumerate(output.hidden_states[:-1]):
                    hidden = hidden[:, :-1]
                    tuned_logits = lens(hidden, layer).float()
                    norm = model.base_model.norm(hidden.float())
                    logit_logits = model.lm_head(norm.to(model.lm_head.weight.dtype)).float()
                    for prefix, logits in (("", tuned_logits), ("logit_", logit_logits)):
                        logp = logits.log_softmax(dim=-1)
                        sums[f"{prefix}kl"][layer] += (final_prob * (final_logp - logp)).sum().item()
                        sums[f"{prefix}nll"][layer] += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum").item()
                        top = logits.topk(5, dim=-1).indices
                        sums[f"{prefix}top1"][layer] += (top[..., :1] == labels.unsqueeze(-1)).any(dim=-1).sum().item()
                        sums[f"{prefix}top5"][layer] += (top == labels.unsqueeze(-1)).any(dim=-1).sum().item()
                        del logits, logp, top
            report_progress()
            save_checkpoint()
    result = {"model": args.model, "lens": args.lens, "data": args.data, "tokens": count, "layers": []}
    for layer in range(layers):
        result["layers"].append({
            "layer": layer,
            "tuned_kl": sums["kl"][layer].item() / count,
            "logit_kl": sums["logit_kl"][layer].item() / count,
            "tuned_nll": sums["nll"][layer].item() / count,
            "logit_nll": sums["logit_nll"][layer].item() / count,
            "tuned_top1": sums["top1"][layer].item() / count,
            "logit_top1": sums["logit_top1"][layer].item() / count,
            "tuned_top5": sums["top5"][layer].item() / count,
            "logit_top5": sums["logit_top5"][layer].item() / count,
        })
    elapsed = time.monotonic() - started
    result.update({
        "input_tokens": input_count,
        "batch_size": args.batch_size,
        "sequence_length": args.length,
        "elapsed_s": elapsed,
        "scored_tokens_per_s": count / elapsed if elapsed else None,
        "events": str(events_path),
        "checkpoint": str(checkpoint_path),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    })
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    write_event(
        events,
        "complete",
        input_tokens=input_count,
        scored_tokens=count,
        elapsed_s=elapsed,
        scored_tokens_per_s=count / elapsed if elapsed else None,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        output=args.out,
    )
    checkpoint_path.unlink(missing_ok=True)
    events.close()
    print(f"[complete] wrote {args.out} tokens={count:,} elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
