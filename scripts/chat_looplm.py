#!/usr/bin/env python3
"""Small interactive LoopLM-SFT chat session for a local CUDA host."""

from __future__ import annotations

import argparse

import torch
import transformers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/alex/models/looplm-135m-naive-sft")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16
    ).cuda().eval()
    # LoopLM normally samples a fresh recurrent initial state. Use zero state
    # for a stable interactive analysis session and reproducible lens probes.
    model.model._h0 = lambda batch, seq_len, device, dtype: torch.zeros(
        batch, seq_len, model.config.d_model, device=device, dtype=dtype
    )
    print("LoopLM ready. Type /exit to quit.", flush=True)

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/exit", "/quit"}:
            break
        prompt = f"### Instruction:\n{user}\n\n### Response:\n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        generated: list[int] = []
        with torch.inference_mode():
            for _ in range(args.max_new_tokens):
                logits = model(input_ids[:, -1024:]).logits[0, -1].float()
                logits = logits / max(args.temperature, 1e-5)
                if generated:
                    logits[torch.tensor(list(set(generated[-20:])), device=logits.device)] /= 1.3
                values, _ = torch.topk(logits, min(args.top_k, logits.numel()))
                logits[logits < values[-1]] = float("-inf")
                next_id = int(torch.multinomial(logits.softmax(dim=-1), 1))
                if next_id == tokenizer.eos_token_id:
                    break
                generated.append(next_id)
                input_ids = torch.cat((input_ids, torch.tensor([[next_id]], device=input_ids.device)), dim=1)
        answer = tokenizer.decode(generated, skip_special_tokens=True)
        if "###" in answer:
            answer = answer.split("###", 1)[0]
        print(f"looplm> {answer.strip()}\n", flush=True)


if __name__ == "__main__":
    main()
