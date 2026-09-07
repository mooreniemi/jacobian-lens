#!/usr/bin/env python3
"""Interactive chat for ByteDance Ouro-2.6B-Thinking."""

from __future__ import annotations

import argparse

import torch
import transformers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/alex/models/ouro-2.6b-thinking")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16
    ).cuda().eval()
    print("Ouro-2.6B-Thinking ready. Type /exit to quit.", flush=True)
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
        messages = [{"role": "user", "content": user}]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(model.device) if hasattr(encoded, "keys") else encoded.to(model.device)
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.7,
            )
        answer = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)
        print(f"ouro> {answer.strip()}\n", flush=True)


if __name__ == "__main__":
    main()
