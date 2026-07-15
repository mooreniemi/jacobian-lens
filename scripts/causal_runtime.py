"""Shared model loading for causal evaluators."""
from __future__ import annotations

import torch
import transformers


def load_causal_model(model_name: str, *, load_in_4bit: bool):
    if load_in_4bit:
        quant = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant,
            device_map="auto",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    return transformers.AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).cuda()
