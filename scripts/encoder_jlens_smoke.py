"""Exploratory encoder score-Jacobian readout for MiniLM cross-encoder.

This is deliberately not part of the decoder J-lens benchmark. It asks whether
an average gradient of the final relevance score with respect to an
intermediate [CLS] state can predict held-out relevance scores.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PAIRS = [
    ("What is the capital of France?", "Paris is the capital city of France.", 1),
    ("What is the capital of France?", "Berlin is the capital city of Germany.", 0),
    ("How do plants make food?", "Plants use photosynthesis to make food from light, water, and carbon dioxide.", 1),
    ("How do plants make food?", "The Pacific Ocean is the largest ocean on Earth.", 0),
    ("What causes rain?", "Rain forms when water vapor condenses into droplets in clouds and falls to Earth.", 1),
    ("What causes rain?", "A mammal is an animal whose young drink milk.", 0),
    ("What is a cross encoder?", "A cross encoder jointly reads a query and document and produces a relevance score.", 1),
    ("What is a cross encoder?", "A tokenizer converts text into a sequence of token IDs.", 0),
    ("Who wrote Hamlet?", "William Shakespeare wrote the play Hamlet.", 1),
    ("Who wrote Hamlet?", "Isaac Newton formulated laws of motion and universal gravitation.", 0),
    ("What is the boiling point of water?", "At standard pressure, water boils at 100 degrees Celsius.", 1),
    ("What is the boiling point of water?", "The Earth revolves around the Sun once each year.", 0),
    ("What is DNA?", "DNA is a molecule that stores genetic instructions in living organisms.", 1),
    ("What is DNA?", "Granite is an igneous rock formed from cooled magma.", 0),
    ("What is the largest planet?", "Jupiter is the largest planet in the Solar System.", 1),
    ("What is the largest planet?", "Venus is the second planet from the Sun.", 0),
    ("What does CPU stand for?", "CPU means central processing unit, the main processor in a computer.", 1),
    ("What does CPU stand for?", "RAM is volatile memory used to hold active data.", 0),
    ("What is photosynthesis?", "Photosynthesis converts light energy into chemical energy in plants.", 1),
    ("What is photosynthesis?", "Mitosis is a process in which one cell divides into two daughter cells.", 0),
    ("What is a glacier?", "A glacier is a persistent mass of ice that moves slowly over land.", 1),
    ("What is a glacier?", "A volcano is an opening through which molten rock reaches the surface.", 0),
    ("What is an HTTP request?", "An HTTP request asks a web server for a resource or action.", 1),
    ("What is an HTTP request?", "An operating system manages hardware and software resources.", 0),
]


def rank_corr(a: list[float], b: list[float]) -> float:
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def main() -> None:
    started = time.monotonic()
    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"loading {name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    model.requires_grad_(True)
    n_layers = model.config.num_hidden_layers
    print(f"model={model.__class__.__name__} layers={n_layers} d_model={model.config.hidden_size} params={model.num_parameters():,}", flush=True)

    rows = []
    for index, (query, document, label) in enumerate(PAIRS):
        batch = tokenizer(query, document, return_tensors="pt", truncation=True, max_length=128)
        outputs = model(**batch, output_hidden_states=True)
        score = outputs.logits[0, 0]
        grads = torch.autograd.grad(score, outputs.hidden_states, retain_graph=False, allow_unused=False)
        hidden = [state[0, 0].detach().float() for state in outputs.hidden_states]
        rows.append({
            "query": query,
            "document": document,
            "label": label,
            "score": float(score.detach()),
            "hidden": [x.tolist() for x in hidden],
            "gradient": [x[0, 0].detach().float().tolist() for x in grads],
        })
        print(f"pair {index + 1}/{len(PAIRS)} score={float(score.detach()):+.4f}", flush=True)

    # Split by query: the first 8 query groups train the average score Jacobian.
    train = rows[:16]
    test = rows[16:]
    results = []
    for layer in range(n_layers + 1):
        h_train = np.asarray([r["hidden"][layer] for r in train], dtype=np.float64)
        y_train = np.asarray([r["score"] for r in train], dtype=np.float64)
        g_train = np.asarray([r["gradient"][layer] for r in train], dtype=np.float64)
        g_mean = g_train.mean(axis=0)
        b_j = float(np.mean(y_train - h_train @ g_mean))
        X = np.column_stack([np.ones(len(h_train)), h_train])
        beta, *_ = np.linalg.lstsq(X, y_train, rcond=None)
        h_test = np.asarray([r["hidden"][layer] for r in test], dtype=np.float64)
        y_test = np.asarray([r["score"] for r in test], dtype=np.float64)
        pred_j = b_j + h_test @ g_mean
        pred_linear = np.column_stack([np.ones(len(h_test)), h_test]) @ beta
        # The classifier head applied to intermediate CLS is a decoder-like
        # baseline, but MiniLM's pooler is only naturally trained at the final layer.
        direct = []
        with torch.no_grad():
            for r in test:
                cls = torch.tensor(r["hidden"][layer]).unsqueeze(0).unsqueeze(0)
                direct.append(float(model.classifier(model.bert.pooler(cls))[0, 0]))
        results.append({
            "layer": layer,
            "jacobian_mse": float(np.mean((pred_j - y_test) ** 2)),
            "linear_mse": float(np.mean((pred_linear - y_test) ** 2)),
            "direct_head_mse": float(np.mean((np.asarray(direct) - y_test) ** 2)),
            "jacobian_rank_corr": rank_corr(pred_j.tolist(), y_test.tolist()),
            "linear_rank_corr": rank_corr(pred_linear.tolist(), y_test.tolist()),
            "direct_head_rank_corr": rank_corr(direct, y_test.tolist()),
            "mean_gradient_norm": float(np.linalg.norm(g_mean)),
        })
        print(f"layer {layer}: J-MSE={results[-1]['jacobian_mse']:.4f} J-rho={results[-1]['jacobian_rank_corr']:.3f}", flush=True)

    out = {
        "model": name,
        "task": "exploratory encoder score-Jacobian readout",
        "pairs": len(PAIRS),
        "train_pairs": len(train),
        "test_pairs": len(test),
        "split_note": "ordered pair split; production experiment must split by query group",
        "elapsed_s": time.monotonic() - started,
        "layers": results,
    }
    path = Path("data/experiments/encoder-jlens-minilm-smoke.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
