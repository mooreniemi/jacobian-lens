# Anthropic J-lens baseline reproduction

Status: baseline evaluator prepared; confirmatory baseline run pending.

## Purpose

This experiment establishes what the ordinary, non-intervened Anthropic-style J-lens reproduction can actually show before we interpret conjunction swaps. It preserves the exact multihop prompt from the walkthrough and reports J-lens and vanilla logit-lens readouts in a systematic table.

## Primary item

The primary item is the walkthrough prompt:

```text
Fact: The capital of Japan is Tokyo.
Fact: The currency used in the country shaped like a boot is
```

Expected next token: `Euro`.

The dataset also contains three clearly labeled local extensions. They are descriptive extensions, not part of the exact reproduction claim.

## Measurements

For each item, the evaluator records:

- model final-position top-k and expected-token rank;
- J-lens top-k at the answer position for every fitted source layer;
- vanilla logit-lens top-k at the answer position for the same layers;
- best expected-token rank anywhere in the prompt for J-lens and logit lens;
- layer and position of each best rank;
- raw layer-by-layer JSON plus human-readable summary tables.

The original walkthrough rendered an interactive position-by-layer visualization. This evaluator adds numerical summaries but does not replace that visualization.

## Interpretation

A successful reproduction requires the exact walkthrough item to run and produce a coherent J-lens readout. The expected token should become substantially more recoverable under J-lens than under the vanilla logit lens at some intermediate layers or positions, consistent with the original qualitative claim.

This is a calibration experiment, not a conjunction or causal intervention test. It establishes model/lens/readout behavior and output resolution. It does not establish that every prompt has a correct top-1 answer or that a lens readout is causally used by the model.

## Command

```bash
uv run python scripts/eval_anthropic_baseline.py
```

The default model is `Qwen/Qwen3.5-4B` with the released `qwen-n1000` lens.
