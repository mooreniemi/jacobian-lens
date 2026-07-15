# Anthropic baseline reproduction log

Protocol: [ANTHROPIC_BASELINE.md](ANTHROPIC_BASELINE.md)  
Dataset: [anthropic-baseline.json](../../data/experiments/anthropic-baseline.json)  
Runner: [eval_anthropic_baseline.py](../../scripts/eval_anthropic_baseline.py)

## 2026-07-12 — Experiment prepared

The exact walkthrough multihop prompt is separated from three local extension prompts. The runner will compare J-lens and vanilla logit-lens top-k/ranks by layer and position, print summary tables, and write detailed JSON. No causal intervention or conjunction swap is involved.


## 2026-07-12 — Corrected baseline run

### Run

Command run in tmux pane `0:1.3`:

```bash
uv run python scripts/eval_anthropic_baseline.py \
  --out data/experiments/anthropic-baseline-results.json
```

The Qwen3.5-4B model and released `qwen-n1000` lens loaded with 23.24 GiB free. The run processed the exact walkthrough item plus three labeled local extensions and wrote `data/experiments/anthropic-baseline-results.json`.

### Bug found and fixed

The first run scored bare strings such as `Euro` rather than the context-appropriate continuation token ` Euro`. This produced internally inconsistent ranks, including a token displayed as top-1 but reported at rank 1238. The evaluator was corrected to score leading-space continuation tokens, and the baseline was rerun. The first output is superseded.

### Corrected findings

- Exact walkthrough multihop: model rank 3 for `Euro`; best J-lens rank 1 at layer 28, position 22; best vanilla logit-lens rank 3.
- Capital-of-boot extension: model and J-lens rank 1 for `Rome`.
- Largest-planet extension: model and J-lens rank 1 for `Jupiter`.
- Japan-currency extension: model rank 4 for `yen`; best J-lens rank 1.

The exact walkthrough result reproduces the qualitative claim that the J-lens can surface an interpretable expected token at an intermediate layer/position even when the final model prediction is not top-1. This establishes baseline readout resolution; it is not a causal result.
