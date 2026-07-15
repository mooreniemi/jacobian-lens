# Causal verbal-report experiment

This is the stronger Anthropic-style test: a spontaneous candidate and a
different candidate are represented in the same prompt, then two J-lens
coordinates are swapped at selected hidden-state layers. The outcome is the
target candidate's output rank, compared with a matched random-vector control.

The intervention is the paper's J-space patch:

```text
h' = h + V (swap(pinv(V) h) - pinv(V) h)
```

The current runner patches all prompt positions at a selected layer and
reports source and target ranks, candidate ranks, and top-k tokens. It is a
small, auditable reproduction, not yet a full statistical replication.

## Current protocol

- Matched model/lens: `Qwen/Qwen3.5-4B` and the released Qwen3.5-4B lens.
- Source: highest-ranked candidate in a category.
- Target: same-category candidate selected outside the initial top-10 where possible.
- Layers: 24, 28, and 30.
- Control: matched random directions with comparable norms.
- Current runtime: `--qwen-kernels off`; optional fast kernels are not installed because the available causal-conv1d build targets CUDA 12.6 while PyTorch is CUDA 13.0.

## Success criterion

Across held-out items, coordinate swaps should improve target rank and increase
target top-1/top-k verbal reports more than matched random, position, and
unrelated-vector controls. Effects matched by controls count as nonspecific
intervention sensitivity.

The paper’s method description is the reference for this implementation:
[Transformer Circuits, Jacobian Lens and verbal report](https://transformer-circuits.pub/2026/workspace/index.html#global-workspace-supports-verbal-report).

The current comparative methods are J-lens coordinates, ordinary logit-lens
directions (`J=I`), and a norm-matched random displacement. A tuned-lens run is
not included until a compatible Qwen3.5 tuned-lens checkpoint is fitted and
validated.
