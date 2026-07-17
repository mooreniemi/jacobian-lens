# H-lens conjunctive interaction experiment log

This chronological log records implementation, validation, and results for
the proposed second-order/Hessian-lens experiment. The protocol is in
[`HLENS_CONJUNCTION.md`](HLENS_CONJUNCTION.md).

## 2026-07-14 — Proposed experiment recorded

The proposed experiment was added after discussion of whether a Hessian-based
extension could reveal conjunction-level interactions missed by first-order
J-lens measurements.

Current decision:

- begin with Hessian-vector products, not full Hessian matrices;
- use answer-logit margins rather than raw logits as the primary scalar score;
- start on Qwen3-0.6B or SmolLM2 in BF16/FP32;
- validate numerically before using the current GPU queue or attempting 27B;
- extend the existing conjunction fixture, then move to randomized Boolean
  truth-table prompts if the pilot survives controls;
- treat the first output as a curvature diagnostic, not yet as a distributable
  fitted H-lens artifact.

No H-lens code or results have been produced yet. The next implementation gate
is a one-prompt, one-layer HVP smoke test with finite-difference validation,
zero-direction checks, and explicit CUDA-memory cleanup.

## 2026-07-15 — Implementation boundary clarified

We clarified that the first H-lens implementation will be a directional
curvature diagnostic, not a full Hessian matrix and not a vocabulary-wide
second-order decoder. For a scalar answer margin `s(h)` and directions `u` and
`v`, the primary statistic is `u · Hessian(s) · v`, computed with one HVP and
a dot product. The directions will initially be frozen J-lens/readout or
explicit property directions.

The staged gates are now: (1) analytic toy-model autograd tests, (2) one
Qwen3-0.6B or SmolLM2 FP32 prompt/layer with finite-difference agreement, (3)
a small answer-position-only conjunction pilot, and (4) the held-out J-only
versus J-plus-H prediction test. Full Hessian materialization, quantized 27B
second-order autograd, and a distributable H-lens checkpoint are explicitly
out of scope until those gates pass.

The naming distinction is now explicit: the HVP implementation is a
**Hessian interaction probe** because it directly measures `uᵀHv`; a future
reusable curvature representation may be called an **H-lens**; and a one-site
matrix calculation on a small model is a separate **full local Hessian
feasibility** experiment. The latter is planned rather than rejected. Its
purpose is to establish what matrix sizes, eigensolvers, memory, and runtime
are practical locally before considering Modal or a low-rank approximation.
