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
