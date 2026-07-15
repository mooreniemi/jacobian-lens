# Causal verbal-report experiment log

## 2026-07-12 — Implementation smoke

The coordinate-swap runner was implemented with a pseudoinverse-based patch,
source/target rank reporting, and a matched random-vector control. A one-item
sport smoke test completed without OOM on the RTX 3090. Basketball started at
rank 60 and Rugby at rank 263; across layers 24/28/30 the coordinate swap put
Rugby at ranks 48/26/41, while the random control put it at 567/257/342.

This was directionally encouraging but not sufficient: one category, no
top-1 target reports, and the vector convention still needs validation against
the exact released-lens implementation.

## 2026-07-12 — Five-category calibration batch

Output: `data/experiments/verbal-report-causal-batch5.json`.

Categories were country, color, fruit, sport, and instrument; each was tested
at layers 24, 28, and 30. The target improved in all 15 coordinate-swap
conditions (median rank improvement 163; no target reached rank 1). The
matched random control improved the target in 0/15 conditions (median change
was a worsening of 410 ranks).

This is strong calibration evidence that the implemented coordinate swap is
more targeted than the current random-direction control, but it is not yet a
confirmatory causal claim. The batch is small, source/target selection is
category-list based, and the result could still depend on patch position,
layer choice, source rank, or the precise J-vector convention.

## Decision

Proceed to a larger preregistered verbal-report batch only after adding the
remaining controls and validating the patch algebra. Keep this causal track
separate from the conjunction experiment: it establishes the intervention
machinery before applying it to genuinely non-redundant conjunction prompts.

## 2026-07-12 — Full 14-category matched-model batch

Output: `data/experiments/verbal-report-causal-full14.json`.

This used the released Qwen3.5-4B lens with the matched Qwen3.5-4B model,
all prompt positions, and layers 24, 28, and 30. It covered the fourteen
categories used by the verbal-report setup. There were 42 coordinate-swap
conditions and 42 matched random-direction controls. The random controls were
norm-matched to the coordinate-swap displacement at each patched position; this
corrected run supersedes the initial calibration output.

| intervention | conditions | target rank improved | median rank change | mean rank change | target ≤10 |
|---|---:|---:|---:|---:|---:|
| coordinate swap | 42 | 40/42 | +243 | +606.9 | 8/42 |
| random matched | 42 | 11/42 | −9.5 | −33.9 | 0/42 |

No condition reached target rank 1. The direction and control separation are
evidence that the implemented patch is more targeted than a norm-matched random
perturbation, but the absence of rank-1 reports means this is not yet a faithful quantitative
reproduction of the paper's strongest verbal-report result. Candidate choice,
vector normalization/convention, layer selection, model differences, and
patching at every position remain live explanations for the gap.

The next required step is a controlled method audit: verify the released lens
vector convention, add position and unrelated-vector controls, and test whether
the target improvement survives when the source/target pair is selected from a
larger held-out candidate inventory rather than the current hand-authored list.

## 2026-07-12 — Logit-lens comparison

The 14-category verbal-report run was repeated with ordinary unembedding
directions as a causal logit-lens baseline:

| method | conditions | target rank improved | median Δrank | target top-1 | target top-10 |
|---|---:|---:|---:|---:|---:|
| J-lens | 42 | 40 | +243 | 0 | 8 |
| logit lens | 42 | 35 | +163.5 | 0 | 5 |
| random matched | 42 | 11 | −9.5 | 0 | 0 |

Both learned/token-associated direction families can move the target, but the
J-lens is stronger on this open-model verbal-report setup. The logit lens is a
meaningful baseline rather than a null.

## 2026-07-13 — Matched Qwen3.5-0.8B replication

Output: `data/experiments/verbal-report-causal-0.8b.json`.

The run used the locally fitted 204-prompt Qwen3.5-0.8B lens and layers
16/19/21, chosen at approximately comparable relative depths to the 4B run.
All 14 categories were usable.

| method | layer conditions | improved | median Δrank | target top-1 | target top-10 |
|---|---:|---:|---:|---:|---:|
| J-lens | 42 | 41 | +1032.5 | 0 | 1 |
| logit lens | 42 | 20 | −158.5 | 0 | 0 |
| random matched | 42 | 24 | +13 | 0 | 0 |

This is a strong small-model separation in rank movement, but it produces very
few top-k reports; it should be treated as a model-scale result, not as direct
evidence that the 0.8B model has the same workspace quality as the 4B model.

MLflow run: `99ac9226435b4a029ca217b2c33e79b9`.

## 2026-07-13 — Matched Qwen3-0.6B replication

Output: `data/experiments/verbal-report-causal-0.6b.json`. This used the
model-matched local 204-prompt lens and layers 19/22/25; all 14 categories were
usable.

| method | conditions | improved | median Δrank | mean Δrank | target top-1 | target top-10 |
|---|---:|---:|---:|---:|---:|---:|
| J-lens | 42 | 35 | +389.5 | +959.7 | 1 | 1 |
| logit lens | 42 | 31 | +266.5 | +613.0 | 2 | 3 |
| random matched | 42 | 16 | −14.0 | −39.6 | 0 | 0 |

MLflow run: `fabb16b5006c4ebfba2924b00454f96a`.

## 2026-07-13 — Matched SmolLM2-135M-Instruct replication

Output: `data/experiments/verbal-report-causal-smollm2-135m.json`. This used a
fresh 204-prompt lens matched to SmolLM2 and layers 21/25/28.

| method | conditions | improved | median Δrank | target top-1 | target top-10 |
|---|---:|---:|---:|---:|---:|
| J-lens | 42 | 37 | +1428.0 | 0 | 0 |
| logit lens | 42 | 34 | +1446.5 | 0 | 0 |
| random matched | 42 | 19 | −39.0 | 0 | 0 |

The runner completed all 14 categories; the exact aggregate is retained in the
JSON artifact because this evaluator's verbal table includes mean rank change
and top-10 in addition to the compact log summary. MLflow run:
`6d4329e5d24f492fb1a562ffa94aedd2`.

## 2026-07-13 — Tuned-lens comparisons across matched smaller models

These runs use model-matched Wikitext-trained tuned lenses. The `tuned` method
uses the exported affine translator basis; the final nonlinear RMSNorm is not
folded into that causal patch basis.

| model | method | conditions | improved | median Δrank | mean Δrank | top-1 | top-10 |
|---|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M | tuned | 42 | 31 | +1425.5 | +5342.7 | 0 | 0 |
| SmolLM2-135M | logit | 42 | 34 | +1446.5 | +4770.4 | 0 | 0 |
| SmolLM2-135M | random | 42 | 19 | −44.0 | +75.6 | 0 | 0 |
| Qwen3-0.6B | tuned | 42 | 30 | +449.5 | −4204.6 | 8 | 11 |
| Qwen3-0.6B | logit | 42 | 31 | +266.5 | +613.0 | 2 | 3 |
| Qwen3-0.6B | random | 42 | 17 | −33.0 | −146.1 | 0 | 0 |
| Qwen3.5-0.8B | tuned | 42 | 22 | +45.5 | −9228.6 | 0 | 2 |
| Qwen3.5-0.8B | logit | 42 | 20 | −158.5 | −1537.3 | 0 | 0 |
| Qwen3.5-0.8B | random | 42 | 25 | +12.5 | −1.7 | 0 | 0 |
| Qwen3.5-4B | tuned | 42 | 26 | +143.5 | −181.7 | 0 | 6 |
| Qwen3.5-4B | logit | 42 | 35 | +163.5 | +599.6 | 0 | 5 |
| Qwen3.5-4B | random | 42 | 12 | −8.0 | −18.1 | 0 | 0 |

MLflow runs: `b9d8d1a566874355a50859d8a9bf2b72`,
`6ab5c11e804f4f0788a9d52662a21f30`, `248fa347433541c389b88839525af458`,
and `ddf7f64f4a974a31a1f4bce7de4f1976`.

## 2026-07-13 — Matched Qwen3.6-27B 4-bit replication

Output: `data/experiments/verbal-report-causal-27b.json`. The released matching
1000-prompt lens was used with the NF4 PyTorch model and layers 16/32/60.

| method | conditions | improved | median Δrank | mean Δrank | target top-1 | target top-10 |
|---|---:|---:|---:|---:|---:|---:|
| J-lens | 42 | 34 | +24.0 | +131.6 | 0 | 10 |
| logit lens | 42 | 28 | +6.5 | +94.0 | 0 | 9 |
| random matched | 42 | 21 | +0.5 | +4.6 | 0 | 0 |

MLflow run: `9721680bc6cf4251b1b299bc1c71aa66`.

## 2026-07-14 — Model-matched Qwen3.6-27B tuned-lens evaluation

The native Wikitext-trained tuned lens was fit remotely on the exact
`Qwen/Qwen3.6-27B` revision in NF4 mode, validated for finite parameters, and
exported as the affine translator basis used by this causal runner. The
nonlinear final RMSNorm is not folded into this patch basis. The same NF4
model, prompts, layers 16/32/60, and all-position patch were used; only the
previously missing tuned row was run.

| method | conditions | improved | median Δrank | mean Δrank | target top-1 | target top-10 |
|---|---:|---:|---:|---:|---:|---:|
| tuned | 42 | 24 | +3.5 | +83.4 | 0 | 7 |

Output: `data/experiments/verbal-report-causal-27b.json` (now includes the
merged `tuned` rows) and tuned-only artifact
`data/experiments/modal-tuned-27b/verbal-report-causal-27b-tuned.json`.
