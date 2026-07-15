# H-lens conjunctive interaction experiment

Status: proposed design; no H-lens results yet  
Summary: this document is the protocol and design record  
Experiment log: [`HLENS_CONJUNCTION_LOG.md`](HLENS_CONJUNCTION_LOG.md)  
Related protocol: [`CONJUNCTION_SWAP.md`](CONJUNCTION_SWAP.md)  
Last updated: 2026-07-14

## 1. Research question

Does a model's curvature along two property directions reveal a conjunction-level
interaction that is not visible in first-order J-lens measurements?

The experiment is motivated by prompts of the form:

```text
The animal that spins webs and has eight legs is a ...
```

The question is not simply whether the model can answer the prompt. It is
whether the effect of property A changes when property B is present, and
whether that interaction can be localized and measured internally.

## 2. Conceptual definition

Let `s(h)` be a scalar model score, such as the logit margin between the
expected answer and a competing answer, and let `u` and `v` be two normalized
activation directions associated with properties A and B.

The J-lens is first-order:

```text
s(h + d) ≈ s(h) + gradient(s) · d
```

The proposed H-lens adds the second-order term:

```text
s(h + d) ≈ s(h) + gradient(s) · d + 1/2 d · Hessian(s) · d
```

For conjunctions, the primary quantity is the cross-Hessian:

```text
cross(A, B) = u · Hessian(s) · v
```

This measures whether the effect of A changes in the presence of B. A full
Hessian is not required: the quantity can be computed with Hessian-vector
products. The initial H-lens is therefore a curvature/interaction readout,
not a claim that we have found a new universal vocabulary decoder.

## 3. Hypotheses

### H1: second-order predictive improvement

For real interventions, a Taylor prediction using the J-lens term plus the
Hessian term predicts the observed answer-score change better than the J-only
prediction.

### H2: conjunction-specific cross-curvature

The cross-Hessian between A and B is larger, more reproducible, or more
directionally structured for genuine conjunctions than for additive,
scrambled, unrelated-property, and nearby-position controls.

### H3: localized interaction

Cross-curvature peaks at a layer and position associated with combining the
two properties, rather than merely tracking the norm of either direction or
generic prompt sensitivity.

### Null hypothesis

The model is locally close to linear in the tested directions, or the relevant
computation is not captured by these directions. The H term then provides no
reliable predictive improvement, and cross-curvature is indistinguishable from
matched controls.

## 4. Concrete experimental design

### Phase 0: numerical and pipeline validation

Use Qwen3-0.6B or SmolLM2 before touching the 27B model. Run one known-good
prompt, one layer, one answer margin, and a small set of directions.

Validate every HVP against a finite-difference estimate:

```text
Hessian · v ≈ [gradient(s at h + epsilon v)
               - gradient(s at h - epsilon v)] / (2 epsilon)
```

Check multiple epsilon values, symmetry where available, dtype behavior, and
that a zero direction gives zero curvature. Fail the run if these checks do
not pass within a documented numerical tolerance.

### Phase 1: controlled natural-language conjunctions

Use the validated conjunction fixture and retain only items where:

- A alone and B alone identify intended intermediates;
- A and B together identify one intended baseline entity;
- the answer has a measurable baseline margin;
- A and B spans are tokenizer-checked;
- swaps are classified as valid, ambiguous, or contradictory.

Evaluate the factorial conditions:

| Condition | Purpose |
|---|---|
| neither | baseline/no-property control |
| A only | isolate A |
| B only | isolate B |
| A and B | conjunction condition |
| A swapped | change A while holding B fixed |
| B swapped | change B while holding A fixed |
| both swapped | test combined effect |
| connective swap | test `and` versus `or` |
| random-word scramble | generic lexical perturbation control |

At each selected layer and relevant token position, compute the answer-margin
gradient, directional curvature for A and B, and the cross-Hessian.

### Phase 2: causal prediction test

For an intervention with displacement `d`, compare:

1. the actual score after applying the activation intervention;
2. the J-only Taylor prediction;
3. the J-plus-H Taylor prediction.

The primary outcome is the held-out prediction error reduction from J-only to
J-plus-H. The intervention direction must be frozen before evaluating the
held-out prompts, and direction norms must be reported.

### Phase 3: randomized Boolean extension

If the natural-language pilot passes its controls, generate opaque-property
truth-table prompts with known AND, OR, XOR, implication, and NOT rules. This
prevents one fact from accidentally determining the answer and makes the
interaction ground truth explicit.

The primary behavioral interaction statistic is:

```text
score(1,1) - score(1,0) - score(0,1) + score(0,0)
```

Compare this statistic with the cross-Hessian measured before intervention and
with the causal answer change after intervention.

## 5. Controls

The minimum control set is:

- matched-norm random directions;
- unrelated semantic directions;
- cross-type substitutions, such as an entity direction replaced by a common
  noun direction;
- nearby non-conjunct positions;
- connective substitution (`and` to `or`);
- reversed conjunct order;
- multiple intervention magnitudes;
- finite-difference and autograd HVP agreement.

If random controls produce effects comparable to the intended A/B directions,
the result is generic curvature or intervention sensitivity, not evidence for
conjunction binding.

## 6. Models and precision

The first smoke test should use a small model in BF16 or FP32. FP32 is
preferable for the numerical HVP check if it fits. Quantized 27B weights may
be acceptable for inference, but second-order autograd through a quantized
model is not the initial target.

The 27B follow-up should be a separate model-matched run. It should report
base-weight precision, activation dtype, device, peak allocated/reserved CUDA
memory, number of HVPs, and wall-clock time. We must not infer H-lens creation
cost from the existing J-lens or tuned-lens runs.

## 7. Analysis and uncertainty

Report per-item and aggregate results. The primary summaries are:

- held-out J-only versus J-plus-H prediction error;
- cross-Hessian effect size versus each control distribution;
- layer/position localization;
- correlation between cross-curvature and Boolean interaction;
- bootstrap confidence intervals over prompt families;
- sensitivity to direction norm, epsilon, dtype, and answer-margin choice;
- peak memory, HVP count, and fit/evaluation time.

The pilot is for feasibility and variance estimation. A confirmatory item count
should be chosen after measuring paired-item variance and the smallest effect
that would be scientifically meaningful.

## 8. Evidence that would change our mind

### Supports the H-lens hypothesis

- J-plus-H predictions improve held-out intervention prediction over J-only;
- cross-curvature is larger for genuine conjunctions than matched controls;
- the effect is localized and survives norm, position, order, and scramble
  controls;
- cross-curvature tracks the known Boolean interaction in the randomized task;
- results replicate across seeds, paraphrases, and at least one second model.

### Weakens or falsifies it

- H terms do not improve prediction beyond J-only;
- cross-curvature is no larger than random or unrelated controls;
- results disappear under small changes in epsilon or direction scale;
- apparent effects are explained by answer-margin size, direction norm, or
  generic perturbation sensitivity;
- HVP validation fails or differs materially across dtype/device settings.

A null would not prove that the model has no nonlinear conjunction mechanism.
It would show that this H-lens construction and these directions do not detect
one reliably.

## 9. Implementation sketch

The first implementation should expose a small, testable API rather than add a
large second-order abstraction immediately:

```python
score = answer_logit - competitor_logit
gradient = torch.autograd.grad(score, hidden, create_graph=True)[0]
hvp = torch.autograd.grad((gradient * direction).sum(), hidden)[0]
directional_curvature = (hvp * direction).sum()
cross_curvature = (hvp_for_v * direction_u).sum()
```

The hidden state must remain connected to the suffix computation for the
backward pass. The full model should not be retained unnecessarily, and each
prompt should release its graph before the next prompt. Unit tests should
cover zero directions, scaling laws, finite-difference agreement, layer and
position indexing, and cleanup after an exception.

The first deliverable is a diagnostic table, not a new fitted checkpoint. A
later “H-lens artifact” would require a separate decision about whether to
store per-layer curvature operators, learned directions, or only a reusable
HVP-based scoring procedure.

## 10. Relationship to existing experiments

This extends the conjunction-swap experiment rather than replacing it. The
existing J-lens experiment asks where first-order information points; H-lens
asks whether the effect of one direction depends on another. The Boolean
extension remains useful because it provides explicit interaction ground truth.

The first result should therefore be reported as:

> second-order curvature and conjunction interaction analysis

and only later renamed an H-lens if it demonstrates a stable, reusable readout
with a clearly defined user-facing interpretation.
