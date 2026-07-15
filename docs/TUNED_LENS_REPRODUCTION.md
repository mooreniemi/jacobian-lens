# Tuned-lens reproduction track

## What is known about Anthropic's tuned lens

Anthropic do not identify a downloadable, model-specific tuned-lens checkpoint
or publish the exact tuned-lens training command used in their J-lens study.
Their report describes tuned lens as the learned affine-translator baseline and
uses it as a comparator, but the released reproduction detail is about the
J-lens corpus and Jacobian construction rather than a tuned-lens artifact.
Therefore we must not call our lens “Anthropic's tuned lens.”

The strongest public recipe we can reproduce is the original tuned-lens
recipe: fit on The Pile validation split and evaluate on the held-out Pile test
split. The official documentation uses `val.jsonl` for fitting and `test.jsonl`
for evaluation, with a 16.4M-token evaluation example. Our model-specific
implementation retains the same affine-translator and KL-to-final-logits
objective, while adding Qwen-compatible final-norm dispatch.

## Artifact names

| Artifact | Meaning |
|---|---|
| `tuned-wiki-small-v0` | Existing Wikitext-2 pilot: 100 steps and 256–512 short chunks; not a strong reproduction |
| `tuned-pile-repro-v1` | Planned model-matched fit on Pile validation, with held-out Pile test metrics |
| `tuned-pile-repro-v1-causal` | The same validated lens used in the causal intervention suites |

Existing Wikitext artifacts remain in place for provenance, but their manifests
and experiment-log labels use `tuned-wiki-small-v0` from this point forward.

## Staged execution

We proceed in increasing resource order and stop on any validation failure:

1. Qwen3-0.6B: local end-to-end Pile fit/evaluation smoke, then full pilot.
2. SmolLM2-135M and Qwen3.5-0.8B: repeat the same validated recipe.
3. Qwen3.5-4B: repeat with the model-matched tokenizer and revision.
4. Qwen3.6-27B: run only after the small-model artifacts pass all checks; use
   the cached local/Modal model and persistent dataset volume.

Every run must record the immutable model revision, tokenizer revision, dataset
file hashes, token count, sequence length, optimizer and learning-rate
schedule, precision/quantization, seed, package versions, and artifact hash.

Precision notation is explicit: `compute_dtype` describes arithmetic used by
the forward/translator computation, while `base_quantization` describes how
the frozen model weights were loaded. Thus `BF16 + NF4` means BF16 compute on
an NF4 4-bit base; it does not mean that the learned translators themselves
are 4-bit. `BF16 + none` means an unquantized BF16 base. Logit lens has no fit
precision; its runtime precision is inherited from the loaded base model.

Before causal use, the held-out test report must include layer-wise KL,
cross-entropy, top-1/top-5 accuracy, entropy, and calibration. The causal
runner must refuse a `tuned-pile-repro-v1` artifact whose model revision or
hidden size does not match the evaluated model.

## Why this is the right next step

Tuned lens is trained to predict the final output, so it should beat logit lens
on that predictive metric. Anthropic's claim for J-lens is different: J-lens
is intended to expose and causally drive intermediate computations, where the
tuned lens can “skip ahead” to the answer. A better Pile fit tests whether our
weak tuned results came from undertraining, without changing that conceptual
distinction.
