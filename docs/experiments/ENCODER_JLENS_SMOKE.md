# Encoder J-lens exploratory smoke

Status: exploratory; excluded from the main decoder report

## Subject and question

We tested whether a Jacobian-style readout can predict a cross-encoder's final
relevance score from intermediate hidden states. The subject is
`cross-encoder/ms-marco-MiniLM-L-6-v2`, loaded from its Hugging Face PyTorch
checkpoint because the Wolf Village copy is ONNX-only and cannot provide
autograd Jacobians.

The model has 22,713,601 parameters, six Transformer layers, and hidden width
384. We used 24 hand-written query/document pairs, with the first 16 pairs for
the readout estimate and the final eight as a held-out smoke set. This is a
feasibility test, not benchmark evidence.

## What was measured

For each pair and layer we recorded the `[CLS]` state and the gradient of the
final relevance score with respect to that layer's `[CLS]` state. We averaged
those score gradients over the training pairs and used the resulting vector
as a first-order score-Jacobian readout. We compared it against:

- an ordinary least-squares linear score readout from the intermediate `[CLS]`;
- the model classifier head applied directly to the intermediate `[CLS]`;
- the final model score.

This is not yet the full decoder-style J-lens construction. A faithful encoder
version should likely compute the Jacobian of final `[CLS]` representation
with respect to intermediate states, then apply the relevance head.

## Smoke result

| Layer | Score-Jacobian MSE | Score-Jacobian rank correlation | Linear-readout MSE | Linear-readout rank correlation |
|---:|---:|---:|---:|---:|
| 0 | 82.63 | −0.095 | 82.63 | −0.095 |
| 1 | 82.65 | −0.571 | 219.73 | 0.000 |
| 2 | 82.62 | 0.167 | 67.20 | 0.619 |
| 3 | 82.68 | −0.524 | 19.20 | 0.905 |
| 4 | 82.26 | 0.476 | 7.34 | 0.786 |
| 5 | 70.62 | 0.929 | 3.65 | 0.976 |
| 6 | 9.09 | 1.000 | 0.24 | 0.976 |

The late layers clearly contain relevance information. The naive average of
per-example score gradients is not a good early-layer transport map: it has
poor score calibration and can have the wrong rank ordering. A learned linear
readout is much stronger by layer 3–5. This is evidence for an encoder
readout opportunity, not evidence that the current score-gradient probe is a
working J-lens.

## Next experiment

Implement the faithful encoder analogue:

1. target the final `[CLS]` representation rather than only the scalar score;
2. estimate the layer-to-final representation Jacobian on grouped training
   pairs;
3. apply the frozen relevance classifier head;
4. evaluate held-out query groups using score MSE, pairwise ranking accuracy,
   Spearman correlation, and early-exit risk/coverage;
5. compare against the linear readout and direct intermediate classifier.

Only if that representation-level map generalizes should we consider causal
activation patching or an early-exit system.

Artifact: `data/experiments/encoder-jlens-minilm-smoke.json`
Runner: `scripts/encoder_jlens_smoke.py`
