# Hypothesis–Test–Evidence Traceability Matrix

This matrix records which Anthropic-style measurement tests address each
hypothesis, what outcome is measured, and the current evidence status. A
`partial` or `TODO` cell is intentional: it means the experiment has not yet
been run or does not yet support the claim.

| Hypothesis | Verbal report | Two-hop reasoning | Flexible generalization | Directed modulation | Selectivity | J-space specificity | Lens comparison |
|---|---|---|---|---|---|---|---|
| H1: representations are verbally reportable | **complete** — 27B, 4B, 0.8B, 0.6B, and Smol J/logit/random swaps | supporting test, not primary | supporting test, not primary | TODO — no modulation protocol | N/A | N/A | partial — tuned complete for four smaller rows; 27B pending |
| H2: representations are load-bearing for internal reasoning | secondary | **complete** — 27B, 4B, 0.8B, 0.6B, and Smol two-hop swaps | supporting test | TODO — no directed intervention protocol | supporting test | partial — no probe decomposition | partial — tuned complete for four smaller rows; 27B pending |
| H3: representations generalize across downstream functions | secondary | secondary | **complete** — 27B, 4B, 0.8B, 0.6B, and Smol function swaps | TODO — no modulation protocol | N/A | TODO — no independent probe split | partial — tuned complete for four smaller rows; 27B pending |
| H4: workspace contents are top-down modulated | TODO — verbal report is not a modulation test | N/A | N/A | TODO — focus/suppress/control runner absent | N/A | TODO — causal modulation readout absent | TODO — no predictive-lens modulation baseline |
| H5: workspace is selective rather than required for routine processing | N/A | partial — reasoning effects measured, no ablation/control battery | N/A | TODO — no ablation protocol | TODO — routine-vs-reasoning task pair absent | TODO — no selective ablation measurement | partial — random controls only |
| H6: J-space is a privileged causal subspace | supporting test | partial — 27B, 4B, 0.8B, 0.6B, and Smol J vs logit/random | partial — 27B, 4B, 0.8B, 0.6B, and Smol J vs logit/random | TODO — directed modulation not run | TODO — no selectivity controls | TODO — J/non-J probe split absent | partial — tuned comparisons complete for four smaller rows; 27B pending |
| H7: J-lens exposes intermediates better than predictive lenses | partial — causal rank outcomes | partial — causal and readout evidence | partial — causal outcomes | TODO — no modulation readout | TODO — no selectivity readout | TODO — no probe decomposition | **partial** — logit comparisons complete for all five rows; tuned comparisons complete for four smaller rows; 27B pending |

## Measurement definitions

- **Verbal report:** target concept rank/top-1/top-5 after a concept swap.
- **Two-hop reasoning:** downstream answer rank/top-1/top-5 after swapping an
  inferred intermediate.
- **Flexible generalization:** whether a swapped argument produces the correct
  answer across multiple functions.
- **Directed modulation:** target lens hit rate under focus/suppress/control
  instructions on unrelated carrier text.
- **Selectivity:** degradation of reasoning versus routine continuation under
  J-space ablation and matched controls.
- **J-space specificity:** causal effect of independently fitted concept-probe
  J-space versus non-J-space components, including clean-coordinate clamping.
- **Lens comparison:** intermediate pass@k, ablation KL, and causal swap
  success for J-lens, logit lens, tuned lens, and controls.

The completed evidence is linked from the per-experiment logs and MLflow
artifacts; this matrix is a coverage map, not a substitute for those results.
