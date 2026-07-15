# Conjunction swap experiment: preregistration draft

Status: draft; no confirmatory run has been started.  
Protocol: [`CONJUNCTION_SWAP.md`](CONJUNCTION_SWAP.md)  
Summary: [`CONJUNCTION_SWAP_SUMMARY.md`](CONJUNCTION_SWAP_SUMMARY.md)

## 1. Confirmatory question

Does changing one conjunct in a model prompt produce a localized, compositional representation change, or does it reveal a bound conjunction-level representation?

The primary comparison is between the A-swap and B-swap conditions relative to the unchanged conjunction condition. The causal intervention analysis is required for a confirmatory representation claim; observational readouts are exploratory and diagnostic.

## 2. Hypotheses

- **H1, compositionality:** an A intervention changes A-local readouts more than B-local readouts, an analogous B intervention changes B-local readouts more than A-local readouts, and the double intervention is approximately predictable from the two single interventions.
- **H2, binding/interaction:** an intervention changes later conjunction/workspace readouts, changes the untouched conjunct, or produces a reliable interaction in the double intervention that is not predicted by the single interventions.
- **H0, no detectable effect:** after valid-item filtering and causal controls, no intervention produces a reliable effect above sham and position controls.

H2 is not accepted merely because an answer changes. The effect must be reproducible, localized or interaction-specific, and survive the controls below.

## 3. Data and exclusions

The initial fixture is `data/experiments/conjunction-swap.json`, currently 12 items across country, element, planet, animal, plant, and person categories.

Before confirmatory scoring, each item must pass the existing strict validator:

```bash
uv run python scripts/validate_conjunction_data.py --strict
```

An item is excluded before looking at intervention results if any of the following holds:

- A or B is absent or ambiguously tokenized;
- single A, single B, or both does not have the intended baseline answer;
- a replacement does not identify a declared alternative, unless explicitly marked as an ambiguity control;
- the prompt is contradictory or has no interpretable expected answer;
- the model's baseline answer has no measurable margin under the predeclared threshold;
- the lens readout or causal hook fails technically.

All exclusions and their reasons must be reported. No item may be excluded because its intervention result is inconvenient.

## 4. Primary outcomes

The primary behavioral outcome is the paired change in expected-answer log probability at the final answer position:

```text
A swap versus both
B swap versus both
both swap versus both
```

Top-1 answer changes are secondary categorical summaries, not the sole outcome.

The primary representation outcome is the difference-in-differences between the intended swapped span and the untouched span, measured as the change in expected-token lens rank/logit across the predeclared layer band.

The primary binding interaction is:

```text
observed effect(both swap)
  - effect(A swap)
  - effect(B swap)
```

computed on the same item and normalized using the sham intervention.

## 5. Controls

The confirmatory run will include:

- no-intervention baseline;
- A-only and B-only conditions;
- sham vector substitution with matched norm and random or unrelated direction;
- type-matched semantic scramble using an unrelated entity of the same type;
- cross-type scramble (for example `Paris` to `bicycle`) as a nonspecific perturbation control, analyzed separately from the semantic hypothesis;
- conjunct-order reversal and connective substitution controls;
- within-span word-order and random-content-word scrambles, repeated over fixed predeclared seeds;
- position controls at nearby non-conjunct tokens;
- multiple intervention strengths selected before inspecting confirmatory outcomes;
- paraphrase or template controls where available;
- held-out items not used to fit the lens.

## 6. Statistical analysis

Results will be analyzed at the item level, preserving the paired structure. We will report:

- per-item effects;
- category-stratified effects as descriptive analyses;
- median and mean paired effects;
- 95% paired bootstrap confidence intervals over items;
- the fraction of items showing the predicted direction;
- permutation or sign-flip tests for the primary paired contrasts;
- effect sizes with units and normalization stated explicitly.

Multiple layers and positions will not be treated as independent item replicates. The layer band and scoring positions must be fixed before confirmatory scoring; exploratory heatmaps may not redefine them after seeing results.

## 7. Power and sensitivity analysis

A conventional power calculation is not reliable before we observe the variance structure of these paired, token-level outcomes. We will therefore use an empirical two-stage plan:

1. **Pilot estimation:** use the smoke test and a separately labeled pilot subset to estimate within-item variance, baseline answer margins, valid-item rate, and the correlation between paired conditions.
2. **Simulation:** simulate paired item effects under a range of standardized effect sizes and valid-item counts, preserving the observed covariance structure. Report power for the predeclared test at each effect-size scenario.
3. **Sensitivity table:** publish the smallest effect detectable with approximately 80% and 90% power for the planned item count, plus how power changes after plausible exclusion rates.
4. **No post-hoc expansion:** the confirmatory item count and stopping rule are frozen before the final run. Additional items may form a separately labeled replication, not silently increase confirmatory power.

Because the primary outcomes are paired, power depends more on within-item correlation and valid-item count than on the raw number of token positions. Lens layers and positions increase measurement precision but do not count as independent samples.

## 8. Decision rules

Evidence favoring H1 requires a consistent localized pattern for both A and B, a predicted answer effect on valid items, and no comparable effect in the untouched span or sham control.

Evidence favoring H2 requires a reproducible conjunction-level or interaction effect in the causal analysis that exceeds sham and position controls and survives the predeclared robustness checks.

Evidence for H0 is a result compatible with zero whose confidence interval excludes effects larger than the predeclared practically meaningful threshold, together with adequate sensitivity according to the power simulation.

If the confidence interval is wide or power is inadequate, the conclusion will be “inconclusive,” not “null.”

## 9. Reproducibility record

Freeze and record before the confirmatory run:

- model and tokenizer revisions;
- lens file and fitting prompt split;
- exact fixture hash;
- validator output;
- layer band, positions, intervention strength grid, and sham procedure;
- software versions and GPU;
- command lines;
- raw JSON results and tabulated report.
