# Conjunction swap experiment: summary

This is the short research brief. The full protocol is in [`CONJUNCTION_SWAP.md`](CONJUNCTION_SWAP.md), and the preregistration draft is in [`CONJUNCTION_SWAP_PREREGISTRATION.md`](CONJUNCTION_SWAP_PREREGISTRATION.md).

## The main claim, result, or open problem you want to build on

We want to determine whether a model represents conjunctions compositionally—as separable conjunct representations—or binds them into a distinct conjunction-level representation.

The smoke test showed that the pipeline works. In the France example, swapping `has the Eiffel Tower` to `has the Colosseum` changed the answer from Paris to Rome. This is not yet evidence about representation because the earlier seed set had weak B-swaps and ambiguous prompts; the fixture has now been repaired and passes strict structural validation.

## A crisp research question or hypothesis

When a model processes `the entity that satisfies property A and property B`, does changing A primarily affect A's representation while leaving B stable, or does it alter a bound conjunction-level representation?

- Compositional hypothesis: A and B can be changed independently, and their combined effects approximately compose.
- Binding hypothesis: changing one conjunct affects later or conjunction-level representations, changes the untouched conjunct, or produces non-compositional interactions.
- Null hypothesis: valid causal interventions produce no reliable lens or behavioral effect.

## A concrete experimental plan

1. Validate prompts where A and B identify the same baseline entity and replacements identify known alternatives.
2. Evaluate A alone, B alone, A and B, A-swapped, B-swapped, and both-swapped conditions.
3. Measure baseline answers, answer margins, observational J-lens readouts, and causal J-lens-vector substitutions.
4. Compare against sham interventions, type-matched and cross-type scrambling controls, position controls, intervention strengths, paraphrases, and held-out prompts.
5. Estimate uncertainty and detectable effect sizes with paired bootstrap intervals and pilot-based simulation.
6. Aggregate across valid items rather than interpreting individual prompts.

## What evidence would make the project succeed, fail, or change your mind

The compositional hypothesis succeeds if A and B swaps produce localized, predictable effects and double swaps approximately combine the single-swap effects. The binding hypothesis gains support if swaps reliably affect later conjunction/workspace positions, alter the untouched conjunct, or produce nonlinear interactions that causal controls reproduce. The project fails to support either claim if valid interventions produce no reliable lens or behavioral effects, or if results disappear under prompt and measurement controls. A surprising answer alone is not sufficient evidence.

## Follow-on Boolean-rule extension

The next extension is a separate, randomized factorial dataset with opaque
properties and known AND, OR, XOR, implication, and NOT rules. It tests all
truth-table rows, operand swaps, connective swaps, and matched random controls,
so neither conjunct accidentally determines the answer by itself. The primary
interaction statistic is the difference-in-differences across the four rows.

For each item we can compare the generated rule, model behavior, held-out
probes/J-lens readouts, causal counterfactuals, and the model's own report of
which facts and connective it used. The full design and falsifiers are in
section 9 of [`CONJUNCTION_SWAP.md`](CONJUNCTION_SWAP.md). This follow-on
should remain separate from the current natural-fact pilot and be
preregistered as its own dataset.
