# Conjunction swap experiment

Status: design and smoke-test stage  
Summary: [`CONJUNCTION_SWAP_SUMMARY.md`](CONJUNCTION_SWAP_SUMMARY.md)  
Preregistration draft: [`CONJUNCTION_SWAP_PREREGISTRATION.md`](CONJUNCTION_SWAP_PREREGISTRATION.md)  
Experiment log: [`CONJUNCTION_SWAP_LOG.md`](CONJUNCTION_SWAP_LOG.md)  
Last updated: 2026-07-12  
Dataset: [`data/experiments/conjunction-swap.json`](../../data/experiments/conjunction-swap.json)  
Runner: [`scripts/eval_conjunction_swap.py`](../../scripts/eval_conjunction_swap.py)  
Results: [`data/experiments/conjunction-swap-results.json`](../../data/experiments/conjunction-swap-results.json)

## 1. Research question

When a model reads a phrase of the form:

> the animal that satisfies property A **and** property B

does it maintain separable representations of A and B, or does it form a conjunction-level representation that binds the two properties together somewhere else in the network?

This is a representation-and-causality question, not merely a question about whether the model can answer a factual prompt. The experiment therefore compares both model behavior and Jacobian-lens readouts, and ultimately requires a causal activation intervention.

## 2. Hypotheses

### H1: compositional conjunct representation

The model represents the two conjuncts sufficiently independently that changing A primarily changes the A-related representation, while B remains comparatively stable. Changing B has the analogous effect. The answer should shift in the direction implied by the changed conjunct when the modified conjunction remains interpretable.

A stronger version of H1 predicts approximate compositionality: the representation and answer effect of changing both conjuncts should be approximately predictable from the two single-conjunct changes.

### H2: bound conjunction representation

The model forms a conjunction-level representation that is not reducible to two independent conjunct representations. A single-conjunct change may therefore alter a later workspace position, change the representation of the untouched conjunct, produce a nonlinear answer effect, or fail to combine predictably with the other single-conjunct change.

H2 is not established merely by observing a surprising answer. It requires a reproducible, localized causal effect that cannot be explained by prompt ambiguity, invalid swaps, tokenization, lens error, or ordinary model uncertainty.

### H3: no detectable conjunct-level effect

The intervention has no reliable effect in the measured representation or behavior. This is a genuine null only if the prompt is valid, the lens has adequate sensitivity, the intervention reaches the intended positions and layers, and the model has enough answer margin for a change to be observable.

A null result does **not** prove that conjunctions are absent from the model. It may mean that the representation is elsewhere, the lens does not recover it, or the causal intervention is too weak or misplaced.

## 3. Concrete example

A clean conceptual example is:

```text
The number of legs on the animal that spins webs and has eight legs is
```

The intended chain is:

```text
spins webs       -> spider
has eight legs   -> spider
both             -> spider -> 8
```

The six conditions are:

| Condition | Prompt fragment | Purpose |
|---|---|---|
| single A | `spins webs` | Measure A alone |
| single B | `has eight legs` | Measure B alone |
| both | `spins webs and has eight legs` | Measure the conjunction |
| swap A | replace `spins webs` with another meaningful property | Test A's effect while B is fixed |
| swap B | replace `has eight legs` with another meaningful property | Test B's effect while A is fixed |
| swap both | apply both replacements | Test compositional combination |

The replacements must be designed so that the partial swaps remain interpretable. For example, `makes honey and has six legs` identifies a bee, whereas `spins webs and has six legs` may be contradictory or ambiguous. Such a contradictory condition can be a useful stress test, but it should not be treated as a clean compositional trial.

The current seed set includes related examples such as:

```text
The capital of the country that has the Eiffel Tower and is in Europe is
```

with the intended baseline `France -> Paris`. Replacing `has the Eiffel Tower` with `has the Colosseum` is intended to produce `Italy -> Rome`.

## 4. What is measured

### Behavioral measurements

For every condition, record:

- the model's top-k next-token predictions;
- probability or logit margin for the expected answer when available;
- whether the expected intermediate and answer are recovered;
- whether the swap changes the answer in the predicted direction;
- whether both swaps resemble the combined single-swap effects.

A top-1 change is useful but not sufficient. A rank change or probability shift may be meaningful even when the top token stays the same.

### Lens measurements

At each source layer and selected token position, record:

- top-k lens tokens at the A span;
- top-k lens tokens at the B span;
- top-k lens tokens at the final answer position;
- similarity or rank of the expected intermediate and its replacement;
- whether changes remain local to the swapped span or propagate to later positions.

The observational runner currently records these readouts. It does not yet substitute a lens vector into a model activation.

### Causal measurements

The decisive follow-up is to take a representation or J-lens vector associated with one condition and substitute it into another condition at specified layers and positions. Compare:

```text
ordinary prompt
causally intervened prompt
sham intervention / position control
```

The intervention should be evaluated against matched controls and multiple strengths, not only one arbitrary scale.

The causal controls should include: (a) a type-matched semantic scramble, such as replacing a country representation with an unrelated country; (b) a cross-type scramble, such as `Paris` to `bicycle`, which tests generic perturbation but is not a semantic-conjunction test; and (c) a positional scramble that applies the same intervention at a nearby non-conjunct token.

A fuller noise-floor battery should also include: (d) conjunct-order reversal; (e) connective substitution such as `and` to `or`; (f) within-span word-order scrambling; (g) random content-word replacement that preserves approximate span length and function-word structure, repeated over several fixed seeds; and, for causal runs, (h) matched-norm random-vector and unrelated-item-vector interventions. These controls test progressively stronger destruction of syntax, lexical identity, semantic type, and intended direction.

The primary swap should be considered specific only if its effect is larger and more directionally interpretable than the relevant control distribution. If random-word or random-vector controls produce comparable effects, the result should be classified as nonspecific intervention sensitivity.

## 5. Predictions and evidence

### Evidence supporting H1

Evidence would include all of the following patterns across many valid items:

1. A swaps produce a reproducible change in A-span lens readouts toward the replacement, while B-span readouts remain stable.
2. B swaps show the corresponding pattern in the opposite direction.
3. The answer shifts toward the replacement's answer when the conjunction remains semantically valid.
4. Double swaps are approximately predictable from the two single swaps.
5. Causal substitution of A's vector changes the answer in the A-predicted direction without requiring a broad unrelated activation change.

### Evidence supporting H2

Evidence would include a reproducible pattern such as:

1. Single-conjunct swaps affect a later conjunction/workspace position more strongly than the local swapped span.
2. The untouched conjunct's representation changes systematically when the other conjunct is swapped.
3. Double swaps produce an interaction that cannot be predicted from the two single swaps.
4. A causal intervention at the candidate conjunction-level position changes the answer, while matched local-span interventions do not.
5. These effects survive prompt paraphrases, position controls, intervention-strength controls, and valid-answer filtering.

This would support a bound or interaction-level representation. It would not by itself establish one unique mechanism.

### Evidence supporting H3 / a null

A meaningful null would look like:

- no reliable lens movement at the swapped span or later candidate workspace positions;
- no reliable causal answer shift across intervention strengths;
- no difference from sham and position controls;
- stable results across valid prompt paraphrases and random seeds.

A null is not interpretable if the baseline answer is already wrong, the replacement does not identify a unique entity, the model has nearly tied answer logits, or the lens fit cannot recover known information in the same prompts.

### Evidence against the current hypothesis framing

The experiment should be revised rather than forced into H1/H2 if:

- single A and single B do not independently identify the intended intermediate;
- the conjunction is logically contradictory or leaves many valid entities;
- answer effects occur but lens readouts do not track any plausible position;
- effects disappear under small paraphrases;
- token-span alignment is wrong;
- the fitted lens fails ordinary factual readout controls.

These would be design or measurement failures, not evidence for a bound conjunction.

## 6. Null-result taxonomy

When a swap has no apparent effect, classify it before interpreting it:

| Type | Meaning | Follow-up |
|---|---|---|
| semantic null | The replacement does not change the answer because both alternatives share the same answer | Keep as a control; do not call it a failed intervention |
| model null | The model ignores the changed property | Test stronger or clearer prompts and answer margins |
| lens null | The model may change behavior, but the lens does not recover the relevant information | Validate lens quality or inspect other positions/layers |
| causal null | The lens vector is recovered but substitution does not change behavior | Test intervention scale, direction, and target position |
| invalid null | Prompt, tokenization, baseline answer, or expected answer is defective | Exclude or repair the item |

## 7. Design requirements before interpretation

Each item should satisfy:

- single A identifies one intended intermediate;
- single B identifies one intended intermediate;
- both conjuncts identify the same intended baseline intermediate;
- A replacement identifies a known alternative;
- B replacement identifies a known alternative;
- partial and double swaps are labeled as valid, ambiguous, or contradictory;
- expected answer tokens are checked with the target tokenizer;
- the baseline model gives the expected answer with a measurable margin;
- A and B token spans are unambiguous;
- no fitting prompts overlap the reported evaluation set.

The current seed set does not yet satisfy all of these requirements. In particular, many entries have no real B replacement, and some swaps create broad or contradictory conditions. Those items are useful for smoke testing the runner but should not be used for a strong scientific conclusion.

## 8. Analysis sequence

1. Repair and validate the prompt set.
2. Run single-A, single-B, and both controls without interventions.
3. Run observational lens readouts and inspect A, B, and answer positions.
4. Add causal vector substitution with sham and position controls.
5. Aggregate by item and category rather than relying on individual examples.
6. Repeat with paraphrases and a held-out prompt set.
7. Record negative results and failed items in the research log.

The two-item smoke test is therefore only a pipeline check. It confirmed that the runner can load the model and lens, process all six conditions, write structured results, and print a summary without an out-of-memory failure.

## 9. Extension: recover and test the latent Boolean rule

The strongest extension is to move from natural-language conjunctions to
controlled Boolean decision problems with known ground truth. The question
becomes:

> Can the model's internal rule be read out and causally manipulated, and does
> the model's own report of that rule agree with the causal evidence?

This turns the experiment into a three-way comparison between the specified
rule, the model's behavior, and its internal representation. It also avoids
the main ambiguity in natural examples: one fact accidentally determining the
answer by itself.

### 9.1 Factorial Boolean dataset

Generate balanced truth-table items with opaque, independently randomized
properties so that lexical world knowledge cannot supply a shortcut. For an
AND rule, the behavioral labels should follow:

| A | B | Expected output |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 1 | 1 |

Use the same item template for AND, OR, XOR, implication, and NOT variants,
with held-out rules and held-out property names. Before any lens analysis,
verify that the model follows the rule above chance and report its answer
margin on every truth-table row.

### 9.2 Intervention matrix

For each rule, intervene on A alone, B alone, both operands, and the connective
itself. Include semantic, type-matched, unrelated-word, position, word-order,
and matched-norm random-vector controls. Record both the behavioral output and
the lens readout for each operand and the connective.

The primary Boolean interaction statistic is the difference-in-differences:

```text
interaction = output(1,1) - output(1,0) - output(0,1) + output(0,0)
```

For ideal binary outputs this is positive for AND, negative for OR, and more
strongly negative for XOR. We should estimate it from answer probabilities or
logit margins rather than only top-1 labels, then bootstrap over independent
items and random rule instantiations.

### 9.3 Internal readout and self-report

At selected layers, fit held-out probes for the truth values of A and B and
for the connective identity. Then ask the model to report:

- the truth value of each proposition;
- which propositions were necessary for the answer;
- whether the rule was AND, OR, XOR, implication, or NOT.

Compare these reports to the generated ground truth, probe predictions, lens
readouts, and causal answer changes. A verbal explanation is not evidence of a
faithful mechanism by itself; it becomes informative when it predicts held-out
counterfactual behavior and agrees with interventions.

### 9.4 Strong evidence and falsifiers

The extension succeeds if the model follows held-out Boolean rules, the
internal readouts recover operands and connective, operand swaps produce the
correct counterfactual outputs, and self-reports predict those outputs. A
particularly informative result would be a mismatch: the model verbally
reports an AND rule while causal swaps reveal OR-like behavior, or the model
answers correctly while the proposed operand representation has no causal
effect.

The conjunction-bound hypothesis gains support if single-operand swaps do not
combine additively, the interaction statistic is localized to a later
workspace position, or connective-level interventions dominate operand-level
interventions. A null remains meaningful only after the model passes the
behavioral truth-table check and the lens passes ordinary operand readout
controls.

This Boolean extension should be treated as a follow-on dataset and not mixed
with the current natural-fact pilot. Its randomized rules, truth tables,
model outputs, probes, intervention strengths, and self-report scoring should
be preregistered separately.
