# Conjunction swap experiment log

This is the chronological record for this experiment. The protocol and preregistration are static documents; this file records what was actually changed, run, observed, and decided.

Related documents:

- [summary](CONJUNCTION_SWAP_SUMMARY.md)
- [protocol](CONJUNCTION_SWAP.md)
- [preregistration draft](CONJUNCTION_SWAP_PREREGISTRATION.md)
- [dataset](../../data/experiments/conjunction-swap.json)

## 2026-07-12 — Preparation and pilot gate

### Completed

- Added tabulated human-readable output to `scripts/eval_conjunction_swap.py`; detailed JSON output remains unchanged.
- Added `scripts/validate_conjunction_data.py` with tokenizer span checks and strict mode.
- Repaired malformed conjunct spans and unchanged B-swaps in the 12-item fixture.
- Strict validation passes all 12 items with zero errors and zero warnings.
- Added the summary and preregistration documents.
- Added an empirical power/sensitivity plan based on pilot variance, paired-item correlation, valid-item rate, and simulation.

### Existing run

The original two-item smoke test completed without an OOM and produced `data/experiments/conjunction-swap-results.json`. It used the pre-repair fixture, so it is a pipeline check only and must not be treated as evidence or as the final pilot result.

### Current interpretation

Twelve items are appropriate for a pilot and prompt/hook validation, not for a broad claim about conjunction representations. The confirmatory claim requires causal intervention and a larger held-out item set.

### Open work

1. Rerun the observational pilot on the repaired fixture and inspect the tabulated output.
2. Implement causal J-lens-vector/activation substitution with sham and position controls.
3. Complete the larger lens fit or otherwise freeze the lens checkpoint used for confirmatory scoring.
4. Estimate power from pilot effects and freeze the confirmatory item count and analysis rules.


## 2026-07-12 — Repaired observational pilot

### Run

Command run in tmux pane `0:1.3`:

```bash
uv run python scripts/eval_conjunction_swap.py \
  --lens data/lenses/qwen3.5-0.8b-lens.pt \
  --data data/experiments/conjunction-swap.json \
  --out data/experiments/conjunction-swap-pilot-results.json
```

The run loaded the Qwen3.5-0.8B model with 23.24 GiB free, processed all 12 items and six conditions per item, and completed without an OOM. Output: `data/experiments/conjunction-swap-pilot-results.json`.

### Findings

The runner produced a tabulated top-1 summary. Several items showed expected directional behavior—for example, France/Italy changed `Paris` to `Rome` under the A swap, and France/Italian-language changed `French` to `Italian` under the B swap. Other items produced weak answer-position predictions such as `a`, `the`, or an empty decoded token. Therefore the run validates the pipeline and reveals prompt/model screening failures, but it is not evidence for the representation hypotheses.

### Decision

Do not use all 12 items for confirmatory statistics. Before causal intervention, add an explicit baseline-quality report using expected answers and answer margins, classify items as valid/ambiguous/invalid, and repair or exclude invalid items according to the preregistered rules. Preserve this complete pilot output as diagnostic data.

### Next step

Implement and unit-test the causal activation/J-lens-vector substitution on a small known-good item, with no-op, sham, nearby-position, and intervention-strength controls before running the full pilot.


## 2026-07-12 — Reporting and scrambling controls

- Added aggregate summary tables to the evaluator: top-1 change counts/fractions relative to the conjunction baseline, plus expected-answer agreement for conditions with declared answers.
- Added three planned nonspecific-intervention controls to the protocol and preregistration:
  - type-matched semantic scramble;
  - cross-type scramble such as `Paris` to `bicycle`;
  - positional scramble at a nearby non-conjunct token.
- The controls are documented but not yet executed; they belong in the causal intervention runner, not in the observational result already collected.


## 2026-07-12 — Noise-floor control design

The pilot should include a graded scramble battery rather than only one unrelated substitution: conjunct-order reversal, connective substitution (`and`/`or`), within-span word-order scrambling, random content-word replacement with fixed seeds, type-matched semantic scrambles, and cross-type scrambles. The causal phase should add matched-norm random-vector, unrelated-item-vector, and nearby-position controls.

The intended swap will only be interpreted as specific if it exceeds the relevant control distribution and has the predicted semantic direction. Comparable effects from random-word or random-vector controls will count as nonspecific intervention sensitivity.


## 2026-07-12 — Expanded observational pilot interpretation

The expanded run saved `data/experiments/conjunction-swap-pilot-controls-results.json` and printed per-item, primary-summary, control-summary, and expected-answer tables.

### Quantitative summary

- A swap changed the top-1 answer relative to `both` on 5/12 items (41.7%).
- B swap changed the top-1 answer on 5/12 items (41.7%).
- Double swap changed the top-1 answer on 8/12 items (66.7%).
- Control-order changed the top-1 answer on 5/12 items (41.7%).
- `and` to `or` changed it on 4/12 items (33.3%).
- Same-type scramble changed it on 5/8 items (62.5%); only eight items had a same-category partner.
- Cross-type scramble changed it on 6/12 items (50.0%).
- Expected baseline answer appeared top-1 for only 4/12 `both` prompts; it appeared in the saved top-5 for the same 4/12.

### Interpretation

The intended swaps are not more behaviorally specific than the simple controls in this pilot. The 8/12 double-swap rate is compatible with generic prompt perturbation because order and scramble controls produce comparable rates. Only a few items have reliable baseline and swapped answers: France/Italy, Canada/US, Germany/France, and France-language. The remaining items need repair, exclusion, or explicit ambiguity-control status.

This is therefore a successful screening/noise-floor pilot, not evidence for compositional or bound conjunction representations. The next analysis should use a baseline-quality and answer-margin filter, then run causal interventions only on valid items with matched-norm random, unrelated-vector, and nearby-position controls.


## 2026-07-12 — Calibrated J-lens/logit-lens pilot

The evaluator was upgraded using the Anthropic baseline lessons: continuation targets now include context-appropriate leading whitespace, expected-token ranks are recorded, best layer/position is reported, and each condition receives a matched vanilla logit-lens comparison. A two-item smoke test passed, followed by the full 12-item run:

`data/experiments/conjunction-swap-calibrated-pilot-results.json`

For the unchanged conjunction conditions:

- expected token was model top-1 on 6/12 items;
- best J-lens rank was 1 on 7/12 items;
- best vanilla logit-lens rank was 1 on 4/12 items;
- best J-lens rank was ≤5 on 8/12 items;
- best vanilla logit-lens rank was ≤5 on 8/12 items.

This confirms that final answer quality and intermediate lens recoverability are distinct. Weak final predictions do not automatically make an item useless for observational lens analysis, but they do make behavioral swap interpretation fragile. The calibrated readout is now the required format for future conjunction analyses. It remains observational and does not establish causality.


## 2026-07-12 — Model/lens pairing and larger fit

The experiment matrix is now explicit:

| run | model | lens |
|---|---|---|
| Anthropic baseline | `Qwen/Qwen3.5-4B` | released matching 4B `qwen-n1000` lens |
| small conjunction | `Qwen/Qwen3.5-0.8B` | locally fitted matching 0.8B lens |
| larger conjunction replication | `Qwen/Qwen3.6-27B` | released matching 27B lens initially; local fit is a separate comparison track |

The preliminary 0.8B lens used in earlier conjunction pilots contains `n_prompts=62`. A balanced 204-prompt, tokenizer-checked corpus is now prepared at `data/lens-prompts/fit-mix-204.json`, with 34 prompts per category. Its matched 0.8B lens fit is running in tmux pane `0:1.3` with checkpointing.

A locally fitted 27B lens would be scientifically useful for comparison with the released 27B lens, but prior attempts OOMed because fitting retains large autograd graphs even with 4-bit weights. It will be attempted as a memory-bounded secondary track and will not replace the released matched lens if it cannot fit.


## 2026-07-12 — External benchmark sources downloaded

We began processing established logical-reasoning datasets for a stronger conjunction corpus:

- **ProofWriter** (`tasksource/proofwriter`): 585,552 train, 85,468 validation, and 174,476 test rows scanned; 15,000 conjunction-bearing candidates saved across splits.
- **FOLIO** (`tasksource/folio`): 1,001 train and 203 validation rows scanned; 1,084 conjunction-bearing candidates saved.

Derived files:

- `data/benchmarks/proofwriter-conjunction-candidates.jsonl`
- `data/benchmarks/folio-conjunction-candidates.jsonl`
- `data/benchmarks/inventory.json`

Provenance is recorded in `inventory.json`. The first filter is intentionally broad: a conjunction appears somewhere in the theory or formal premises, but the rows have not yet been shown to satisfy the stronger design criterion that A alone is ambiguous, B alone is ambiguous, and A∧B uniquely identifies one entity. The next processing step is structural extraction and intersection filtering. ProofWriter is the primary source for controlled cases; FOLIO is reserved for natural-language robustness checks.

## 2026-07-12 — Structural ProofWriter extraction

Added `scripts/extract_proofwriter_strong_conjunctions.py`. It conservatively
parses simple unary ProofWriter facts and rules, retaining cases with explicit
A and B facts for an entity, no explicit C fact, and no detected A→C or B→C
singleton rule. It produced 260 candidates at
`data/benchmarks/proofwriter-strong-conjunctions.jsonl`.

These are not yet experiment-ready. The next CPU validation must deduplicate
stories, check derivation depth and negation, tokenize the actual Qwen prompt,
and test that A-only and B-only ablations are genuinely ambiguous while A∧B
has a unique answer. Only then should examples enter the held-out conjunction
causal set.

## 2026-07-13 — Strict factorial screen completed

The static factorial validator was run on the current 390 extracted
ProofWriter candidates. Only 9 survived all strict criteria: A ambiguous, B
ambiguous, a unique A∧B intersection, and no explicit conclusion fact. The
failure counts were 121 for A ambiguity, 113 for B ambiguity, 184 for a unique
intersection, and 264 for an explicit conclusion; counts overlap because a row
can fail multiple criteria.

The current 12 hand-authored conjunction fixtures remain clean: tokenizer-span
validation found 0 structural errors and 0 scientific warnings. The strict
ProofWriter survivors are therefore a held-out seed, not yet a 100-item
benchmark. Expansion should use fresh controlled templates or a less strict,
explicitly labeled design rather than silently mixing redundant examples into
the primary factorial analysis.

## 2026-07-14 — Boolean-rule and self-report extension added to design

The experiment protocol now specifies a follow-on randomized Boolean track.
It will use opaque independently randomized properties and balanced truth
tables for AND, OR, XOR, implication, and NOT, with operand/connective swaps
and matched semantic, positional, word-order, and random-vector controls.

The intended evidence chain is: generated rule → model behavior → probe/J-lens
readout → causal counterfactual → model self-report. The self-report is not
treated as evidence by itself; it must predict held-out counterfactuals and
agree with the causal intervention. The primary interaction statistic is the
four-cell difference-in-differences, and this dataset will remain separate
from the natural-fact conjunction pilot until independently preregistered.
