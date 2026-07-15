# Two-hop causal J-lens experiment log

## 2026-07-12 — Smoke test

Added `scripts/eval_multihop_causal.py`, reusing the verified pseudoinverse
coordinate patch and norm-matched random control. Three items ran cleanly with
the matched Qwen3.5-4B model/released lens. The smoke result was weak and was
not interpreted as evidence: coordinate swaps improved the intended answer in
4/9 layer conditions versus 3/9 for random controls.

## 2026-07-12 — Full 90-item run

Output: `data/experiments/multihop-causal-full90.json`.

The corpus contained 90 prompts. Sixty-eight were usable single-token cases;
22 were skipped because at least one intermediate or answer tokenized into
multiple tokens. We tested layers 24, 28, and 30 at all prompt positions.

Layer-level summary (204 conditions per intervention):

| intervention | target rank improved | median Δrank | target top-1 | target top-5 |
|---|---:|---:|---:|---:|
| coordinate swap | 150/204 | +7 | 20 | 82 |
| norm-matched random | 55/204 | 0 | 6 | 42 |

To avoid pseudoreplication across layers, the item-level summary is the primary
readout: a coordinate swap improved the intended swapped answer at the median
across layers for 51/68 items, reached target top-1 at some tested layer for
12/68, and reached target top-5 for 32/68. The random control did so for 20/68,
2/68, and 14/68 respectively.

This supports a real causal effect of the J-space intervention on downstream
two-hop answers, above the current random control. It is not yet a perfect
paper reproduction because 22 cases are multi-token, the prompt corpus is a
local 90-item set rather than the paper's exact 50-item release, and the
Qwen3.5-4B model is not Anthropic's Claude model. The result should therefore
be reported as a matched open-model replication of the causal pattern, not as
an exact quantitative replication of the paper's success rate.

## 2026-07-12 — Logit-lens comparison

The same 68 usable items and three layers were rerun with ordinary unembedding
directions (`J = I`) as a causal comparison. The random control remained
matched to the J-lens displacement.

| method | conditions | target rank improved | median Δrank | target top-1 | target top-5 |
|---|---:|---:|---:|---:|---:|
| J-lens | 204 | 150 | +7 | 20 | 82 |
| logit lens | 204 | 146 | +4 | 9 | 66 |
| random matched | 204 | 55 | 0 | 6 | 42 |

The logit lens is therefore a meaningful causal baseline, not a null control:
it also redirects answers, but the J-lens produces more exact top-1 and top-5
successes. This supports J-lens specificity while showing that the effect is
not exclusive to every possible residual-space token direction.

At the time of this original 4B comparison, a tuned lens was not included:
there was no compatible fitted Qwen3.5 tuned-lens checkpoint locally. The later
Wikitext-trained tuned comparison is recorded below as a separate run, with its
translator-basis limitation documented explicitly.

### Paired item-level comparison

Because the three layers are repeated measurements on each prompt, the primary
comparison is paired by item. The median target-rank improvement across layers
was 7.0 for J-lens and 4.5 for logit lens. J-lens was better on 40/68 items,
tied on 21, and worse on 7. Using the exploratory best-of-three-layer success
criterion, J-lens reached top-1 on 12/68 items versus 4/68 for logit lens, and
top-5 on 32/68 versus 25/68. Exact paired sign tests for these selected-layer
outcomes were p=0.0039 and p=0.0078 respectively, but these p-values are
descriptive only because the layer-selection rule was not preregistered and
the dataset was used to develop the comparison.

MLflow provenance:

- two-hop comparative run: `5e40ab79943c47778de82d2c7645fc37`
- verbal-report comparative run: `8847f3f96e8242d5bb4664e8afafcac1`
- experiment: `anthropic-causal-reproduction`
- tracking store: local `mlflow.db`

## 2026-07-13 — Matched Qwen3.5-0.8B replication

Output: `data/experiments/multihop-causal-0.8b.json`.

The 0.8B run used the 204-prompt fitted lens and layers 16/19/21. It retained
68 usable single-token items and 22 multi-token skips.

| method | item median improved | item median Δrank | any target top-1 | any target top-5 |
|---|---:|---:|---:|---:|
| J-lens | 49/68 | +9.5 | 2/68 | 15/68 |
| logit lens | 46/68 | +8.0 | 2/68 | 9/68 |
| random matched | 30/68 | 0 | 1/68 | 4/68 |

MLflow run: `b1181700409a4e6889feaebad516fa4b`.

## 2026-07-13 — Matched Qwen3-0.6B replication

Output: `data/experiments/multihop-causal-0.6b.json`. The model-matched local
204-prompt lens used layers 19/22/25; 68 items were usable and 22 were skipped
for multi-token concepts.

| method | conditions | improved | median Δrank | target top-1 | target top-5 |
|---|---:|---:|---:|---:|---:|
| J-lens | 204 | 118 | +2 | 3 | 32 |
| logit lens | 204 | 116 | +1 | 3 | 27 |
| random matched | 204 | 86 | 0 | 3 | 27 |

MLflow run: `50b8c08602924e0ca23b0fb8b60a1b79`.

## 2026-07-13 — Matched SmolLM2-135M-Instruct replication

Output: `data/experiments/multihop-causal-smollm2-135m.json`. The matched local
204-prompt lens used layers 21/25/28; 66 items were usable and 24 were skipped
for multi-token concepts.

| method | conditions | improved | median Δrank | target top-1 | target top-5 |
|---|---:|---:|---:|---:|---:|
| J-lens | 198 | 130 | +18.5 | 13 | 21 |
| logit lens | 198 | 115 | +5 | 7 | 20 |
| random matched | 198 | 74 | −1 | 7 | 13 |

MLflow run: `d210071231a3408e9d20eb58a18beea7`.

## 2026-07-13 — Tuned-lens comparisons across matched smaller models

| model | method | conditions | improved | median Δrank | top-1 | top-5 |
|---|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | tuned | 198 | 108 | +4 | 8 | 26 |
| SmolLM2-135M | logit | 198 | 115 | +5 | 7 | 20 |
| SmolLM2-135M | random | 198 | 62 | −5 | 4 | 9 |
| Qwen3-0.6B | tuned | 204 | 96 | 0 | 5 | 26 |
| Qwen3-0.6B | logit | 204 | 116 | +1 | 3 | 27 |
| Qwen3-0.6B | random | 204 | 83 | 0 | 3 | 26 |
| Qwen3.5-0.8B | tuned | 204 | 124 | +3.5 | 2 | 18 |
| Qwen3.5-0.8B | logit | 204 | 137 | +7 | 4 | 17 |
| Qwen3.5-0.8B | random | 204 | 80 | 0 | 3 | 12 |
| Qwen3.5-4B | tuned | 204 | 126 | +2 | 7 | 58 |
| Qwen3.5-4B | logit | 204 | 146 | +4 | 9 | 66 |
| Qwen3.5-4B | random | 204 | 55 | 0 | 6 | 42 |

MLflow runs: `3ffa5c9e4112415ca06e1ba48d590cea`,
`168fb89b66784a099eed5f788e56c096`, `018d4e9fb0ff4b6897f6d38085153830`,
and `a65c7b162d3743ba884546abc852486d`.

## 2026-07-13 — Matched Qwen3.6-27B 4-bit replication

Output: `data/experiments/multihop-causal-27b.json`. The released matching
1000-prompt lens was used with NF4 PyTorch weights and layers 16/32/60. There
were 68 usable items and 22 multi-token skips.

| method | conditions | improved | median Δrank | target top-1 | target top-5 |
|---|---:|---:|---:|---:|---:|
| J-lens | 204 | 155 | +11 | 23 | 48 |
| logit lens | 204 | 134 | +3 | 4 | 22 |
| random matched | 204 | 72 | 0 | 3 | 19 |

MLflow run: `36ffac85a4f74495a2682f7d2a5ff7a5`.

## 2026-07-14 — Model-matched Qwen3.6-27B tuned-lens evaluation

The native Wikitext-trained tuned lens was fit on the exact Qwen3.6-27B
checkpoint in NF4 mode, validated, and exported to the same affine translator
basis convention used by the smaller-model tuned comparisons. The existing
two-hop fixture, 68 usable items, 22 multi-token skips, and layers 16/32/60
were retained; only the missing tuned row was evaluated.

| method | conditions | improved | median Δrank | target top-1 | target top-5 |
|---|---:|---:|---:|---:|---:|
| tuned | 204 | 121 | +2 | 4 | 22 |

Output: `data/experiments/multihop-causal-27b.json` (now includes the merged
`tuned` rows) and tuned-only artifact
`data/experiments/modal-tuned-27b/multihop-causal-27b-tuned.json`.
