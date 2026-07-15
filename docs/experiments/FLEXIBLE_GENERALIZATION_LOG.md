# Flexible generalization experiment log

## 2026-07-12 — Causal evaluator and run started

This is Anthropic's third major causal test: swap the representation of one
argument into prompts that apply different downstream functions, testing
whether the replacement is correctly used by each function.

### Fixture

Source: `data/experiments/flexible-generalization.json`.

- categories: countries, months, animals, numbers;
- four arguments per category;
- four downstream functions per category;
- every source argument swapped with each of the other three arguments;
- 192 prompt trials total (`4 × 4 × 4 × 3`).

### Run configuration

- model: `Qwen/Qwen3.5-4B`;
- lens: released matching Qwen3.5-4B lens, revision `qwen-n1000`;
- layers: 24, 28, 30;
- positions: all prompt positions;
- methods: J-lens, ordinary logit-lens, norm-matched random control;
- Qwen kernels: off, because compatible fast kernels are not installed;
- execution session: `69178`;
- output: `data/experiments/flexible-causal-full48.json` (historical filename;
  the current evaluator contains 192 trials).

The smoke test passed on two trials before the full run. Completion results,
skips, and MLflow provenance will be appended here.

## 2026-07-12 — Full causal run completed

Output: `data/experiments/flexible-causal-full48.json`.

The 192 generated trials produced 170 usable single-token trials and 22 skips.
With three layers per trial, this yielded 510 conditions per method.

| method | conditions | target rank improved | median Δrank | target top-1 | target top-5 |
|---|---:|---:|---:|---:|---:|
| J-lens | 510 | 409 | +19 | 18 | 48 |
| logit lens | 510 | 414 | +13 | 0 | 20 |
| random matched | 510 | 203 | 0 | 0 | 9 |

At the item level, taking the median across the three layers, J-lens improved
139/170 items with median improvement +22; logit lens improved 140/170 with
median +15; random improved 66/170 with median 0. Using the exploratory
best-of-three-layer criterion, J-lens reached target top-1 on 11/170 items and
top-5 on 27/170; logit reached 0 and 11; random reached 0 and 4.

This supports flexible reuse of the swapped representation, with J-lens
producing stronger exact successes than logit and random controls. The raw rank
movement is not uniquely J-lens-specific—logit is also a strong intervention—
so the top-k and item-level comparisons are more informative than the raw
“improved” count.

MLflow provenance:

- experiment: `anthropic-causal-reproduction`;
- run ID: `59f8188ab46c42f5b03995a9be509524`;
- artifact: the JSON output above, with command, git commit, package versions,
  result hash, and run parameters.

## 2026-07-13 — Matched Qwen3.5-0.8B replication

Output: `data/experiments/flexible-causal-0.8b.json`.

The same 192-trial fixture yielded 170 usable trials and 22 multi-token skips,
using layers 16/19/21.

| method | item median improved | item median Δrank | any target top-1 | any target top-5 |
|---|---:|---:|---:|---:|
| J-lens | 108/170 | +3 | 13/170 | 30/170 |
| logit lens | 107/170 | +3 | 8/170 | 28/170 |
| random matched | 55/170 | 0 | 0/170 | 17/170 |

MLflow run: `b62be3848c0f4188b10ef2d7444a02cf`.

## 2026-07-13 — Matched Qwen3-0.6B replication

Output: `data/experiments/flexible-causal-0.6b.json`. The model-matched local
204-prompt lens used layers 19/22/25; 170 trials were usable and 22 were
skipped for multi-token concepts.

| method | conditions | improved | median Δrank | target top-1 | target top-5 |
|---|---:|---:|---:|---:|---:|
| J-lens | 510 | 352 | +4 | 23 | 76 |
| logit lens | 510 | 309 | +3 | 7 | 59 |
| random matched | 510 | 173 | −1 | 2 | 24 |

MLflow run: `1163758863d84c21b71d79a1e13f6f37`.

## 2026-07-13 — Matched SmolLM2-135M-Instruct replication

Output: `data/experiments/flexible-causal-smollm2-135m.json`. The matched local
204-prompt lens used layers 21/25/28; 170 trials were usable and 22 were
skipped for multi-token concepts.

| method | conditions | improved | median Δrank | target top-1 | target top-5 |
|---|---:|---:|---:|---:|---:|
| J-lens | 510 | 302 | +2 | 33 | 143 |
| logit lens | 510 | 254 | 0 | 13 | 117 |
| random matched | 510 | 184 | 0 | 8 | 107 |

MLflow run: `618bfbc97a00493784115fa14b15f8e1`.

## 2026-07-12 — 27B Transformers checkpoint download started

The official `Qwen/Qwen3.6-27B` Transformers repository is approximately
55.6 GB. It is being downloaded to `/home/alex/models/qwen3.6-27b-hf` for later
4-bit bitsandbytes loading in PyTorch. The existing 17 GB llama.cpp GGUF is
unchanged.

- execution session: `15778`;
- download script: `scripts/download_qwen36_hf.py`;
- download log: `/home/alex/models/qwen3.6-27b-hf-download.log`;
- disk guard: refuse if less than 120 GiB free;
- free space at start: 462.2 GiB.

The checkpoint is not yet a causal experiment result; after download we must
perform a one-item VRAM/load smoke test before any larger run.

## 2026-07-13 — 27B checkpoint download completed

The full `Qwen/Qwen3.6-27B` Transformers repository downloaded successfully to
`/home/alex/models/qwen3.6-27b-hf` (approximately 55.6 GB from the Hub, about
52 GB occupied locally). It has not yet been loaded into PyTorch. The next gate
is a 4-bit bitsandbytes load smoke test with a free-VRAM check, followed by a
single causal prompt before any batch experiment.

## 2026-07-13 — 27B PyTorch 4-bit smoke test completed

Script: `scripts/smoke_qwen36_27b_4bit.py`. The guarded test loaded the local
`Qwen3.6-27B` Transformers checkpoint with bitsandbytes NF4, starting with
23.24 GiB free and ending model load with 6.48 GiB free. One deterministic
prompt generated `Paris.` followed by the model's reasoning markers, and the
script released the model and CUDA cache successfully. This establishes that
the PyTorch path is viable for one-prompt inference; it does not yet establish
that a 27B J-lens fit or full causal batch fits in the remaining VRAM.

## 2026-07-13 — Tuned-lens comparisons across matched smaller models

| model | method | conditions | improved | median Δrank | top-1 | top-5 |
|---|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | tuned | 510 | 257 | +1 | 13 | 115 |
| SmolLM2-135M | logit | 510 | 254 | 0 | 13 | 117 |
| SmolLM2-135M | random | 510 | 205 | 0 | 6 | 110 |
| Qwen3-0.6B | tuned | 510 | 295 | +2 | 13 | 55 |
| Qwen3-0.6B | logit | 510 | 309 | +3 | 7 | 59 |
| Qwen3-0.6B | random | 510 | 173 | −1 | 2 | 24 |
| Qwen3.5-0.8B | tuned | 510 | 301 | +2.5 | 10 | 68 |
| Qwen3.5-0.8B | logit | 510 | 323 | +3 | 10 | 61 |
| Qwen3.5-0.8B | random | 510 | 195 | 0 | 0 | 51 |
| Qwen3.5-4B | tuned | 510 | 394 | +10 | 4 | 20 |
| Qwen3.5-4B | logit | 510 | 414 | +13 | 0 | 20 |
| Qwen3.5-4B | random | 510 | 194 | 0 | 0 | 8 |

MLflow runs: `a255f568aacc471ebfbbc4f6115309f7`,
`1f9fdfe87fc943008f53827b5930abae`, `3ee15dc832f34131a40c5659ccf5ba90`,
and `a0cbe0a733fa4ae5a76141b8fa4f42c7`.

## 2026-07-13 — Matched Qwen3.6-27B 4-bit replication

Output: `data/experiments/flexible-causal-27b.json`. The released matching
1000-prompt lens was used with NF4 PyTorch weights and layers 16/32/60. There
were 170 usable trials and 22 multi-token skips.

| method | conditions | improved | median Δrank | target top-1 | target top-5 |
|---|---:|---:|---:|---:|---:|
| J-lens | 510 | 394 | +9 | 32 | 53 |
| logit lens | 510 | 301 | +1 | 0 | 6 |
| random matched | 510 | 190 | 0 | 0 | 3 |

MLflow run: `9fee44e0623c400c84fc4e8c44d29308`.

## 2026-07-14 — Model-matched Qwen3.6-27B tuned-lens evaluation

The native Wikitext-trained tuned lens was fit on the exact Qwen3.6-27B
checkpoint in NF4 mode, validated, and exported to the same affine translator
basis convention used by the smaller-model tuned comparisons. The existing
fixture, 170 usable trials, 22 multi-token skips, and layers 16/32/60 were
retained; only the missing tuned row was evaluated.

| method | conditions | improved | median Δrank | target top-1 | target top-5 |
|---|---:|---:|---:|---:|---:|
| tuned | 510 | 283 | +1 | 0 | 4 |

Output: `data/experiments/flexible-causal-27b.json` (now includes the merged
`tuned` rows) and tuned-only artifact
`data/experiments/modal-tuned-27b/flexible-causal-27b-tuned.json`.
