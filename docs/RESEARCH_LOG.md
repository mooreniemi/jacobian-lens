# J-Lens Research & Experiments Log

This log records the local reproduction of Anthropic's Jacobian-lens work, changes made for this machine, and experiments added afterward.

## Status

- Last updated: 2026-07-15
- Workspace: `/home/alex/Code/jlens/jacobian-lens`
- Hardware: NVIDIA GeForce RTX 3090, 24 GB VRAM, 31 GB system RAM
- Primary environment: project `.venv`, PyTorch 2.12.0+cu130, Transformers 5.9.0
- Notebook: [`walkthrough.ipynb`](../walkthrough.ipynb)

## 1. Original Anthropic reproduction

Source repository: [`/home/alex/Code/jlens/jacobian-lens`](.)

The original walkthrough loads a Hugging Face causal language model, wraps it with `jlens.from_hf`, downloads a pretrained Jacobian lens, applies the lens to a prompt, and renders a layer × position visualization.

The original notebook’s default model is `Qwen/Qwen3.5-4B`; Qwen3.6-27B is an alternate commented model. The original model-loading cell uses BF16 weights directly on CUDA.

### Reproduced successfully

- Created the project environment with `uv sync --extra dev`.
- Installed CUDA-enabled PyTorch and Transformers.
- Added IPython/Jupyter tooling.
- Loaded `Qwen/Qwen3.5-4B` through the original Hugging Face adapter.
- Auto-detected the Qwen text-decoder layout.
- Downloaded the 1,000-prompt pretrained lens from `neuronpedia/jacobian-lens`.
- Ran a real J-lens readout successfully on the RTX 3090.
- Observed roughly 7.9 GiB CUDA allocation for the 4B inference/readout path.

Smoke-test result: the original core reproduction works locally.

## 2. Changes required for this machine

### CUDA/PyTorch environment

The machine already had a CUDA-enabled llama.cpp build, but no system-level PyTorch installation. The project now has its own CUDA PyTorch environment.

Added development dependencies:

- `ipython`
- `jupyter`
- `bitsandbytes`
- `accelerate`

### Qwen3.6-27B loading

The original BF16 27B load does not fit on the RTX 3090. The notebook now has separate optional 4-bit Qwen3.6-27B cells using bitsandbytes, leaving the original 4B cells intact.

The 27B model is suitable for inference/readout, but not practical for Jacobian fitting on a 24 GB GPU. A fitting attempt exhausted approximately 24 GB because the retained autograd graph dominates memory even when weights are 4-bit quantized.

### Notebook GPU lifecycle

Notebook cells now include a `release_gpu(...)` helper. Important implementation detail: bare final expressions such as `model_27b` are retained by IPython in `Out[...]`, so model variables alone are not sufficient for cleanup. The helper clears named objects, IPython outputs, exception traceback references, Python garbage, and CUDA allocator caches.

A full kernel restart remains the reliable fallback.

## 3. Small-model lens fitting

The first fresh lens target is `Qwen/Qwen3.5-0.8B`, because it shares the Qwen3.5/Qwen3.6 hybrid architecture family while fitting comfortably on the RTX 3090.

### Preliminary fit

An initial run used 150 mixed evaluation prompts and the default `skip_first=16` rule. Short prompts were skipped, leaving:

- 62 usable prompts
- 23 source layers
- output: `data/lenses/qwen3.5-0.8b-lens.pt`

This is considered preliminary only.

### Current fitting split

The fitting corpus is now separated from the short evaluation probes:

- file: `data/lens-prompts/fit-mix-120.json`
- this is an exploratory split sampled from the evaluation fixtures; it is not yet a strict held-out benchmark split
- 20 prompts per category
- 120 prompts total
- prompts are tokenizer-checked for at least 18 tokens
- short probes receive a neutral prefix in their fitting copy
- original evaluation prompts are preserved as `evaluation_prompt`
- standard `skip_first=16` is retained

The six categories are:

- multihop reasoning
- multilingual transformations
- order-of-operations arithmetic
- associations
- poetry
- typo correction

The notebook fit cell uses this split, `dim_batch=64`, `max_seq_len=64`, and checkpointing every 10 prompts.

Baseline calibration experiment: [`docs/experiments/ANTHROPIC_BASELINE.md`](experiments/ANTHROPIC_BASELINE.md); per-experiment log: [`docs/experiments/ANTHROPIC_BASELINE_LOG.md`](experiments/ANTHROPIC_BASELINE_LOG.md)

## 4. New conjunction experiment

Detailed protocol and falsifiable predictions: [`docs/experiments/CONJUNCTION_SWAP.md`](experiments/CONJUNCTION_SWAP.md)
Per-experiment chronological log: [`docs/experiments/CONJUNCTION_SWAP_LOG.md`](experiments/CONJUNCTION_SWAP_LOG.md)


Goal: test whether a conjunction is represented compositionally or as a bound entity-level representation.

Dataset: [`data/experiments/conjunction-swap.json`](../data/experiments/conjunction-swap.json)

The seed set contains 12 controlled items. Intended conditions:

1. single conjunct A;
2. single conjunct B;
3. A AND B;
4. A swapped while B remains fixed;
5. B swapped while A remains fixed;
6. both conjuncts swapped.

The evaluator records model predictions and J-lens top-k readouts at the conjunct spans and answer position:

```bash
uv run python scripts/eval_conjunction_swap.py \
  --lens data/lenses/qwen3.5-0.8b-lens.pt \
  --max-items 2
```

Current evaluator: [`scripts/eval_conjunction_swap.py`](../scripts/eval_conjunction_swap.py)

This is initially observational. The next stage is causal activation/J-lens-vector substitution, followed by comparison of single-conjunct and conjunction-level effects.

## 5. Proposed H-lens extension

The proposed second-order experiment is documented separately:

- protocol: [`docs/experiments/HLENS_CONJUNCTION.md`](experiments/HLENS_CONJUNCTION.md)
- chronological log: [`docs/experiments/HLENS_CONJUNCTION_LOG.md`](experiments/HLENS_CONJUNCTION_LOG.md)

The first implementation will use Hessian-vector products rather than full
Hessian matrices. Its primary question is whether the cross-Hessian between
two property directions predicts conjunction-specific answer interactions
better than the first-order J-lens alone and better than matched scrambling
controls. This is a proposed experiment only; no H-lens result is being folded
into the current evidence tables yet.


### Working hypotheses

We expect an independently compositional representation to show a localized response: changing one conjunct should primarily alter the lens readout associated with that conjunct, preserve the other conjunct, and shift the answer in the direction implied by the replacement. The two-conjunct swap should approximately combine the two single-conjunct effects.

A bound conjunction would look different: a conjunct swap could alter a conjunction-level or later workspace position, fail to predictably compose with the untouched conjunct, or produce a nonlinear answer shift. A null or noisy result is also informative only after checking prompt validity, token spans, greedy baseline answers, lens fit quality, and causal intervention strength.

The experiment therefore has three gates: (1) validate the seed prompts and controls, (2) compare observational lens readouts across the six conditions, and (3) perform causal J-lens-vector substitution. Only the third gate can support a causal claim.

### Conjunction smoke test

The two-item smoke test completed without an OOM and wrote `data/experiments/conjunction-swap-results.json`. The runner now prints a compact findings summary as well as the detailed JSON. For the first France/Italy item, the baseline conjunction predicted `Paris`; swapping A changed the top prediction to `Rome`, while swapping B left it unchanged. The Canada/US item produced a weak baseline (`the`) but still showed an A-swap change (`New`), so it should be treated as a prompt-quality warning rather than a result. These observations motivate validating all seed controls before expanding or interpreting the experiment.

## 2026-07-12 — Strong causal reproduction results

The matched Qwen3.5-4B/released-lens verbal-report experiment ran across all
14 categories with norm-matched random controls. Coordinate swaps improved the
target rank in 40/42 layer conditions and reached target top-10 in 8/42;
random controls improved 11/42 and reached top-10 in 0/42. No target reached
top-1 in this open-model reproduction.

The stronger two-hop causal experiment then ran on the existing 90-item
`probe-swap.json` corpus. After excluding 22 multi-token cases, coordinate
swaps improved the intended downstream answer at the item level for 51/68
items, reached top-1 for 12/68 and top-5 for 32/68. Norm-matched random
controls reached top-1 for 2/68 and top-5 for 14/68. Full details are in
[`MULTIHOP_CAUSAL_LOG.md`](experiments/MULTIHOP_CAUSAL_LOG.md) and
[`VERBAL_REPORT_CAUSAL_LOG.md`](experiments/VERBAL_REPORT_CAUSAL_LOG.md).

These results support the causal pattern but do not establish exact numerical
replication of Anthropic's Claude results: our model, prompt corpus, lens, and
single-token filtering differ.

The two-hop run was extended with an ordinary logit-lens causal baseline:
J-lens swaps improved 150/204 layer conditions and reached top-1 on 20;
logit-lens swaps improved 146/204 and reached top-1 on 9; random matched
controls improved 55/204 and reached top-1 on 6. The result file and detailed
comparison are recorded in `MULTIHOP_CAUSAL_LOG.md`.

Anthropic's flexible-generalization causal reproduction is now running from
the existing four-category fixture. It expands to 192 source→target prompt
trials across countries, months, animals, and numbers, tested with J-lens,
logit-lens, and random controls at three layers. Its per-experiment log is
[`FLEXIBLE_GENERALIZATION_LOG.md`](experiments/FLEXIBLE_GENERALIZATION_LOG.md).

That flexible-generalization run has completed: 170/192 single-token trials
were usable. J-lens reached target top-1 on 11/170 items and top-5 on 27/170;
logit lens reached 0/170 and 11/170; random reached 0/170 and 4/170. The
MLflow run is `59f8188ab46c42f5b03995a9be509524`.

In parallel, the official 55.6 GB Qwen3.6-27B Transformers checkpoint is being
downloaded for a later PyTorch 4-bit run; this is separately logged and guarded
by a 120 GiB minimum-free-space check.

The 27B download has since completed successfully at
`/home/alex/models/qwen3.6-27b-hf`; PyTorch loading and causal smoke testing
remain TODO.

Local MLflow tracking is now configured with SQLite and ignored local artifact
state. The two comparative runs are logged under the
`anthropic-causal-reproduction` experiment. The latest provenance-complete
two-hop run ID is `5e40ab79943c47778de82d2c7645fc37`; the corresponding
verbal-report run ID is `8847f3f96e8242d5bb4664e8afafcac1`.

## Model × intervention experimental design matrix

This is the planned comparison matrix for the causal reproduction. A tuned
lens is model-specific and must be fitted separately; it is not interchangeable
with a J-lens or logit-lens direction.

| Model | Size | Clean baseline | J-lens swap | Logit-lens swap | `tuned-wiki-small-v0` swap | `tuned-pile-repro-v1` swap | Random matched |
|---|---:|---|---|---|---|---|
| SmolLM2-135M-Instruct | 135M | T1–T3 complete | T1–T3 complete; local 204-prompt lens | T1–T3 complete | T1–T3 complete; Wikitext pilot | TODO | T1–T3 complete |
| Qwen3-0.6B | 0.6B | T1–T3 complete | T1–T3 complete; local 204-prompt lens | T1–T3 complete | T1–T3 complete; Wikitext pilot | TODO: fit complete, held-out validation pending | T1–T3 complete |
| Qwen3.5-0.8B | 0.8B | T1–T3 complete | T1–T3 complete; 204-prompt lens | T1–T3 complete | T1–T3 complete; Wikitext pilot | TODO | T1–T3 complete |
| Qwen3.5-4B | 4B | T1–T3 complete | T1–T3 complete; released lens | T1–T3 complete | T1–T3 complete; Wikitext pilot | TODO | T1–T3 complete |
| Qwen3.6-27B | 27B | T1–T3 complete via guarded 4-bit NF4 runner | T1–T3 complete; released matching 1000-prompt lens | T1–T3 complete | T1–T3 complete; model-matched Wikitext pilot | TODO | T1–T3 complete |

The same prompt sets, layer aggregation rule, and item-level metrics should be
used across rows wherever the model tokenizer permits. Multi-token concepts
must be handled with an explicit span intervention or recorded as skips; they
must not be silently truncated.

### Design correction — balanced model × intervention coverage

The Qwen3-1.7B short-2M run is not a replacement for balanced coverage. It
adds a new model/intervention dimension that must be completed symmetrically:

| Dimension | Existing models | Qwen3-1.7B |
|---|---|---|
| Pile tuned lens, short-2M causal suite | Run the same three causal tasks for SmolLM2-135M, Qwen3-0.6B, Qwen3.5-0.8B, Qwen3.5-4B, and Qwen3.6-27B where resources permit | Already complete provisionally |
| J-lens | Existing model-matched artifacts/runs | Obtain or fit a model-matched 1.7B J-lens before claiming a full intervention comparison |
| Wikitext tuned lens | Existing pilot rows | Run the same Wikitext pilot if it is retained as a comparison condition |
| Logit and random controls | Existing rows | Already present in the short-2M causal suite |

The report coverage matrix now exposes these as separate statuses rather than
making the 1.7B addition look like a complete new model row. Cross-model plots
show the 1.7B short-2M bars with their explicit provenance label; they are not
silently treated as equivalent to the Wikitext tuned bars.

### 2026-07-15 — Qwen3-1.7B model-matched J-lens and causal suites

The model-matched Qwen3-1.7B J-lens fit completed locally on the RTX 3090 in
18m23s using the 204-prompt balanced fit mix, 27 source layers, BF16 model
compute, `dim_batch=64`, and checkpointing every 10 prompts. The resulting
artifact is `data/lenses/qwen3-1.7b-fit-204-lens.pt` with `d_model=2048`.

We then ran the same three causal protocols at layers 8/14/20 with all-position
patching and J-lens, logit, and random controls:

| Task | J-lens | Logit | Random |
|---|---|---|---|
| Verbal report | 27/42 improved; median Δrank +42.5; top-10 0 | 18/42; −16.5; top-10 2 | 21/42; +0.5; top-10 0 |
| Two-hop reasoning | 111/204; median +1; top-5 47 | 102/204; +0.5; top-5 40 | 64/204; 0; top-5 34 |
| Flexible generalization | 266/510; median +1; top-5 56 | 282/510; +1; top-5 42 | 186/510; 0; top-5 28 |

These are model-matched J-lens causal results. The tuned comparison in the
same plot group remains the separately labelled Qwen3-1.7B
`tuned-pile-repro-v1-short-2m` result, so the plot now distinguishes both
intervention identity and tuned-lens training provenance.

### 2026-07-15 — Qwen3-1.7B Wikitext versus Pile tuned-lens ablation

To isolate training-corpus effects at fixed model scale, we fit the same
100-step, 512-chunk `tuned-wiki-small-v0` pilot for Qwen3-1.7B. The fit took
19.3s on the RTX 3090, used BF16 compute, and peaked at 5.09 GiB. We exported
its patch basis and ran the same causal fixtures, layers, positions, and
logit/random controls as the Pile short-2M run.

| Task | Wikitext tuned | Pile short-2M tuned | Logit | Random |
|---|---|---|---|---|
| Verbal report | 22/42 improved; median +2; top-10 5 | 14/42; median −28; top-10 3 | 18/42; median −16.5; top-10 2 | 13/42; median −8.5; top-10 0 |
| Two-hop reasoning | 87/204; median 0; top-5 39 | 95/204; median 0; top-5 33 | 102/204; median +0.5; top-5 40 | 64/204; median −1; top-5 31 |
| Flexible generalization | 250/510; median 0; top-5 47 | 224/510; median 0; top-5 36 | 282/510; median +1; top-5 42 | 167/510; median −2; top-5 24 |

The controls are not numerically identical in the Wikitext output's random
draws, so comparisons should use the paired item-level outputs rather than
only these aggregate counts. The important design result is that both
model-matched tuned variants now appear separately in the 1.7B plot group;
Wikitext is the small pilot, while Pile short-2M is the larger-corpus but
still incompletely validated track.

The corresponding hypothesis-to-test table is maintained in
[`docs/HYPOTHESIS_TEST_MATRIX.md`](HYPOTHESIS_TEST_MATRIX.md).

### 2026-07-13 — Qwen3-0.6B matched T1–T3 cells

The Qwen3-0.6B model was evaluated with its own freshly fitted 204-prompt
J-lens (`data/lenses/qwen3-0.6b-fit-204-lens.pt`), using relative-depth layers
19/22/25. All three interventions were run on the same fixtures: J-lens,
ordinary logit-lens, and norm-matched random control. Results are recorded in
the three per-experiment logs and MLflow runs `fabb16b5006c4ebfba2924b00454f96a`,
`50b8c08602924e0ca23b0fb8b60a1b79`, and `1163758863d84c21b71d79a1e13f6f37`.

This fills the Qwen3-0.6B J/logit/random row for T1–T3. The tuned-lens cells
were subsequently completed with the local Wikitext trainer documented below;
the original reference trainer remains unusable because it imports the removed
`torchdata.dataloader2` API.

### 2026-07-13 — Tuned-lens intervention cells (`tuned-wiki-small-v0`)

The reference tuned-lens package was incompatible with the installed
`torchdata`, and also lacked Qwen3/Qwen3.5 model-surgery dispatch. We added a
local trainer that preserves its affine-translator/KL objective on Wikitext,
patched only the final-norm dispatch for Qwen3-family models, and exported the
translator basis `I + Wᵀ` for the causal runners. The nonlinear final RMSNorm
is not folded into that linear patch basis; this limitation is recorded in each
artifact manifest.

Completed pilot tuned T1–T3 rows (`tuned-wiki-small-v0`):

- SmolLM2-135M-Instruct: `data/lenses/smollm2-135m-tuned-wikitext`
- Qwen3-0.6B: `data/lenses/qwen3-0.6b-tuned-wikitext`
- Qwen3.5-0.8B: `data/lenses/qwen3.5-0.8b-tuned-wikitext`
- Qwen3.5-4B: `data/lenses/qwen3.5-4b-tuned-wikitext`

All corresponding JSON results and MLflow runs are recorded in the three
per-experiment logs. These short Wikitext-2 fits are sanity-check artifacts,
not the strong tuned-lens reproduction. The model-matched Pile validation/test
track is specified in `docs/TUNED_LENS_REPRODUCTION.md`.

### 2026-07-13 — Qwen3.6-27B causal cells

The official matching lens was downloaded and validated locally:
`n_prompts=1000`, `d_model=5120`, source layers 0–62. The causal runners were
extended with a guarded bitsandbytes NF4 path; the model loaded with 6.48 GiB
free after quantized placement. All three T1–T3 protocols completed with the
same relative-depth layers (16/32/60), and the full JSON outputs and MLflow
runs are recorded in the per-experiment logs.

The 27B tuned-lens cell remains TODO. A 4-bit inference model leaves too little
headroom for a trustworthy 845M-parameter translator fit, and no compatible
released tuned lens exists locally. This is a genuine resource TODO, not a
substituted approximation.

### 2026-07-14 — Qwen3.6-27B tuned lens fit, validation, and causal evaluation

The model-matched native tuned lens is now complete. The fit used the exact
`Qwen/Qwen3.6-27B` revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, NF4
bitsandbytes weights, BF16 compute, 512 Wikitext chunks of length 128, 100
steps, learning rate 0.001, seed 42, and the per-layer KL-to-final-logits
objective. It ran on one Modal A100-80GB with no retry; peak allocation was
34.852 GiB and the fit completed in about 193 seconds. The artifact is
`data/lenses/qwen3.6-27b-tuned-wikitext-nf4/` and its translator-basis export
is `data/lenses/qwen3.6-27b-tuned-wikitext-basis.pt`.

Validation passed for required files, model/revision/precision metadata,
finite parameters, and a smaller-model strict translator load plus forward
pass. The 27B artifact contains 128 finite translator tensors and has SHA256
`7efffd0bc78befbe7a8806612f7fe420d39ac41a2a7e33fd28dd158d3eacd48c`.

The tuned-only causal evaluation then used the same NF4 base model, fixtures,
all-position intervention, and relative-depth layers 16/32/60 as the existing
27B J/logit/random runs. The merged canonical result files now contain the
new tuned rows:

| experiment | conditions | improved | median Δrank | mean Δrank | top-1 | top-5/top-10 |
|---|---:|---:|---:|---:|---:|---:|
| verbal report | 42 | 24 | +3.5 | +83.4 | 0 | 7 top-10 |
| two-hop | 204 | 121 | +2 | — | 4 | 22 top-5 |
| flexible generalization | 510 | 283 | +1 | — | 0 | 4 top-5 |

Tuned-only provenance artifacts are in `data/experiments/modal-tuned-27b/`;
the three canonical JSON files under `data/experiments/` include the merged
`tuned` method. This is an affine translator-basis causal treatment: the
native tuned lens's nonlinear final RMSNorm is not folded into the patch.

### 2026-07-14 — Training-data exploration notebook and report examples

Added [`notebooks/training_data_explorer.ipynb`](../notebooks/training_data_explorer.ipynb), a read-only notebook that inventories fitting data versus causal evaluation fixtures, displays local J-lens prompt mixes, reads tuned-lens fit manifests, and optionally samples raw Wikitext. The HTML report now has matching intervention-data examples and explicitly marks logit/random as unfitted controls and the released 27B J-lens prompt text as unavailable in this checkout.

## 5. Reproduction discipline

For every experiment, record:

- model identifier and checkpoint revision;
- tokenizer and chat-template settings;
- lens file, source layers, and number of fitting prompts;
- prompt corpus and holdout split;
- `skip_first`, `dim_batch`, and maximum sequence length;
- GPU memory peak and software versions;
- exact command or notebook cell;
- qualitative result and any failed assumptions.

Do not use evaluation prompts in the fitting split when reporting held-out lens quality.

## 6. Human-readable findings report

### 2026-07-13 — Reproducible HTML report added

The current findings are now rendered as a human-readable HTML report with
tables and plots:
[`reports/research/index.html`](../reports/research/index.html). It is
generated from the recorded JSON artifacts by
[`scripts/generate_research_report.py`](../scripts/generate_research_report.py)
rather than by copying result values into a hand-maintained page.

The report includes model/intervention coverage, Qwen3.6-27B headline
comparisons, cross-model causal plots, tuned-versus-logit coverage for smaller
models, current conclusions, limitations, and the validation gate required
before publishing a 27B tuned lens. It is intended to be served locally and
viewed through SSH port forwarding.

The first report build exposed a selection bug: the cross-model plots chose the
tuned 4B artifact before the historical 4B J/logit/random artifact, and skipped
the latter because its filename contains `full`/`methods`. The generator now
selects an artifact containing the requested method and includes the recorded
4B runs. The corrected plots show J-lens values for all five model rows.

The report now also records uncertainty using a deterministic 2,000-replicate
cluster bootstrap over prompts/items. All scored layers belonging to one prompt
are resampled together, rather than treated as independent observations. It
reports 95% intervals for improvement rates and paired percentage-point
contrasts of J-lens or tuned lens versus the random matched control. These are
sample uncertainty intervals, not independent replication or a universal
significance claim. Three parameter-count summaries are now included: absolute
improvement rate, improvement above random in percentage points, and median
rank change. They are descriptive rather than scaling laws because model
family, tokenizer, prompt set, and lens fit differ across rows.

The report presentation was revised to a paper-like HTML layout: an abstract
style opening summary based on the reproduction and conjunction briefs, a
linked table of contents, restrained serif/ sans typography, and
closed-by-default disclosure sections for the large tables. This keeps the
plots visible while allowing detailed numerical tables to be opened on demand.

## 7. Modal tuned-lens preparation

### 2026-07-13 — Guarded remote fit prepared; same-model preflight submitted

The standard dense tuned-lens fit for Qwen3.6-27B now has a guarded Modal
runner in [`scripts/modal_fit_tuned_lens.py`](../scripts/modal_fit_tuned_lens.py).
It uses one exact `A100-80GB`, 128 GiB requested host memory, four CPU cores,
`retries=0`, `max_containers=1`, persistent model/output volumes, an 8-hour
execution timeout, and a 30-minute startup timeout. It requires an explicit
`--confirm-budget 25` before submitting work.

The conservative GPU/CPU/memory estimate for the configured maximum is
`$16.20`, leaving a margin below the `$25` ceiling. The remote trainer uses the
frozen base model in NF4 4-bit mode while keeping the dense tuned translators
trainable in BF16; the resulting artifact must be labeled as fit against the
NF4 model and validated before publication.

Before submission, the local checks passed: 40 tests, Ruff clean, and the
budget guard reported `$16.20`. A one-step, one-chunk same-model preflight was
then submitted from tmux pane `0:1.4`; this is intentionally bounded and is not
the full fit.

The runner was then tightened before preflight: the Hub model revision is
resolved and passed explicitly to both download and artifact metadata; the
model cache volume is committed after download; the output is checked for
`config.json`, `params.pt`, `fit_manifest.json`, matching model/revision
metadata, NF4 provenance, and finite parameters before the output volume is
committed; and the Modal image pins the local Torch/Transformers/
bitsandbytes/tuned-lens versions. The safety suite now has 40 passing tests.

The next submission will require the Modal secret `huggingface-secret`, exposing
only its `HF_TOKEN` variable to the container. The runner refuses anonymous
Hub access and never logs the token value. The already-running preflight was
submitted before this requirement was added and is unaffected; we will not
rerun it merely to change download authentication unless it fails.

### 2026-07-14 — Hard $5 Modal ceiling applied

Before any further GPU submission, the runner was changed to require exactly
`--confirm-budget 5`. The fit now has a 90-minute timeout, a 20-minute startup
timeout, 96 GiB host memory, one A100, four CPUs, zero retries, and one
container. Its conservative worst-case resource estimate is `$3.02`, leaving
the remaining budget as margin. The image-only build has no GPU allocation and
is not counted as a fit submission.

Two image-only attempts reached OCI rootfs unpacking and were then stopped by
the local command session. Modal logs show the first external shutdown came
from the interrupted local runner; the second was the explicit Ctrl-C used to
stop it. No model download or GPU fit occurred. Future image-only checks use
detached mode so terminal interruption cannot terminate the build.

The safety suite also now checks the model-cache contract: the named persistent
volume and stable cache path are used, an existing cached `config.json` takes
the no-download branch only when the complete weight set is present (including
all shards), and the volume is committed only after a successful download.
This is a regression check for reuse, not a substitute for observing
the first authenticated Modal run. The cache remains available until the Modal
volume is explicitly deleted or otherwise expires under the account's storage
policy.

### 2026-07-14 — Preflight exposed incomplete-cache validation bug

The authenticated preflight successfully reused the Modal volume and skipped
the download, but then failed because the prior interrupted anonymous download
had left `config.json` and metadata without any weight files. The old cache
condition incorrectly treated metadata alone as a complete model. The runner
now requires either a complete Safetensors shard index or a nonempty model
weight file before skipping download, and tests cover both incomplete and
sharded caches. No tuned-lens artifact was produced by this failed preflight.

Additional hardening now binds cache reuse to a `.jlens_revision` marker, fits
into a hidden partial output directory, refuses to overwrite a nonempty final
artifact, validates the manifest revision, and publishes only after artifact
checks pass. Gradient clearing uses `set_to_none=True` to reduce unnecessary
memory writes during fitting.

The Modal entrypoint now supports an explicit cache-only mode, allowing the
authenticated model download and volume commit to be observed and validated
without allocating an A100. It still requires the explicit budget confirmation
argument as a cost-control checkpoint.

### 2026-07-14 — Local-to-Modal model transfer and cache-path decision

The raw Qwen3.6-27B Transformers checkpoint was downloaded locally with the
authenticated high-performance Xet path and validated: 15 Safetensors shards,
about 51.8 GiB, and the immutable revision marker. The first `modal volume put`
completed under `Qwen--Qwen3.6-27B/qwen3.6-27b-hf/` rather than directly under
the runner's expected path. A server-side copy into the root path was started
but stopped after proving it would duplicate the large files unnecessarily.
The runner now recognizes the complete nested upload path directly, avoiding
another transfer. This is the chosen cost/time-saving path for tonight's fit.

The v1 model Volume remains within Modal's included 1 TiB storage allowance;
the current concern is transfer and compute time, not persistent-volume cost.

### 2026-07-14 — Local Hub authentication and transfer preflight passed

Using a permission-600 token file outside the repository, the exact Modal
download path was tested locally without downloading the 27B weights: Hub
metadata access succeeded at revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, an authenticated `config.json`
download succeeded, and a 29-file snapshot dry-run completed with 16 workers
and Xet disabled. The token value was never printed or logged. The full local
safety suite remains green at 45 tests.

### 2026-07-14 — Local tuned-lens smoke gate passed

Before another remote attempt, the exact local trainer path was run on the
cached `Qwen/Qwen3.5-0.8B` model with NF4 loading, one Wikitext chunk, and one
optimization step. Model loading, Qwen-family final-norm surgery, translator
construction, KL backpropagation, checkpoint writing, JSONL events, and finite
artifact output all succeeded. The smoke artifact contained `config.json`,
`params.pt`, `fit_manifest.json`, and `run-events.jsonl`; peak RTX 3090
allocation was 1.52 GiB. The 27B remote fit remains gated until the
authenticated cache-only step succeeds with the pinned Hub/Xet configuration.

### 2026-07-14 — Slow Qwen fallback caught before training

The first 27B load emitted Transformers' warning that the Qwen hybrid
linear-attention fast path was unavailable and would fall back to the Torch
implementation. The run was stopped during weight loading, before training.
The Modal image now pins `flash-linear-attention==0.5.1`, and the remote trainer
passes `--require-fast-kernels`; it will fail before training if `fla` cannot be
imported, rather than silently spending GPU time on the slow fallback.

The bounded smoke also showed that this Transformers build checks
`causal-conv1d`; the image now pins `causal-conv1d==1.6.2.post1` and the guard
requires both packages. The Wikitext loader was additionally changed to stop
tokenizing after the requested chunk budget instead of materializing the full
2.5M-token split, removing an avoidable tokenizer warning and CPU-memory cost.

Modal observability will provide the live application log, call history, CPU/
RAM/GPU resource metrics, GPU-memory metrics, and GPU-health events. The fit
itself now emits structured JSON events for preflight, model load, fit setup,
periodic loss/memory checkpoints, and completion; these are saved as
`run-events.jsonl` beside the lens artifact. The manifest also records the
software versions and peak CUDA allocation/reservation. Modal's dashboard is
the live source; the JSONL and manifest are the durable experiment provenance.

### 2026-07-14 — Local 27B one-step capacity probe reached backward pass

The complete local Qwen3.6-27B Transformers checkpoint was loaded in NF4
mode using the maintained NVIDIA PyTorch 25.11 image. The checkpoint and
architecture path are valid: model loading took about 105 seconds, leaving
6.48 GiB free on the 24 GiB RTX 3090. The trainer constructed the expected
64-layer, 5120-dimensional tuned lens with 2,949,452,800 parameters.

With one short Wikitext chunk and one optimization step, the run failed at
the first backward pass when allocating an additional 50 MiB. This confirms
that the local GPU limit is training memory for the dense lens plus gradients
and optimizer state, not checkpoint compatibility or model loading. No final
artifact was written. This is positive evidence that the guarded Modal
A100-80GB path is the appropriate next capacity test; the local run remains
useful for software and failure-boundary validation.

### 2026-07-14 — Proper tuned-lens reproduction track started

The existing Wikitext artifacts and causal rows are now explicitly labelled
`tuned-wiki-small-v0`. They remain useful sanity checks, but are not treated as
the original tuned-lens training recipe. Anthropic did not release an exact
model-specific tuned-lens checkpoint or training command for the J-lens study.

We are beginning `tuned-pile-repro-v1`, following the public tuned-lens recipe:
fit on The Pile validation split and evaluate on held-out Pile test text. The
first staged subject is Qwen3-0.6B, followed by the locally available
SmolLM2-135M-Instruct and Qwen3-1.7B. The Qwen3.5-0.8B, Qwen3.5-4B, and
Qwen3.6-27B checkpoints are not currently present as local Transformers
weights, so they remain future acquisition/remote cells rather than being
claimed as local Pile runs. The trainer now accepts explicit local
JSONL corpora so a run cannot silently use Wikitext when the Pile path was
intended. Dataset hashes, model revisions, token counts, held-out layer-wise
metrics, and artifact hashes are required before a lens enters causal tests.

The first `tuned-pile-repro-v1` fit completed for Qwen3-0.6B. It used 16,384
128-token chunks (2,097,152 sampled training tokens), 4,096 optimizer steps,
BF16 compute, and the unfrozen model in full precision. The artifact is
`data/lenses/qwen3-0.6b-tuned-pile-repro-v1` and the run completed in about
507 seconds without memory pressure. This counts as one fitted model, not yet
one validated model: held-out Pile-test scoring is the next gate.

The held-out evaluator was smoke-tested on 130,048 scored next-token positions
and showed the expected pattern: tuned lens substantially beats raw logit lens
in early layers and converges toward it near the final layer. The official
16.4M-input-token Pile-test evaluation is now running with the tested
memory-safe batch-size-32 setting; its output is reserved at
`data/experiments/qwen3-0.6b-tuned-pile-repro-v1-eval.json`. The evaluator
reports layer-wise KL, NLL, top-1, and top-5 for tuned and logit lenses.

The same guarded queue is prepared behind this evaluation: after the Qwen3-0.6B
held-out artifact appears, it fits and evaluates SmolLM2-135M-Instruct and
then Qwen3-1.7B with the same 16,384-chunk/4,096-step Pile recipe. Only one
CUDA workload is admitted at a time; the queue is deliberately sequential to
avoid an accidental OOM or an unrecorded model swap.

### 2026-07-14 — Preliminary efficiency plots and Modal billing provenance

The HTML report now includes an efficiency section at `#efficiency` with three
views: measured saved-artifact footprint, measured fit wall-time/peak GPU
memory, and a descriptive causal-effectiveness-versus-footprint scatter. The
plots are explicitly preliminary: the existing fit records use different
hardware and training regimes, so they are not pooled as a matched benchmark.
The report also records that GPT-5.6 Luna Medium, operating through OpenAI
Codex, assisted with implementation and documentation under the researcher's
direction; it did not generate the model outputs.

The Modal billing API was queried for the two successful 27B app IDs. The fit
was billed `$0.2458`, causal evaluation `$0.5071`, for a successful-run subtotal
of `$0.7528`; the full same-day workspace report was `$1.9379` across 23
entries, including failed/aborted attempts and shell usage. These are separate
from the conservative worst-case ceilings. The billing summary is archived at
`data/provenance/modal-billing-2026-07-14.json`.

The efficiency report initially omitted larger J-lens bars because those
artifacts were loaded from the published `neuronpedia/jacobian-lens` repository
during causal evaluation rather than copied into the local checkout. We
verified the repository metadata at revision `qwen-n1000` and added the
model-matched artifact sizes to the report: 406.3 MB for Qwen3.5-4B and 3,303.0
MB for Qwen3.6-27B. They are labelled as remote artifact metadata, not local
disk measurements.

### 2026-07-14 — Efficiency view refocused on artifact creation cost

The report's primary efficiency view was revised to match the research
question: creation wall time and peak GPU memory as functions of base-model
parameter count. The old artifact-footprint, quality-per-footprint, and
model-grouped cost plots were removed from the report because saved size is a
secondary storage/provenance fact, not the main cost of creating a lens.

The new log-log plot currently contains only measured completed fits: the
Qwen3-0.6B Pile tuned fit on an RTX 3090 and the Qwen3.6-27B Wikitext tuned
fit on an A100-SXM4-80GB. Missing model/method points are left blank rather
than imputed. The provenance table retains fit time, peak VRAM, hardware/source,
and artifact size (secondary), so future model-matched runs can be added
without conflating scale effects with intervention effects.

The efficiency visualization was then normalized to avoid implying that local
wall-clock time is a cash-price comparison with Modal. The primary time panel
now reports minutes per 1,000 optimizer steps, while the memory panel reports
peak VRAM as a fraction of installed device VRAM. The provenance table retains
total GPU-hours, VRAM-hours, optimizer-step count, and billed USD where known.
The Qwen3-0.6B local RTX 3090 row is explicitly “not priced”; the $0.2458
number applies only to the separately billed 27B Modal A100 fit.

No local 27B J-lens creation-time or GPU-memory telemetry is currently
recorded, so it is shown as an unmeasured row rather than a measured point.
The external M4 reference is retained only as a separately labelled baseline.
Logit lens is represented in the interpretation as zero artifact-creation/
training cost: it has no learned artifact, while its causal effectiveness
remains compared in the separate J-lens/logit/tuned/random result plots.

The post-validation causal queue is prepared in
`scripts/run_pile_causal_queue.sh`. It waits for complete 16.4M-token Pile
test artifacts for Qwen3-0.6B, SmolLM2-135M, and Qwen3-1.7B; checks finite
layer-wise metrics and early-layer tuned-vs-logit improvement; exports each
validated translator basis; and runs the verbal, two-hop, and flexible suites
sequentially with tuned/logit/random methods. It is waiting behind the active
Qwen3-0.6B predictive evaluation, so no second CUDA workload is currently
running.

The report generator now recognizes Qwen3-1.7B Pile artifacts and reads fit
telemetry for all three queued local Pile models (Qwen3-0.6B, SmolLM2-135M,
and Qwen3-1.7B) automatically when their manifests appear. This prevents a
completed queued cell from failing report generation or being omitted from
the creation-scaling view.

The primary efficiency section is now a 27B evidence table. It keeps the
recorded J-lens fit, tuned Wikitext pilot, zero-cost logit control, and missing
27B Pile fit visible while marking the hardware/workload mismatch explicitly.
The broader model-scale resource plot is retained as a folded exploratory
view, not presented as a method ranking.

### 2026-07-14 — Correction: 27B J-lens hardware provenance

We have not run any experiment on an Apple M4 Pro. The `164.7 min` J-lens
figure from `jlens-qwen36/docs/perf/fit-05.md`—20 prompts × 63 layers—is an
external baseline we were told about, not one of our measurements. It is
excluded from the report's empirical efficiency plots and GPU-cost comparison;
the report's runtime-environment table labels it as an external reference
only. There is currently no completed local RTX 3090 27B J-lens fit record
with wall-time or VRAM telemetry.

The successful 27B tuned Wikitext pilot was a separate Modal A100-SXM4-80GB
run: 193 seconds, 34.852 GiB peak allocated memory, and $0.2458 billed. A
local RTX 3090 tuned attempt reached fit-ready model loading but OOMed on the
first backward pass. This supports the hypothesis that J-lens may be more
memory-feasible on the 3090, but does not yet measure a local 27B J-lens cost.

Until a matched benchmark exists, we will report raw wall time together with
hardware, workload units, peak memory, and actual price where available. We
will not convert M4 time into a fictitious 3090 cost or claim that J-lens is
slower/faster than tuned from these unmatched runs. A proper comparison needs
the same model and defined fitting workload on the same device, or an
explicitly measured power/price model; FLOP-only conversion is not reliable
enough here because the Jacobian and Qwen3.6 architecture workloads differ.

### 2026-07-14 — Precision provenance made explicit

All efficiency and experiment records must distinguish compute dtype from base
weight quantization. The local Qwen3-0.6B Pile tuned fit used BF16 compute with
an unquantized base on the RTX 3090. The successful Qwen3.6-27B Wikitext pilot
used BF16 compute with a bitsandbytes NF4 4-bit frozen base on the Modal
A100-SXM4-80GB; its dense tuned translators were trained in the reported
compute dtype. The local 27B tuned attempt used the same NF4/BF16 setup and
OOMed during backward.

Logit lens has no fitted precision or training artifact: at application time
it uses the precision of the loaded base model and its unembedding. J-lens
precision and peak memory are not yet recorded for a completed local 27B run.
The report now exposes precision in the runtime, creation, and memory tables,
and newly generated causal JSON results include BF16/NF4 provenance. Older
causal JSON files without that field remain historical results and must not be
treated as precision-complete until re-run or separately annotated.

The runtime-environment table also records the published M4 Pro hardware
facts—up to a 20-core GPU, up to 64 GiB of unified memory, and 273 GB/s memory
bandwidth—while leaving ML-specific TFLOPS and the upstream benchmark's
precision as not published. These describe the external reference platform,
not a machine used in our experiments.

### 2026-07-14 — Browser-readable documentation export

The report generator now renders every Markdown file under `docs/` into a
matching HTML page under `reports/research/docs/`. The served report has a
Documentation and logs section plus a documentation index; relative links
between Markdown documents are rewritten to their rendered HTML counterparts.
This makes the research log, experiment protocols, preregistration drafts, and
per-experiment logs readable through the same SSH-forwarded report server.

### 2026-07-14 — Runner observability and interrupted 1.7B evaluation

The first Qwen3-1.7B held-out evaluation was started with `batch-size 8` as a
conservative setting, but without a fixed throughput benchmark, durable
progress events, or resumable aggregate checkpointing. After approximately
four hours it was stopped because the GPU was using only about 9.7 GiB of 24
GiB and the expected completion time was too uncertain. No final evaluation
artifact was written and the GPU was released cleanly.

This is recorded as a process failure, not an experimental result. The Pile
evaluator now writes flushed JSONL events containing batches, input/scored
tokens, tokens/second, ETA, and peak VRAM. It also writes atomic checkpoints
and validates configuration identity on `--resume`. The operating standard is
documented in [`docs/RUNNER_OPERATIONS.md`](RUNNER_OPERATIONS.md).

Before restarting Qwen3-1.7B, we will benchmark batch sizes 16 and 32 on a
fixed small token sample, test interruption/resume, choose the largest safe
configuration, and record the measured throughput and peak memory. This is
required before the Pile causal queue is restarted.

### 2026-07-14 — Pile predictive evaluations completed for two models

The Qwen3-0.6B and SmolLM2-135M `tuned-pile-repro-v1` held-out evaluations
completed successfully. Each scored 16,271,875 next-token positions across
28 and 30 layers respectively. These are predictive checks, not causal
intervention results.

| Model | Fit time | Fit peak VRAM | Evaluation | Result artifact |
|---|---:|---:|---|---|
| Qwen3-0.6B | 506.9 s | 2.04 GiB | 16,271,875 positions, complete | `data/experiments/qwen3-0.6b-tuned-pile-repro-v1-eval.json` |
| SmolLM2-135M | 204.7 s | 0.53 GiB | 16,271,875 positions, complete | `data/experiments/smollm2-135m-tuned-pile-repro-v1-eval.json` |

Qwen3-0.6B's tuned/logit KL at first, middle, and final layers was
`4.369 / 2.598 / 0.383` versus `98.127 / 6.774 / 0.423`. SmolLM2's was
`4.675 / 2.918 / 0.392` versus `103.246 / 16.856 / 2.502`. Lower is better.
SmolLM2's best-layer top-1 was 37.4% tuned versus 28.6% logit; Qwen3-0.6B's
final-layer top-1 was 37.1% tuned versus 38.2% logit, so tuned is not uniformly
better on every endpoint metric.

Qwen3-1.7B fitting then completed in 676.5 s with 5.09 GiB peak VRAM. The
canonical 16.4M-token held-out evaluation was started but intentionally stopped
after its runtime estimate reached roughly 6.5 hours; its checkpoint remains
available. A short 200k/1M/2M predictive ladder is now complete, and any causal
suite using it must be labeled provisional rather than canonical.

### 2026-07-14 — Report refresh process documented

The repository now documents the Markdown-to-HTML boundary in
[`docs/REPOSITORY_GUIDE.md`](REPOSITORY_GUIDE.md). Markdown remains the
editable, reviewable source; the browser-facing report is generated with
`make report`. The generated report now reflects the completed Qwen3-0.6B and
SmolLM2 predictive artifacts and distinguishes them from the still-running
Qwen3-1.7B evaluation. Pile predictive artifacts do not change causal plots
until their model-matched causal suites have run.
### 2026-07-15 — Qwen3-1.7B short predictive evaluation ladder

The canonical 16.4M-token held-out Pile evaluation was intentionally stopped
after its runtime estimate reached roughly 6.5 hours. Its checkpoint and event
stream remain preserved, but no final artifact was admitted. We then ran a
three-budget predictive ladder on the same Qwen3-1.7B tuned Pile lens using the
same 3090, BF16 compute, 128-token chunks, and batch size 16:

| Input budget | Scored tokens | Runtime | Peak VRAM | Status |
|---:|---:|---:|---:|---|
| 200,000 | 198,501 | 290.6 s | 12.89 GiB | complete |
| 1,000,000 | 992,251 | about 24 min | 12.89 GiB | complete |
| 2,000,000 | 1,984,375 | 2,892.1 s | 12.89 GiB | complete |

The 200k and 1M estimates were already close: final-layer tuned top-1 was
39.7% and 39.9%, while final-layer logit top-1 was 43.8% and 43.7%. Early and
middle-layer tuned KL remained far below logit KL. These are predictive
precision/sensitivity checks, not causal evidence; the Pile-trained causal
suite is the next guarded step and will be labeled short-2M unless the full
held-out gate is later completed.

### 2026-07-15 — Provisional Qwen3-1.7B Pile causal smoke and suite queue

We exported a model-matched patch basis from the Qwen3-1.7B
`tuned-pile-repro-v1` translator and ran guarded causal smoke tests with BF16
compute, an unquantized model, layers 8/14/20, patching at all positions, and
`tuned`, `logit`, and norm-matched `random` controls. The smoke outputs are
stored separately from validated causal results because the canonical 16.4M
held-out predictive gate is incomplete.

| Task | Items / scored conditions | Tuned | Logit | Random | Interpretation |
|---|---:|---:|---:|---:|---|
| Verbal report, sport | 3 / 3 per method | improved 2; median Δrank 4 | improved 0; median −107 | improved 1; median −91 | diagnostic only; tiny sample |
| Two-hop, 5-item smoke | 5 / 15 per method | improved 5; median Δrank 0 | improved 2; median 0 | improved 4; median 0 | diagnostic only; layer observations are clustered |
| Flexible generalization, 5-item smoke | 5 / 15 per method | improved 6; median −5 | improved 11; median 119 | improved 4; median −12 | logit moved targets more on this tiny sample |

The full provisional runner, `scripts/run_qwen3_1.7b_short_pile_causal.sh`,
then completed all three suites. It writes one independently named JSON result
per causal task and an event stream beside each output, skips already-valid
outputs, performs a free-VRAM preflight, and uses the `short-2m` label
throughout.

| Task | Scored conditions | Skipped | Tuned | Logit | Random |
|---|---:|---:|---|---|---|
| Verbal report | 42 per method | 0 | improved 14; median Δrank −28 | improved 18; median −16.5 | improved 14; median −11 |
| Two-hop reasoning | 204 per method | 22 items | improved 95; median 0; top-5 33 | improved 102; median 0.5; top-5 40 | improved 65; median −2; top-5 29 |
| Flexible generalization | 510 per method | 22 items | improved 224; median 0; top-5 36 | improved 282; median 1; top-5 42 | improved 158; median −3; top-5 21 |

These outputs are diagnostic, not validated reproduction evidence: the
canonical 16.4M-token predictive gate remains unfinished. On this short
track, logit has the largest raw improvement rate in the two-hop and flexible
tasks, while tuned exceeds random; verbal report is inconclusive and tuned's
median movement is not better than the controls. No claim that tuned beats
logit or J-lens should be made from this run.

### 2026-07-15 — Validated small-model Pile causal cells

The existing full Pile predictive artifacts for SmolLM2-135M and Qwen3-0.6B
were followed by their model-matched causal suites. The queue took about
1m17s on the RTX 3090, including basis export and all three tasks per model.
These are now the first validated `tuned-pile-repro-v1` causal rows; they are
separate from the Qwen3-1.7B `short-2m` provisional rows.

| Model | Task | Tuned | Logit | Random |
|---|---|---|---|---|
| Qwen3-0.6B | Verbal report | 30/42; median +517; top-10 10 | 31/42; +266.5; top-10 3 | 13/42; −141; 0 |
| Qwen3-0.6B | Two-hop | 82/204; median −4; top-5 25 | 116/204; +1; 27 | 69/204; −3; 15 |
| Qwen3-0.6B | Flexible | 261/510; median +1; top-5 46 | 309/510; +3; 59 | 148/510; −4; 23 |
| SmolLM2-135M | Verbal report | 26/42; median +227.5; top-10 5 | 34/42; +1446.5; 0 | 18/42; −103.5; 0 |
| SmolLM2-135M | Two-hop | 99/198; median +0.5; top-5 16 | 115/198; +5; 20 | 50/198; −11; 11 |
| SmolLM2-135M | Flexible | 239/510; median 0; top-5 103 | 254/510; 0; 117 | 185/510; −1; 98 |

The Qwen3-0.6B and SmolLM2 coverage cells now read validated predictive plus
causal complete. The next Pile gaps are new fits for Qwen3.5-0.8B and
Qwen3.5-4B; Qwen3.6-27B remains a separate remote-resource decision.

### 2026-07-16 — Daytime Qwen3.5 Pile queue

The Qwen3.5-0.8B Pile lens fit completed locally on the RTX 3090 in 850.2s
with BF16 weights, peak allocated VRAM 2.764 GiB, and peak reserved VRAM
2.891 GiB. Its guarded 16.4M-token held-out predictive evaluation is now
running with a resumable checkpoint and JSONL progress log. A background queue
will validate that artifact before exporting its patch basis and running the
verbal-report, two-hop, and flexible-generalization causal suites sequentially.
Moneypenny notifications are emitted after predictive validation, each causal
suite, and report/PDF regeneration.

After that queue completes, a separate guarded Qwen3.5-4B Pile fit is queued;
it will not compete for the GPU and will notify on completion. We are not
launching a local 27B fit: that remains a Modal/A100 job. The current report
and PDF were regenerated from all completed artifacts before the 0.8B
predictive result exists, so 0.8B causal bars will appear only after that
queue passes its gates.

### 2026-07-19 — Task-construct overlap and next evaluation families

We reviewed the three current causal task protocols—verbal report, two-hop
reasoning, and flexible generalization—and separated the terminology:

* a **task/protocol** is the prompt fixture;
* a **construct** is the capability the fixture is intended to test;
* a **measurement/estimand** is the statistic, such as improved-condition
  rate, target rank change, or excess over random.

The current fixtures all perform a closely related counterfactual operation:
replace an argument-like entity and test whether a downstream answer changes
appropriately. The two-hop fixture adds a nested relation, while flexible
generalization presents the relation as a named function. Their current
J-lens aggregate rates correlate at about `r = 0.76` across six models, which
is suggestive of a shared construct but is not sufficient to establish task
equivalence or independence. The report now documents this as a task-
discriminability question rather than treating the three suites as three
independent capabilities.

Candidate evaluation families that would add more independent coverage:

| Family | New construct | Priority |
|---|---|---|
| Conjunction / non-separability | Whether two facts are jointly represented and neither alone determines the answer | First new causal suite |
| Role/filler binding | Whether entities remain bound to semantic roles such as agent, recipient, and object | First new causal suite |
| Negation and quantifier scope | Whether operators such as `not`, `every`, and `some` bind correctly | High |
| Interference/selectivity | Whether a swap changes the intended concept without moving unrelated distractors | High control |
| Two-swap interaction | Whether simultaneous interventions combine additively, bind, or interfere | High; connects to H-lens |
| Paraphrase/multilingual invariance | Whether the same concept survives surface-form changes | Medium; robustness more than a new reasoning construct |
| Confidence/calibration | Whether intermediate readouts track uncertainty rather than only target rank | Separate measurement family |

The hardest and most scientifically valuable first target is the conjunction /
non-separability suite, strengthened with role-binding and two-swap controls.
It is difficult because the examples must prevent a single fact, lexical cue,
or memorized answer from determining the result. Success should require both
facts, and the interaction of two interventions should be measured explicitly.
This is a better stress test of compositional representation than adding more
direct substitution prompts.

Benchmark/data assessment:

* **ProofWriter** is the strongest immediate source for controlled proof depth,
  conjunctions, distractors, and open-/closed-world variants. We already have
  `15,000` saved candidates (`data/benchmarks/proofwriter-strong-conjunctions.jsonl`,
  `proofwriter-factorial-conjunctions.jsonl`, and related candidate files),
  so it is the practical first source for a large matched causal suite. Its
  generated proof graphs make it possible to certify that both conjuncts are
  necessary. See the [ProofWriter paper](https://aclanthology.org/2021.findings-acl.317/).
* **FOLIO** is the strongest natural-language complement for expert-written
  first-order logic, negation, quantifiers, and entailment/contradiction/
  unknown judgments. We have `1,084` saved conjunction candidates, but its
  smaller size makes it better for a carefully audited natural-language suite
  than the sole source of statistical power. See the [FOLIO paper](https://arxiv.org/abs/2209.00840).
* **CLUTRR** is the best additional benchmark for role/relational binding and
  systematic generalization over held-out relation combinations. It is not
  currently downloaded, so it remains a follow-up acquisition rather than an
  immediate run. See the [CLUTRR paper](https://arxiv.org/abs/1908.06177).

Planned order: build the first model-matched conjunction suite from the
ProofWriter candidates, audit it against the strong-conjunction criteria,
run it across the existing small-to-27B model slate where artifacts permit,
then add a smaller FOLIO natural-language audit and a CLUTRR-style
role-binding suite. This keeps the first new experiment both difficult and
well-powered without pretending that the current two-hop and flexible suites
already cover these constructs.

### 2026-07-19 — SmolLM2 ProofWriter factorial smoke

We added `scripts/eval_proofwriter_factorial_smoke.py` and ran an
observational benchmark smoke on the local `SmolLM2-135M-Instruct` checkpoint
using 100 held-out test rows from
`data/benchmarks/proofwriter-strong-conjunctions.jsonl`. Each row was scored
in four conditions: full theory, remove the target entity's A fact, remove its
B fact, and remove both facts. The script computes exact positive-fragment
labels under ProofWriter-style open-world semantics, scores next-token
evidence for `True`, `False`, and `Unknown`, prints a compact tabulated
summary, writes a JSON artifact, and logs progress. It is intentionally not a
J-lens causal result yet.

| Pilot | Result |
|---|---|
| Items / conditions | 100 / 400 |
| Model | SmolLM2-135M-Instruct, BF16, RTX 3090 |
| Runtime | about 2 seconds for scoring after model load |
| Exact full labels | `True` on 100/100 |
| Exact ablation labels | `Unknown` on 59/100 after A removal, 63/100 after B removal, 73/100 after both removal |
| Model full-condition top-1 | `True` on 98/100 |
| Model ablation top-1 | `True` on 98/100 in each ablation condition |
| Model/exact agreement | 39% after A removal, 35% after B removal, 25% after both removal |

The immediate conclusion is that the exact benchmark transformations are
working, but final answer-token behavior is a poor conjunction-sensitivity
readout for this Smol checkpoint. The symbolic labels often change to
`Unknown`, while the model continues to assign its highest answer-token
probability to `True`. Possible causes include answer-format priors, residual
derivability through other facts/rules, and the gap between logical
uncertainty and the model's three-token answer calibration. Before causal
fitting, the next check is therefore to inspect layer-level logit-lens and
J-lens evidence, with exact labels retained as the benchmark target rather
than treating the final answer token as sufficient.

The smoke artifact is
`data/experiments/proofwriter-smollm2-factorial-smoke.json`; it remains a
diagnostic and is not pooled into the headline causal tables.
