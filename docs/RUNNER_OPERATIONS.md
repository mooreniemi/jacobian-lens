# Experiment-runner operations standard

Every long-running experiment must be observable, resumable, and auditable.
This is part of the experiment design, not an optional convenience.

## Required runner features

Long-running fit, evaluation, download, and causal runners should provide:

- a preflight phase that prints model, data, precision, batch size, device,
  free memory, and output paths;
- human-readable progress with completed work, total work, elapsed time,
  throughput, ETA, and peak memory where applicable;
- a flushed JSONL event stream beside the output artifact;
- atomic checkpoints at a bounded interval;
- `--resume` behavior that validates model/data/config identity before reuse;
- a final result containing the run configuration, timing, throughput, and
  peak resource measurements;
- explicit interrupted/failed status when possible;
- no overwrite of a completed artifact without an explicit new output path.

## Current implementation status

| Runner family | Progress | Durable events | Resume/checkpoint | Status |
|---|---|---|---|---|
| Pile tuned-lens evaluation | yes | yes | yes | reference implementation |
| Tuned-lens fitting | yes | yes | checkpointed fit state | existing implementation; audit resume semantics |
| J-lens fitting | checkpoint cadence | partial | checkpoint files | audit before new long run |
| Causal task evaluators | tabulated progress | result JSON | rerun from output boundary | add chunk-level checkpoints if runs grow |
| Modal fit/evaluation | Modal logs + events | yes | remote artifact guard | verify local resume/retry semantics |

## Pile evaluator command pattern

The Pile evaluator writes progress events and a checkpoint by default. To
resume an interrupted run, repeat the same command with `--resume` and the
same output/configuration arguments:

```bash
uv run --extra tuned python scripts/eval_tuned_lens_pile.py \
  --model /home/alex/models/qwen3-1.7b-hf \
  --lens data/lenses/qwen3-1.7b-tuned-pile-repro-v1 \
  --data data/tuned-lens-pile/test.jsonl \
  --out data/experiments/qwen3-1.7b-tuned-pile-repro-v1-eval.json \
  --tokens 16400000 --batch-size 16 --dtype bf16 \
  --progress-every-batches 100 --resume
```

The checkpoint is deleted only after the final result is written. A changed
model, lens, data path, token budget, sequence length, or batch size is
rejected rather than silently producing a mixed result.

## Pre-run checklist

Before submitting or starting a long job:

1. Run a fixed small benchmark for candidate batch sizes.
2. Confirm the output, event, and checkpoint paths are unique.
3. Confirm free GPU memory and expected peak memory.
4. Confirm the command prints a useful ETA within the first progress interval.
5. Confirm interruption and resume on the small benchmark.
6. Record the chosen configuration and the reason in the experiment log.

## Incident record

The Qwen3-1.7B held-out evaluation was started with a conservative
`batch-size 8` without a throughput benchmark or resumable aggregate
checkpoint. It ran for about four hours before being stopped to benchmark a
larger batch size. This was recoverable because no final artifact was claimed,
but the wasted compute and missing ETA are process failures. Future runs must
pass the checklist above.
