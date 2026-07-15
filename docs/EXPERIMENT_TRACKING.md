# Local experiment tracking

We use MLflow locally for run metadata and result artifacts. It is free,
works without an account, and keeps the experiment database on this machine.
The JSON files remain the canonical, reviewable outputs; MLflow is an index and
comparison view rather than the source of truth.

Install the optional dependency:

```bash
uv sync --extra tracking
```

Log a completed run:

```bash
uv run --extra tracking python scripts/log_experiment_mlflow.py \
  --result data/experiments/multihop-causal-methods-full90.json \
  --experiment anthropic-causal-reproduction
```

This creates `mlflow.db` and local artifact storage. Start the UI when useful:

```bash
uv run --extra tracking mlflow ui --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 --port 5000
```

The UI can then be reached through the existing SSH port-forwarding workflow.
Do not commit `mlflow.db` or MLflow artifact directories; they are local
runtime state.

Tuned-lens remains a separate model-specific fitting treatment. The optional
`tuned` dependency provides the reference affine-translator implementation;
there are still no compatible released Qwen3.5/Qwen3/Qwen3.6 tuned-lens
checkpoints. The existing locally fitted Wikitext artifacts are now labelled
`tuned-wiki-small-v0`; they are pilot/sanity-check artifacts, not the strong
reproduction. The staged Pile validation/test track is specified in
[`TUNED_LENS_REPRODUCTION.md`](TUNED_LENS_REPRODUCTION.md).
