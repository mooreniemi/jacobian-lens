# Repository and fork guide

This checkout began as Anthropic's reference repository and now contains a
local research extension. The source of truth remains reviewable code,
Markdown protocols/logs, compact JSON fixtures/results, and run manifests.

## Markdown and HTML

Markdown is the editable and version-controlled source. The browser-facing
HTML is generated, never hand-edited:

```text
docs/*.md
    ↓  make report
reports/research/docs/*.html
```

The generated `reports/research/` directory is intentionally tracked so a
checkout contains the same readable report that was used for the research
review. GitHub's file browser does not serve HTML as a site; use the local
server command below, or publish this directory with GitHub Pages when a
hosted report is wanted.

Refresh the report from the repository root with:

```bash
make report
```

The report generator reads result JSON and fit manifests, renders every
Markdown document under `docs/`, and rewrites relative Markdown links to the
corresponding HTML pages. This keeps the browser view reproducible after a
fork while preserving clean diffs and mergeable research notes.

## Repository layout

Track these as source or reviewable research records:

- `jlens/`: reusable library and model adapters;
- `scripts/`: fit, evaluation, validation, and report entry points;
- `tests/`: preflight and regression tests;
- `notebooks/` and `walkthrough.ipynb`: exploratory/reproduction notebooks;
- `docs/`: protocols, preregistrations, logs, and provenance;
- `data/experiments/`: compact fixtures and JSON result artifacts;
- `data/evaluations/`, `data/benchmarks/`, and `data/lens-prompts/`:
  declared datasets and prompt inputs;
- `pyproject.toml`, `uv.lock`, and `Makefile`: environment/check commands;
- `docker/`: reproducible local/Modal image recipes.

Long-running runner requirements and the current audit are in
[`RUNNER_OPERATIONS.md`](RUNNER_OPERATIONS.md).

Artifact publication and retrieval instructions are in
[`ARTIFACTS.md`](ARTIFACTS.md).

Generated or machine-local material includes rendered reports, downloaded
corpus shards, model weights, binary lens parameters, MLflow state, caches,
and notebook checkpoints. Run manifests, event JSONL, and compact evaluation
JSON should remain reviewable even when large binary artifacts are stored
elsewhere.

## Large-file policy

Do not put Docker images, downloaded model checkpoints, Pile shards, or fitted
multi-gigabyte lens tensors in Git or Git LFS. Git LFS stores pointer files in
Git but still requires separate LFS storage/bandwidth, and GitHub Free permits
individual LFS files up to 2 GB. Our 3.2 GB 27B lens artifact exceeds that
limit. Store those artifacts in Hugging Face, Modal volumes, object storage, or
release assets, and commit the manifest, hash, source revision, and retrieval
command instead.

Git LFS is technically suitable for a small, intentionally published binary
artifact under 2 GB, but we do not currently need it. Compact prompt datasets,
JSON results, manifests, Markdown, Dockerfiles, and scripts belong in ordinary
Git. Docker images themselves should be rebuilt from the tracked Dockerfile or
published to a container registry; they should never be committed as image
archives.

## Preparing a personal fork

The current `origin` points to the upstream Anthropic repository. Do not push
research changes there. After creating a personal GitHub fork, use a separate
branch and remotes like:

```bash
git remote rename origin upstream
git remote add origin https://github.com/<your-account>/jacobian-lens.git
git fetch upstream
git switch -c research-extension
```

Before the first push, review `git status`, run `make check`, and separate
commits into infrastructure, experiment protocols/logs, and result artifacts.
Keep the upstream history intact.

## Current publishing boundary

The Pile predictive artifacts for Qwen3-0.6B and SmolLM2-135M are recorded
locally; Qwen3-1.7B predictive evaluation is still running. These are not yet
Pile-trained causal results. The H-lens experiment is also design-only.
