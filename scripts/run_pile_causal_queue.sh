#!/usr/bin/env bash
set -euo pipefail

# This queue intentionally starts only after the full held-out Pile evaluations
# exist. A predictive smoke test is not sufficient evidence for causal use.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

log() { echo "[pile-causal-queue $(date '+%H:%M:%S')] $*"; }

wait_for_file() {
    local path="$1"
    log "waiting for completed held-out artifact: $path"
    while [[ ! -s "$path" ]]; do sleep 60; done
}

validate_eval() {
    local path="$1"
    python - "$path" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
assert data["tokens"] >= 16_000_000, (path, data.get("tokens"))
layers = data["layers"]
assert layers, path
for row in layers:
    for key in ("tuned_kl", "logit_kl", "tuned_nll", "logit_nll"):
        assert math.isfinite(row[key]), (path, row["layer"], key, row[key])
early = layers[: max(1, len(layers) // 2)]
assert any(row["tuned_kl"] < row["logit_kl"] for row in early), path
print(f"validated {path}: tokens={data['tokens']:,} layers={len(layers)}")
PY
}

run_model() {
    local slug="$1" model="$2" lens_dir="$3" jlens_path="$4" layers="$5"
    local basis="data/lenses/${slug}-tuned-pile-repro-v1-patch-basis.pt"
    log "exporting tuned Pile basis for $slug"
    uv run python scripts/export_tuned_patch_basis.py \
        --model "$model" --tuned-dir "$lens_dir" --out "$basis"

    local common=(
        --model "$model" --lens-local "$basis" --layers "$layers"
        --methods tuned,logit,random --qwen-kernels off --min-free-gib 6
    )
    log "running verbal causal suite for $slug"
    uv run python scripts/eval_verbal_report_causal.py "${common[@]}" \
        --data data/experiments/verbal-report.json \
        --categories country color fruit sport instrument planet tree bird language profession beverage organ city river \
        --patch-positions all \
        --out "data/experiments/verbal-report-causal-${slug}-tuned-pile-repro-v1.json"
    log "running two-hop causal suite for $slug"
    uv run python scripts/eval_multihop_causal.py "${common[@]}" \
        --data data/experiments/probe-swap.json --patch-positions all \
        --out "data/experiments/multihop-causal-${slug}-tuned-pile-repro-v1.json"
    log "running flexible-generalization causal suite for $slug"
    uv run python scripts/eval_flexible_causal.py "${common[@]}" \
        --data data/experiments/flexible-generalization.json \
        --out "data/experiments/flexible-causal-${slug}-tuned-pile-repro-v1.json"
    log "completed causal Pile suites for $slug"
}

wait_for_file data/experiments/qwen3-0.6b-tuned-pile-repro-v1-eval.json
wait_for_file data/experiments/smollm2-135m-tuned-pile-repro-v1-eval.json
wait_for_file data/experiments/qwen3-1.7b-tuned-pile-repro-v1-eval.json

log "validating all held-out predictive gates before causal work"
validate_eval data/experiments/qwen3-0.6b-tuned-pile-repro-v1-eval.json
validate_eval data/experiments/smollm2-135m-tuned-pile-repro-v1-eval.json
validate_eval data/experiments/qwen3-1.7b-tuned-pile-repro-v1-eval.json

# The first two models have existing model-matched J-lens artifacts in this
# checkout. The Pile causal rows deliberately use the new tuned basis alongside
# logit/random; J-lens remains the separately recorded comparison row.
run_model qwen3-0.6b /home/alex/models/qwen3-0.6b \
    data/lenses/qwen3-0.6b-tuned-pile-repro-v1 \
    data/lenses/qwen3-0.6b-fit-204-lens.pt 19,22,25
run_model smollm2-135m /home/alex/models/smollm2-135m-instruct \
    data/lenses/smollm2-135m-tuned-pile-repro-v1 \
    data/lenses/smollm2-135m-fit-204-lens.pt 21,25,28
run_model qwen3-1.7b /home/alex/models/qwen3-1.7b-hf \
    data/lenses/qwen3-1.7b-tuned-pile-repro-v1 \
    "" 19,22,25

log "all validated local Pile causal suites complete"
