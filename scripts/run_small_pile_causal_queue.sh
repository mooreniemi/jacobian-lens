#!/usr/bin/env bash
set -euo pipefail

# Fill validated Pile causal rows for models whose full Pile predictive
# artifacts already passed validation. This queue deliberately excludes the
# unfinished 1.7B predictive gate and larger models without Pile fits.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
log() { echo "[small-pile-causal $(date '+%H:%M:%S')] $*"; }

run_model() {
  local slug="$1" model="$2" lens_dir="$3" layers="$4"
  local basis="data/lenses/${slug}-tuned-pile-repro-v1-patch-basis.pt"
  log "preflight ${slug}"
  local free
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {print $1/1024}')"
  awk "BEGIN { exit !($free >= 6) }" || { log "insufficient free VRAM: ${free} GiB"; exit 2; }
  if [[ ! -s "data/experiments/${slug}-tuned-pile-repro-v1-eval.json" ]]; then
    log "missing validated predictive artifact for ${slug}"; exit 2
  fi
  log "exporting tuned basis for ${slug}"
  uv run python scripts/export_tuned_patch_basis.py --model "$model" --tuned-dir "$lens_dir" --out "$basis"
  local common=(--model "$model" --lens-local "$basis" --layers "$layers" --methods tuned,logit,random --qwen-kernels off --min-free-gib 6)
  log "verbal suite ${slug}"
  uv run python scripts/eval_verbal_report_causal.py "${common[@]}" \
    --data data/experiments/verbal-report.json \
    --categories country color fruit sport instrument planet tree bird language profession beverage organ city river \
    --patch-positions all --out "data/experiments/verbal-report-causal-${slug}-tuned-pile-repro-v1.json"
  log "two-hop suite ${slug}"
  uv run python scripts/eval_multihop_causal.py "${common[@]}" \
    --data data/experiments/probe-swap.json --patch-positions all \
    --out "data/experiments/multihop-causal-${slug}-tuned-pile-repro-v1.json"
  log "flexible suite ${slug}"
  uv run python scripts/eval_flexible_causal.py "${common[@]}" \
    --data data/experiments/flexible-generalization.json \
    --out "data/experiments/flexible-causal-${slug}-tuned-pile-repro-v1.json"
  log "completed ${slug}"
}

run_model qwen3-0.6b /home/alex/models/qwen3-0.6b \
  data/lenses/qwen3-0.6b-tuned-pile-repro-v1 19,22,25
run_model smollm2-135m /home/alex/models/smollm2-135m-instruct \
  data/lenses/smollm2-135m-tuned-pile-repro-v1 21,25,28
log "validated small-model Pile causal queue complete"
