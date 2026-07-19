#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
QUEUE_LOG="$ROOT/data/experiments/qwen35-0.8b-pile-queue.log"
OUT="data/lenses/qwen3.5-4b-tuned-pile-repro-v1"
EVENTS="data/lenses/qwen3.5-4b-tuned-pile-repro-v1.events.jsonl"
log() { echo "[qwen35-4b-pile $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
notify() { "$HOME/.local/bin/notify-me" "$*" || log "notification failed"; }
log_mlflow() { uv run python scripts/log_experiment_mlflow.py --result "$1" --experiment "$2" || log "MLflow logging failed for $1"; }
trap 'rc=$?; notify "Qwen3.5-4B Pile fit failed (exit $rc). Check the 4B fit log."; exit $rc' ERR

log "waiting for Qwen3.5-0.8B predictive/causal queue to finish"
while ! grep -q "queue complete" "$QUEUE_LOG" 2>/dev/null; do sleep 60; done
notify "Qwen3.5-0.8B queue completed; starting guarded Qwen3.5-4B Pile fit on the 3090."

uv run python scripts/fit_tuned_lens_local.py \
  --model Qwen/Qwen3.5-4B \
  --out "$OUT" \
  --data-path data/tuned-lens-pile/val.jsonl \
  --dataset-label "EleutherAI/pile_val_test:val.jsonl" \
  --max-length 128 --max-chunks 16384 --steps 4096 --lr 1e-3 \
  --dtype bf16 --min-free-gib 6 \
  --events-out "$EVENTS" \
  > data/lenses/qwen3.5-4b-tuned-pile-repro-v1.log 2>&1
log_mlflow "$OUT/fit_manifest.json" lens-fits

notify "Qwen3.5-4B Pile fit finished. Artifact: $OUT. Predictive evaluation is next; no 27B local fit was launched."
log "fit complete"
