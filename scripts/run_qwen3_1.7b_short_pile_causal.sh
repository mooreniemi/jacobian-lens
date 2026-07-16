#!/usr/bin/env bash
set -euo pipefail

# Provisional causal suite for the Qwen3-1.7B Pile-trained lens.
# The 2M predictive ladder is intentionally not the canonical 16.4M held-out
# gate, so every output is labelled short-2m and must not enter validated plots.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="/home/alex/models/qwen3-1.7b-hf"
BASIS="data/lenses/qwen3-1.7b-tuned-pile-repro-v1-short-2m-patch-basis.pt"
PREFIX="data/experiments"
COMMON=(--model "$MODEL" --lens-local "$BASIS" --layers 8,14,20
  --methods tuned,logit,random --qwen-kernels off --min-free-gib 6)

log() { echo "[qwen3-1.7b-short-2m $(date '+%H:%M:%S')] $*"; }

check_output() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  uv run python - "$path" <<'PY'
import json, math, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
assert d.get("n_used", 0) > 0 or d.get("aggregate"), p
for row in d.get("aggregate", []):
    for value in row[1:]:
        if isinstance(value, (int, float)):
            assert math.isfinite(value), (p, row)
print(f"validated {p.name}")
PY
}

run_once() {
  local label="$1" output="$2"; shift 2
  if check_output "$output"; then
    log "skip existing valid output: $output"
    return
  fi
  log "starting $label; output=$output"
  /usr/bin/time -v uv run python "$@" 2>&1 | tee "${output%.json}.events.jsonl"
  check_output "$output"
  log "completed $label"
}

free_gib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {print $1/1024}')"
log "preflight free VRAM=${free_gib} GiB; requiring >= 6 GiB"
awk "BEGIN { exit !($free_gib >= 6) }" || { log "refusing to start: insufficient free VRAM"; exit 2; }
[[ -s "$BASIS" ]] || { log "missing tuned patch basis: $BASIS"; exit 2; }

run_once verbal "$PREFIX/verbal-report-causal-qwen3-1.7b-tuned-pile-repro-v1-short-2m.json" \
  scripts/eval_verbal_report_causal.py "${COMMON[@]}" \
  --data data/experiments/verbal-report.json \
  --categories country color fruit sport instrument planet tree bird language profession beverage organ city river \
  --patch-positions all \
  --out "$PREFIX/verbal-report-causal-qwen3-1.7b-tuned-pile-repro-v1-short-2m.json"

run_once multihop "$PREFIX/multihop-causal-qwen3-1.7b-tuned-pile-repro-v1-short-2m.json" \
  scripts/eval_multihop_causal.py "${COMMON[@]}" \
  --data data/experiments/probe-swap.json --patch-positions all \
  --out "$PREFIX/multihop-causal-qwen3-1.7b-tuned-pile-repro-v1-short-2m.json"

run_once flexible "$PREFIX/flexible-causal-qwen3-1.7b-tuned-pile-repro-v1-short-2m.json" \
  scripts/eval_flexible_causal.py "${COMMON[@]}" \
  --data data/experiments/flexible-generalization.json \
  --out "$PREFIX/flexible-causal-qwen3-1.7b-tuned-pile-repro-v1-short-2m.json"

log "all provisional Qwen3-1.7B short-2m causal suites complete"
