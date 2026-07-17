#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential post-evaluation queue for the Qwen3.5-0.8B Pile lens.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="Qwen/Qwen3.5-0.8B"
SLUG="qwen3.5-0.8b"
LENS="data/lenses/qwen3.5-0.8b-tuned-pile-repro-v1"
EVAL="data/experiments/qwen3.5-0.8b-tuned-pile-repro-v1-eval.json"
BASIS="data/lenses/qwen3.5-0.8b-tuned-pile-repro-v1-patch-basis.pt"
LAYERS="16,19,21"
log() { echo "[qwen35-0.8b-pile $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
notify() { "$HOME/.local/bin/notify-me" "$*" || log "notification failed"; }
trap 'rc=$?; notify "Qwen3.5-0.8B Pile queue failed (exit $rc). Check the queue log."; exit $rc' ERR

validate_eval() {
  uv run python - "$EVAL" <<'PY'
import json, math, sys
d = json.load(open(sys.argv[1]))
assert d["tokens"] >= 16_000_000, d.get("tokens")
assert d["layers"]
for row in d["layers"]:
    for k in ("tuned_kl", "logit_kl", "tuned_nll", "logit_nll"):
        assert math.isfinite(row[k]), (row["layer"], k, row[k])
early = d["layers"][:max(1, len(d["layers"]) // 2)]
assert any(r["tuned_kl"] < r["logit_kl"] for r in early)
print(f"validated predictive eval: {d['tokens']:,} tokens, {len(d['layers'])} layers")
PY
}

while [[ ! -s "$EVAL" ]]; do sleep 60; done
validate_eval
notify "Qwen3.5-0.8B Pile predictive evaluation finished and passed gates (16.4M tokens). Starting causal suites."

if [[ ! -s "$BASIS" ]]; then
  log "exporting tuned patch basis"
  uv run python scripts/export_tuned_patch_basis.py --model "$MODEL" --tuned-dir "$LENS" --out "$BASIS"
fi
COMMON=(--model "$MODEL" --lens-local "$BASIS" --layers "$LAYERS" --methods tuned,logit,random --qwen-kernels off --min-free-gib 6)

log "starting verbal-report causal suite"
uv run python scripts/eval_verbal_report_causal.py "${COMMON[@]}" \
  --data data/experiments/verbal-report.json \
  --categories country color fruit sport instrument planet tree bird language profession beverage organ city river \
  --patch-positions all \
  --out data/experiments/verbal-report-causal-${SLUG}-tuned-pile-repro-v1.json
notify "Qwen3.5-0.8B Pile verbal-report causal suite finished."

log "starting two-hop causal suite"
uv run python scripts/eval_multihop_causal.py "${COMMON[@]}" \
  --data data/experiments/probe-swap.json --patch-positions all \
  --out data/experiments/multihop-causal-${SLUG}-tuned-pile-repro-v1.json
notify "Qwen3.5-0.8B Pile two-hop causal suite finished."

log "starting flexible-generalization causal suite"
uv run python scripts/eval_flexible_causal.py "${COMMON[@]}" \
  --data data/experiments/flexible-generalization.json \
  --out data/experiments/flexible-causal-${SLUG}-tuned-pile-repro-v1.json
notify "Qwen3.5-0.8B Pile flexible-generalization causal suite finished."

notify "Qwen3.5-0.8B Pile causal queue finished. Regenerating research report and PDF now."
make report-pdf
notify "Qwen3.5-0.8B Pile results are in the research report and PDF."
log "queue complete"
