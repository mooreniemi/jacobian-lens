#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Use the local token file for all subsequent HF requests without printing it.
if [[ -z "${HF_TOKEN:-}" && -s "$HOME/.config/huggingface/modal-token" ]]; then
  export HF_TOKEN="$(<"$HOME/.config/huggingface/modal-token")"
fi
log() { echo "[pile-day-queue $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
notify() { "$HOME/.local/bin/notify-me" "$*" || log "notification failed"; }
trap 'rc=$?; notify "Pile daytime queue failed (exit $rc). Check the pane/log."; exit $rc' ERR

validate_eval() {
  uv run python - "$1" <<'PY'
import json, math, sys
d = json.load(open(sys.argv[1]))
assert d["tokens"] >= 16_000_000, d.get("tokens")
assert d["layers"]
for row in d["layers"]:
    for key in ("tuned_kl", "logit_kl", "tuned_nll", "logit_nll"):
        assert math.isfinite(row[key]), (row["layer"], key, row[key])
early = d["layers"][:max(1, len(d["layers"]) // 2)]
assert any(row["tuned_kl"] < row["logit_kl"] for row in early)
print(f"validated {sys.argv[1]}: {d['tokens']:,} tokens, {len(d['layers'])} layers")
PY
}

notify_eval_table() {
  local path="$1"
  uv run python - "$path" <<'PY' | "$HOME/.local/bin/notify-me" --table || log "results-table notification failed"
import json, sys
d = json.load(open(sys.argv[1]))
rows = d["layers"]
best_tuned = min(rows, key=lambda r: r["tuned_kl"])
best_logit = min(rows, key=lambda r: r["logit_kl"])
last = rows[-1]
print(f"predictive evaluation: {d.get('model', 'model')}")
print(f"tokens\t{d['tokens']:,}\tlayers\t{len(rows)}")
print("metric\tbest layer\tvalue\tmatched value")
print(f"tuned KL\t{best_tuned['layer']}\t{best_tuned['tuned_kl']:.6f}\tlogit KL at same layer {best_tuned['logit_kl']:.6f}")
print(f"logit KL\t{best_logit['layer']}\t{best_logit['logit_kl']:.6f}\ttuned KL at same layer {best_logit['tuned_kl']:.6f}")
print(f"final layer\t{last['layer']}\ttuned KL {last['tuned_kl']:.6f}\tlogit KL {last['logit_kl']:.6f}")
PY
}

wait_for() {
  local path="$1"
  log "waiting for $path"
  while [[ ! -s "$path" ]]; do sleep 60; done
}

run_causal() {
  local slug="$1" model="$2" lens="$3" layers="$4"
  local basis="data/lenses/${slug}-tuned-pile-repro-v1-patch-basis.pt"
  uv run python scripts/export_tuned_patch_basis.py --model "$model" --tuned-dir "$lens" --out "$basis"
  local common=(--model "$model" --lens-local "$basis" --layers "$layers" --methods tuned,logit,random --qwen-kernels on --min-free-gib 6)
  uv run python scripts/eval_verbal_report_causal.py "${common[@]}" --data data/experiments/verbal-report.json --categories country color fruit sport instrument planet tree bird language profession beverage organ city river --patch-positions all --out "data/experiments/verbal-report-causal-${slug}-tuned-pile-repro-v1.json"
  notify "$slug Pile verbal-report causal suite finished."
  uv run python scripts/eval_multihop_causal.py "${common[@]}" --data data/experiments/probe-swap.json --patch-positions all --out "data/experiments/multihop-causal-${slug}-tuned-pile-repro-v1.json"
  notify "$slug Pile two-hop causal suite finished."
  uv run python scripts/eval_flexible_causal.py "${common[@]}" --data data/experiments/flexible-generalization.json --out "data/experiments/flexible-causal-${slug}-tuned-pile-repro-v1.json"
  notify "$slug Pile flexible-generalization causal suite finished."
}

FOUR_EVAL="data/experiments/qwen3.5-4b-tuned-pile-repro-v1-eval.json"
wait_for "$FOUR_EVAL"
validate_eval "$FOUR_EVAL"
notify_eval_table "$FOUR_EVAL"
notify "Qwen3.5-4B Pile predictive gate passed; starting its causal suites."
run_causal qwen3.5-4b Qwen/Qwen3.5-4B data/lenses/qwen3.5-4b-tuned-pile-repro-v1 24,28,30
make report-pdf
notify "Qwen3.5-4B Pile results are plotted and the report/PDF was regenerated."

ONE7_EVAL="data/experiments/qwen3-1.7b-tuned-pile-repro-v1-eval.json"
ONE7_CHECKPOINT="data/experiments/qwen3-1.7b-tuned-pile-repro-v1-eval.checkpoint.pt"
log "resuming Qwen3-1.7B full Pile predictive evaluation"
uv run python scripts/eval_tuned_lens_pile.py --model /home/alex/models/qwen3-1.7b-hf --lens data/lenses/qwen3-1.7b-tuned-pile-repro-v1 --data data/tuned-lens-pile/test.jsonl --out "$ONE7_EVAL" --tokens 16400000 --length 128 --batch-size 16 --dtype bf16 --progress-every-batches 100 --events-out data/experiments/qwen3-1.7b-tuned-pile-repro-v1-eval.events.jsonl --checkpoint-out "$ONE7_CHECKPOINT" --resume
validate_eval "$ONE7_EVAL"
notify_eval_table "$ONE7_EVAL"
notify "Qwen3-1.7B full Pile predictive gate passed; starting its causal suites."
run_causal qwen3-1.7b /home/alex/models/qwen3-1.7b-hf data/lenses/qwen3-1.7b-tuned-pile-repro-v1 19,22,25
make report-pdf
notify "Qwen3-1.7B Pile results are plotted and the report/PDF was regenerated."
log "4B and 1.7B Pile queue complete"
