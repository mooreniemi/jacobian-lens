"""Log a JSON experiment result to a local MLflow tracking store.

Usage:
  uv run --extra tracking python scripts/log_experiment_mlflow.py \
    --result data/experiments/multihop-causal-full90.json \
    --experiment anthropic-causal-reproduction

The default SQLite database and artifact directory are local to the repo and
are intentionally not required for the experiment runners themselves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--experiment", default="anthropic-causal-reproduction")
    ap.add_argument("--tracking-uri", default="sqlite:///mlruns.db")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    try:
        import mlflow
    except ImportError as exc:
        raise SystemExit("Install the optional tracker with: uv sync --extra tracking") from exc

    result_path = Path(args.result).resolve()
    result = json.loads(result_path.read_text())
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    name = args.run_name or result_path.stem
    with mlflow.start_run(run_name=name) as run:
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=result_path.parent.parent.parent, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = "unknown"
        try:
            import importlib.metadata as metadata
            package_versions = {
                "torch": metadata.version("torch"),
                "transformers": metadata.version("transformers"),
                "mlflow": metadata.version("mlflow"),
            }
        except Exception:
            package_versions = {}
        mlflow.set_tags({
            "git_commit": git_commit,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "command": " ".join(sys.argv),
            **{f"package_{k}": v for k, v in package_versions.items()},
        })
        mlflow.log_param("result_sha256", hashlib.sha256(result_path.read_bytes()).hexdigest())
        params = {
            key: value
            for key, value in {
                "model": result.get("model"),
                "lens_file": result.get("lens_file"),
                "kernel_mode": result.get("kernel_mode"),
                "patch_positions": result.get("patch_positions"),
                "layers": result.get("layers"),
                "methods": result.get("methods"),
                "n_input": result.get("n_input"),
                "n_used": result.get("n_used"),
                "n_skipped": len(result.get("skipped", [])),
            }.items()
            if value is not None
        }
        mlflow.log_params({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in params.items()})
        for row in result.get("aggregate", []):
            # Multihop rows have six fields; verbal-report rows may also have
            # a seventh top-10 field. Log the shared metrics and any extras.
            if len(row) == 7:
                method, n, improved, median_delta, mean_delta, top1, *extras = row
            else:
                method, n, improved, median_delta, top1, *extras = row
                mean_delta = None
            prefix = str(method)
            metrics = {
                f"{prefix}/n_conditions": n,
                f"{prefix}/improved": improved,
                f"{prefix}/median_delta_rank": median_delta,
                f"{prefix}/top1": top1,
            }
            if mean_delta is not None:
                metrics[f"{prefix}/mean_delta_rank"] = mean_delta
            if extras:
                metrics[f"{prefix}/top5_or_top10"] = extras[0]
            if len(extras) > 1:
                metrics[f"{prefix}/top10"] = extras[1]
            mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(result_path), artifact_path="results")
        print(json.dumps({"run_id": run.info.run_id, "experiment": args.experiment, "run_name": name}, indent=2))


if __name__ == "__main__":
    main()
