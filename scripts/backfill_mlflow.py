#!/usr/bin/env python3
"""Backfill MLflow from durable causal and Pile-evaluation JSON artifacts."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def experiment_for(path: Path) -> str | None:
    name = path.name
    if "smoke" in name or "short-2m" in name:
        return None
    if "-causal-" in name:
        return "causal-evaluations"
    if "-eval" in name and name.endswith(".json"):
        return "pile-predictive-evaluations"
    if name == "fit_manifest.json" and path.parent.parent.name == "lenses":
        return "lens-fits"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="sqlite:///mlruns.db")
    args = parser.parse_args()
    logger = ROOT / "scripts/log_experiment_mlflow.py"
    paths = list((ROOT / "data/experiments").glob("*.json"))
    paths.extend((ROOT / "data/lenses").glob("*/fit_manifest.json"))
    results = [(p, experiment_for(p)) for p in sorted(paths)]
    results = [(p, e) for p, e in results if e]
    print(f"backfilling {len(results)} result artifacts into MLflow")
    failures = 0
    for path, experiment in results:
        command = [sys.executable, str(logger), "--result", str(path), "--experiment", experiment,
                   "--tracking-uri", args.tracking_uri, "--run-name", path.stem]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if completed.returncode:
            failures += 1
            print(f"FAIL {path.name}: {completed.stderr.strip()}")
        else:
            print(f"OK   {path.name}")
    if failures:
        raise SystemExit(f"{failures} MLflow backfill jobs failed")


if __name__ == "__main__":
    main()
