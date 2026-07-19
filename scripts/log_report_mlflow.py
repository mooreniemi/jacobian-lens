#!/usr/bin/env python3
"""Upload the generated research report and analysis outputs to MLflow."""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="sqlite:///mlruns.db")
    parser.add_argument("--experiment", default="research-report")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="research-report-current"):
        report = ROOT / "reports/research"
        for path in sorted((report / "plots").glob("*.png")):
            mlflow.log_artifact(str(path), artifact_path="plots")
        for path in (report / "research-report.pdf", ROOT / "data/analysis/paired_causal_effects.json", ROOT / "data/analysis/causal_protocol_audit.json"):
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="report-data")
        mlflow.set_tag("git_commit", __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
        print("uploaded research plots, PDF, and analysis artifacts")


if __name__ == "__main__":
    main()
