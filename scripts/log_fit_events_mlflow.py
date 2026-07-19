#!/usr/bin/env python3
"""Import stepwise fit events into a single MLflow lens-fit run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--experiment", default="lens-fits")
    parser.add_argument("--tracking-uri", default="sqlite:///mlruns.db")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name) as run:
        for line in args.events.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") != "step" or "step" not in event:
                continue
            step = int(event["step"])
            metrics = {
                key: float(value)
                for key, value in event.items()
                if key not in {"event", "time", "step", "steps"}
                and isinstance(value, (int, float))
            }
            if metrics:
                mlflow.log_metrics(metrics, step=step)
        mlflow.log_artifact(str(args.events), artifact_path="events")
        print(run.info.run_id)


if __name__ == "__main__":
    main()
