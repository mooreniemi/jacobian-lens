#!/usr/bin/env python3
"""Audit causal result artifacts for model/method/protocol mismatches.

This is deliberately metadata- and schema-focused: it does not run a model.
It checks that methods inside each saved causal artifact saw the same items,
layers, patch setting, and tokenization-validity decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/experiments"


def item_signature(data: dict) -> tuple[str, ...]:
    return tuple(item.get("name", f"item-{i}") for i, item in enumerate(data.get("items", [])))


def audit(path: Path) -> dict:
    data = json.loads(path.read_text())
    errors: list[str] = []
    warnings: list[str] = []
    methods = list(data.get("methods", []))
    declared_layers = tuple(str(x) for x in data.get("layers", []))
    items = data.get("items", [])
    if not data.get("model"):
        errors.append("missing model")
    if not methods:
        errors.append("missing methods")
    if not declared_layers:
        errors.append("missing layers")
    if not items:
        warnings.append("no scored items")

    schema_kinds: set[str] = set()
    for index, item in enumerate(items):
        item_layers = tuple(str(x) for x in item.get("layers", {}).keys())
        if item_layers != declared_layers:
            errors.append(f"item {index} layer set {item_layers} != declared {declared_layers}")
        for layer, layer_data in item.get("layers", {}).items():
            present = set(layer_data)
            missing = set(methods) - present
            extra = present - set(methods)
            if missing:
                errors.append(f"item {index} layer {layer} missing methods {sorted(missing)}")
            if extra:
                warnings.append(f"item {index} layer {layer} has undeclared methods {sorted(extra)}")
            for method in methods:
                record = layer_data.get(method, {})
                if "target_rank_after" in record:
                    schema_kinds.add("verbal")
                elif "swap_answer_rank_after" in record:
                    schema_kinds.add("multihop")
                elif "target_answer_rank_after" in record:
                    schema_kinds.add("flexible")
                else:
                    errors.append(f"item {index} layer {layer} method {method} has no recognized rank field")
    if len(schema_kinds) > 1:
        errors.append(f"mixed task schemas: {sorted(schema_kinds)}")
    if data.get("n_used") is not None and data["n_used"] != len(items):
        errors.append(f"n_used={data['n_used']} but items={len(items)}")
    if data.get("aggregate"):
        aggregate_methods = {row[0] for row in data["aggregate"]}
        if aggregate_methods != set(methods):
            errors.append(f"aggregate methods {sorted(aggregate_methods)} != methods {sorted(methods)}")
    return {
        "file": str(path.relative_to(ROOT)),
        "model": data.get("model"),
        "methods": methods,
        "layers": list(declared_layers),
        "n_items": len(items),
        "n_input": data.get("n_input"),
        "precision": data.get("precision"),
        "kernel_mode": data.get("kernel_mode"),
        "patch_positions": data.get("patch_positions"),
        "task_schema": sorted(schema_kinds),
        "errors": errors,
        "warnings": warnings,
        "status": "FAIL" if errors else ("WARN" if warnings else "PASS"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*-causal-*.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/analysis/causal_protocol_audit.json")
    args = parser.parse_args()
    paths = sorted(RESULTS.glob(args.glob))
    records = []
    for path in paths:
        if any(tag in path.name for tag in ("smoke", "short-2m")):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not data.get("items") or not data.get("aggregate"):
            continue
        records.append(audit(path))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2) + "\n")
    counts = {status: sum(row["status"] == status for row in records) for status in ("PASS", "WARN", "FAIL")}
    print(f"audited {len(records)} artifacts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    for row in records:
        if row["status"] != "PASS":
            print(f"{row['status']} {row['file']}: {'; '.join(row['errors'] + row['warnings'])}")
    if counts["FAIL"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
