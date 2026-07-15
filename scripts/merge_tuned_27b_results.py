"""Merge a validated tuned-only causal run into a canonical result JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = json.loads(Path(args.base).read_text())
    tuned = json.loads(Path(args.tuned).read_text())
    if "qwen3.6-27b" not in str(base.get("model")).lower() or "qwen3.6-27b" not in str(tuned.get("model")).lower():
        raise SystemExit("base and tuned artifacts are not Qwen3.6-27B results")
    if tuned.get("methods") != ["tuned"]:
        raise SystemExit(f"expected tuned-only artifact, got {tuned.get('methods')}")
    if len(base["items"]) != len(tuned["items"]):
        raise SystemExit("base and tuned item counts differ")
    for left, right in zip(base["items"], tuned["items"], strict=True):
        if left.get("layers", {}).keys() != right.get("layers", {}).keys():
            raise SystemExit("base and tuned layer keys differ")
        for layer in left["layers"]:
            if set(right["layers"][layer]) != {"tuned"}:
                raise SystemExit("tuned artifact contains unexpected methods")
            left["layers"][layer]["tuned"] = right["layers"][layer]["tuned"]
    # Make the merge idempotent so a validation rerun cannot duplicate a row.
    base["methods"] = [method for method in base["methods"] if method != "tuned"] + ["tuned"]
    base["aggregate"] = [row for row in base["aggregate"] if row[0] != "tuned"]
    base["aggregate"].extend(row for row in tuned["aggregate"] if row[0] == "tuned")
    base["model"] = "Qwen/Qwen3.6-27B"
    base["tuned_lens_local"] = "data/lenses/qwen3.6-27b-tuned-wikitext-nf4"
    base["tuned_patch_basis"] = "data/lenses/qwen3.6-27b-tuned-wikitext-basis.pt"
    base["tuned_eval_artifact"] = str(Path(args.tuned).resolve())
    base["tuned_eval_manifest"] = {
        "model": tuned["model"],
        "layers": tuned["layers"],
        "n_used": tuned.get("n_used", len(tuned["items"])),
        "skipped": len(tuned.get("skipped", [])),
        "aggregate": tuned["aggregate"],
    }
    Path(args.out).write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
