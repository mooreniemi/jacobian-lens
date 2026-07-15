"""Download and inventory public logical-reasoning datasets.

Raw datasets remain in the Hugging Face datasets cache. This writes only small
filtered candidate JSONL files and an inventory under data/benchmarks/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/benchmarks")
    parser.add_argument("--max-candidates", type=int, default=5000)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inventory = {
        "sources": {
            "proofwriter": {
                "dataset": "tasksource/proofwriter",
                "paper": "https://aclanthology.org/2021.findings-acl.317/",
                "dataset_card": "https://huggingface.co/datasets/tasksource/proofwriter",
            },
            "folio": {
                "dataset": "tasksource/folio",
                "paper": "https://arxiv.org/abs/2209.00840",
                "dataset_card": "https://huggingface.co/datasets/tasksource/folio",
            },
        },
        "datasets": {},
    }

    proof_candidates = []
    for split in ("train", "validation", "test"):
        ds = load_dataset("tasksource/proofwriter", split=split)
        rows = []
        for row in ds:
            theory = str(row.get("theory", ""))
            proofs = str(row.get("allProofs", ""))
            # ProofWriter marks conjunctive proof nodes with '&'; retain rows
            # for later structural filtering rather than claiming uniqueness.
            if "&" in proofs or " and " in theory.lower():
                rows.append({"split": split, **dict(row)})
                if len(rows) >= args.max_candidates:
                    break
        proof_candidates.extend(rows)
        inventory["datasets"].setdefault("proofwriter", {})[split] = {
            "rows_seen": len(ds), "conjunction_candidates_saved": len(rows)
        }

    folio_candidates = []
    for split in ("train", "validation"):
        ds = load_dataset("tasksource/folio", split=split)
        rows = []
        for row in ds:
            premises = str(row.get("premises", ""))
            premises_fol = str(row.get("premises-FOL", ""))
            if "∧" in premises_fol or " and " in premises.lower():
                rows.append({"split": split, **dict(row)})
                if len(rows) >= args.max_candidates:
                    break
        folio_candidates.extend(rows)
        inventory["datasets"].setdefault("folio", {})[split] = {
            "rows_seen": len(ds), "conjunction_candidates_saved": len(rows)
        }

    write_jsonl(out / "proofwriter-conjunction-candidates.jsonl", proof_candidates)
    write_jsonl(out / "folio-conjunction-candidates.jsonl", folio_candidates)
    inventory["outputs"] = {
        "proofwriter_candidates": len(proof_candidates),
        "folio_candidates": len(folio_candidates),
    }
    (out / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
