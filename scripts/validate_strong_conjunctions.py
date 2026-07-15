"""Filter ProofWriter conjunction candidates for a factorial intersection design.

Required structure:
  * at least two entities have A;
  * at least two entities have B;
  * exactly one entity has both A and B;
  * no entity has the conclusion C as an explicit positive fact;
  * no detected singleton rule maps A or B directly to C.

This is a static screen. The survivors still need tokenizer and model
behavioral validation before entering a causal experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_proofwriter_strong_conjunctions import parse_theory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/benchmarks/proofwriter-strong-conjunctions.jsonl")
    ap.add_argument("--output", default="data/benchmarks/proofwriter-factorial-conjunctions.jsonl")
    args = ap.parse_args()

    kept = []
    rejected = {}
    seen = set()
    for line in Path(args.input).open():
        row = json.loads(line)
        facts, rules = parse_theory(row["theory"])
        a, b, c = row["conjunct_a"], row["conjunct_b"], row["conclusion"]
        a_entities = {e for e, ps in facts.items() if a in ps}
        b_entities = {e for e, ps in facts.items() if b in ps}
        both = a_entities & b_entities
        c_entities = {e for e, ps in facts.items() if c in ps}
        reasons = []
        if len(a_entities) < 2:
            reasons.append("A_not_ambiguous")
        if len(b_entities) < 2:
            reasons.append("B_not_ambiguous")
        if both != {row["entity"]}:
            reasons.append("intersection_not_unique")
        if c_entities:
            reasons.append("C_explicit_somewhere")
        if reasons:
            for reason in reasons:
                rejected[reason] = rejected.get(reason, 0) + 1
            continue
        key = (row["theory"], row["entity"], a, b, c)
        if key in seen:
            continue
        seen.add(key)
        row["factorial_validation"] = {
            "A_entities": sorted(a_entities),
            "B_entities": sorted(b_entities),
            "A_and_B_entities": sorted(both),
            "C_entities": sorted(c_entities),
            "status": "static_pass",
        }
        kept.append(row)

    Path(args.output).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept))
    print(json.dumps({"output": args.output, "kept": len(kept), "rejected": rejected}, indent=2))


if __name__ == "__main__":
    main()
