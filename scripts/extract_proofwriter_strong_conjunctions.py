"""Extract structurally non-redundant conjunction examples from ProofWriter.

This is deliberately conservative: it only keeps simple unary rules of the
form A(x) & B(x) -> C(x), where the story explicitly gives A and B for an
entity, does not give C directly, and has no matching A->C or B->C rule.
The result is a screening corpus for manual/model validation, not a claim
that the natural-language story is semantically perfect.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

FACT = re.compile(r"^(?:The )?([A-Za-z][A-Za-z0-9 -]*?) is (not )?([A-Za-z][A-Za-z0-9 -]*)\.$")
RULE_IF = re.compile(
    r"^If (?:someone|something) is ([A-Za-z][A-Za-z0-9 -]*) and (?:they|it) are? ([A-Za-z][A-Za-z0-9 -]*) then (?:they|it) are? ([A-Za-z][A-Za-z0-9 -]*)\.$",
    re.I,
)
RULE_IF_GENERIC = re.compile(
    r"^If (?:someone|something) is (not )?([A-Za-z][A-Za-z0-9 -]*) and (not )?([A-Za-z][A-Za-z0-9 -]*) then (?:they|it) (?:is|are) (not )?([A-Za-z][A-Za-z0-9 -]*)\.$",
    re.I,
)
RULE_ALL = re.compile(
    r"^(?:All|Every) ([A-Za-z][A-Za-z0-9 -]*), ([A-Za-z][A-Za-z0-9 -]*) (?:people|things|animals) are ([A-Za-z][A-Za-z0-9 -]*)\.$",
    re.I,
)
RULE_COMMA = re.compile(
    r"^([A-Za-z][A-Za-z0-9 -]*), ([A-Za-z][A-Za-z0-9 -]*) (?:people|things|animals) are ([A-Za-z][A-Za-z0-9 -]*)\.$",
    re.I,
)
RULE_IF_SINGLE = re.compile(
    r"^If (?:someone|something) is ([A-Za-z][A-Za-z0-9 -]*) then (?:they|it) are ([A-Za-z][A-Za-z0-9 -]*)\.$", re.I)
RULE_CLASS_SINGLE = re.compile(
    r"^(?:All|Every) ([A-Za-z][A-Za-z0-9 -]*) (?:people|things|animals) are ([A-Za-z][A-Za-z0-9 -]*)\.$", re.I)
RULE_THING_SINGLE = re.compile(
    r"^([A-Za-z][A-Za-z0-9 -]*) (?:people|things|animals) are ([A-Za-z][A-Za-z0-9 -]*)\.$", re.I)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def parse_theory(theory: str):
    facts = defaultdict(set)
    rules = []
    # ProofWriter stores several sentences on one line in many records.
    clauses = re.split(r"(?<=\.)\s+", theory.replace("\n", " "))
    for raw in clauses:
        line = raw.strip()
        m = FACT.match(line)
        if m and not line.lower().startswith(("if ", "all ", "every ")):
            entity, neg, pred = m.groups()
            if not neg:
                facts[norm(entity)].add(norm(pred))
            continue
        m = RULE_IF.match(line)
        if m:
            rules.append(tuple(norm(x) for x in m.groups()))
            continue
        m = RULE_IF_GENERIC.match(line)
        if m:
            neg_a, a, neg_b, b, neg_c, c = m.groups()
            if not any((neg_a, neg_b, neg_c)):
                rules.append((norm(a), norm(b), norm(c)))
            continue
        m = RULE_ALL.match(line)
        if m:
            rules.append(tuple(norm(x) for x in m.groups()))
            continue
        m = RULE_COMMA.match(line)
        if m:
            rules.append(tuple(norm(x) for x in m.groups()))
            continue
        for single_re in (RULE_IF_SINGLE, RULE_CLASS_SINGLE, RULE_THING_SINGLE):
            m = single_re.match(line)
            if m:
                a, c = (norm(x) for x in m.groups())
                rules.append((a, None, c))
                break
    return facts, rules


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/benchmarks/proofwriter-conjunction-candidates.jsonl")
    ap.add_argument("--output", default="data/benchmarks/proofwriter-strong-conjunctions.jsonl")
    ap.add_argument("--max-output", type=int, default=500)
    args = ap.parse_args()

    grouped = {}
    with Path(args.input).open() as f:
        for line in f:
            row = json.loads(line)
            grouped.setdefault(row.get("theory", ""), row)

    out = []
    seen = set()
    for theory, row in grouped.items():
        facts, rules = parse_theory(theory)
        for a, b, c in rules:
            if b is None:
                continue
            if a == b or c in (a, b):
                continue
            for entity, predicates in facts.items():
                if not {a, b}.issubset(predicates) or c in predicates:
                    continue
                if any((x == c and y in (a, b)) or (y is None and x in (a, b) and z == c)
                       for x, y, z in rules):
                    continue
                key = (theory, entity, a, b, c)
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "source_split": row.get("split"),
                    "source_id": row.get("id", row.get("example_id")),
                    "entity": entity,
                    "conjunct_a": a,
                    "conjunct_b": b,
                    "conclusion": c,
                    "question": f"{entity} is {c}.",
                    "theory": theory,
                    "source_answer": row.get("answer"),
                    "criterion": "A_and_B_explicit; C_not_explicit; no_A_or_B_singleton_rule_to_C",
                })
                if len(out) >= args.max_output:
                    break
            if len(out) >= args.max_output:
                break
        if len(out) >= args.max_output:
            break

    Path(args.output).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(json.dumps({"output": args.output, "n": len(out)}, indent=2))


if __name__ == "__main__":
    main()
