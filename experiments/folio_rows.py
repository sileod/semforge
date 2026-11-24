"""Corrected FOLIO validation (semforge.domains.folio; "Fixing FOLIO and MALLS", 2606.02837) as test rows in
the LLM experiments' format: the story's English as premises, the conclusion's English as hypothesis, the
corrected label as gold (True/False/Uncertain -> entailment/contradiction/neutral).

FOLIO has no calculus of ours, so there is no certified step count: `mandatory` is None. For E/C rows that
z3 decides, `mandatory_premises` is the number of premises in every minimal support (the semantic audit's
measure, docs/audit.md), and `z3_agrees` says whether z3 gives the corrected label from the corrected
formulas.

  python -m experiments.folio_rows --out results/llm_mix/data/test_folio.jsonl

`--dev N` instead writes N rows of FOLIO's original train split (tasksource/folio; stories disjoint from
validation, labels uncorrected) for selecting training distributions, e.g. to val_folio.jsonl.
"""
from __future__ import annotations

import argparse
import collections
import json

from experiments.llm_common import FORMAT
from semforge.domains import folio


def mandatory_premises(prem, goal, gold):
    from semforge.core.lattice import Oracle, enumerate_lattice
    try:
        lab = folio.label(prem, goal)
    except folio.Undecided:
        return None, None
    if gold == "neutral" or lab != gold:
        return None, lab == gold
    g = goal if gold == "entailment" else ("not", goal)
    sups, _, complete = enumerate_lattice(prem, Oracle(lambda s: folio.entails(list(s), g)), limit=64)
    if not complete or not sups:
        return None, True
    return len(set.intersection(*map(set, sups))), True


def dev_rows(n, seed=0):
    import random
    from datasets import load_dataset
    gold = {"True": "entailment", "False": "contradiction", "Uncertain": "neutral"}
    rows = [{"gold": gold[r["label"]], "mandatory": None, "id": r["example_id"],
             "text": FORMAT.format(premises=r["premises"].strip(), hypothesis=r["conclusion"].strip())}
            for r in load_dataset("tasksource/folio", split="train")]
    return random.Random(seed).sample(rows, min(n, len(rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/llm_mix/data/test_folio.jsonl")
    ap.add_argument("--dev", type=int, default=0, help="write this many FOLIO train rows instead (see above)")
    args = ap.parse_args()
    if args.dev:
        rows = dev_rows(args.dev)
        with open(args.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        return print(len(rows), dict(collections.Counter(r["gold"] for r in rows)))
    raw = {json.loads(line)["id"]: json.loads(line) for line in open(folio.PATH)}
    rows = []
    for id_, prem, goal, gold, r in folio.load():
        story = raw["story_" + id_.split("_")[1]]
        m, agrees = mandatory_premises(prem, goal, gold) if prem is not None else (None, None)
        rows.append({"gold": gold, "mandatory": None, "mandatory_premises": m, "z3_agrees": agrees, "id": id_,
                     "text": FORMAT.format(premises=story["NL_sentence"].strip(), hypothesis=r["NL_sentence"].strip())})
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(len(rows), dict(collections.Counter(r["gold"] for r in rows)),
          "z3 agrees", sum(r["z3_agrees"] is True for r in rows),
          "with mandatory premises", sum(r["mandatory_premises"] is not None for r in rows))


if __name__ == "__main__":
    main()
