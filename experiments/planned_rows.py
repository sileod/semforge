"""The audit's finding as a training contrast: rows a generator plans as deep, split by certified depth.

FLD×2's depth-5 rows make two mandatory steps on average (benchmarks/audit_fld_proofs.py --depth 5): a
generator that trusts its planned depth ships shallow rows labelled deep. Here both arms are what a
forward generator would call depth 5-9 data, matched per label and planned depth; they differ only in
what certification says about each row.

  planshal  forward monadic rows planned at depth 5-9 whose certified depth is at most 3
  plandeep  the same planned depths, certified at least planned depth - 1

Neutrals come from the same generator (one mandatory premise dropped) and are split the same way by the
depth of their counterfactual proof, so neutral depth is no cue to the label within an arm.

  python -m experiments.planned_rows --out results/llm_mix/data_planned

--polarity balances the polarity cue: within each label and planned depth, half the rows of each arm have a
negated hypothesis (cue_reliability.negated), so negation says nothing about the label in either arm. Arms
are then planshal_pm / plandeep_pm.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil

from experiments.external_rows import write
from experiments.cue_reliability import negated
from experiments.llm_common import LABELS, generate

PLANNED = range(5, 10)


class PlannedDeep:
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, r):
        return r["gold"] in self.labels and r.get("planned") in PLANNED and r.get("mandatory") is not None


def arm_of(r):
    if r["mandatory"] <= 3:
        return "planshal"
    if r["mandatory"] >= r["planned"] - 1:
        return "plandeep"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/llm_mix/data_planned")
    ap.add_argument("--n", type=int, default=3000, help="rows per arm, a third per label")
    ap.add_argument("--pool", type=int, default=60000, help="E/C rows generated (neutrals: half as many)")
    ap.add_argument("--tests-from", default="results/llm_mix/data_packmix")
    ap.add_argument("--polarity", action="store_true", help="half negated hypotheses per label and depth")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = generate("monadic", "forward", 70, args.pool, PlannedDeep(("entailment", "contradiction")), jobs=32)
    rows += generate("monadic", "forward_neutral", 71, args.pool // 2, PlannedDeep(("neutral",)), jobs=32)
    by = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: r["text"]):
        a = arm_of(r)
        if a:
            by[a, r["gold"], (r["planned"], negated(r["text"])) if args.polarity else r["planned"]].append(r)
    rng = random.Random(0)
    for v in by.values():
        rng.shuffle(v)
    per_label = args.n // 3
    arms = {"planshal": [], "plandeep": []}
    for lab in LABELS:
        if args.polarity:  # same planned-depth profile in both halves; each half is per_label / 2
            halves = [(per_label // 2 + per_label % 2, True), (per_label // 2, False)]
            # depth profile: matched counts of the scarcer polarity half, so both halves can follow it
            prof = {p: min(len(by[a, lab, (p, n)]) for a in arms for n in (True, False)) for p in PLANNED}
        else:
            halves = [(per_label, None)]
            prof = {p: min(len(by[a, lab, p]) for a in arms) for p in PLANNED}
        for want, neg in halves:
            key = (lambda p: (p, neg)) if args.polarity else (lambda p: p)
            total = sum(prof.values())
            assert total >= want, f"{lab} neg={neg}: only {total} matched rows; raise --pool"
            take = {p: prof[p] * want // total for p in PLANNED}
            for p in sorted(PLANNED, key=lambda p: -prof[p])[:want - sum(take.values())]:
                take[p] += 1
            name = (lambda a: a + "_pm") if args.polarity else (lambda a: a)
            for a in arms:
                arms[a] += [{**r, "cell": name(a)} for p in PLANNED for r in by[a, lab, key(p)][:take[p]]]
    stats = {}
    for a, rs in arms.items():
        rng.shuffle(rs)
        a = a + "_pm" if args.polarity else a
        write(f"{args.out}/train_{a}.jsonl", rs)
        mean = lambda k: round(sum(r[k] for r in rs) / len(rs), 2)
        stats[a] = {"rows": len(rs), "labels": collections.Counter(r["gold"] for r in rs),
                    "planned": collections.Counter(r["planned"] for r in rs), "mean_planned": mean("planned"),
                    "mean_mandatory": mean("mandatory"), "mean_premises": mean("premises"),
                    "mean_chars": round(sum(len(r["text"]) for r in rs) / len(rs)),
                    "negated": {lab: round(sum(negated(r["text"]) for r in rs if r["gold"] == lab) / max(1, sum(r["gold"] == lab for r in rs)), 3) for lab in LABELS}}
    for f in os.listdir(args.tests_from):
        if f.startswith(("test_", "val_")):
            shutil.copy(f"{args.tests_from}/{f}", args.out)
    print(json.dumps(stats, indent=1))
    json.dump(stats, open(f"{args.out}/stats{'_pm' if args.polarity else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
