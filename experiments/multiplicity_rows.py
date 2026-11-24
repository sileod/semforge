"""Proof multiplicity as a training contrast: one proof against several, at the same proof length.

The audit finds datasets where the goal has several minimal proofs (re-routed or over-determined), and
gpt-oss errs more on FLD×2 rows whose premises can be re-routed (experiments/llm_realized_depth.py).
Whether training on such rows helps or hurts is open (paper, Limitations). Both arms are forward monadic
rows, matched per label and minimum proof size; they differ in how many minimal proofs certification
finds.

  plansing  exactly one minimal proof
  planmult  at least three minimal proofs

Fewer steps are mandatory when proofs multiply (re-routing bypasses them), so the contrast is
multiplicity together with its loss of forced steps, at a fixed shortest proof: the situation the audit
describes. Neutrals come from the same generator (one mandatory premise dropped) and are split by the
multiplicity of the row they were made from. Only rows with an exact profile are kept.

  python -m experiments.multiplicity_rows --out results/llm_mix/data_multiplicity
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil

from experiments.external_rows import write
from experiments.llm_common import LABELS, generate

SIZES = range(3, 13)


class Profiled:
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, r):
        return r["gold"] in self.labels and r.get("exact") and r.get("min_size") in SIZES and arm_of(r)


def arm_of(r):
    if r["proofs"] == 1:
        return "plansing"
    if r["proofs"] >= 3:
        return "planmult"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/llm_mix/data_multiplicity")
    ap.add_argument("--n", type=int, default=3000, help="rows per arm, a third per label")
    ap.add_argument("--pool", type=int, default=60000, help="E/C rows generated (neutrals: half as many)")
    ap.add_argument("--tests-from", default="results/llm_mix/data_packmix")
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = generate("monadic", "forward", 80, args.pool, Profiled(("entailment", "contradiction")),
                    jobs=args.jobs, profile=True)
    rows += generate("monadic", "forward_neutral", 81, args.pool // 2, Profiled(("neutral",)),
                     jobs=args.jobs, profile=True)
    by = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: r["text"]):
        by[arm_of(r), r["gold"], r["min_size"]].append(r)
    rng = random.Random(0)
    for v in by.values():
        rng.shuffle(v)
    per_label = args.n // 3
    arms = {"plansing": [], "planmult": []}
    for lab in LABELS:
        matched = {s: min(len(by["plansing", lab, s]), len(by["planmult", lab, s])) for s in SIZES}
        total = sum(matched.values())
        assert total >= per_label, f"{lab}: only {total} matched rows; raise --pool"
        take = {s: matched[s] * per_label // total for s in SIZES}
        for s in sorted(SIZES, key=lambda s: -matched[s])[:per_label - sum(take.values())]:
            take[s] += 1
        for a in arms:
            arms[a] += [{**r, "cell": a} for s in SIZES for r in by[a, lab, s][:take[s]]]
    stats = {}
    for a, rs in arms.items():
        rng.shuffle(rs)
        write(f"{args.out}/train_{a}.jsonl", rs)
        mean = lambda k: round(sum(r[k] for r in rs) / len(rs), 2)
        stats[a] = {"rows": len(rs), "labels": collections.Counter(r["gold"] for r in rs),
                    "min_size": collections.Counter(r["min_size"] for r in rs), "mean_min_size": mean("min_size"),
                    "mean_proofs": mean("proofs"), "mean_supports": mean("supports"),
                    "mean_mandatory": mean("mandatory"), "mean_planned": mean("planned"),
                    "mean_premises": mean("premises"), "mean_chars": round(sum(len(r["text"]) for r in rs) / len(rs))}
    for f in os.listdir(args.tests_from):
        if f.startswith(("test_", "val_")):
            shutil.copy(f"{args.tests_from}/{f}", args.out)
    print(json.dumps(stats, indent=1))
    json.dump(stats, open(f"{args.out}/stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
