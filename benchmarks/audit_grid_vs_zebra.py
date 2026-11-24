"""ZebraLogic cells against our grid rows, at the proof level in the same propagation rules.

ZebraLogic cells are certified as in audit_zebra_proofs.py. Our rows come from packs/grid.py, whose
rules are the same, with the same kind names. Each source gets the proof-level measures of
audit_fld_proofs.py, at an equal count and at an equal mandatory-step distribution.

  python -m benchmarks.audit_grid_vs_zebra --n 150 --out results/audit/grid_vs_zebra.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter

from benchmarks.audit_fld_proofs import measures, size_matched
from benchmarks.audit_zebra_proofs import certify
from semforge.construct.calculus import analyze
from semforge.domains.constructed import ConstructedDomain
from semforge.domains.zebralogic import load


def zebra(n, rng):
    puzzles, rows, cache = load()[0], [], {}
    while len(rows) < n:
        p = rng.choice(puzzles)
        key, label = rng.choice(list(p.variables)), rng.choice(["entailment", "contradiction"])
        house = p.solution[key] if label == "entailment" else rng.choice([h for h in range(1, p.n + 1) if h != p.solution[key]])
        r = certify(p, key, house, label, cache)
        if r:
            rows.append(r)
    return rows


def ours(strategy, n, rng):
    d, rows, t, requests = ConstructedDomain("grid", strategy=strategy), [], time.time(), 0
    while len(rows) < n:
        requests += 1
        rows += [inst for inst, lab in d.sample(rng, rng.randint(0, 5)) if lab != "neutral"]
    for inst in rows:  # forward instances are analyzed on demand (as synth.export_certificate does)
        if inst.analysis is None:
            inst.analysis = analyze(d.pack.calculus, inst.premises, inst.conclusion, inst.universe)
    return rows[:n], {"requests": requests, "rows_per_s": round(n / (time.time() - t), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    groups, cost = {"zebralogic": zebra(args.n, random.Random(args.seed))}, {}
    for s in ("shape", "forward"):
        groups[f"ours_{s}"], cost[f"ours_{s}"] = ours(s, args.n, random.Random(args.seed))
        print(s, cost[f"ours_{s}"], flush=True)
    report = {"cost": cost, "equal_n": {k: measures(v) for k, v in groups.items()},
              "size_matched": size_matched(groups, random.Random(args.seed)),
              "mandatory_kinds": {k: dict(Counter(x for r in v for x in r.analysis.mandatory_kinds).most_common())
                                  for k, v in groups.items()}}
    print(json.dumps(report, indent=1))
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
