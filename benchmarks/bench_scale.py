"""How construction cost grows with instance size: planned proof steps and distractor derivations.

  python -m benchmarks.bench_scale --pack propositional [--steps 4 8 12 16 20] [--n 10] [--budget 60]

Per cell: success rate, seconds per attempt, the share of time spent in structural analysis, and the
instance size (premises). A cell stops early when it exceeds its time budget, so the table also shows
where construction stops being practical.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

from semforge.construct import synth as S
from semforge.construct.packs import get
from semforge.construct.target import Distractors, Target


def cell(syn, steps, distractors, n, budget, rng):
    spent = {"analyze": 0.0}
    analyze = S.analyze

    def timed(*a, **kw):
        t = time.perf_counter()
        try:
            return analyze(*a, **kw)
        finally:
            spent["analyze"] += time.perf_counter() - t
    S.analyze = timed
    made, sizes, t0, tried = 0, [], time.perf_counter(), 0
    stats0 = dict(syn.stats)
    try:
        for _ in range(n):
            if time.perf_counter() - t0 > budget:
                break
            tried += 1
            target = Target(rng.choice(["entailment", "contradiction", "neutral"]),
                            rules=syn.sample_plan(rng, steps),
                            distractors=Distractors(irrelevant=distractors, near_miss=distractors))
            inst = syn.realize(target, rng)
            if inst is not None:
                made += 1
                sizes.append(len(inst.premises))
    finally:
        S.analyze = analyze
    total = time.perf_counter() - t0
    why = {k: v - stats0.get(k, 0) for k, v in syn.stats.items() if v - stats0.get(k, 0)}
    attempts = why.pop("attempts", 0)
    return dict(steps=steps, distractors=distractors, requests=tried, success=made / max(tried, 1),
                s_per_attempt=total / max(attempts, 1), s_per_instance=total / made if made else None,
                analyze_share=spent["analyze"] / total if total else 0.0,
                premises=sum(sizes) / len(sizes) if sizes else None, timed_out=tried < n, outcomes=why)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="propositional")
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 8, 12, 16, 20, 30])
    ap.add_argument("--n", type=int, default=10, help="requests per cell")
    ap.add_argument("--budget", type=float, default=60, help="seconds per cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-atlas", action="store_true", help="construct without the shortcut atlas")
    ap.add_argument("--atlas-size", type=int, default=3, help="largest precomputed nogood")
    ap.add_argument("--no-learn", action="store_true", help="do not learn nogoods from shortcut failures")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    syn, rng, rows = S.Synthesizer(get(args.pack), atlas=not args.no_atlas, atlas_size=args.atlas_size,
                                            learn=not args.no_learn), random.Random(args.seed), []
    print(f"{'steps':>5} {'distr':>5} {'success':>8} {'s/attempt':>10} {'s/inst':>8} {'analyze':>8} {'premises':>9}")
    for steps in args.steps:
        r = cell(syn, steps, steps // 2, args.n, args.budget, rng)
        rows.append(r)
        print(f"{steps:>5} {r['distractors']:>5} {r['success']:>8.0%} {r['s_per_attempt']:>10.3f} "
              f"{r['s_per_instance'] or float('nan'):>8.2f} {r['analyze_share']:>8.0%} "
              f"{r['premises'] or float('nan'):>9.1f}{'  (budget hit)' if r['timed_out'] else ''}  {r['outcomes']}", flush=True)
    out = args.out or f"results/construct/scale_{args.pack}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    learned = [syn.atlas.describe(q) for q in syn.atlas.nogoods] if syn.atlas else []
    print(f"atlas: {len(learned)} nogoods; learned during run: {syn.stats['learned']}")
    json.dump(dict(pack=args.pack, args=vars(args), rows=rows, nogoods=learned, learned=syn.stats["learned"]),
              open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
