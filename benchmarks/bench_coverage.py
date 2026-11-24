"""Coverage of certified structure per strategy, at an equal number of instances.

  python -m benchmarks.bench_coverage --pack propositional [--n 600]

Strategies:
  forward          FLD-style random proof trees (1-20 steps), labels by the oracle, certified post hoc
  random_plans     targets with random sampled plans (Archive.draw), no coverage steering
  plan_archive     least-covered niche among random candidate plans (Archive.propose)
  random_shapes    shape targets from random skeletons (1-20 steps, random branching)
  shape_archive    TopologyArchive: least-covered (label, spine, joins) cell inverted into a shape target
  shape_archive_kinds  the same, also aiming each request at the least-covered set of required kinds
  shape_archive_fm / shape_archive_kinds_fm  the same archives, with rule choices ordered by an online
                   factorization machine (construct.policy.FMPolicy) rewarded for certified, novel kind sets

Metrics on what was certified: grid cells covered (label × spine ≤ 12 × joins ≤ 6), distinct
topologies, distinct mandatory-step patterns (kinds of the steps in every proof; E/C), distinct mandatory proof
structures (archive.structure: the rule-labelled tree of mandatory steps modulo renaming), the same two
counts on an equal number of E/C instances, entropy over cells, instances per second.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter

import benchmarks.bench_baselines as B
from semforge.construct.archive import Archive, TopologyArchive, structure, topology_niche
from semforge.construct.build import random_skeleton
from semforge.construct.packs import get
from semforge.construct.policy import FMPolicy, novelty_reward
from semforge.construct.synth import Synthesizer
from semforge.construct.target import Distractors, Target

GRID = set(TopologyArchive().cells)


class _Cert:  # what forward samples certify, shaped like an Instance for the metrics
    def __init__(self, label, analysis):
        self.label, self.analysis = label, analysis


def entropy(c: Counter) -> float:
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0


EQUAL_EC = 380  # structures are also counted on the first EQUAL_EC E/C instances, equal for every strategy


def metrics(insts, seconds):
    cells = Counter(topology_niche(i) for i in insts)
    ec = [i for i in insts if i.label != "neutral"]
    structs = [structure(i.analysis) for i in ec]
    return dict(n=len(insts), grid_cells=len(set(cells) & GRID), grid_size=len(GRID),
                grid_cells_ec=len({c for c in cells if c in GRID and c[0] != "neutral"}),
                topologies=len({c[1:] for c in cells}), cell_entropy=round(entropy(cells), 2),
                patterns=len({tuple(sorted(i.analysis.mandatory_kinds.items())) for i in ec}), n_ec=len(ec),
                kind_sets=len({frozenset(i.analysis.mandatory_kinds) for i in ec}),
                structures=len(set(structs)), structures_at_equal_ec=len(set(structs[:EQUAL_EC])),
                kind_sets_at_equal_ec=len({frozenset(i.analysis.mandatory_kinds) for i in ec[:EQUAL_EC]}),
                mean_mandatory=round(sum(len(i.analysis.mandatory_steps) for i in insts) / max(len(insts), 1), 1),
                max_spine=max((c[1] for c in cells), default=0), per_second=round(len(insts) / seconds, 1))


def size_matched(made: dict, rng) -> dict:
    """Diversity at an equal *size* distribution: each strategy's E/C instances are subsampled so that
    every count of mandatory steps occurs equally often in all strategies (the per-count minimum), so
    larger proofs cannot inflate the counts."""
    by = {s: {} for s in made}
    for s, insts in made.items():
        for i in insts:
            if i.label != "neutral":
                by[s].setdefault(len(i.analysis.mandatory_steps), []).append(i)
    common = {k: min(len(by[s].get(k, ())) for s in by) for k in set().union(*map(set, by.values()))}
    out = {}
    for s in by:
        ec = [i for k, m in common.items() if m for i in rng.sample(by[s][k], m)]
        cells = {topology_niche(i) for i in ec}
        out[s] = dict(n=len(ec), cells_ec=len(cells & GRID), structures=len({structure(i.analysis) for i in ec}),
                      kind_sets=len({frozenset(i.analysis.mandatory_kinds) for i in ec}),
                      patterns=len({tuple(sorted(i.analysis.mandatory_kinds.items())) for i in ec}),
                      mean_mandatory=round(sum(len(i.analysis.mandatory_steps) for i in ec) / max(len(ec), 1), 1))
    return out


def run(strategy, pack, n, rng):
    fm = strategy.endswith("_fm")
    strategy = strategy.removesuffix("_fm")
    syn = Synthesizer(pack, policy=FMPolicy(), reward=novelty_reward()) if fm else Synthesizer(pack)
    made, t0, tries = [], time.perf_counter(), 0
    plan_arch = Archive(pack.calculus.rules, sampler=syn.sample_plan, kinds=pack.calculus.kinds)
    shape_arch = TopologyArchive(calculus=pack.calculus) if strategy == "shape_archive_kinds" else TopologyArchive()
    while len(made) < n and tries < 20 * n:
        tries += 1
        if strategy == "forward":
            s = B.forward(rng, rng.randint(1, 20))
            if s is None:
                continue
            premises, hyp, label, rules, height, universe, d = s
            a = B.certify(premises, hyp, label, rules, universe, d.conclusion)
            if a is not None:
                made.append(_Cert(label, a))
            continue
        if strategy == "random_plans":
            inst = syn.realize(plan_arch.draw(rng, level=rng.choice((0, 1, 2))), rng)
        elif strategy == "plan_archive":
            inst = syn.realize(plan_arch.propose(rng, level=len(made) % 3), rng)
            if inst is not None:
                plan_arch.add(inst)
        elif strategy == "random_shapes":
            sk = random_skeleton(rng, rng.randint(1, 20), branch=rng.random())
            inst = syn.realize(Target(rng.choice(("entailment", "contradiction", "neutral")), rules=(), shape=sk,
                                      distractors=Distractors(rng.randint(0, 2), rng.randint(0, 2))), rng)
        else:
            inst = syn.realize(shape_arch.propose(rng), rng)
            shape_arch.add(inst)
        if inst is not None:
            made.append(inst)
    return metrics(made, time.perf_counter() - t0), made


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", default="propositional")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--strategies", nargs="+", default=["forward", "random_plans", "plan_archive", "random_shapes", "shape_archive", "shape_archive_kinds"])
    args = ap.parse_args()
    pack = get(args.pack)
    B.PACK = pack
    rows, made = [], {}
    for strategy in args.strategies:
        m, made[strategy] = run(strategy, pack, args.n, random.Random(args.seed))
        rows.append(dict(strategy=strategy, **m))
        print(json.dumps(rows[-1]), flush=True)
    matched = size_matched(made, random.Random(args.seed))
    for r in rows:
        r["size_matched"] = matched[r["strategy"]]
    print("size-matched:", json.dumps(matched), flush=True)
    out = args.out or f"results/construct/coverage_{args.pack}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(pack=args.pack, args=vars(args), rows=rows), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
