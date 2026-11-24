"""Competing generation strategies, re-implemented on the same pack and scored by the same certifier.

  python -m benchmarks.bench_baselines --pack propositional [--steps 8 16 30] [--n 20] [--budget 60]

The task: an instance whose proof makes n rule applications, *every one of them necessary* (no proof
skips any), with label entailment or contradiction and n/2 + n/2 distractor derivations. Strategies:

  forward         random proof trees with distractors, labelled by the oracle, nothing certified:
                  the FLD×2 / RuleTaker / ProofWriter recipe. Reported as is: how often the
                  intended structure is the certified one (all n steps required, intended depth).
  forward+filter  forward, resampled until the certificate meets the target: sample-and-select.
  reject          backward construction toward the target, each candidate certified and rejected
                  (or repaired by deleting distractors) on a shortcut: per-instance search over
                  proofs, as in Smith et al. 2013 (quantifying over solutions), without an atlas.
  semforge        the same construction with the shortcut atlas and conflict learning.

Per strategy: success within the time budget, seconds per hit, and the certified structure of what it
produces.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

from semforge.construct.build import Fresh, derive
from semforge.construct.calculus import Universe, analyze
from semforge.construct.packs import get
from semforge.construct.synth import Synthesizer, negation_depth
from semforge.construct.target import Distractors, Target

PACK = None


def forward(rng, n, max_negations=2):
    """One FLD-style sample: a random n-step proof tree plus n/2 random 1-2 step derivations as
    distractors. Returns (premises, hypothesis, label, intended rules, intended depth, universe, tree)."""
    calc, fresh = PACK.calculus, Fresh()
    ok = lambda f: negation_depth(f) <= max_negations
    d = derive(calc, [None] * n, rng, fresh, admissible=ok, deep=rng.random() < 0.5, max_depth=max(6, n // 3 + 4))
    if d is None:
        return None
    premises, inter = list(d.premises), list(d.intermediates)
    for _ in range(n // 2):
        e = derive(calc, [None] * rng.randint(1, 2), rng, fresh, admissible=ok)
        if e is not None:
            premises += [p for p in e.premises if p not in premises]
            inter += e.intermediates
    rng.shuffle(premises)
    label = rng.choice(["entailment", "contradiction"])
    hyp = d.conclusion if label == "entailment" else PACK.negate(d.conclusion)
    height = {0: 0}
    for _, at, ids in reversed(d.steps):
        height[at] = 1 + max(height.get(i, 0) for i in ids)
    return premises, hyp, label, d.rules, height[0], Universe([*premises, hyp, *inter], PACK.negate, PACK.closed), d


def certify(premises, hyp, label, rules, universe, goal):
    if PACK.label(premises, hyp) != label:
        return None
    a = analyze(PACK.calculus, premises, goal, universe)
    return a if a.derivable else None


def run_forward(rng, n, budget, filtered):
    t0, hits, samples, exact, depth_ok, labels_ok = time.perf_counter(), 0, 0, 0, 0, 0
    tries_per_hit = []
    k = 0
    while time.perf_counter() - t0 < budget and (hits < REQ if filtered else samples < REQ):
        s = forward(rng, n)
        k += 1
        if s is None:
            continue
        premises, hyp, label, rules, height, universe, d = s
        samples += 1
        a = certify(premises, hyp, label, rules, universe, d.conclusion)
        if a is None:
            continue
        labels_ok += 1
        full = a.requires(PACK.calculus.kinds(rules)) is True
        exact += full
        depth_ok += a.depth == height
        if full:
            hits += 1
            tries_per_hit.append(k)
            k = 0
    dt = time.perf_counter() - t0
    out = dict(samples=samples, label_kept=labels_ok / max(samples, 1), all_steps_required=exact / max(samples, 1),
               intended_depth_certified=depth_ok / max(samples, 1))
    if filtered:
        out.update(success=hits / REQ, s_per_hit=dt / hits if hits else None,
                   samples_per_hit=sum(tries_per_hit) / len(tries_per_hit) if tries_per_hit else None)
    return out


def run_synth(rng, n, budget, atlas):
    syn = Synthesizer(PACK, atlas=atlas, learn=atlas)
    t0, hits, req = time.perf_counter(), 0, 0
    while time.perf_counter() - t0 < budget and req < REQ:
        req += 1
        target = Target(rng.choice(["entailment", "contradiction"]), rules=syn.sample_plan(rng, n),
                        distractors=Distractors(n // 2, n // 2))
        hits += syn.realize(target, rng) is not None
    dt = time.perf_counter() - t0
    return dict(requests=req, success=hits / max(req, 1), s_per_hit=dt / hits if hits else None,
                attempts=syn.stats["attempts"], shortcut_rejections=syn.stats["shortcut"],
                repaired=syn.stats["repaired"], learned=syn.stats["learned"])


REQ = 20


def main():
    global PACK, REQ
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", default="propositional")
    ap.add_argument("--steps", type=int, nargs="+", default=[8, 16, 30])
    ap.add_argument("--n", type=int, default=20, help="hits (or samples) wanted per cell")
    ap.add_argument("--budget", type=float, default=60, help="seconds per strategy and cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    PACK, REQ = get(args.pack), args.n
    rows = []
    for n in args.steps:
        for name, fn in [("forward", lambda r: run_forward(r, n, args.budget, False)),
                         ("forward+filter", lambda r: run_forward(r, n, args.budget, True)),
                         ("reject", lambda r: run_synth(r, n, args.budget, False)),
                         ("semforge", lambda r: run_synth(r, n, args.budget, True))]:
            r = dict(steps=n, strategy=name, **fn(random.Random(args.seed + n)))
            rows.append(r)
            print(json.dumps(r), flush=True)
    out = args.out or f"results/construct/baselines_{args.pack}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(pack=args.pack, args=vars(args), rows=rows), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
