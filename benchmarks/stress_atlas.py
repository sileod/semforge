"""Stress test of shortcut-atlas soundness: look for an instance of a nogood that has no shortcut.

  python -m benchmarks.stress_atlas --pack propositional [--subs 200] [--embedded 2000]

The soundness argument (the paper, Appendix A) says every instance of a nogood, in any context, is bypassed.
Two attacks on it:

  substitution  each most general instance of each nogood under adversarial substitutions of its
                constants, sort-respecting: constants identified with each other, propositional
                atoms replaced by negations (including of other atoms), conjunctions, disjunctions
                and implications. Analysed alone and with random extra premises.
  embedded      nogood instances inside random derivations of 2-30 steps built without the atlas,
                with distractor derivations, analysed as a whole: the planned steps must not all be
                required.

A counterexample is an instance where every planned application is still mandatory. Substitutions that
identify constants are reported apart: the theorem covers injective ones (the paper, Appendix A). Instances outside
a guarded nogood's guard make no claim and are counted apart.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

from semforge.construct.atlas import atlas_for, instances, rules_of
from semforge.construct.build import Fresh, derive, ground
from semforge.construct.calculus import Universe, analyze
from semforge.construct.packs import get
from semforge.construct.signature import PROP


def replace(x, sigma):
    """Simultaneous replacement of constants (never of head symbols)."""
    if isinstance(x, tuple):
        return (x[0], *(replace(a, sigma) for a in x[1:]))
    return sigma.get(x, x)


def _value(rng, c, same, fresh, connectives, depth=0):
    """An adversarial replacement for constant c, of the same sort."""
    r = rng.random()
    if r < 0.3 and same:
        return rng.choice(same)                       # identification
    if not (connectives and depth < 2) or r < 0.45:
        return fresh.atom() if r < 0.45 else c        # fresh atom, or unchanged
    op = rng.choice([x for x in connectives if x != "not"] + ["not"] * 2)
    arg = lambda: _value(rng, c, same, fresh, connectives, depth + 1)
    return ("not", arg()) if op == "not" else (op, arg(), arg())


def substitutions(pack, rng, n, extra_premises=False):
    calc, sig = pack.calculus, pack.calculus.signature
    atlas, out = atlas_for(pack), Counter()
    for q in atlas.nogoods:
        want = calc.kinds(rules_of(q))
        for nodes, s, steps in instances(calc, q, Fresh("a")):
            base = ground(calc, nodes, s, Fresh("b"), rng, leaves=False)
            sorts = sig.infer(base)
            derived = {at for _, at, _ in steps}
            for _ in range(n):
                fresh = Fresh("z")
                consts = sorted(sorts)
                sigma = {}
                for c in consts:
                    same = [d for d in consts if d != c and sorts[d] == sorts[c]]
                    conn = [x for x in sig.connectives if x != "not"] if sorts[c] == PROP else []
                    sigma[c] = _value(rng, c, same, fresh, conn + (["not"] if sorts[c] == PROP else []))
                g = [calc.canonical(replace(x, sigma)) for x in base]
                if not atlas.violated(steps, 0, g, {}):  # outside a guarded nogood's guard: no claim
                    out["outside_guard"] += 1
                    continue
                premises = [g[i] for i in range(len(g)) if i not in derived]
                if extra_premises:  # unrelated context: monotonicity
                    premises += [x for x in base if x not in premises and rng.random() < 0.3]
                a = analyze(calc, premises, g[0], Universe(g + premises, pack.negate, pack.closed), cap=256)
                if not a.derivable:
                    out["not_derivable"] += 1   # identification can make a premise inconsistent with rule shape
                    continue
                verdict = a.requires(want)  # every planned application mandatory?
                injective = len({repr(v) for v in sigma.values()}) == len(sigma)
                tag = "" if injective else "_identifying"  # the theorem covers injective substitutions
                out[("COUNTEREXAMPLE" if verdict else "bypassed") + tag] += 1
                if verdict:
                    out.setdefault("examples", []).append((atlas.describe(q), premises, g[0]))
    return out


def embedded(pack, rng, n):
    calc, atlas, out = pack.calculus, atlas_for(pack), Counter()
    for _ in range(n):
        fresh, k = Fresh(), rng.randint(2, 30)
        d = derive(calc, [None] * k, rng, fresh, max_depth=max(6, k // 3 + 4))
        if d is None:
            continue
        if not any(atlas.violated(d.steps, at, d.nodes, {}) for _, at, _ in d.steps):
            out["no_nogood"] += 1
            continue
        premises, inter = list(d.premises), list(d.intermediates)
        for _ in range(k // 2):
            e = derive(calc, [None] * rng.randint(1, 3), rng, fresh)
            if e is not None:
                premises += [p for p in e.premises if p not in premises]
                inter += e.intermediates
        a = analyze(calc, premises, d.conclusion, Universe([*premises, *inter], pack.negate, pack.closed), cap=256)
        verdict = a.requires(calc.kinds(d.rules))
        out["COUNTEREXAMPLE" if verdict else "bypassed"] += 1
        if verdict:
            out.setdefault("examples", []).append((d.rules, premises, d.conclusion))
        out["with_nogood"] += 1
        out["steps_total"] += k
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", default="propositional")
    ap.add_argument("--subs", type=int, default=200, help="substitutions per most general instance")
    ap.add_argument("--embedded", type=int, default=2000, help="random derivations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pack, rng = get(args.pack), random.Random(args.seed)
    res = dict(pack=args.pack, nogoods=len(atlas_for(pack).nogoods),
               substitution=substitutions(pack, rng, args.subs),
               substitution_with_context=substitutions(pack, rng, args.subs // 4, extra_premises=True),
               embedded=embedded(pack, rng, args.embedded))
    for k in ("substitution", "substitution_with_context", "embedded"):
        ex = res[k].pop("examples", [])[:5]
        res[k] = dict(res[k], examples=[repr(e) for e in ex])
        print(k, {x: v for x, v in res[k].items() if x != "examples"}, *ex[:3], sep="\n  ")
    out = args.out or f"results/construct/stress_atlas_{args.pack}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
