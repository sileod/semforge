"""How much do structural measures depend on the calculus? Re-analyze the same rows under variant calculi.

Mandatory steps, minimum depth and minimum size are exact relative to a pack's rules. A reviewer may ask
whether they are properties of the problem or of that rule set. We take certified monadic rows (E/C,
shape construction) and analyze them, unchanged, under three sound calculi:
  base     the pack's rules
  extended base + new primitive rules (modus ponens on a negated disjunct, double negation)
  macro    base + derived rules fusing frequent two-step patterns (instantiation chains, Barbara then
           instantiation, and-elimination then disjunctive syllogism): a coarser grain of "step"
Mandatory counts are compared per row: a macro that replaces two mandatory steps by one alternative
route can make both non-mandatory, so a coarser calculus can lower the count by more than the fused steps.

  python -m experiments.calculus_robustness --n 300 --out results/construct/calculus_robustness.json
"""
import argparse
import json
import random
import statistics

from semforge.construct.calculus import Calculus, analyze
from semforge.construct.check import prove_rule
from semforge.construct.packs import get
from semforge.construct.signature import rule
from semforge.domains.constructed import ConstructedDomain

EXTENDED = [
    rule("or_modus_ponens", "(or (not P) Q), P |- Q"),
    rule("double_negation", "not (not P) |- P"),
]
MACRO = [
    rule("instantiation_chain", "(all A B), (all B C), (is A X) |- is C X"),
    rule("barbara_instantiation", "(is A X), (all A B), (all B C) |- is C X", kind="instantiation_chain"),
    rule("conjunct_syllogism", "(and (or P Q) R), not P |- Q"),
    rule("syllogism_conjunct", "(or P Q), (and (not P) R) |- Q", kind="conjunct_syllogism"),
]


def spearman(x, y):
    rank = lambda v: {i: r for r, i in enumerate(sorted(range(len(v)), key=lambda i: v[i]))}
    rx, ry = rank(x), rank(y)
    n = len(x)
    return 1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/construct/calculus_robustness.json")
    args = ap.parse_args()
    pack = get("monadic")
    for r in EXTENDED + MACRO:
        assert prove_rule(pack, r) == "proved", r.name
    base = list(pack.calculus.rules.values())
    calculi = {"base": pack.calculus, "extended": Calculus(base + EXTENDED, pack.signature),
               "macro": Calculus(base + MACRO, pack.signature)}
    d, rng, rows = ConstructedDomain("monadic", strategy="shape"), random.Random(args.seed), []
    while len(rows) < args.n:
        for inst, label in d.sample(rng, rng.randint(0, 5)):
            if label == "neutral":
                continue
            row = {}
            for name, calc in calculi.items():
                a = analyze(calc, inst.premises, inst.conclusion, inst.universe)
                row[name] = {k: a.profile[k] for k in ("mandatory", "min_depth", "min_size", "spine", "exact")}
            rows.append(row)
    out = {"n": len(rows), "pack": "monadic", "strategy": "shape"}
    for name in ("extended", "macro"):
        cmp = {}
        for m in ("mandatory", "min_depth", "min_size", "spine"):
            x = [r["base"][m] for r in rows]
            y = [r[name][m] for r in rows]
            cmp[m] = {"base_mean": round(statistics.mean(x), 2), "mean": round(statistics.mean(y), 2),
                      "unchanged": round(sum(a == b for a, b in zip(x, y)) / len(x), 3),
                      "spearman": round(spearman(x, y), 3),
                      "ratio_median": round(statistics.median(b / a for a, b in zip(x, y) if a), 3)}
        cmp["inexact_rows"] = sum(not r[name]["exact"] for r in rows)
        out[name] = cmp
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
