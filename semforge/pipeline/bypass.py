"""Paired arms that differ only in whether the planted steps are necessary: the FLD×2 gap as a treatment.

  python -m semforge.pipeline.bypass --pack monadic --n 1000 --out results/transfer/data_bypass

Each certified E/C row (shape construction) gets one extra premise, in two versions:
  bypass     an intermediate conclusion of the row's proof, chosen to leave the fewest mandatory steps:
             the planted chain is still there, but the hypothesis now follows almost directly;
  necessary  the same formula with its constants renamed to other constants of the row (same sorts):
             same kind of sentence and vocabulary, no shortcut (mandatory steps unchanged, checked).
Both versions keep the label (checked with the oracle), have the same premises in the same order (the
extra one at the same position, the same names for shared constants), and share the neutral rows. So
arms match on planted depth, length, label and surface; they differ in necessary depth.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

from semforge.construct.calculus import Universe, analyze
from semforge.domains.constructed import ConstructedDomain

from .common import write_jsonl


def rename(f, mapping):
    if isinstance(f, tuple):
        return (f[0], *(rename(x, mapping) for x in f[1:]))
    return mapping.get(f, f)


def constants(f):
    return {f} if not isinstance(f, tuple) else set().union(*map(constants, f[1:]))


def pair(pack, inst, rng, tries=20):
    """(bypass formula, necessary formula, mandatory before, after bypass) or None."""
    a, goal = inst.analysis, inst.conclusion
    if a is None or not a.derivable:
        return None
    edges, _ = a._witness()
    derived = {a.edges[k][1] for k in edges} - {a.goal} - set(a.premises)
    best = None
    for f in sorted(derived, key=repr):
        u = Universe([*inst.universe.literals, f], pack.negate, pack.closed)
        m = len(analyze(pack.calculus, [*inst.premises, f], goal, u).mandatory_steps)
        if best is None or m < best[1]:
            best = (f, m)
    if best is None:
        return None
    f, after = best
    if pack.label([*inst.premises, f], inst.hypothesis) != inst.label:
        return None
    sorts = pack.signature.infer(list(inst.premises) + [inst.hypothesis])
    by_sort = {}
    for c, s in sorts.items():
        by_sort.setdefault(s, []).append(c)
    before = len(a.mandatory_steps)
    for _ in range(tries):
        mapping = {c: rng.choice([x for x in by_sort[sorts[c]] if x != c] or [c]) for c in constants(f) if c in sorts}
        g = rename(f, mapping)
        if g == f or g in inst.premises:
            continue
        u = Universe([*inst.universe.literals, g], pack.negate, pack.closed)
        if len(analyze(pack.calculus, [*inst.premises, g], goal, u).mandatory_steps) == before \
                and pack.label([*inst.premises, g], inst.hypothesis) == inst.label:
            return f, g, before, after
    return None


def render(pack, premises, extra, hypothesis, pos, seed):
    """The extra premise is rendered last (so shared constants get the same names in both arms), then
    moved to position `pos`."""
    text, hyp = pack.render([*premises, extra], hypothesis, random.Random(seed))
    lines, theory = text.split("\n"), len(pack.signature.theory)  # theory sentences come first
    lines.insert(theory + pos, lines.pop())
    return "\n".join(lines), hyp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default="monadic")
    ap.add_argument("--n", type=int, default=1000, help="paired E/C rows")
    ap.add_argument("--neutral", type=float, default=0.5, help="neutral rows per E/C row (shared by both arms)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    d, rng = ConstructedDomain(args.pack, strategy="shape"), random.Random(args.seed)
    arms, neutral, stats = {"bypass": [], "necessary": []}, [], Counter()
    while len(arms["bypass"]) < args.n or len(neutral) < args.n * args.neutral:
        for inst, label in d.sample(rng, rng.randint(0, 5)):
            if label == "neutral":
                if len(neutral) < args.n * args.neutral:
                    p, h = d.render(inst)
                    neutral.append({"premises": p, "hypothesis": h, "label": label})
                continue
            if len(arms["bypass"]) >= args.n:
                continue
            got = pair(d.pack, inst, rng)
            stats["tried"] += 1
            if got is None:
                continue
            f, g, before, after = got
            pos, seed = rng.randint(0, len(inst.premises)), rng.random()
            for arm, extra, m in (("bypass", f, after), ("necessary", g, before)):
                p, h = render(d.pack, inst.premises, extra, inst.hypothesis, pos, seed)
                arms[arm].append({"premises": p, "hypothesis": h, "label": label, "mandatory": m, "planted": before})
            stats["paired"] += 1
    os.makedirs(args.out, exist_ok=True)
    for arm, rows in arms.items():
        rows = rows + [dict(r, mandatory=None, planted=None) for r in neutral]
        random.Random(args.seed).shuffle(rows)
        write_jsonl(f"{args.out}/src_{arm}.jsonl", rows)
    mean = lambda arm: round(sum(r["mandatory"] for r in arms[arm]) / len(arms[arm]), 2)
    summary = {**stats, "neutral": len(neutral), "mandatory_bypass": mean("bypass"),
               "mandatory_necessary": mean("necessary"),
               "bypass_zero_or_one": round(sum(r["mandatory"] <= 1 for r in arms["bypass"]) / len(arms["bypass"]), 3)}
    json.dump(summary, open(f"{args.out}/bypass.json", "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
