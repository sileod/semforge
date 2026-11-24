"""ZebraLogic at the proof level: certify its cells in a propagation calculus for grid puzzles.

Each cell "value v is (not) in house h" becomes a goal over ground atoms (at v h). Clues become
relation facts:
- (same a b), (notsame a b);
- (left a b): a somewhere left of b;
- (dleft a b): a directly left of b;
- (dist k a b): k-1 houses between a and b, so k=1 is "next to";
- (at a h) and (not (at a h)) for the house clues.
The puzzle's frame is a set of ground background facts: (rival a b) for two values of one category,
(other h h2) for two houses, (succ h h2), (le h h2), (first h), (last h), (far k h h2) when |h-h2| != k,
(only k h h2) when h2 is the one house at distance k from h, and (others h h1 ... hn-1) and
(rivals a b1 ... bn-1) for the singles.

The rules are the propagation a person applies, with no case splits: equalities and exclusions
through clues, column and category exclusion, naked singles (a value excluded from all but one house)
and hidden singles (a house excluded for all but one value of a category). Cells that need search are
not derivable and are counted.

On each derivable cell the usual certified measures are read off: mandatory steps, spine, E/C cells,
structures, kind sets. Background facts are premises like the clues, but only the clues count as
the puzzle's premises.

  python -m benchmarks.audit_zebra_proofs --n 400 --out results/audit/zebra_proofs.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from benchmarks.audit_fld_proofs import measures
from semforge.construct.calculus import Calculus, Closure, Universe, analyze
from semforge.construct.signature import negate, rule
from semforge.domains.zebralogic import load


def singles(n):
    hs, bs = " ".join(f"H{i}" for i in range(1, n)), " ".join(f"B{i}" for i in range(1, n))
    return [rule(f"naked_single_{n}", ", ".join([f"(others H {hs})"] + [f"not (at A H{i})" for i in range(1, n)])
                 + " |- at A H", kind="naked_single"),
            rule(f"hidden_single_{n}", ", ".join([f"(rivals A {bs})"] + [f"not (at B{i} H)" for i in range(1, n)])
                 + " |- at A H", kind="hidden_single")]


RULES = [
    rule("same", "(same A B), (at A H) |- at B H"),
    rule("same_back", "(same A B), (at B H) |- at A H", kind="same"),
    rule("same_exclusion", "(same A B), not (at A H) |- not (at B H)"),
    rule("same_exclusion_back", "(same A B), not (at B H) |- not (at A H)", kind="same_exclusion"),
    rule("notsame", "(notsame A B), (at A H) |- not (at B H)"),
    rule("notsame_back", "(notsame A B), (at B H) |- not (at A H)", kind="notsame"),
    rule("category", "(rival A B), (at A H) |- not (at B H)"),
    rule("column", "(other H H2), (at A H) |- not (at A H2)"),
    rule("left", "(left A B), (at A H), (le H2 H) |- not (at B H2)"),
    rule("left_back", "(left A B), (at B H), (le H H2) |- not (at A H2)", kind="left"),
    rule("left_bound", "(left A B), (first H) |- not (at B H)"),
    rule("left_bound_back", "(left A B), (last H) |- not (at A H)", kind="left_bound"),
    rule("dleft", "(dleft A B), (at A H), (succ H H2) |- at B H2"),
    rule("dleft_back", "(dleft A B), (at B H2), (succ H H2) |- at A H", kind="dleft"),
    rule("dleft_exclusion", "(dleft A B), not (at B H2), (succ H H2) |- not (at A H)"),
    rule("dleft_exclusion_back", "(dleft A B), not (at A H), (succ H H2) |- not (at B H2)", kind="dleft_exclusion"),
    rule("dleft_bound", "(dleft A B), (first H) |- not (at B H)"),
    rule("dleft_bound_back", "(dleft A B), (last H) |- not (at A H)", kind="dleft_bound"),
    rule("distance", "(dist K A B), (at A H), (far K H H2) |- not (at B H2)"),
    rule("distance_back", "(dist K A B), (at B H), (far K H H2) |- not (at A H2)", kind="distance"),
    rule("distance_forced", "(dist K A B), (at A H), (only K H H2) |- at B H2"),
    rule("distance_forced_back", "(dist K A B), (at B H), (only K H H2) |- at A H2", kind="distance_forced"),
    *(r for n in range(2, 7) for r in singles(n)),
]
CALCULUS = Calculus(RULES)


def name(var):
    return var.name


def clue(f):
    t = type(f).__name__
    if t == "EqVar":
        return ("same", name(f.x), name(f.y))
    if t == "NeVar":
        return ("notsame", name(f.x), name(f.y))
    if t == "Lt":
        return ("left", name(f.x), name(f.y))
    if t == "Linear":  # c0*v0 + c1*v1 == rhs with {c0, c1} = {1, -1} and |rhs| = 1: v_right - v_left = 1
        assert sorted(f.coeffs) == [-1, 1] and abs(f.rhs) == 1 and f.op == "==", vars(f)
        right = f.vars[0] if f.coeffs[0] == f.rhs else f.vars[1]
        left = f.vars[1] if right is f.vars[0] else f.vars[0]
        return ("dleft", name(left), name(right))
    if t == "Distance":
        return ("dist", f"k{f.k}", name(f.x), name(f.y))
    if t == "Eq":
        return ("at", name(f.x), f"h{f.value}")
    if t == "Ne":
        return ("not", ("at", name(f.x), f"h{f.value}"))
    raise KeyError(t)


def frame(p):
    n, hs = p.n, [f"h{i}" for i in range(1, p.n + 1)]
    out = [("first", "h1"), ("last", f"h{n}")]
    for i in range(1, n + 1):
        others = [f"h{j}" for j in range(1, n + 1) if j != i]
        out.append(("others", f"h{i}", *others))
        for j in range(1, n + 1):
            if i != j:
                out.append(("other", f"h{i}", f"h{j}"))
            if j <= i:
                out.append(("le", f"h{j}", f"h{i}"))
            if j == i + 1:
                out.append(("succ", f"h{i}", f"h{j}"))
            for k in range(1, n):
                if abs(i - j) != k:
                    out.append(("far", f"k{k}", f"h{i}", f"h{j}"))
        for k in range(1, n):
            at_k = [j for j in (i - k, i + k) if 1 <= j <= n]
            if len(at_k) == 1:
                out.append(("only", f"k{k}", f"h{i}", f"h{at_k[0]}"))
    for c, vals in p.categories.items():
        vs = [name(p.variables[(c, v)]) for v in vals]
        for a in vs:
            out.append(("rivals", a, *[b for b in vs if b != a]))
            out += [("rival", a, b) for b in vs if b != a]
    return out, [("at", name(v), h) for v in p.variables.values() for h in hs]


class _Row:
    def __init__(self, label, analysis):
        self.label, self.analysis = label, analysis


class _ClueSupports:
    """The analysis with premise supports counted over the clues only. The puzzle frame (house order,
    rivals, ...) is passed as premises so that rules can use it, but it is theory, not a premise: one clue
    set reached through two frame facts is one minimal justification, not two. `supports` is 1 when the
    clues needed by the goal (those it cannot do without) derive it alone, the exact condition for a unique
    minimal clue set (closure is monotone), and 2 (at least two) otherwise; no enumeration, so no cap."""
    def __init__(self, analysis, unique):
        self._a, self._unique = analysis, unique

    def __getattr__(self, k):
        if k.startswith("_"):
            raise AttributeError(k)
        return getattr(self._a, k)

    @property
    def profile(self):
        return {**self._a.profile, "supports": 1 if self._unique else 2}


class _Snap:
    """The fields `measures` reads, computed once in a worker so the row can cross processes."""
    FIELDS = ("derivable", "mandatory_steps", "mandatory_kinds", "topology", "profile", "given", "goal")

    def __init__(self, a):
        for k in self.FIELDS:
            setattr(self, k, getattr(a, k))


def certify(p, key, house, label, cache):
    if p.pid not in cache:
        cache[p.pid] = frame(p)
    background, atoms = cache[p.pid]
    clues = list(dict.fromkeys(map(clue, p.clues)))
    premises = [*clues, *background]
    atom = ("at", name(p.variables[key]), f"h{house}")
    goal = atom if label == "entailment" else negate(atom)
    universe = Universe([*premises, *atoms], negate, closed=())
    a = analyze(CALCULUS, premises, goal, universe)
    if not a.derivable:
        return None
    g, can = CALCULUS.canonical(goal), CALCULUS.canonical
    derives = lambda cs: g in Closure(CALCULUS, [can(x) for x in [*cs, *background]], universe).round
    needed = [c for c in clues if not derives([d for d in clues if d != c])]
    return _Row(label, _ClueSupports(a, derives(needed)))


_PUZZLES, _CACHE = None, {}


def _evaluate(cand):
    global _PUZZLES
    if _PUZZLES is None:
        _PUZZLES = {p.pid: p for p in load()[0]}
    pid, key, house, label = cand
    r = certify(_PUZZLES[pid], key, house, label, _CACHE)
    return r and _Row(r.label, _Snap(r.analysis))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=1400, help="goals drawn, enough to contain n derivable")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    puzzles, rng = load()[0], random.Random(args.seed)
    cands = []  # the draw sequence does not depend on which goals turn out derivable
    for _ in range(args.draws):
        p = rng.choice(puzzles)
        key = rng.choice(list(p.variables))
        label = rng.choice(["entailment", "contradiction"])
        house = p.solution[key] if label == "entailment" else rng.choice([h for h in range(1, p.n + 1) if h != p.solution[key]])
        cands.append((p.pid, key, house, label))
    with ProcessPoolExecutor(args.jobs) as ex:
        results = list(ex.map(_evaluate, cands, chunksize=1))
    rows, seen, by_size, sizes = [], Counter(), Counter(), {p.pid: p.n for p in puzzles}
    for (pid, *_), r in zip(cands, results):
        if len(rows) == args.n:
            break
        seen["derivable" if r else "not_derivable"] += 1
        by_size[(sizes[pid], bool(r))] += 1
        if r:
            rows.append(r)
    assert len(rows) == args.n, f"only {len(rows)} derivable goals in {args.draws} draws; raise --draws"
    report = {"coverage": dict(seen),
              "derivable_by_houses": {n: round(by_size[(n, True)] / max(1, by_size[(n, True)] + by_size[(n, False)]), 3)
                                      for n in range(2, 7)},
              "measures": measures(rows),
              "mandatory_kinds": dict(Counter(k for r in rows for k in r.analysis.mandatory_kinds).most_common()),
              "mandatory_steps_hist": dict(sorted(Counter(min(len(r.analysis.mandatory_steps), 20) for r in rows).items()))}
    print(json.dumps(report, indent=1))
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
