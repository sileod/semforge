"""Grid puzzles (ZebraLogic-style): N houses in a row, categories of N values each, every value in exactly
one house and every house holding exactly one value per category.

The clues are:
- "v is (not) in house h";
- "v is (not) w";
- "v is somewhere / directly left of w";
- "v and w are next to each other";
- "there are one / two houses between v and w".

Houses and values are closed sorts with fixed meanings, and the frame (house order, successors,
distances, which values share a category) is background theory: ground facts every closure has, which
construction binds rule premises to and never states. The rules are the propagation of
benchmarks/audit_zebra_proofs.py, the calculus ZebraLogic cells were certified in: exclusions and
equalities through clues, column and category exclusion, naked and hidden singles, and no case
splits.
"""
from __future__ import annotations

from itertools import combinations

import z3

from ..pack import declare
from ..signature import Signature, Sort, Symbol, rule

N = 5
CATEGORIES = [
    ("{}", ["Alice", "Bob", "Carol", "David", "Eve"]),
    ("the person in the {} house", ["red", "green", "blue", "yellow", "white"]),
    ("the {} owner", ["cat", "dog", "bird", "fish", "horse"]),
    ("the {} drinker", ["tea", "coffee", "milk", "juice", "water"]),
]
ORDINAL = ["first", "second", "third", "fourth", "fifth", "sixth"]
HOUSES = [f"h{i}" for i in range(1, N + 1)]
VALUES = [[f"v{c}_{i}" for i in range(N)] for c in range(len(CATEGORIES))]
GAPS = {"next": 1, "gap1": 2, "gap2": 3}  # clue -> distance between the houses


# Semantics are symbolic, never computed from constant names, so every rule schema can be proved sound
# for all instances (check.prove_rule): a house has an integer position `num` in 1..N, a value a
# category `cat` and a house `pos`; background predicates are formulas over these.

def _num(m, h):
    return m.fn("num", "House", "Int")(h)


def _cat(m, v):
    return m.fn("cat", "Value", "Int")(v)


def _pos(m, v):
    return m.fn("pos", "Value", "House")(v)


def _apart(m, a, b, d):
    return z3.Or(_num(m, _pos(m, a)) - _num(m, _pos(m, b)) == d, _num(m, _pos(m, b)) - _num(m, _pos(m, a)) == d)


def _frame(m):
    """Houses are N distinct positions; a category's values are in distinct houses; every house holds a
    value of every category (one axiom per category: no quantifier ranges over integers, so countermodels
    stay finite and checkable); the fixed constants have their intended positions and categories."""
    h, k = z3.Const("_h", m.sort("House")), z3.Const("_k", m.sort("House"))
    a, b = z3.Const("_a", m.sort("Value")), z3.Const("_b", m.sort("Value"))
    named = [z3.Const(f"House_{x}", m.sort("House")) for x in HOUSES]
    return z3.And(
        z3.ForAll([h], z3.And(1 <= _num(m, h), _num(m, h) <= N)),
        z3.ForAll([h, k], z3.Implies(_num(m, h) == _num(m, k), h == k)),
        z3.ForAll([a, b], z3.Implies(z3.And(_cat(m, a) == _cat(m, b), a != b), _pos(m, a) != _pos(m, b))),
        *(z3.ForAll([h], z3.Exists([a], z3.And(_cat(m, a) == c, _pos(m, a) == h))) for c in range(len(CATEGORIES))),
        z3.ForAll([a], z3.And(0 <= _cat(m, a), _cat(m, a) < len(CATEGORIES))),
        *(_num(m, x) == i for i, x in enumerate(named, 1)),
        *(_cat(m, z3.Const(f"Value_{v}", m.sort("Value"))) == ci for ci, cat in enumerate(VALUES) for v in cat))


def _left_of(m, a, b):
    return _num(m, _pos(m, a)) < _num(m, _pos(m, b))


def _only(d):
    def f(m, h, k):
        n = _num(m, h)
        return z3.And(z3.Or(_num(m, k) - n == d, n - _num(m, k) == d), z3.Or(n - d < 1, n + d > N))
    return f


def _all_of_category(m, a, *bs):
    v = z3.Const("_v", m.sort("Value"))
    return z3.And(z3.Distinct(a, *bs), *(_cat(m, b) == _cat(m, a) for b in bs),
                  z3.ForAll([v], z3.Implies(_cat(m, v) == _cat(m, a), z3.Or([v == x for x in (a, *bs)]))))


S = lambda name, args, templates=(), negated=(), z3f=None, unordered=(): \
    Symbol(name, args, templates, negated, z3f, unordered=unordered, leaf=False)

SYMBOLS = [
    Symbol("at", ("Value", "House"), ("{0} is in {1}", "{0} lives in {1}"), ("{0} is not in {1}", "{0} does not live in {1}"),
           lambda m, a, h: _pos(m, a) == h),
    S("same", ("Value", "Value"), ("{0} is {1}",), ("{0} is not {1}",), lambda m, a, b: _pos(m, a) == _pos(m, b), ((0, 1),)),
    S("notsame", ("Value", "Value"), ("{0} is not {1}",), ("{0} is {1}",), lambda m, a, b: _pos(m, a) != _pos(m, b), ((0, 1),)),
    S("left", ("Value", "Value"), ("{0} is somewhere to the left of {1}",), ("{0} is not to the left of {1}",), _left_of),
    S("dleft", ("Value", "Value"), ("{0} is directly left of {1}",), ("{0} is not directly left of {1}",),
      lambda m, a, b: _num(m, _pos(m, b)) - _num(m, _pos(m, a)) == 1),
    S("next", ("Value", "Value"), ("{0} and {1} are next to each other",), ("{0} and {1} are not next to each other",),
      lambda m, a, b: _apart(m, a, b, 1), ((0, 1),)),
    S("gap1", ("Value", "Value"), ("there is one house between {0} and {1}",), (),
      lambda m, a, b: _apart(m, a, b, 2), ((0, 1),)),
    S("gap2", ("Value", "Value"), ("there are two houses between {0} and {1}",), (),
      lambda m, a, b: _apart(m, a, b, 3), ((0, 1),)),
    # background theory
    S("first", ("House",), z3f=lambda m, h: _num(m, h) == 1),
    S("last", ("House",), z3f=lambda m, h: _num(m, h) == N),
    S("succ", ("House", "House"), z3f=lambda m, h, k: _num(m, k) == _num(m, h) + 1),
    S("le", ("House", "House"), z3f=lambda m, h, k: _num(m, h) <= _num(m, k)),
    S("other", ("House", "House"), z3f=lambda m, h, k: h != k, unordered=((0, 1),)),
    S("rival", ("Value", "Value"), z3f=lambda m, a, b: z3.And(_cat(m, a) == _cat(m, b), a != b), unordered=((0, 1),)),
    S("others", ("House",) * N, z3f=lambda m, *hs: z3.Distinct(*hs), unordered=(tuple(range(1, N)),)),
    S("rivals", ("Value",) * N, z3f=_all_of_category, unordered=(tuple(range(1, N)),)),
    *(S(f"far_{g}", ("House", "House"), z3f=lambda m, h, k, d=d: z3.And(_num(m, h) - _num(m, k) != d, _num(m, k) - _num(m, h) != d))
      for g, d in GAPS.items()),
    *(S(f"only_{g}", ("House", "House"), z3f=_only(d)) for g, d in GAPS.items()),
]


def _background():
    hs, idx = HOUSES, {h: i + 1 for i, h in enumerate(HOUSES)}
    out = [("first", hs[0]), ("last", hs[-1])]
    out += [("succ", a, b) for a, b in zip(hs, hs[1:])]
    out += [("le", a, b) for a in hs for b in hs if idx[a] <= idx[b]]
    out += [("other", a, b) for a, b in combinations(hs, 2)]
    out += [("others", h, *[k for k in hs if k != h]) for h in hs]
    for cat in VALUES:
        out += [("rival", a, b) for a, b in combinations(cat, 2)]
        out += [("rivals", a, *[b for b in cat if b != a]) for a in cat]
    for g, d in GAPS.items():
        out += [(f"far_{g}", a, b) for a in hs for b in hs if abs(idx[a] - idx[b]) != d]
        for a in hs:
            near = [b for b in hs if abs(idx[a] - idx[b]) == d]
            if len(near) == 1:
                out.append((f"only_{g}", a, near[0]))
    return tuple(out)


_list = lambda ws: ", ".join(ws[:-1]) + " and " + ws[-1]
THEORY = (f"There are {N} houses in a row, from the first (leftmost) to the {ORDINAL[N - 1]} (rightmost), and one "
          "person lives in each.",
          f"Each of {_list(CATEGORIES[0][1])} lives in exactly one house.",
          f"Each house has exactly one of the colours {_list(CATEGORIES[1][1])}, and each colour is on exactly one house.",
          f"Each person owns exactly one of these animals: {_list(['a ' + w for w in CATEGORIES[2][1]])}; no two own "
          "the same one.",
          f"Each person drinks exactly one of {_list(CATEGORIES[3][1])}; no two drink the same one.",
          "A name, a colour, an animal and a drink may describe the same person.")

SIGNATURE = Signature.of(
    sorts=[Sort("House", fixed=tuple((h, f"the {ORDINAL[i]} house") for i, h in enumerate(HOUSES))),
           Sort("Value", fixed=tuple((v, phrase.format(word)) for (phrase, words), cat in zip(CATEGORIES, VALUES)
                                     for v, word in zip(cat, words)))],
    symbols=SYMBOLS, connectives=("not",), axioms=(_frame,), background=_background(), theory=THEORY)

HS, BS = " ".join(f"H{i}" for i in range(1, N)), " ".join(f"B{i}" for i in range(1, N))
RULES = [
    rule("same", "(same A B), (at A H) |- at B H"),
    rule("same_exclusion", "(same A B), not (at A H) |- not (at B H)"),
    rule("notsame", "(notsame A B), (at A H) |- not (at B H)"),
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
    *(rule(f"distance_{g}", f"({g} A B), (at A H), (far_{g} H H2) |- not (at B H2)", kind="distance") for g in GAPS),
    *(rule(f"distance_forced_{g}", f"({g} A B), (at A H), (only_{g} H H2) |- at B H2", kind="distance_forced")
      for g in GAPS),
    rule("naked_single", ", ".join([f"(others H {HS})"] + [f"not (at A H{i})" for i in range(1, N)]) + " |- at A H"),
    rule("hidden_single", ", ".join([f"(rivals A {BS})"] + [f"not (at B{i} H)" for i in range(1, N)]) + " |- at A H"),
]

PACK = declare("grid", SIGNATURE, RULES, atlas_size=2)  # size 3 is 128k fragments; larger nogoods are learned
entails, render = PACK.entails, PACK.render
