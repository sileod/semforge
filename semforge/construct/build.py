"""Backward construction of a derivation that uses a requested multiset of rules.

Start from a conclusion slot (a metavariable, or a given formula), pick a planned rule whose conclusion
unifies with an open slot, open its premises as new slots, and repeat (backtracking on dead ends) until
the plan is used up; open slots left over become the given premises. Remaining metavariables are bound
to fresh atoms, so the derivation shares no vocabulary with anything else unless asked to. The tree is
kept, so later constructions can attach below a chosen node (e.g. an alternative proof of a subgoal,
leaving the steps above it shared). Nothing here is trusted: the synthesizer re-analyses every instance.
"""
from __future__ import annotations

import z3

from collections import Counter
from functools import lru_cache
from itertools import permutations
from dataclasses import dataclass
from typing import Optional

from .calculus import Calculus
from .terms import is_var, rename, subst, unifiers, variables


class Fresh:
    """Fresh atoms and rename tags, unique within one instance."""

    def __init__(self, prefix="p"):
        self.prefix, self.n = prefix, 0

    def _next(self):
        self.n += 1
        return self.n

    def atom(self):
        return f"{self.prefix}{self._next()}"

    def var(self):
        return f"?v{self._next()}"

    def tag(self):
        return str(self._next())


@dataclass
class Derivation:
    nodes: list           # ground formulas; nodes[0] is the conclusion
    steps: list           # (rule name, conclusion node, premise nodes), in construction order
    leaves: list          # node indices left open: the given premises

    @property
    def conclusion(self):
        return self.nodes[0]

    @property
    def premises(self) -> list:
        return list(dict.fromkeys(self.nodes[i] for i in self.leaves))

    @property
    def rules(self) -> list:
        return [r for r, _, _ in self.steps]

    @property
    def intermediates(self) -> list:
        return list(self.nodes)

    def above(self, node: int) -> Counter:
        """Rule applications on the path from `node` up to the conclusion (excluding any concluding it)."""
        parent = {p: (r, c) for r, c, ps in self.steps for p in ps}
        out = Counter()
        while node in parent:
            r, node = parent[node]
            out[r] += 1
        return out


def derive(calculus: Calculus, plan, rng, fresh: Fresh, root=None, max_depth: int = 6,
           budget: int = 200, admissible=None, deep: bool = False, ordered: bool = False,
           script=None, forbid=None) -> Optional[Derivation]:
    """A derivation applying every rule of `plan` once (a multiset), or None if they cannot be placed.
    A `None` entry in the plan is a wildcard (any rule): a plan of n wildcards grows a random n-step
    derivation, whose rules are then a placeable plan by construction.

    Randomized backtracking over (open slot, unused planned rule) pairs that unify, at most `budget`
    placements tried; a slot is only a leaf once the plan is used up. `admissible` (formula -> bool,
    checked on every partially instantiated node) prunes constructions early, e.g. unreadable nesting.
    `deep` expands the deepest open slots first, giving chains rather than bushy trees. `ordered`
    places the plan's rules in the given order (top-down), branching only over slots: the order a
    derivation was built in (`Derivation.rules`) always replays, which free placement may miss.
    `script` (a derivation's `steps`) replays its exact tree with fresh constants, with no search.
    `forbid(steps, node, nodes, s)` rejects a placement at `node` given the steps and formulas so far (e.g. Atlas.violated:
    fragments the calculus can always bypass). With a
    signature on the calculus, conclusions match modulo unordered arguments, and an unconstrained
    formula slot becomes a leaf symbol over fresh constants (e.g. "c is a P"); without one, a fresh
    propositional atom. Ground nodes are canonical."""
    root = fresh.var() if root is None else root
    tries = [budget]

    def place(open_, plan, s, nodes, steps):
        if not plan:
            return open_, s, nodes, steps
        wild = plan.index(None) if None in plan else None  # wildcards are interchangeable: try one
        if script is not None:
            name, at, _ = script[len(steps)]
            moves = [(i, 0, name) for i, (node, _) in enumerate(open_) if node == at]
        else:
            moves = [(i, k, name) for i, (_, d) in enumerate(open_) if d < max_depth
                     for k in (range(1) if ordered else range(len(plan)))
                     if plan[k] is not None or k == wild
                     for name in ([plan[k]] if plan[k] is not None else calculus.rules)]
            rng.shuffle(moves)
            if deep:
                moves.sort(key=lambda m: -open_[m[0]][1])
        for i, k, name in moves:
            if tries[0] <= 0:
                return None
            rule, tag = calculus.rules[name], fresh.tag()
            node, depth = open_[i]
            for s2 in unifiers(nodes[node], rename(rule.conclusion, tag), s, calculus.unordered):
                tries[0] -= 1
                new = [rename(p, tag) for p in rule.premises]
                if admissible and not all(admissible(subst(x, s2)) for x in nodes + new):
                    continue
                ids = list(range(len(nodes), len(nodes) + len(new)))
                placed = steps + [(name, node, tuple(ids))]
                if forbid is not None and forbid(placed, node, nodes + new, s2):
                    continue
                found = place(open_[:i] + open_[i + 1:] + [(j, depth + 1) for j in ids], plan[:k] + plan[k + 1:],
                              s2, nodes + new, placed)
                if found:
                    return found
        return None

    if script is not None:
        plan = [name for name, _, _ in script]
    found = place([(0, 0)], list(plan), {}, [root], [])
    if found is None:
        return None
    open_, s, nodes, steps = found
    return _derivation(calculus, ground(calculus, nodes, s, fresh, rng), steps, [i for i, _ in open_])


def ground(calculus: Calculus, nodes, s, fresh: Fresh, rng, leaves: bool = True) -> Optional[list]:
    """Schematic nodes under substitution s, made ground: a free formula slot becomes a leaf symbol
    over fresh constants (or, without `leaves` or leaf symbols, a fresh atom), every other metavariable a
    fresh constant; canonical. With background facts, nodes over background predicates are first
    matched to true facts (None if impossible), and metavariables of closed sorts (`Sort.fixed`) are
    bound to their constants, distinct ones where possible."""
    sig = calculus.signature
    if sig is not None and sig.solved_grounding:
        s = _bind_satisfiable(calculus, nodes, s, rng)
        if s is None:
            return None
    elif sig is not None and sig.background_heads:
        s = _solve_background(calculus, nodes, s, rng)
        if s is None:
            return None
        s = _bind_closed(sig, nodes, s, rng)
    if sig is not None and leaves:
        for v in sorted(set().union(*(sig.formula_vars(subst(x, s)) for x in nodes))):
            leaf = sig.leaf(fresh, rng)
            if leaf is not None:
                s = {**s, v: leaf}
    names = {v: fresh.atom() for v in sorted(set().union(*(variables(subst(x, s)) for x in nodes)))}
    return [calculus.canonical(subst(subst(x, s), names)) for x in nodes]


def _bind_satisfiable(calculus, nodes, s, rng):
    """Closed-sort metavariables bound one at a time, in random order, each to a random constant (unused
    ones first) that keeps all nodes jointly satisfiable; background nodes must then be facts. Random
    binding of every value independently makes a proof over functional symbols almost always
    inconsistent. None when some variable has no consistent value."""
    from .oracle import Model
    sig = calculus.signature
    xs = [subst(x, s) for x in nodes]
    xs = [x for x in xs if not sig.formula_vars(x)]  # free formula slots are filled later, by leaves
    sorts = sig.infer(xs)
    free = [v for v in sorted(set().union(*(variables(x) for x in xs)))
            if sig.sorts.get(sorts.get(v)) is not None and sig.sorts[sorts[v]].fixed]
    m, solver = Model(sig, xs), z3.Solver()
    solver.set("timeout", 1000)
    const = lambda srt, c: z3.Const(f"{srt}_{c}", m.sort(srt))
    for srt in {sorts[v] for v in free} | {sorts[c] for c in sorts if c not in free and sorts[c] in sig.sorts}:
        spec = sig.sorts[srt]
        if spec.fixed and spec.distinct and len(spec.fixed) > 1:
            solver.add(z3.Distinct([const(srt, c) for c, _ in spec.fixed]))
    solver.add(*(a(m) for a in sig.axioms), *(m.formula(x) for x in xs))
    solver.add(*(z3.Or([const(sorts[v], v) == const(sorts[v], c) for c, _ in sig.sorts[sorts[v]].fixed]) for v in free))
    used ={c for c, srt in sorts.items() if c not in free}
    rng.shuffle(free)
    for v in free:
        consts = [c for c, _ in sig.sorts[sorts[v]].fixed]
        rng.shuffle(consts)
        for c in sorted(consts, key=lambda c: c in used):
            solver.push()
            solver.add(const(sorts[v], v) == const(sorts[v], c))
            if solver.check() == z3.sat:
                s, used = {**s, v: c}, used | {c}
                break
            solver.pop()
        else:
            return None
    facts, heads = calculus.background_set, sig.background_heads
    if any(x[0] in heads and subst(x, s) not in facts for x in nodes if isinstance(x, tuple)):
        return None
    return s


def _solve_background(calculus, nodes, s, rng):
    heads, facts = calculus.signature.background_heads, calculus.background_set
    todo = list(dict.fromkeys(x for x in nodes if isinstance(x, tuple) and x[0] in heads))
    tries = [2000]

    def go(i, s):
        if i == len(todo):
            return s
        x = subst(todo[i], s)
        if facts.interprets(x):
            options = [f for f, _ in facts.solutions(x, s)]
        else:
            options = [f for f in facts.listed if f[0] == x[0]]
        rng.shuffle(options)
        for f in options:
            for s2 in unifiers(x, f, s, calculus.unordered):
                tries[0] -= 1
                if tries[0] < 0:
                    return None
                found = go(i + 1, s2)
                if found is not None:
                    return found
        return None
    return go(0, s)


def _bind_closed(sig, nodes, s, rng):
    free = set().union(*(variables(subst(x, s)) for x in nodes))
    sorts = sig.infer([subst(x, s) for x in nodes])  # metavariables are strings, so they get sorts too
    used = {c for c, srt in sorts.items() if c not in free and sig.sorts.get(srt, None) is not None}
    for v in sorted(free):
        spec = sig.sorts.get(sorts.get(v))
        if spec is None or not spec.fixed:
            continue
        consts = [c for c, _ in spec.fixed]
        unused = [c for c in consts if c not in used]
        c = rng.choice(unused or consts)
        used.add(c)
        s = {**s, v: c}
    return s


# --- shapes ------------------------------------------------------------------------------------------
# A skeleton is an unlabeled proof tree: a tuple of child skeletons, one per *derived* premise of a step
# (other premises are given). () is a step with only given premises; ((), ()) one joining two
# sub-derivations; (((),),) a chain of three.

def skeleton_size(sk) -> int:
    return 1 + sum(map(skeleton_size, sk))


def shape_topology(sk) -> dict:
    """The certified topology (Analysis.topology) a skeleton should have if all its steps are mandatory."""
    def height(k):
        return 1 + max(map(height, k), default=0)

    def joins(k):
        return (len(k) >= 2) + sum(map(joins, k))
    return {"spine": height(sk), "joins": joins(sk)}


def random_skeleton(rng, n: int, max_children: int = 2, branch: float = 0.3):
    """A random skeleton with n steps; `branch` is the chance that a new step opens a sibling branch
    rather than extending the current chain (0 gives a chain, 1 a bushy tree)."""
    kids = {0: []}
    tips = [0]
    for i in range(1, n):
        open_ = [j for j in kids if len(kids[j]) < max_children]
        parent = rng.choice([j for j in open_ if kids[j]] or open_) if rng.random() < branch else \
            next((t for t in reversed(tips) if len(kids[t]) < max_children), open_[0])
        kids[parent].append(i)
        kids[i] = []
        tips.append(i)

    def build(j):
        return tuple(build(c) for c in kids[j])
    return build(0)


def _outer(f) -> tuple:
    """A formula pattern's outer symbols, through negations, down to a symbol or "?" for a variable:
    `not (at A H)` -> ("not", "at"), `?P` -> ("?",). Two patterns can unify only if these agree up to
    the first "?"."""
    if is_var(f):
        return ("?",)
    if isinstance(f, str):
        return (f,)
    return (f[0],) + (_outer(f[1]) if f[0] == "not" else ())


def _agree(a: tuple, b: tuple) -> bool:
    for x, y in zip(a, b):
        if "?" in (x, y):
            return True
        if x != y:
            return False
    return len(a) == len(b)


@lru_cache(maxsize=None)
def _fillers(calculus: Calculus) -> tuple:
    """Per rule: the outer symbols of its conclusion and of the premises a sub-derivation may fill."""
    bg = calculus.signature.background_heads if calculus.signature is not None else frozenset()
    return tuple((_outer(r.conclusion), tuple(_outer(p) for p in r.premises if not (isinstance(p, tuple) and p[0] in bg)))
                 for r in calculus.rules.values())


def shape_feasible(calculus: Calculus, skeleton) -> bool:
    """False when no assignment of rules can fill the skeleton even on outer symbols: some step needs a
    rule whose derivable premises take its sub-derivations, and none concludes what they would have to.
    Conservative (variables match anything, sorts are ignored), so True does not promise a derivation;
    False is certain, and construction can skip the search (e.g. nested joins in the grid pack)."""
    fillers = _fillers(calculus)

    @lru_cache(maxsize=None)
    def fits(want, sk):
        return any(_agree(head, want) and any(all(fits(slot, child) for slot, child in zip(chosen, sk))
                                              for chosen in permutations(slots, len(sk)))
                   for head, slots in fillers if len(slots) >= len(sk))
    return fits(("?",), skeleton)


def derive_shape(calculus: Calculus, skeleton, rng, fresh: Fresh, plan=None, root=None, budget: int = 400,
                 admissible=None, forbid=None, prefer=None, policy=None) -> Optional[Derivation]:
    """A derivation with exactly the skeleton's shape: every step has as many derived premises as its
    skeleton node has children (the remaining premises are given). Rules are free, or drawn from
    `plan` (a multiset whose size equals the skeleton's), or free but trying rules of the kinds in
    `prefer` first (a soft steer: the realized mix is whatever fits). Randomized backtracking over rule
    choices and over which premise slots carry the sub-derivations; same pruning hooks as `derive`.
    `policy` (construct.policy) orders the candidate rules instead (it is told `prefer` by the caller)."""
    if not shape_feasible(calculus, skeleton):
        return None
    root = fresh.var() if root is None else root
    tries = [budget]
    names = sorted(calculus.rules)
    bg = calculus.signature.background_heads if calculus.signature is not None else frozenset()

    def derivable(rule):  # premise positions a sub-derivation may fill (not background facts)
        return [i for i, p in enumerate(rule.premises) if not (isinstance(p, tuple) and p[0] in bg)]

    def place(todo, plan, s, nodes, steps):
        if not todo:
            return s, nodes, steps
        (node, sk), rest = todo[0], todo[1:]
        pool = sorted(set(plan)) if plan is not None else names
        pool = [n for n in pool if len(derivable(calculus.rules[n])) >= len(sk)]
        if policy is not None:
            pool = policy.order(calculus, pool, steps, node, sk, rng)
        else:
            rng.shuffle(pool)
            if prefer:
                pool.sort(key=lambda n: calculus.rules[n].family not in prefer)
        for name in pool:
            rule, tag = calculus.rules[name], fresh.tag()
            for s2 in unifiers(nodes[node], rename(rule.conclusion, tag), s, calculus.unordered):
                if tries[0] <= 0:
                    return None
                tries[0] -= 1
                new = [rename(p, tag) for p in rule.premises]
                if admissible and not all(admissible(subst(x, s2)) for x in nodes + new):
                    continue
                ids = tuple(range(len(nodes), len(nodes) + len(new)))
                placed = steps + [(name, node, ids)]
                if forbid is not None and forbid(placed, node, nodes + new, s2):
                    continue
                slots = list(permutations(derivable(rule), len(sk)))
                rng.shuffle(slots)
                rest_plan = None if plan is None else _minus(plan, name)
                for chosen in slots[:6]:
                    found = place(rest + [(ids[i], c) for i, c in zip(chosen, sk)], rest_plan, s2,
                                  nodes + new, placed)
                    if found:
                        return found
        return None

    found = place([(0, skeleton)], None if plan is None else list(plan), {}, [root], [])
    if found is None:
        return None
    s, nodes, steps = found
    derived = {at for _, at, _ in steps}
    return _derivation(calculus, ground(calculus, nodes, s, fresh, rng), steps,
                       [i for i in range(len(nodes)) if i not in derived])


def _derivation(calculus, nodes, steps, leaves) -> Optional[Derivation]:
    """Background facts among the leaves are theory, not given premises."""
    if nodes is None:
        return None
    bg = calculus.background_set
    return Derivation(nodes, steps, [i for i in leaves if nodes[i] not in bg])


def _minus(plan, name):
    out = list(plan)
    out.remove(name)
    return out
