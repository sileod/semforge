"""Shortcut atlas: the proof fragments a calculus can always bypass, computed once per pack.

A fragment is the top of a derivation tree: a rule, and for each of its premises either nothing (the
premise may be given or derived any way) or a sub-fragment deriving it. Instantiate a fragment in its
most general way (fresh constants everywhere, every unifier of the rule schemas) and analyze it: if
every instance has a proof that skips one of the fragment's rule applications, the fragment is a
*nogood*.

Nogoods are sound everywhere. Let D be any derivation containing an instance of nogood F, in any
context with any extra premises. The schemas are closed under substitution, so the shortcut proof of
F's most general instance maps to a shortcut proof of F's root inside D, from F's leaf formulas. Those
formulas are available in D, derived by the rest of the tree or given. Analysis is monotone, so the
shortcut stays a proof in D, and the step it skips is not required. Construction that never builds a
nogood therefore never loses a planned step to that shortcut. Certification still checks everything
(shortcuts across distant branches and through distractors remain possible); `Atlas.learn` adds the
fragments it finds there. That is conflict-driven construction: certification failures become
schematic nogoods that prune all later constructions.
"""
from __future__ import annotations

import random
from functools import lru_cache
from itertools import product
from typing import Optional

from .build import Fresh
from .calculus import Calculus, Universe, analyze, step_id
from .signature import PROP
from .terms import atoms, matches, rename, subst, unifiers, variables

Pattern = tuple  # (rule name, (child pattern or None per premise))


def size(p) -> int:
    return 0 if p is None else 1 + sum(size(c) for c in p[1])


def rules_of(p) -> list:
    return [] if p is None else [p[0], *(r for c in p[1] for r in rules_of(c))]


def fragments(calculus: Calculus, n: int) -> list:
    """Every fragment with exactly n rule applications."""
    memo = {}

    def gen(n):
        if n not in memo:
            out = []
            for name in sorted(calculus.rules):
                k = len(calculus.rules[name].premises)
                for split in _compositions(n - 1, k):
                    for kids in product(*([None] if m == 0 else gen(m) for m in split)):
                        out.append((name, tuple(kids)))
            memo[n] = out
        return memo[n]
    return gen(n)


def _compositions(total, parts):
    if parts == 0:
        if total == 0:
            yield ()
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def instances(calculus: Calculus, pattern: Pattern, fresh: Fresh, cap: int = 16):
    """Most general instantiations of a fragment: (schematic nodes, substitution, steps)."""
    out = []

    def go(todo, s, nodes, steps):
        if len(out) >= cap:
            return
        if not todo:
            out.append((nodes, s, steps))
            return
        (pat, at), rest = todo[0], todo[1:]
        rule, tag = calculus.rules[pat[0]], fresh.tag()
        for s2 in unifiers(nodes[at], rename(rule.conclusion, tag), s, calculus.unordered):
            new = [rename(p, tag) for p in rule.premises]
            ids = tuple(range(len(nodes), len(nodes) + len(new)))
            go(rest + [(c, i) for c, i in zip(pat[1], ids) if c is not None], s2, nodes + new,
               steps + [(rule.name, at, ids)])
    go([(pattern, 0)], {}, [fresh.var()], [])
    return out


def _verdict(pack, nodes, s, steps, want, schematic_vars, fresh) -> bool:
    """Is this schematic instance bypassed? Metavariables become fresh atoms; those in
    `schematic_vars` stand for arbitrary formulas, so their negations are left out of the universe."""
    calc = pack.calculus
    names = {v: fresh.atom() for v in sorted(set().union(*(variables(subst(x, s)) for x in nodes)))}
    g = [calc.canonical(subst(subst(x, s), names)) for x in nodes]
    derived = {at for _, at, _ in steps}
    premises = [g[i] for i in range(len(g)) if i not in derived]
    a = analyze(calc, premises, g[0], Universe(g, pack.negate, pack.closed, {names[v] for v in schematic_vars}))
    return a.derivable and not a.requires(want)  # an alternative proof bypasses a planned application


def _formula_vars(calc, nodes, s) -> list:
    sig = calc.signature
    if sig is None:  # every metavariable is a formula
        return sorted(set().union(*(variables(subst(x, s)) for x in nodes)))
    return sorted(set().union(*(sig.formula_vars(subst(x, s)) for x in nodes)))


def bypassed(pack, pattern: Pattern, rng=None, guarded: bool = False):
    """True if every instance of the fragment has a shortcut, False if not certainly, None if the
    fragment cannot be built at all.

    Most general instances replace formula metavariables by atoms whose negations are left out of the
    analysis universe: negation cancels (the negation of "not q" is q), so a shortcut through the
    negation of an atom need not survive substitution. Without that restriction, a verdict holds only
    for instances where the atom's image is not itself a negation. With `guarded`, such fragments
    return their *guards* instead of False: the list of (steps, schematic nodes, plain variables) cases
    in which they are bypassed, each formula metavariable being either plain (not a negation) or
    "not w" for an arbitrary w. Every case is closed under substitution, so the claim is exact."""
    calc, fresh = pack.calculus, Fresh("a")
    cap = 64
    insts = instances(calc, pattern, fresh, cap=cap)
    if not insts:
        return None
    if len(insts) >= cap:  # unifiers possibly missed: no claim
        return False
    want = calc.kinds(rules_of(pattern))
    if all(_verdict(pack, nodes, s, steps, want, _formula_vars(calc, nodes, s), fresh) for nodes, s, steps in insts):
        return True
    if not guarded:
        return False
    cases = []
    for nodes, s, steps in insts:
        fv = _formula_vars(calc, nodes, s)
        if len(fv) > 6 or not _verdict(pack, nodes, s, steps, want, (), fresh):
            continue  # only fragments bypassed when nothing is a negation are split (sound, incomplete)
        for neg in product((False, True), repeat=len(fv)):
            s2 = {**s, **{v: ("not", fresh.var()) for v, n in zip(fv, neg) if n}}
            plain = {v for v, n in zip(fv, neg) if not n}
            inner = set(_formula_vars(calc, nodes, s2)) - plain
            if _verdict(pack, nodes, s2, steps, want, inner, fresh):
                cases.append((steps, [subst(x, s2) for x in nodes], frozenset(plain)))
    return cases or False


def contains(q: Pattern, p) -> bool:
    """q matches the top of p (None in q matches anything)."""
    return p is not None and q[0] == p[0] and all(qc is None or contains(qc, pc) for qc, pc in zip(q[1], p[1]))


class Atlas:
    def __init__(self, pack, max_size: int = 3):
        """Minimal nogoods up to `max_size` rule applications: a fragment containing a smaller nogood
        is not tested or stored, since matching the smaller one already prunes it."""
        self.pack, self.nogoods, self.tested = pack, [], 0
        self.guards = {}  # guarded nogood -> the cases in which it is bypassed
        self._mgis = {}   # unconditional nogood -> its most general instances (see cases)
        for n in range(1, max_size + 1):
            for p in fragments(pack.calculus, n):
                if any(self._within(q, p) for q in self.nogoods if q not in self.guards):
                    continue
                self.tested += 1
                # beyond a guarded nogood, only an unconditional one adds anything
                weaker = any(self._within(q, p) for q in self.guards)
                self._add(p, bypassed(pack, p, guarded=not weaker))
        self.depth = max((self._height(q) for q in self.nogoods), default=0)

    def _add(self, p, verdict) -> bool:
        if not verdict:
            return False
        if verdict is not True:
            self.guards[p] = verdict
        self.nogoods.append(p)
        return True

    @staticmethod
    def _height(p):
        return 0 if p is None else 1 + max((Atlas._height(c) for c in p[1]), default=0)

    def _within(self, q, p) -> bool:
        return p is not None and (contains(q, p) or any(self._within(q, c) for c in p[1]))

    def learn(self, pattern: Pattern) -> bool:
        """Add a fragment found by certification, after verifying it on its most general instance."""
        if pattern in self.nogoods or any(self._within(q, pattern) for q in self.nogoods if q not in self.guards):
            return False
        weaker = any(self._within(q, pattern) for q in self.guards)
        if not self._add(pattern, bypassed(self.pack, pattern, guarded=not weaker)):
            return False
        self.depth = max(self.depth, self._height(pattern))
        return True

    def lift(self, steps, missing, max_size: int = 5, analysis=None, nodes=None) -> Optional[Pattern]:
        """Conflict analysis: the intended derivation `steps` lost rule applications of the kinds in
        `missing` to a shortcut. When a certified analysis and ground nodes are available, extract
        actual bypass proofs and search their surrounding intended fragments, including asymmetric
        branches and ancestors within the size budget. Every proposed pattern still goes through the
        usual schematic soundness check; a concrete bypass is never itself a nogood certificate."""
        by_node = {at: (name, ids) for name, at, ids in steps}
        parent = {i: at for _, at, ids in steps for i in ids}
        kind = {n: self.pack.calculus.rules[n].family for n in self.pack.calculus.rules}

        if analysis is not None and nodes is not None and analysis.derivable:
            candidates = {}

            @lru_cache(None)
            def variants(at, budget):
                if at not in by_node or budget == 0:
                    return ((None, frozenset()),)
                name, ids = by_node[at]
                out = [((name, (None,) * len(ids)), frozenset((at,)))]
                for j, child in enumerate(ids):
                    expanded = list(out)
                    for pat, included in out:
                        remaining = budget - len(included)
                        if remaining == 0:
                            continue
                        for sub, subnodes in variants(child, remaining):
                            if sub is None:
                                continue
                            kids = list(pat[1])
                            kids[j] = sub
                            expanded.append(((name, tuple(kids)), included | subnodes))
                    out = expanded
                return tuple(out)

            calc = self.pack.calculus
            for name, at, ids in steps:
                if kind[name] not in missing:
                    continue
                planned = step_id(calc.rules[name], nodes[at], tuple(nodes[i] for i in ids))
                witness = analysis.proof_without(planned)
                if witness is None:
                    continue
                used = {f for s in witness for f in (s[2], *s[3])}
                root = at
                for _ in range(max_size):
                    for pat, included in variants(root, max_size):
                        if pat is None or at not in included:
                            continue
                        score = len({nodes[i] for i in included} & used)
                        candidates[pat] = max(candidates.get(pat, -1), score)
                    if root not in parent:
                        break
                    root = parent[root]

            for p in sorted(candidates, key=lambda p: (size(p), -candidates[p], repr(p)))[:128]:
                if self.learn(p):
                    return p

        def frag(at, depth):
            if at not in by_node or depth == 0:
                return None
            name, ids = by_node[at]
            return (name, tuple(frag(i, depth - 1) for i in ids))
        candidates = set()
        for at, (name, _) in by_node.items():
            if kind[name] not in missing:
                continue
            root = at
            for up in range(3):
                for depth in range(1, 4):
                    p = frag(root, depth + up)
                    if p is not None and size(p) <= max_size:
                        candidates.add(p)
                if root not in parent:
                    break
                root = parent[root]
        for p in sorted(candidates, key=lambda p: (size(p), repr(p))):
            if self.learn(p):
                return p
        return None

    # --- pruning partial derivations (build.derive's `forbid`) ---------------------------------------

    def violated(self, steps, node, nodes=None, s=None) -> bool:
        """Does the partial derivation `steps` contain a nogood rooted at `node` or one of its
        ancestors (up to the atlas depth)? Called after each placement, with the node it expanded.
        Guarded nogoods also need the derivation's formulas (`nodes` under substitution `s`); a slot
        still open counts as plain."""
        by_node = {at: (name, ids) for name, at, ids in steps}
        parent = {i: at for _, at, ids in steps for i in ids}
        root, roots = node, [node]
        for _ in range(self.depth - 1):
            if root not in parent:
                break
            root = parent[root]
            roots.append(root)

        def match(q, at):
            if at not in by_node:
                return False
            name, ids = by_node[at]
            return q[0] == name and all(c is None or match(c, i) for c, i in zip(q[1], ids))

        def hit(q, r):
            if not match(q, r):
                return False
            if nodes is None:  # structure only: unconditional nogoods match, guarded ones cannot be checked
                return q not in self.guards
            return any(self._satisfies(case, by_node, r, nodes, s or {}) for case in self.cases(q))
        return any(hit(q, r) for r in roots for q in self.nogoods)

    def cases(self, q) -> list:
        """The schematic cases a placement must instantiate for nogood q to apply: its guards, or (for
        an unconditional nogood) its most general instances, with no plain-variable condition."""
        if q in self.guards:
            return self.guards[q]
        if q not in self._mgis:
            self._mgis[q] = [(steps, [subst(x, s) for x in nodes], frozenset())
                             for nodes, s, steps in instances(self.pack.calculus, q, Fresh("m"))]
        return self._mgis[q]

    def _satisfies(self, case, by_node, root, nodes, s) -> bool:
        """Is the matched fragment an instance of this case: its formulas match the case's under an
        injective binding of the case's metavariables, plain ones bound to non-negations?"""
        steps, pattern_nodes, plain = case
        mine = {at: ids for _, at, ids in steps}
        pairs, todo = [(0, root)], [(0, root)]
        while todo:
            m, d = todo.pop()
            if m in mine:
                for mi, di in zip(mine[m], by_node[d][1]):
                    pairs.append((mi, di))
                    todo.append((mi, di))
        unordered = self.pack.calculus.unordered

        def go(k, b):
            if k == len(pairs):  # injective (the soundness theorem's condition), plain variables plain
                return len(set(b.values())) == len(b) and \
                    all(not (isinstance(b.get(v), tuple) and b[v][0] == "not") for v in plain)
            m, d = pairs[k]
            return any(go(k + 1, b2) for b2 in matches(pattern_nodes[m], subst(nodes[d], s), b, unordered))
        return go(0, {})

    def describe(self, p) -> str:
        return _describe(p) + (" [guarded]" if p in self.guards else "")


def _describe(p) -> str:
    if p is None:
        return "_"
    return p[0] if all(c is None for c in p[1]) else f"{p[0]}({', '.join(_describe(c) for c in p[1])})"


_ATLASES = {}


def atlas_for(pack, max_size: int = 3) -> Atlas:
    """The pack's atlas, computed once per process (about a second)."""
    key = (pack.name, max_size)
    if key not in _ATLASES:
        _ATLASES[key] = Atlas(pack, max_size)
    return _ATLASES[key]


# --- certificates -------------------------------------------------------------------------------------

def to_json(f):
    return [*map(to_json, f)] if isinstance(f, tuple) else f


def certificate_fields(pack, premises, goal, universe) -> dict:
    """What every semforge.certcheck certificate holds: the calculus, premises, goal and universe."""
    calc = pack.calculus
    return {"pack": pack.name,
            "rules": [{"name": r.name, "kind": r.family, "premises": to_json(r.premises), "conclusion": to_json(r.conclusion)}
                      for r in calc.rules.values()],
            "unordered": {h: [list(g) for g in gs] for h, gs in calc.unordered.items()},
            "universe": {"literals": sorted(map(to_json, universe.literals), key=repr), "closed": sorted(universe.closed)},
            "premises": to_json(tuple(premises)), "goal": to_json(goal),
            # constants of the premises too: a neutral certificate's premises include the withheld ones, whose
            # constants (e.g. a number) may appear nowhere in the visible problem's universe
            "background": to_json(tuple(sorted(calc.background_set.facts(
                _constants(universe) | {a for f in (*premises, goal) for a in atoms(f)}), key=repr)))}


def _constants(universe) -> set:
    """Constants of the universe's formulas: the arguments interpreted facts may take in a derivation."""
    return {a for f in universe.literals for a in atoms(f)}


def _steps_json(ids) -> list:
    """Step ids as certificate steps, each after the steps producing its parents."""
    made, out, done = {x[2]: x for x in ids}, [], set()

    def visit(x):
        if x not in done:
            done.add(x)
            for p in x[3]:
                if p in made:
                    visit(made[p])
            out.append(x)
    for x in ids:
        visit(x)
    return [{"kind": x[1], "head": to_json(x[2]), "parents": to_json(x[3])} for x in out]


def nogood_certificates(atlas, q) -> list:
    """One certificate per most general instance (or guard case) of nogood q, for semforge.certcheck:
    the planned derivation, a planned step, and a proof of the same goal avoiding it. None in place of a
    certificate if no planned step has a bypassing proof (the nogood would be wrong on that instance)."""
    pack, calc, fresh = atlas.pack, atlas.pack.calculus, Fresh("a")
    if q in atlas.guards:
        items = [(nodes, {}, steps, set(_formula_vars(calc, nodes, {})) - plain) for steps, nodes, plain in atlas.guards[q]]
    else:
        items = [(nodes, s, steps, _formula_vars(calc, nodes, s)) for nodes, s, steps in instances(calc, q, fresh)]
    out = []
    for nodes, s, steps, schematic in items:
        names = {v: fresh.atom() for v in sorted(set().union(*(variables(subst(x, s)) for x in nodes)))}
        g = [calc.canonical(subst(subst(x, s), names)) for x in nodes]
        derived = {at for _, at, _ in steps}
        premises = [g[i] for i in range(len(g)) if i not in derived]
        universe = Universe(g, pack.negate, pack.closed, {names[v] for v in schematic})
        a = analyze(calc, premises, g[0], universe)
        planned = [step_id(calc.rules[n], g[at], tuple(g[i] for i in ids)) for n, at, ids in steps]
        cert = None
        for p in planned:
            bypass = a.proof_without(p) if a.derivable else None
            if bypass is not None:
                cert = {**certificate_fields(pack, premises, g[0], universe), "nogood": _describe(q),
                        "planned": _steps_json(planned), "avoids": _steps_json([p])[0], "proof": _steps_json(bypass)}
                break
        out.append(cert)
    return out
