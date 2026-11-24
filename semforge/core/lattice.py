"""Premise lattices: minimal supports, necessary premises and minimal corrections.

Everything here works on a monotone property of premise sets: if `holds(S)` and S is a
subset of T, then `holds(T)`. "The premises entail the answer", "the clues force a unique
value" and "the clues refute x = 3" are all of this kind. A minimal support is an
inclusion-minimal set on which the property holds (a MUS when the property is
unsatisfiability); a minimal correction is an inclusion-minimal set whose removal breaks it
(an MCS). Necessary premises are the ones in every minimal support. The two families are
minimal hitting sets of each other, which is what the enumeration below exploits.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import z3


class Oracle:
    """A monotone property over frozensets of premises, with call accounting.

    `core(S)`, when given, returns a subset of S on which the property still holds (for
    instance the premises in an unsat core), or None when it does not hold on S. Core-guided
    algorithms use it to drop many premises per call.
    """

    def __init__(self, holds: Callable[[frozenset], bool],
                 core: Optional[Callable[[frozenset], Optional[frozenset]]] = None):
        self._holds, self._core = holds, core
        self.calls = 0
        self._cache = {}

    def holds(self, s) -> bool:
        s = frozenset(s)
        if s not in self._cache:
            self.calls += 1
            self._cache[s] = bool(self._holds(s))
        return self._cache[s]

    def core(self, s) -> Optional[frozenset]:
        s = frozenset(s)
        if self._core is None:
            return s if self.holds(s) else None
        if self._cache.get(s) is False:
            return None
        self.calls += 1
        c = self._core(s)
        self._cache[s] = c is not None
        if c is not None:
            self._cache[frozenset(c)] = True
        return None if c is None else frozenset(c)


def deletion(items: Iterable, oracle: Oracle, use_core=True) -> Optional[frozenset]:
    """One minimal support by linear deletion; with cores, each success also drops what the core omits."""
    order = list(items)
    current = oracle.core(order) if use_core else (frozenset(order) if oracle.holds(order) else None)
    if current is None:
        return None
    for x in order:
        if x not in current:
            continue
        trial = current - {x}
        if use_core:
            c = oracle.core(trial)
            if c is not None:
                current = c
        elif oracle.holds(trial):
            current = trial
    return current


def quickxplain(items: Iterable, oracle: Oracle, background=frozenset()) -> Optional[frozenset]:
    """One minimal support (Junker 2004): O(k log(n/k)) checks for a support of size k among n."""
    items = list(items)
    background = frozenset(background)
    if not oracle.holds(background | set(items)):
        return None
    if oracle.holds(background):
        return frozenset()

    def qx(b, delta, c):
        if delta and oracle.holds(b):
            return frozenset()
        if len(c) == 1:
            return frozenset(c)
        c1, c2 = c[:len(c) // 2], c[len(c) // 2:]
        d2 = qx(b | set(c1), bool(c1), c2)
        d1 = qx(b | d2, bool(d2), c1)
        return d1 | d2

    return qx(background, False, items)


def necessary(items: Iterable, oracle: Oracle, support: Optional[frozenset] = None) -> frozenset:
    """Premises in every minimal support. Only members of one support can qualify, so k checks, not n."""
    items = frozenset(items)
    support = quickxplain(items, oracle) if support is None else support
    if support is None:
        return frozenset()
    return frozenset(x for x in support if not oracle.holds(items - {x}))


def enumerate_lattice(items: Iterable, oracle: Oracle, limit=None, shrink=quickxplain):
    """MARCO (Liffiton et al.): enumerate minimal supports and minimal corrections together.

    Returns (supports, corrections, complete). With complete=True the lists are exhaustive, so the
    smallest support is the true minimum and the necessary premises are the intersection of all of
    them. `limit` bounds the number of supports.
    """
    items = list(items)
    var = {x: z3.Bool(f"m{i}") for i, x in enumerate(items)}
    seeds = z3.Solver()
    supports, corrections = [], []
    while limit is None or len(supports) < limit:
        if seeds.check() != z3.sat:
            return supports, corrections, True
        model = seeds.model()
        seed = {x for x in items if z3.is_true(model.eval(var[x], model_completion=True))}
        if oracle.holds(seed):
            s = shrink(seed, oracle)
            supports.append(s)
            seeds.add(z3.Or([z3.Not(var[x]) for x in s]) if s else z3.BoolVal(False))
        else:
            for x in items:
                if x not in seed and not oracle.holds(seed | {x}):
                    seed.add(x)
            mcs = frozenset(items) - seed
            corrections.append(mcs)
            seeds.add(z3.Or([var[x] for x in mcs]) if mcs else z3.BoolVal(False))
    return supports, corrections, False


def minimum_support(items: Iterable, oracle: Oracle) -> Optional[frozenset]:
    """A cardinality-minimum support, as a minimum hitting set of the minimal corrections (implicit hitting sets)."""
    items = list(items)
    if not oracle.holds(items):
        return None
    if oracle.holds(frozenset()):
        return frozenset()
    var = {x: z3.Bool(f"h{i}") for i, x in enumerate(items)}
    corrections = []
    while True:
        opt = z3.Optimize()
        for mcs in corrections:
            opt.add(z3.Or([var[x] for x in mcs]))
        opt.minimize(z3.Sum([z3.If(var[x], 1, 0) for x in items]))
        opt.check()
        model = opt.model()
        hit = {x for x in items if z3.is_true(model.eval(var[x], model_completion=True))}
        if oracle.holds(hit):
            return frozenset(hit)
        for x in items:
            if x not in hit and not oracle.holds(hit | {x}):
                hit.add(x)
        corrections.append(frozenset(items) - hit)
