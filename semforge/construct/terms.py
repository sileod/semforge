"""Formulas as nested tuples, with metavariables and unification.

A formula is an atom (a string not starting with "?") or a tuple (connective, *arguments). Strings
starting with "?" are metavariables, used in rule schemas and in partially built derivations.
Connectives are free: a pack decides which ones exist and what they mean.
"""
from __future__ import annotations

from itertools import permutations, product
from typing import Iterator, Optional, Union

Formula = Union[str, tuple]
Subst = dict


def is_var(x) -> bool:
    return isinstance(x, str) and x.startswith("?")


def walk(x, s: Subst):
    while is_var(x) and x in s:
        x = s[x]
    return x


def subst(x, s: Subst):
    x = walk(x, s)
    return tuple(subst(a, s) if i else a for i, a in enumerate(x)) if isinstance(x, tuple) else x


def variables(x) -> set:
    if is_var(x):
        return {x}
    return set().union(*(variables(a) for a in x[1:])) if isinstance(x, tuple) else set()


def is_ground(x) -> bool:
    return not variables(x)


def _occurs(v, x, s) -> bool:
    x = walk(x, s)
    return x == v or (isinstance(x, tuple) and any(_occurs(v, a, s) for a in x[1:]))


def _orders(x, unordered):
    """x with its unordered argument groups permuted in every way (x itself first)."""
    groups = (unordered or {}).get(x[0], ())
    if not groups:
        yield x
        return
    pools = [permutations([x[i] for i in g]) for g in groups]
    for choice in product(*pools):
        y = list(x)
        for g, vals in zip(groups, choice):
            for i, v in zip(g, vals):
                y[i] = v
        yield tuple(y)


def matches(p, f, s: Subst, unordered: Optional[dict] = None) -> Iterator[Subst]:
    """One-way matching of a pattern against a ground formula: every extension of s with
    subst(p, s') == f, modulo `unordered` argument groups (as in `unifiers`, but no occurs check and
    no variables on the right, which makes it the fast path of the chase)."""
    if is_var(p):
        if p in s:
            if s[p] == f:
                yield s
        else:
            yield {**s, p: f}
        return
    if not isinstance(p, tuple):
        if p == f:
            yield s
        return
    if not (isinstance(f, tuple) and len(p) == len(f) and p[0] == f[0]):
        return
    if unordered and p[0] in unordered:
        seen = []
        for ff in _orders(f, unordered):
            for s2 in _match_args(p, ff, 1, s, unordered):
                if s2 not in seen:
                    seen.append(s2)
                    yield s2
    else:
        yield from _match_args(p, f, 1, s, unordered)


def _match_args(p, f, i, s, unordered):
    if i == len(p):
        yield s
        return
    for s2 in matches(p[i], f[i], s, unordered):
        yield from _match_args(p, f, i + 1, s2, unordered)


def unifiers(a, b, s: Optional[Subst] = None, unordered: Optional[dict] = None) -> Iterator[Subst]:
    """Every most general unifier extending s; `unordered` maps a head symbol to groups of argument
    positions whose order does not matter (symmetric relations, sets), unified modulo permutation."""
    s = {} if s is None else s
    a, b = walk(a, s), walk(b, s)
    if a == b:
        yield s
        return
    if is_var(a):
        if not _occurs(a, b, s):
            yield {**s, a: b}
        return
    if is_var(b):
        if not _occurs(b, a, s):
            yield {**s, b: a}
        return
    if not (isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b) and a[0] == b[0]):
        return
    seen = []
    for bb in _orders(b, unordered):
        for r in _unify_args(a[1:], bb[1:], s, unordered):
            if r not in seen:
                seen.append(r)
                yield r


def _unify_args(xs, ys, s, unordered):
    if not xs:
        yield s
        return
    for s2 in unifiers(xs[0], ys[0], s, unordered):
        yield from _unify_args(xs[1:], ys[1:], s2, unordered)


def unify(a, b, s: Optional[Subst] = None, unordered: Optional[dict] = None) -> Optional[Subst]:
    """A most general unifier extending s, or None."""
    return next(unifiers(a, b, s, unordered), None)


def canonical(x, unordered: Optional[dict] = None):
    """Ground formula with unordered argument groups sorted, so equal-modulo-symmetry means equal."""
    if not isinstance(x, tuple):
        return x
    y = [x[0], *(canonical(a, unordered) for a in x[1:])]
    for g in (unordered or {}).get(x[0], ()):
        for i, v in zip(g, sorted((y[i] for i in g), key=repr)):
            y[i] = v
    return tuple(y)


def rename(x, tag: str):
    """Rename metavariables apart: ?A -> ?A#tag."""
    if is_var(x):
        return f"{x}#{tag}"
    return tuple(rename(a, tag) if i else a for i, a in enumerate(x)) if isinstance(x, tuple) else x


def subformulas(x) -> Iterator[Formula]:
    yield x
    if isinstance(x, tuple):
        for a in x[1:]:
            yield from subformulas(a)


def atoms(x) -> set:
    return {f for f in subformulas(x) if isinstance(f, str)}


def size(x) -> int:
    return 1 + sum(size(a) for a in x[1:]) if isinstance(x, tuple) else 1
