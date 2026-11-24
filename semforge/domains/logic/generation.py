"""logic_depth adapters: conflict provenance and repair-instead-of-reject for sampled theories.

`generate_case` discards a theory as soon as the chase finds a contradiction; at level 6 that is two
draws in three. The provenance chase names every minimal premise set that derives a contradiction
(complementary atoms, or a denial body). Removing a minimum hitting set of those sets, never touching
background axioms or the planted backbone, yields a consistent theory that keeps everything else.
"""
from __future__ import annotations

import random
from contextlib import contextmanager

import z3

from ...core.provenance import IncompleteProvenance, provenance_chase

PROTECTED_RULE_NAMES = {"backbone", "spatial", "kin"}


def _L():
    from reasoning_core.tasks import logic_depth as L
    return L


def conflict_sets(theory, protected=PROTECTED_RULE_NAMES, cap=256):
    """Inclusion-minimal sets of removable premises (facts and sampled rules) that derive a contradiction."""
    L = _L()
    false = L.Atom("__false__", ())
    rules = list(theory.rules) + [L.Rule(d.body, false) for d in theory.denials]

    def pid(x):
        if isinstance(x, L.Rule) and (x.name in protected or x.head == false):
            return None
        return x

    prov = provenance_chase(theory.facts, rules, L._rule_instances, pid, cap)
    if prov.truncated:
        raise IncompleteProvenance("conflict enumeration was capped")
    sets = list(prov.supports.get(false, {}))
    for a, sa in prov.supports.items():
        if a.sign and a.pred != "__false__" and L.opposite(a) in prov.supports:
            sets += [s | t for s in sa for t in prov.supports[L.opposite(a)]]
    sets = set(sets)
    return [s for s in sets if not any(t < s for t in sets)]


def repair(theory, rng=random, rounds=3):
    """Delete a minimum hitting set of the conflicts; None if a conflict uses only protected premises."""
    L = _L()
    for _ in range(rounds):
        try:
            conflicts = conflict_sets(theory)
        except IncompleteProvenance:
            return None
        if not conflicts:
            return theory
        if any(not c for c in conflicts):
            return None
        items = sorted(set().union(*conflicts), key=repr)
        var = {x: z3.Bool(f"d{i}") for i, x in enumerate(items)}
        opt = z3.Optimize()
        for c in conflicts:
            opt.add(z3.Or([var[x] for x in c]))
        # Equal-size hitting sets are broken at random so no premise kind is systematically kept.
        opt.minimize(z3.Sum([z3.If(var[x], 1, 0) for x in items]))
        opt.minimize(z3.Sum([z3.If(var[x], rng.randrange(10), 0) for x in items]))
        opt.check()
        m = opt.model()
        drop = {x for x in items if z3.is_true(m.eval(var[x], model_completion=True))}
        theory = L.Theory([a for a in theory.facts if a not in drop], [r for r in theory.rules if r not in drop],
                          theory.denials, theory.pred_sigs, theory.entities, theory.domain_pack)
    return theory if not L.chase(theory).inconsistent else None


@contextmanager
def repaired_sampling(stats=None):
    """Patch logic_depth.sample_theory so every draw is repaired to consistency before the usual pipeline."""
    L = _L()
    original = L.sample_theory

    def sample(cfg):
        theory = original(cfg)
        fixed = repair(theory)
        if stats is not None:
            removed = len(theory.facts) + len(theory.rules) - (
                len(fixed.facts) + len(fixed.rules) if fixed else 0)
            stats.setdefault("draws", 0)
            stats["draws"] += 1
            stats.setdefault("repaired", 0)
            stats["repaired"] += fixed is not None and removed > 0
            stats.setdefault("removed", []).append(removed if fixed else None)
        return fixed if fixed is not None else theory

    L.sample_theory = sample
    try:
        yield
    finally:
        L.sample_theory = original


def delta_inconsistent(theory, res, extra):
    """Whether closure(theory) plus `extra` becomes inconsistent, chasing only from the new atoms.

    The chase is monotone, so closure(T + extra) is the fixpoint reached from closure(T) + extra, and any
    new rule instance or denial hit must use at least one new atom. Equivalent to
    close_with(theory, extra).inconsistent for a consistent `res`, without re-deriving closure(T).
    """
    L = _L()
    closure = set(res.closure)
    index = {}
    for a in closure:
        index.setdefault((a.pred, a.sign), []).append(a)
    false = L.Atom("__false__", ())
    rules = list(theory.rules) + [L.Rule(d.body, false) for d in theory.denials]
    agenda = [a for a in extra if a not in closure]
    for a in agenda:
        closure.add(a)
        index.setdefault((a.pred, a.sign), []).append(a)

    def instances(rule, new):
        atoms = [b for b in rule.body if isinstance(b, L.Atom)]
        checks = [b for b in rule.body if not isinstance(b, L.Atom)]

        def rec(i, env, pinned):
            if i == len(atoms):
                if any(c[0] == "!=" and env.get(c[1]) == env.get(c[2]) for c in checks):
                    return
                yield L._subst(rule.head, env)
                return
            if i == pinned:
                yield from rec(i + 1, env, pinned)
                return
            for fact in index.get((atoms[i].pred, atoms[i].sign), ()):
                env2 = L._match(atoms[i], fact, env)
                if env2 is not None:
                    yield from rec(i + 1, env2, pinned)

        for j, pat in enumerate(atoms):
            env = L._match(pat, new, {})
            if env is not None:
                yield from rec(0, env, j)

    while agenda:
        new = agenda.pop()
        if L.opposite(new) in closure:
            return True
        for rule in rules:
            for head in instances(rule, new):
                if head == false:
                    return True
                if any(L.is_var(x) for x in head.args) or head in closure:
                    continue
                closure.add(head)
                index.setdefault((head.pred, head.sign), []).append(head)
                agenda.append(head)
    return False


@contextmanager
def delta_neutral_checks(verify=False):
    """Route choose_example's consistency probes through delta_inconsistent; other close_with calls are untouched."""
    L = _L()
    original_choose, original_close = L.choose_example, L.close_with
    current = {}

    class _Probe:
        def __init__(self, inconsistent): self.inconsistent = inconsistent

    def close_with(theory, extra):
        if current.get("theory") is theory:
            fast = delta_inconsistent(theory, current["res"], extra)
            if verify:
                assert fast == original_close(theory, extra).inconsistent
            return _Probe(fast)
        return original_close(theory, extra)

    def choose_example(theory, res, *a, **k):
        current.update(theory=theory, res=res)
        try:
            return original_choose(theory, res, *a, **k)
        finally:
            current.clear()

    L.choose_example, L.close_with = choose_example, close_with
    try:
        yield
    finally:
        L.choose_example, L.close_with = original_choose, original_close
