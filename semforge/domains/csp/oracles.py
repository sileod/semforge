"""Core-guided oracles over reasoning_core's CSPSolver (guard literals on one persistent z3 solver)."""
from __future__ import annotations

from contextlib import contextmanager

import z3

from ...core.lattice import Oracle

Z3_CHECKS = {"calls": 0}  # solver checks made here, comparable to counted() on the baseline


def _unsat_core(solver, formulas):
    """Clues of an unsat core of domains + base + formulas, or None when satisfiable."""
    guards = {solver._guard(f).get_id(): f for f in formulas}
    literals = [solver._guard(f) for f in (*solver.base, *formulas)]
    z = solver._persistent()
    Z3_CHECKS["calls"] += 1
    verdict = z.check(*literals)
    if verdict == z3.unknown:
        raise RuntimeError(f"CSP solver returned unknown: {z.reason_unknown()}")
    if verdict == z3.sat:
        return None
    return frozenset(guards[l.get_id()] for l in z.unsat_core() if l.get_id() in guards)


def refutes(solver, extra) -> Oracle:
    """Clue sets that, with the base, rule out the formulas in `extra` (e.g. Eq(var, wrong_value))."""
    extra = tuple(extra)

    def core(clues):
        c = _unsat_core(solver, (*clues, *extra))
        return None if c is None else c - set(extra)

    return Oracle(lambda clues: core(clues) is not None, core)


def forces_value(solver, var, value, *, pool) -> Oracle:
    """Uniqueness oracle restricted to subsets of a validated satisfiable pool.

    Refuting every alternative means unique value only when clues are consistent.
    Validate the full pool once; all its subsets are then consistent too.
    """
    from reasoning_core.tasks._csp_utils import Eq
    pool = frozenset(pool)
    if value not in var.domain:
        raise ValueError("value is outside the variable domain")
    if _unsat_core(solver, pool) is not None:
        raise ValueError("forces_value requires a satisfiable pool")
    wrong = [Eq(var, w) for w in var.domain if w != value]

    def core(clues):
        if not clues <= pool:
            raise ValueError("clues must be a subset of the validated pool")
        out = set()
        for w in wrong:
            c = _unsat_core(solver, (*clues, w))
            if c is None:
                return None
            out |= c - {w}
        return frozenset(out)

    return Oracle(lambda clues: core(clues) is not None, core)


def counted(solver):
    """Wrap solver._check so the baseline's oracle calls can be counted like ours."""
    original = solver._check
    box = {"calls": 0}

    def check(formulas):
        box["calls"] += 1
        return original(formulas)

    solver._check = check
    return box


@contextmanager
def core_guided_minimization():
    """Trial integration for UniqueValue only; retain all downstream quality filters.

    Other objectives keep reasoning_core's implementation. This changes support
    selection and therefore distributions; it is not a seed-equivalent speed patch.
    Module patches are intended for single-threaded benchmarks, not serving.
    """
    from reasoning_core.tasks import _csp_utils as C

    original = C.minimize_for_objective

    def minimize(solver, pool, objective, rng, n_orders=6):
        if not isinstance(objective, C.UniqueValue):
            return original(solver, pool, objective, rng, n_orders)
        pool = C._stratified_pool(pool, rng)
        # Match the baseline's initial semantic check, which rejects inconsistent
        # pools as well as pools that do not fix the requested value.
        if not objective.holds(solver, pool):
            return []
        oracle = forces_value(solver, objective.var, objective.value, pool=pool)
        from ...core.lattice import deletion
        out = []
        for _ in range(n_orders):
            order = list(pool)
            rng.shuffle(order)
            order.sort(key=lambda c: (len(c.variables()) > 1, c.complexity()))
            support = deletion(order, oracle)
            if support is not None:
                out.append([c for c in order if c in support])
        return out

    C.minimize_for_objective = minimize
    try:
        yield
    finally:
        C.minimize_for_objective = original
