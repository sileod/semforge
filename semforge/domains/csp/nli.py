"""reasoning_core CSPs as certified three-way entailment, with justification patterns.

Premises are the displayed clues; base constraints (domains, all-different) are background, shown as
the preamble and always available. The hypothesis is `var = value`:
  entailment     the clues force the value (every other value is refuted)
  contradiction  the clues refute the value
  neutral        the value is possible but not forced
Labels and justifications come from z3 over guard literals (semforge.csp). The pattern of an E/C
instance is the multiset of clue operators common to all minimal justifications; `rung` records
whether local propagation alone decides it ("propagation") or it needs search ("search").
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ...core import patterns as P
from .oracles import forces_value, refutes

ANSWER = {"entailment": "Yes", "contradiction": "No", "neutral": "Maybe"}


def _U():
    from reasoning_core.tasks import _csp_utils as U
    return U


@dataclass
class CSPInstance:
    family: str
    world: object
    base: list
    clues: list
    var: object
    value: int
    renderer: object
    meta: dict = field(default_factory=dict)


def solver_of(inst):
    U = _U()
    return U.CSPSolver(inst.world.variables if hasattr(inst.world, "variables") else
                       U.get_family(inst.family).variables(inst.world), inst.base)


def label(inst) -> Optional[str]:
    s = solver_of(inst)
    values = s.possible_values(inst.var, inst.clues)
    if not values:
        return None
    if inst.value not in values:
        return "contradiction"
    return "entailment" if len(values) == 1 else "neutral"


def oracle(inst, lab=None):
    U = _U()
    lab = lab or label(inst)
    s = solver_of(inst)
    if lab == "entailment":
        return forces_value(s, inst.var, inst.value, pool=inst.clues)
    if lab == "contradiction":
        return refutes(s, [U.Eq(inst.var, inst.value)])
    return None


def kind(clue) -> str:
    return _U().operator_name(clue)


def rung(inst, lab=None) -> str:
    """'propagation' if local support propagation alone decides the label, else 'search'."""
    U = _U()
    lab = lab or label(inst)
    variables = list(solver_of(inst).variables) if hasattr(solver_of(inst), "variables") else list(inst.world.variables)
    domains, _, _ = U.propagate(variables, [*inst.base, *inst.clues])
    d = domains[inst.var]
    if lab == "entailment" and d == [inst.value]:
        return "propagation"
    if lab == "contradiction" and inst.value not in d:
        return "propagation"
    return "search"


def pattern(inst, limit=64):
    lab = label(inst)
    o = oracle(inst, lab)
    if o is None:
        return None
    return P.pattern(inst.clues, o, kind, limit)


def render(inst):
    U = _U()
    fam = U.get_family(inst.family)
    pre = "\n".join(filter(None, (fam.domains_text(inst.world), fam.invariant(inst.world))))
    clues = "\n".join(c.render(inst.renderer) for c in inst.clues)
    return (pre + "\n" + clues).strip(), U.Eq(inst.var, inst.value).render(inst.renderer) + "."


def from_generated(g, rng: random.Random, want: Optional[str] = None):
    """A hypothesis over a generated instance; `want` picks the label (None: any)."""
    U = _U()
    inst = CSPInstance(g.family, g.world, list(g.base), list(g.clues), g.query.var, g.query.answer, g.renderer)
    s = solver_of(inst)
    options = []
    for v in U.get_family(g.family).variables(g.world):
        values = s.possible_values(v, inst.clues)
        for x in v.domain:
            lab = ("entailment" if values == [x] else "neutral") if x in values else "contradiction"
            options.append((lab, v, x))
    if want:
        options = [o for o in options if o[0] == want]
    if not options:
        return None
    lab, v, x = rng.choice(options)
    inst.var, inst.value = v, x
    inst.meta = {"label": lab}
    return inst


def generate(family, rng, level=0, **kw):
    U = _U()
    from reasoning_core.tasks.constraint_satisfaction import ConstraintSatisfactionConfig as C
    c = C()
    c.set_level(level) if hasattr(c, "set_level") else c.apply_difficulty(level)
    return U.generate_instance(family, rng, max(3, int(c.n_vars) + (1 if family != "numeric" else 0)),
                               max_tries=int(c.max_tries), n_orders=int(c.minimization_orders),
                               max_domain=max(2, int(c.max_domain)), coef_bound=int(c.coef_bound),
                               difficulty=level, n_constraints=int(c.n_constraints),
                               max_arity=int(c.max_arity), **kw)
