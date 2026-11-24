"""reasoning_core constraint satisfaction: hypotheses `var = value`; justifications are minimal clue sets
forcing or refuting it (MARCO over z3 guard literals); rung is propagation vs search.

  oracles  core-guided monotone oracles over reasoning_core's CSPSolver
  nli      CSP instances as three-way entailment (labels, rendering, rung)
"""
from __future__ import annotations

from dataclasses import replace

from ..base import Domain


class CSPDomain(Domain):
    name = "csp"
    families = ("assignment", "graph", "grid", "numeric", "scheduling", "sets")

    def __init__(self, families=None):
        from . import nli
        self.C = nli
        self.families = tuple(families or self.families)

    def sample(self, rng, level):
        fam = rng.choice(self.families)
        for _ in range(8):
            try:
                g = self.C.generate(fam, rng, level=level)
                break
            except Exception:
                continue
        else:
            return []
        out = []
        for want in ("entailment", "contradiction", "neutral"):
            inst = self.C.from_generated(g, rng, want)
            if inst is not None:
                out.append((inst, want))
        return out

    def family(self, inst):
        return inst.family

    def label(self, inst):
        return self.C.label(inst)

    def justifications(self, inst, label=None, limit=64):
        from ...core.lattice import enumerate_lattice
        o = self.C.oracle(inst, label)
        if o is None or not o.holds(inst.clues):
            return None, False
        sups, _, complete = enumerate_lattice(inst.clues, o, limit=limit)
        return sups, complete

    def kind(self, premise):
        return self.C.kind(premise)

    def rung(self, inst, label):
        return self.C.rung(inst, label)

    def premises(self, inst):
        return list(inst.clues)

    def without(self, inst, drop):
        return replace(inst, clues=[c for c in inst.clues if c not in drop])

    def render(self, inst):
        return self.C.render(inst)
