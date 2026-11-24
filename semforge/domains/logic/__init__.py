"""reasoning_core logic_depth: Datalog with signed atoms; justifications are minimal proofs.

  generation  conflict repair and delta-chase neutral checks for logic_depth sampling
  variants    certified orbits (renaming, distractors, near misses, twins) and pattern edits
"""
from __future__ import annotations

import random

from ..base import Domain


class LogicDomain(Domain):
    name = "logic"

    def __init__(self):
        from reasoning_core.tasks import logic_depth as L
        from . import variants as V
        self.L, self.V = L, V

    def sample(self, rng, level):
        from .generation import delta_neutral_checks
        random.seed(rng.random())
        cfg, state = self.L.MultistepNLIConfig(level=level), {}
        while True:
            with delta_neutral_checks():
                case, _ = self.L.generate_case(cfg, state=state)
            if case is None:
                continue
            inst = self.V.from_case(case)
            lab = self.V.label(inst)
            if lab is not None:
                return [(inst, lab)]

    def family(self, inst):
        return getattr(inst.theory.domain_pack, "name", self.name)

    def label(self, inst):
        return self.V.label(inst)

    def justifications(self, inst, label=None):
        from ...core.provenance import IncompleteProvenance, logic_depth_provenance
        label = label or self.label(inst)
        target = self.L.target_for(label, inst.hyp) if label else None
        if target is None:
            return None, False
        prov = logic_depth_provenance(inst.theory, {x: i for i, x in enumerate(inst.order)})
        try:
            sups = prov.minimal_supports(target)
        except IncompleteProvenance:
            return None, False
        return [frozenset(inst.order[i] for i in s) for s in sups], True

    def kind(self, premise):
        if isinstance(premise, self.L.Rule):
            return f"{premise.shape}{'' if premise.head.sign else '-'}"
        return None

    def premises(self, inst):
        return list(inst.order)

    def without(self, inst, drop):
        return self.V._with(inst, facts=[a for a in inst.theory.facts if a not in drop],
                            rules=[r for r in inst.theory.rules if r not in drop],
                            order=[x for x in inst.order if x not in drop])

    def render(self, inst):
        lines, hyp = self.V.render(inst)
        return "\n".join(lines), hyp

    def expand(self, inst, label, rng, kinds_seen):
        """Proof expansion: replace a fact of the unique proof by a rule of the least-seen shape."""
        from ...core import patterns as P
        required = lambda e: (lambda q: q.signature if q else None)(P.analyze(self, e, label))
        for shape in sorted(self.V.SHAPES, key=lambda k: (kinds_seen[k], rng.random())):
            e, _ = self.V.expand_fact(inst, rng, shape, required)
            if e is not None:
                kinds_seen[shape] += 1
                return e
        return None
