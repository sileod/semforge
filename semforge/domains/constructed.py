"""Target-conditioned constructed problems (semforge.construct) as a semforge domain.

Strategies (domain constructed_<pack>[_<strategy>]):
  plan     (default) the plan archive's least-covered niche, synthesized and certified;
  shape    the topology archive: the least-covered (label, spine, joins) cell and kind set, inverted into
           a shape target and constructed under the shortcut atlas;
  forward  FLD-style random proof trees with distractors (construct.forward), nothing certified at
           generation time: the causal control for the same calculus, realizer and pipeline. Justifications are the calculus' minimal proofs (premises and rule applications); the kind of
a step is its rule name and premises have no kind, so the dependency pattern of a row is the multiset of
rule applications every proof makes (exact relative to the pack's calculus). Labels come from the pack's
complete semantics. Annotations keep the intended target next to the realized certificate (E/C) or the
counterfactual one (N), never in the same field.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace

from ..construct.archive import Archive, TopologyArchive
from ..construct.atlas import to_json
from ..construct.forward import forward
from ..construct.oracle import Undecided
from ..construct.packs import get as get_pack
from ..construct.calculus import analyze, rule_of
from ..construct.synth import Synthesizer
from .base import Domain


class ConstructedDomain(Domain):
    def __init__(self, pack: str = "propositional", strict: bool = True, strategy: str = "plan"):
        self.pack, self.strategy = get_pack(pack), strategy
        self.name = f"constructed_{self.pack.name}" + ("" if strategy == "plan" else f"_{strategy}")
        self.synth = Synthesizer(self.pack, strict=strict)
        if strategy == "plan":
            self.archive = Archive(self.pack.calculus.rules, sampler=self.synth.sample_plan,
                                   kinds=self.pack.calculus.kinds)
        elif strategy == "shape":
            self.archive = TopologyArchive(calculus=self.pack.calculus)
        elif strategy != "forward":
            raise KeyError(f"unknown strategy {strategy!r}")

    def sample(self, rng, level):
        try:
            return self._sample(rng, level)
        except Undecided:  # z3 undecided (quantifiers): no certifiable label, dropped
            return []

    def _sample(self, rng, level):
        if self.strategy == "forward":  # 1-20 planned steps, as the shape archive's cells
            out = forward(self.pack, rng, rng.randint(1, 20))
            inst = out and out[0]
            self.planned = out and len(out[1].steps)  # the intended derivation's size, uncertified
        else:
            target = self.archive.propose(rng) if self.strategy == "shape" else self.archive.propose(rng, level=1 + level)
            inst = self.synth.realize(target, rng)
            if self.strategy == "shape" or inst is not None:
                self.archive.add(inst)
        if inst is None or not self._renders(inst):
            return []
        return [(inst, inst.label)]

    def _renders(self, inst) -> bool:
        try:  # more constants of a sort than its lexicon has words: dropped, whatever the strategy
            self.render(inst)
            return True
        except ValueError:
            return False

    def family(self, inst):
        return self.pack.name

    def label(self, inst):
        return self.pack.label(inst.premises, inst.hypothesis)

    def justifications(self, inst, label=None):
        goal = inst.hypothesis if label == "entailment" else self.pack.negate(inst.hypothesis)
        a = analyze(self.pack.calculus, inst.premises, goal, inst.universe)
        return (a.proofs, a.exact) if a.derivable else ([], a.exact)

    def kind(self, x):
        return rule_of(x)

    def premises(self, inst):
        return list(inst.premises)

    def without(self, inst, drop):
        return replace(inst, premises=[p for p in inst.premises if p not in drop], analysis=None,
                       roles={p: r for p, r in inst.roles.items() if p not in drop})

    def render(self, inst):
        return self.pack.render(inst.premises, inst.hypothesis, random.Random(repr((inst.premises, inst.hypothesis))))

    def annotations(self, inst, pattern):
        t, r = inst.target, inst.realized()
        intended = t and {"required_rules": dict(sorted(t.required.items())), "plan": list(t.rules),
                          "proofs": t.proofs, "min_depth": t.min_depth}
        post = None  # the same post-hoc certificate for every strategy: certified depth and topology
        if inst.label != "neutral":
            goal = inst.hypothesis if inst.label == "entailment" else self.pack.negate(inst.hypothesis)
            a = analyze(self.pack.calculus, inst.premises, goal, inst.universe)
            post = {"depth": a.depth, **a.topology, "mandatory": len(a.mandatory_steps)} if a.derivable else None
        said = self.pack.surface(inst.hypothesis, "hypothesis")  # the equivalent form the hypothesis is worded as
        return {"intended": intended, "realized": r["realized"], "counterfactual": r["counterfactual"],
                "withheld": r["withheld"], "roles": dict(Counter(inst.roles.values())), "certified": post,
                "hypothesis_surface": to_json(said) if said != inst.hypothesis else None}
