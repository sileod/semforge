"""A pack is everything logic-specific about construction, declared as data.

    PACK = declare("csp", signature, rules)

gives a calculus (structure), a z3 oracle compiled from the signature (labels) and a template realizer
(text); a pack may override `entails` or `render` when its semantics or language needs more (an external
prover, a grammar). New logics, embedded ones included (modal logic through its standard translation,
many-valued logics as finite domains), are new packs; nothing else changes. `semforge.construct.check`
validates a pack: rule soundness against the oracle, requireability, rendering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .calculus import Analysis, Calculus, Closure, Universe
from .oracle import z3_entails
from .realize import Realizer, classical_surface, verbatim
from .signature import CONNECTIVES, Signature, negate as _negate, parse
from .terms import matches


@dataclass(frozen=True)
class Pack:
    name: str
    calculus: Calculus
    negate: Callable          # formula -> its negation (no stacked double negations)
    entails: Callable         # (premises, formula) -> bool; must be complete: labels rest on it
    render: Callable          # (premises, hypothesis, rng) -> (premises text, hypothesis text)
    closed: tuple = ("->",)   # what the analysis universe closes under (formulas rules may build)
    signature: Optional[Signature] = None
    atlas_size: int = 3       # shortcut atlas enumerated up to this many steps (larger ones are learned)
    surface: Callable = verbatim  # (formula, "premise" | "hypothesis") -> the equivalent form rendered

    def answers(self, premises, query) -> list:
        """Every derivable instance of `query` (a formula with metavariables, or DSL text such as
        "ancestor ann X"): [(binding, analysis)], where binding maps each metavariable name to its value
        and analysis certifies the instance (depth, mandatory steps, proofs). Theory facts are not answers."""
        calc, q = self.calculus, parse(query) if isinstance(query, str) else query
        premises = [calc.canonical(p) for p in premises]
        closure = Closure(calc, premises, Universe(premises, self.negate, self.closed))
        out = []
        for f in sorted(closure.round, key=repr):
            if f in calc.background_set:
                continue
            for s in matches(q, f, {}, calc.unordered):
                out.append(({v[1:]: x for v, x in sorted(s.items())}, Analysis(calc, closure, premises, f)))
        return out

    def label(self, premises, hypothesis):
        if self.entails(premises, hypothesis):
            return "entailment"
        if self.entails(premises, self.negate(hypothesis)):
            return "contradiction"
        return "neutral"


def declare(name: str, signature: Signature, rules, builds: tuple = (), entails: Optional[Callable] = None,
            render: Optional[Callable] = None, atlas_size: int = 3,
            surface: Optional[Callable] = classical_surface) -> Pack:
    """A pack from a signature and rules. `builds`: connectives that rules construct (e.g. "->" for
    hypothetical syllogism), so the analysis universe admits them; symbol applications always are.
    `surface`: equivalent forms per role for rendering (classical by default, like the negation;
    None renders every formula as is)."""
    assert not set(builds) - set(CONNECTIVES), builds
    surface = surface or verbatim
    return Pack(name, Calculus(rules, signature), _negate, entails or z3_entails(signature),
                render or Realizer(signature, surface).render, closed=(*signature.symbols, *builds),
                signature=signature, atlas_size=atlas_size, surface=surface)
