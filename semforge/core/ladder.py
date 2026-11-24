"""Reasoner ladders: difficulty as the weakest reasoner that decides an instance.

A ladder is an ordered list of sound but incomplete reasoners, from surface heuristics to the
complete oracle. Each returns an answer or None (abstains). An instance's rung is the first sound
reasoner that answers correctly; an instance whose rung is below the intended one is a shortcut
for that reasoner. Sound reasoners never answer wrongly; heuristics may, which is the point of
including them. The order is supplied by the caller; it does not imply that
restricted reasoners have nested capabilities.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Sequence


@dataclass(frozen=True)
class Reasoner:
    name: str
    decide: Callable[[object], Optional[str]]
    sound: bool = True


class Ladder:
    def __init__(self, reasoners: Sequence[Reasoner]):
        self.reasoners = list(reasoners)

    def trace(self, instance) -> list:
        return [(r.name, r.decide(instance)) for r in self.reasoners]

    def rung(self, instance, gold: str) -> Optional[str]:
        """First sound reasoner answering correctly in the supplied order, or None."""
        for r in self.reasoners:
            if r.sound and r.decide(instance) == gold:
                return r.name
        return None

    def audit(self, instances, golds) -> dict:
        """Per reasoner: coverage, accuracy when answering, and the distribution of rungs."""
        instances, golds = list(instances), list(golds)
        if not golds or len(instances) != len(golds):
            raise ValueError("instances and golds must have equal, nonzero lengths")
        out = {}
        for r in self.reasoners:
            answers = [r.decide(x) for x in instances]
            answered = [(a, g) for a, g in zip(answers, golds) if a is not None]
            out[r.name] = {
                "sound": r.sound,
                "coverage": len(answered) / len(golds),
                "accuracy_when_answering": (sum(a == g for a, g in answered) / len(answered)) if answered else None,
                "accuracy_overall": sum(a == g for a, g in zip(answers, golds)) / len(golds),
            }
        out["rungs"] = dict(Counter(self.rung(x, g) for x, g in zip(instances, golds)))
        return out
