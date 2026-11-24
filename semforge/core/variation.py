"""Certified semantic variation: an instance together with variants whose answers are verified.

A variant is an edited instance plus the answer a complete oracle assigns to it. Invariant variants
must keep the answer (renaming, reordering, distractors); twins are minimal edits that must change
it. Operators propose edits; nothing is kept unless the oracle confirms the intended relation, so an
orbit is a set of certified claims about which edits matter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Variant:
    name: str            # operator, e.g. "rename", "distractor:near_miss_1", "twin:remove_necessary"
    kind: str            # "base" | "invariant" | "twin"
    instance: Any
    answer: str
    meta: dict = field(default_factory=dict)


@dataclass
class Orbit:
    base: Variant
    variants: list

    def all(self):
        return [self.base, *self.variants]


def certify(base: Variant, edited, operator: str, kind: str, oracle: Callable[[Any], Optional[str]],
            meta=None) -> Optional[Variant]:
    """Keep `edited` only if the oracle's answer stands in the intended relation to the base answer."""
    if edited is None:
        return None
    answer = oracle(edited)
    if answer is None:
        return None
    if kind == "invariant" and answer != base.answer:
        return None
    if kind == "twin" and answer == base.answer:
        return None
    return Variant(operator, kind, edited, answer, meta or {})
