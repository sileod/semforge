"""Nested modal S4: `modal_nested` over reflexive, transitive accessibility, with the T and 4 rules.

Free slots get plain atoms (nesting 0); nesting still arises through the rules (4 adds a box). With
modal leaves, z3 rarely settles transitive frames within its timeout (55 s per 8-step request)."""
from __future__ import annotations

from dataclasses import replace

from ..oracle import reflexive, transitive
from ..pack import declare
from ..signature import rule
from .modal_nested import RULES as K_RULES, SIGNATURE as K_SIGNATURE

THEORY = ("Whatever must be true is true.", "Whatever must be true must necessarily be true.")

SIGNATURE = replace(K_SIGNATURE, axioms=(reflexive("access", "World"), transitive("access", "World")),
                    nesting=0, theory=THEORY)

RULES = K_RULES + [
    rule("truth", "(box A) |- A"),
    rule("positive_introspection", "(box A) |- box (box A)"),
    rule("possibility", "A |- dia A"),
]

PACK = declare("modal_nested_s4", SIGNATURE, RULES, builds=("and",))
