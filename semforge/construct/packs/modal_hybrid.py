"""Hybrid modal logic K: nested modalities at named worlds ("in Tarn, it must be that the bell rings").

The `@` operator of hybrid logic over `modal_nested`: worlds are named constants of the world sort, `at W A`
evaluates the formula A at W, and accessibility between named worlds is a fact. Symbols take an entity
argument and a world-relative formula argument together, so no machinery is specific to this pack: the
standard translation of `modal_nested` with `at` as `A(W)`. Labels are z3's on this guarded fragment.
Rules are the K principles relativised to named worlds; diamond elimination (a fresh witness) is absent.
Rule patterns place the `at` wrappers; free slots get atoms or one modality over an atom (nesting 1).
"""
from __future__ import annotations

from ..pack import declare
from ..signature import PROP, Signature, Sort, Symbol, rule
from .modal import WORLDS
from .modal_nested import SIGNATURE as NESTED, access
from .propositional import facts

SIGNATURE = Signature.of(
    sorts=[Sort("World", WORLDS, distinct=False)],
    symbols=[
        Symbol("at", ("World", PROP), ("in {0}, {1}",), ("in {0}, it is not the case that {1}",),
               lambda m, w, p, _: p(w), leaf=False),
        Symbol("sees", ("World", "World"), ("{1} is reachable from {0}",), ("{1} is not reachable from {0}",),
               lambda m, w, v, _: access(m)(w, v), leaf=False),
        NESTED.symbols["box"],
        NESTED.symbols["dia"],
    ],
    prop_lexicon=facts,
    world="World",
    nesting=1,
)

RULES = [
    rule("necessity_elimination", "(at W (box A)), (sees W V) |- at V A"),
    rule("possibility_introduction", "(at V A), (sees W V) |- at W (dia A)"),
    rule("impossibility_elimination", "(at W (not (dia A))), (sees W V) |- at V (not A)"),
    rule("necessity_refutation", "(at V (not A)), (sees W V) |- at W (not (box A))"),
    rule("box_modus_ponens", "(at W (box (-> A B))), (at W (box A)) |- at W (box B)"),
    rule("modus_ponens", "(at W (-> A B)), (at W A) |- at W B"),
    rule("disjunctive_syllogism", "(at W (or A B)), (at W (not A)) |- at W B"),
    rule("and_elimination", "(at W (and A B)) |- at W A"),
]

PACK = declare("modal_hybrid", SIGNATURE, RULES)
