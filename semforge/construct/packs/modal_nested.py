"""Nested modal logic K (unlabelled): necessity and possibility over arbitrary formulas, any depth.

"It must be that (it might be that the bridge is open)". Unlike the labelled `modal` pack, formulas carry
no world names: they are evaluated at a world (`Signature.world`), boxes and diamonds take formula
arguments, and entailment is local consequence at the actual world under the standard translation. Rules
are the premise-bearing K principles (necessitation, which needs theoremhood, is absent). Free formula
slots are filled with atoms or modal formulas over atoms (nesting 1), so rules apply at depth; deeper
leaves make z3's quantifier instantiation the bottleneck (0.35 s per 30-step request at nesting 1).
`modal_nested_s4` adds reflexive, transitive accessibility with the T and 4 rules.
"""
from __future__ import annotations

import z3

from ..pack import declare
from ..signature import Signature, Sort, Symbol, rule
from .propositional import facts


def access(m):
    return m.fn("access", "World", "World", "Prop")


def necessarily(m, p, w):
    x = m.var("World")
    return z3.ForAll([x], z3.Implies(access(m)(w, x), p(x)))


def possibly(m, p, w):
    x = m.var("World")
    return z3.Exists([x], z3.And(access(m)(w, x), p(x)))


SIGNATURE = Signature.of(
    sorts=[Sort("World", (), distinct=False)],
    symbols=[
        Symbol("box", ("Prop",), ("it must be that {0}", "necessarily, {0}"), (), necessarily),
        Symbol("dia", ("Prop",), ("it might be that {0}", "possibly, {0}"), (), possibly),
    ],
    prop_lexicon=facts,
    world="World",
    nesting=1,
)

RULES = [
    rule("box_modus_ponens", "(box (-> A B)), (box A) |- box B"),
    rule("box_conjunction", "(box A), (box B) |- box (and A B)"),
    rule("box_diamond", "(box A), (dia B) |- dia (and A B)"),
    rule("diamond_split", "(dia (and A B)) |- dia A"),
    rule("box_duality", "not (dia A) |- box (not A)"),
    rule("diamond_duality", "not (box A) |- dia (not A)"),
    rule("modus_ponens", "A, (-> A B) |- B"),
    rule("disjunctive_syllogism", "(or A B), not A |- B"),
    rule("and_elimination", "(and A B) |- A"),
]

PACK = declare("modal_nested", SIGNATURE, RULES, builds=("and",))
