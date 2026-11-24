"""Modal logic K, labelled: facts at named worlds, accessibility, necessity and possibility (a nest probe).

"In Norvale, the bell rings", "Tarn is reachable from Norvale", "in every situation reachable from
Norvale, the bell rings". Semantics is the standard translation (worlds an uninterpreted sort,
accessibility a binary relation), so z3 labels are exact on this guarded fragment. Modalities apply to
atomic facts only (`modal_hybrid` has nested modal formulas at named worlds). Rules are monotone
labelled deduction; diamond elimination (a fresh witness world) is out of reach of a chase and absent.
"""
from __future__ import annotations

import z3

from ..pack import declare
from ..signature import Signature, Sort, Symbol, rule

WORLDS = ["Norvale", "Tarn", "Ostrel", "Brisk", "Calder", "Fenwick", "Hollin", "Merrow", "Quarry", "Selby", "Ulm",
          "Wexley", "Yarrow", "Dunmore", "Ashby", "Pell"]
FACTS = ["the bell rings", "the bridge is open", "the lamps are lit", "the river floods", "the market is busy",
         "the gate is locked", "the tower is guarded", "the well is dry", "the ferry runs", "the mill turns",
         "the road is blocked", "the harbour is calm", "the archive is sealed", "the fields are flooded",
         "the inn is full", "the signal fires burn", "the pass is snowed in", "the garrison is awake"]


def access(m, w, v):
    return m.fn("access", "World", "World", "Prop")(w, v)


def necessarily(m, p, w):
    x = m.var("World")
    return z3.ForAll([x], z3.Implies(access(m, w, x), p(x)))


def possibly(m, p, w):
    x = m.var("World")
    return z3.Exists([x], z3.And(access(m, w, x), p(x)))


SIGNATURE = Signature.of(
    sorts=[Sort("World", WORLDS, distinct=False), Sort("Fact", FACTS, kind="predicate", over="World")],
    symbols=[
        Symbol("holds", ("Fact", "World"), ("in {1}, {0}",), ("in {1}, it is not the case that {0}",),
               lambda m, p, w: p(w)),
        Symbol("access", ("World", "World"), ("{1} is reachable from {0}",), ("{1} is not reachable from {0}",),
               access, leaf=False),
        Symbol("box", ("Fact", "World"), ("in every situation reachable from {1}, {0}",),
               ("in some situation reachable from {1}, it is not the case that {0}",), necessarily, leaf=False),
        Symbol("dia", ("Fact", "World"), ("in some situation reachable from {1}, {0}",),
               ("in no situation reachable from {1} is it the case that {0}",), possibly, leaf=False),
    ],
    connectives=("not", "and", "or"),
)

RULES = [
    rule("necessity_elimination", "(box P W), (access W V) |- holds P V"),
    rule("possibility_introduction", "(holds P V), (access W V) |- dia P W"),
    rule("impossibility_elimination", "not (dia P W), (access W V) |- not (holds P V)"),
    rule("necessity_refutation", "not (holds P V), (access W V) |- not (box P W)"),
    rule("disjunctive_syllogism", "(or P Q), not P |- Q"),
    rule("and_elimination", "(and P Q) |- P"),
]

PACK = declare("modal", SIGNATURE, RULES)
