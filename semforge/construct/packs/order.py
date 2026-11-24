"""Strict orders: "Ada is taller than Bo", chains and refutations through transitivity (a nest probe).

The relation is a strict partial order (irreflexive and transitive, given as theory axioms), so z3 labels
are exact. Unlike the logics, the reasoning is relational: long chains through shared individuals, and
refutations that run a chain backwards ("Ada is taller than Bo but not taller than Cy, so Bo is not
taller than Cy"). Transitivity is associative, so two-step chains have shortcuts the atlas must learn.
"""
from __future__ import annotations

from ..oracle import irreflexive, transitive
from ..pack import declare
from ..signature import Signature, Sort, Symbol, rule

PEOPLE = ["Ada", "Bo", "Cy", "Dee", "Eli", "Fay", "Gus", "Hal", "Ida", "Jon", "Kay", "Lou", "Mo", "Ned", "Oz",
          "Pia", "Quin", "Rue", "Sol", "Tam"]

SIGNATURE = Signature.of(
    sorts=[Sort("Person", PEOPLE)],
    symbols=[Symbol("taller", ("Person", "Person"), ("{0} is taller than {1}", "{1} is shorter than {0}"),
                    ("{0} is not taller than {1}",), lambda m, x, y: m.fn("taller", "Person", "Person", "Prop")(x, y))],
    connectives=("not", "and", "or"),
    axioms=(irreflexive("taller", "Person"), transitive("taller", "Person")),
    theory=("No one is taller than themselves.",
            "For any people X, Y and Z: if X is taller than Y and Y is taller than Z, then X is taller than Z."),
)

RULES = [
    rule("chain", "(taller X Y), (taller Y Z) |- taller X Z"),
    rule("asymmetry", "(taller X Y) |- not (taller Y X)"),
    rule("chain_refutation_left", "(taller X Y), not (taller X Z) |- not (taller Y Z)"),
    rule("chain_refutation_right", "(taller Y Z), not (taller X Z) |- not (taller X Y)"),
    rule("disjunctive_syllogism", "(or P Q), not P |- Q"),
    rule("and_elimination", "(and P Q) |- P"),
]

PACK = declare("order", SIGNATURE, RULES)
