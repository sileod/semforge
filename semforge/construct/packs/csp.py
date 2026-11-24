"""Finite-domain constraint propagation (Zebra-like clues).

People and house colours (colours have unique names). "x's house is v", "x and y live in houses of
different / the same colour" (symmetric), "x's house is a, b or c" (an unordered set), and
"either ... or ...". Labels come from z3; the calculus is the propagation a person applies. Symmetric
and set-valued arguments are declared, so each reasoning kind is one rule. Case splits are absent:
instances needing them are labelled but not derivable, and the synthesizer rejects them.
"""
from __future__ import annotations

import z3

from ..pack import declare
from ..signature import Signature, Sort, Symbol, rule

PEOPLE = ["Alice", "Bob", "Chloe", "Dmitri", "Emma", "Farid", "Grace", "Hugo", "Ines", "Jonas", "Kenji", "Lena",
          "Mateo", "Nora", "Omar", "Priya"]
COLOURS = ["red", "blue", "green", "yellow", "white", "black", "orange", "purple", "grey", "pink", "brown", "teal",
           "ivory", "navy", "olive", "coral", "amber", "lilac"]


def colour(m, x):
    return m.fn("colour", "Person", "Colour")(x)


SIGNATURE = Signature.of(
    sorts=[Sort("Person", PEOPLE), Sort("Colour", COLOURS)],
    symbols=[
        Symbol("eq", ("Person", "Colour"), ("{0}'s house is {1}", "{0} lives in the {1} house"),
               ("{0}'s house is not {1}", "{0} does not live in the {1} house"),
               lambda m, x, v: colour(m, x) == v),
        Symbol("ne", ("Person", "Person"), ("{0} and {1} live in houses of different colours",),
               ("{0} and {1} live in houses of the same colour",),
               lambda m, x, y: colour(m, x) != colour(m, y), unordered=((0, 1),), leaf=False),
        Symbol("same", ("Person", "Person"), ("{0} and {1} live in houses of the same colour",),
               ("{0} and {1} live in houses of different colours",),
               lambda m, x, y: colour(m, x) == colour(m, y), unordered=((0, 1),), leaf=False),
        Symbol("one_of", ("Person", "Colour", "Colour", "Colour"), ("{0}'s house is {1}, {2} or {3}",), (),
               lambda m, x, a, b, c: z3.Or(colour(m, x) == a, colour(m, x) == b, colour(m, x) == c),
               unordered=((1, 2, 3),), leaf=False),
    ],
    connectives=("not", "or"),
)

RULES = [
    rule("inequality", "(eq X V), (ne X Y) |- not (eq Y V)"),
    rule("equality", "(eq X V), (same X Y) |- eq Y V"),
    rule("naked_single", "(one_of X A B C), not (eq X B), not (eq X C) |- eq X A"),
    rule("disjunctive_syllogism", "(or P Q), not P |- Q"),
]

PACK = declare("csp", SIGNATURE, RULES)
