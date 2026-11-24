"""Monadic first-order logic with constants: syllogisms, instantiation, existentials (FLD×2's ground).

"c is a P", "every P is a Q", "no P is a Q", "something is a P", with not/and/or. Labels come from z3
over an uninterpreted domain (complete for this fragment). Nonce nouns (as in PrOntoQA) keep world
knowledge out, and negation reads naturally ("Rex is not a wumpus").
"""
from __future__ import annotations

from dataclasses import replace

import z3

from ..pack import declare
from ..signature import Signature, Sort, Symbol, rule

NOUNS = ["wumpus", "yumpus", "zumpus", "dumpus", "rompus", "numpus", "tumpus", "vumpus", "impus", "jompus",
         "gorpus", "lempus", "sterpus", "brimpus", "lorpus", "grimpus", "shumpus", "kerpus", "felpus", "porpus",
         "quimpus", "harpus", "ompus", "belpus"]
NAMES = ["Rex", "Sam", "Wren", "Max", "Fae", "Polly", "Stella", "Alex", "Sally", "Fred", "Nell", "Otto",
         "Bea", "Ivo", "Juno", "Kit"]


def every(m, p, q, positive=True):
    x = m.var("Thing")
    return z3.ForAll([x], z3.Implies(p(x), q(x) if positive else z3.Not(q(x))))


def some(m, p):
    x = m.var("Thing")
    return z3.Exists([x], p(x))


SIGNATURE = Signature.of(
    sorts=[Sort("Thing", NAMES), Sort("Class", NOUNS, kind="predicate", over="Thing")],
    symbols=[
        Symbol("is", ("Class", "Thing"), ("{1} is {0:a}",), ("{1} is not {0:a}",), lambda m, p, c: p(c)),
        Symbol("all", ("Class", "Class"), ("every {0} is {1:a}", "all {0:s} are {1:s}"),
               ("not every {0} is {1:a}",), every, leaf=False),
        Symbol("no", ("Class", "Class"), ("no {0} is {1:a}",), ("some {0} is {1:a}",),
               lambda m, p, q: every(m, p, q, positive=False), unordered=((0, 1),), leaf=False),
        Symbol("some", ("Class",), ("something is {0:a}", "there is {0:a}"), ("nothing is {0:a}",), some,
               leaf=False),
    ],
    connectives=("not", "and", "or"),
)

RULES = [
    rule("instantiation", "(all A B), (is A X) |- is B X"),
    rule("instantiation_tollens", "(all A B), not (is B X) |- not (is A X)"),
    rule("exclusion", "(no A B), (is A X) |- not (is B X)"),
    rule("barbara", "(all A B), (all B C) |- all A C"),
    rule("celarent", "(all A B), (no B C) |- no A C"),
    rule("existential_generalization", "(is A X) |- some A"),
    rule("darii", "(some A), (all A B) |- some B"),
    # refuting a universal by a witness: without these, a universal hypothesis is never a contradiction,
    # a cue a hypothesis-only classifier reads
    rule("counterexample", "(is A X), not (is B X) |- not (all A B)"),
    rule("overlap", "(is A X), (is B X) |- not (no A B)"),
    rule("disjunctive_syllogism", "(or P Q), not P |- Q"),
    rule("and_elimination", "(and P Q) |- P"),
]

PACK = declare("monadic", SIGNATURE, RULES)

# The same logic without connectives and with ground rules only: every step applies a universal or an
# existential to an individual, so certified depth is quantifier reasoning, not unpacking compound
# premises (80-87% of monadic's mandatory steps). Without Barbara and Celarent a chain of instantiations
# is not bypassed by composing the universals first.
SYLLOGISTIC = declare("monadic_syllogistic", replace(SIGNATURE, connectives=("not",)),
                      [r for r in RULES if r.name in ("instantiation", "instantiation_tollens", "exclusion",
                                                      "existential_generalization", "darii")])
