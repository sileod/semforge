"""Propositional logic: natural-deduction-style rules over atoms, English atoms "<name> <predicate>".

The calculus is sound but deliberately incomplete; labels never rest on it (z3 decides them), only the
structural claims do, and those are stated relative to these rules. Contraposition is kept out of the
default set: with it, modus tollens always has the alternative contraposition + modus ponens, so it is
never required (`semforge.construct.check` reports such redundancy).
"""
from __future__ import annotations

from ..pack import declare
from ..realize import Phrase
from ..signature import Signature, negate, rule

SUBJECTS = ["Alice", "Bob", "Chloe", "Dmitri", "Emma", "Farid", "Grace", "Hugo", "Ines", "Jonas", "Kenji",
            "Lena", "Mateo", "Nora", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara"]
PREDICATES = ["is tired", "is happy", "plays chess", "sings", "is at home", "owns a dog", "is late", "reads novels",
              "drinks tea", "is hungry", "swims", "paints", "is busy", "writes code", "cooks", "is calm",
              "travels", "studies law", "rides a bike", "grows tomatoes"]


def negative(predicate: str) -> str:
    """ "is tired" -> "is not tired", "studies law" -> "does not study law", "swims" -> "does not swim"."""
    verb, _, rest = predicate.partition(" ")
    if verb == "is":
        return f"is not {rest}"
    base = verb[:-3] + "y" if verb.endswith("ies") else verb[:-1]
    return " ".join(["does not", base, rest]).strip()


def facts(rng, n):
    return [Phrase(f"{s} {p}", f"{s} {negative(p)}")
            for s, p in rng.sample([(s, p) for s in SUBJECTS for p in PREDICATES], n)]


RULES = [
    rule("modus_ponens", "A, (-> A B) |- B"),
    rule("modus_tollens", "(-> A B), not B |- not A"),
    rule("hypothetical_syllogism", "(-> A B), (-> B C) |- -> A C"),
    rule("disjunctive_syllogism", "(or A B), not A |- B"),
    rule("constructive_dilemma", "(or A B), (-> A C), (-> B C) |- C"),
    rule("and_elimination", "(and A B) |- A"),
    rule("and_introduction", "A, B |- and A B"),
    rule("double_negation", "not (not A) |- A"),
]
EXTRA = [rule("contraposition", "(-> A B) |- -> (not B) (not A)")]

SIGNATURE = Signature.of(prop_lexicon=facts)
PACK = declare("propositional", SIGNATURE, RULES, builds=("->",))
entails, render = PACK.entails, PACK.render
