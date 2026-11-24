"""Bounded arithmetic (an extension probe): people's scores and the differences between them.

The premises give some scores ("Ann's score is 3"), differences ("Bob's score is 2 higher than Ann's")
and comparisons ("Ann has a higher score than Cy"). Conclusions are computed: a score from another
score and a difference, a difference from two scores, a difference from two differences (adding them),
and comparisons from either. Scores are 0..N-1.

What the probe reuses and what it adds, against the other packs:
- reused: numbers are a fixed sort; arithmetic is interpreted symbols (`plus`, `less`, `positive`,
  `Symbol.holds`), theory facts computed when a rule needs them; side conditions are premises over them
  ("(plus A D C)"), so computed conclusions need no new rule machinery. z3 semantics give numbers their
  integer values, check.py verifies that they define the same relations as `holds`, and every rule is
  proved sound by check.prove_rule.
- added: `Signature.solved_grounding`. Scores are functional (one per person), so a proof's leaves
  constrain each other through paths the proof does not use (Fay is 9, Fay is Jo + 4, Jo is Max + 0,
  Max is Kai + 1 force Kai to 4, whatever leaf gives Kai's score). Binding numbers at random, as the
  other packs do, made 60% of attempts inconsistent; binding them one at a time under z3 removes that.
- not solved: construction. Once the leaves are consistent, a quantity fixed twice is derivable two
  ways, so the planned steps are bypassed (every attempt in a 2-minute run ended as "shortcut"), and
  the dense closure (every pair of scored people is compared) makes proof enumeration the main cost.
  Certified necessity needs each quantity determined once (one anchored path per connected group of
  people), which tree-shaped construction does not see. Forward generation works. A functional pack
  needs that constraint in the shape search, or a quantity-level skeleton, before it can be constructed.
- interpreted symbols replaced a background table that grew as N^2 (all sums and comparisons below N).
"""
from __future__ import annotations

import z3

from ..pack import declare
from ..signature import Signature, Sort, Symbol, rule

N = 10  # numbers 0..N-1
PEOPLE = ("Ann Bob Cy Dee Eve Finn Gus Hal Ivy Jo Kai Lea Max Nia Otto Pia Quin Rex Sam Tia Uma "
          "Vic Wes Xena Yara Zed Abe Bea Cal Dora Eli Fay Gil Hana Ike Jade Kit Lou Mia Ned").split()  # one per role in a long proof
NUMS = [f"n{i}" for i in range(N)]


def _val(m, n):
    return m.fn("val", "Num", "Int")(n)


def _score(m, p):
    return m.fn("score", "Person", "Int")(p)


def _axioms(m):
    """Each number constant has its intended value."""
    return z3.And(*(_val(m, z3.Const(f"Num_{n}", m.sort("Num"))) == i for i, n in enumerate(NUMS)))


S = lambda name, args, templates=(), negated=(), z3f=None, holds=None: \
    Symbol(name, args, templates, negated, z3f, leaf=False, holds=holds)
VALUE = {n: i for i, n in enumerate(NUMS)}

SYMBOLS = [
    S("score", ("Person", "Num"), ("{0}'s score is {1}", "{0} scored {1}"),
      ("{0}'s score is not {1}", "{0} did not score {1}"), lambda m, p, n: _score(m, p) == _val(m, n)),
    S("more", ("Person", "Person", "Num"), ("{0}'s score is {2} higher than {1}'s",
                                            "{0} scored {2} more than {1}"),
      ("{0}'s score is not {2} higher than {1}'s", "{0} did not score {2} more than {1}"),
      lambda m, p, q, d: _score(m, p) == _score(m, q) + _val(m, d)),
    S("higher", ("Person", "Person"), ("{0} has a higher score than {1}", "{0} scored higher than {1}"),
      ("{0} does not have a higher score than {1}", "{0} did not score higher than {1}"),
      lambda m, p, q: _score(m, p) > _score(m, q)),
    # interpreted arithmetic
    S("plus", ("Num", "Num", "Num"), z3f=lambda m, a, b, c: _val(m, a) + _val(m, b) == _val(m, c),
      holds=lambda a, b, c: VALUE[a] + VALUE[b] == VALUE[c]),
    S("less", ("Num", "Num"), z3f=lambda m, a, b: _val(m, a) < _val(m, b), holds=lambda a, b: VALUE[a] < VALUE[b]),
    S("positive", ("Num",), z3f=lambda m, a: _val(m, a) > 0, holds=lambda a: VALUE[a] > 0),
]


THEORY = ("Each person has exactly one score, a whole number.",
          "One score being some amount higher than another means it equals the other plus that amount.")

SIGNATURE = Signature.of(
    sorts=[Sort("Person", fixed=tuple((p, p) for p in PEOPLE)),
           Sort("Num", fixed=tuple((n, str(i)) for i, n in enumerate(NUMS)))],
    symbols=SYMBOLS, connectives=("not",), axioms=(_axioms,), theory=THEORY,
    solved_grounding=True)

RULES = [
    rule("add", "(score Q A), (more P Q D), (plus A D C) |- score P C"),
    rule("subtract", "(score P C), (more P Q D), (plus A D C) |- score Q A"),
    rule("difference", "(score P C), (score Q A), (plus A D C) |- more P Q D"),
    rule("chain", "(more P Q D), (more Q R E), (plus D E F) |- more P R F"),
    rule("gap", "(more P Q D), (positive D) |- higher P Q"),
    rule("compare", "(score P B), (score Q A), (less A B) |- higher P Q"),
    rule("transitive", "(higher P Q), (higher Q R) |- higher P R"),
    rule("asymmetric", "(higher P Q) |- not (higher Q P)"),
    rule("unique", "(score P A), (less A B) |- not (score P B)"),
    rule("unique_below", "(score P B), (less A B) |- not (score P A)", kind="unique"),
    rule("not_higher", "(score P A), (score Q B), (less A B) |- not (higher P Q)"),
]

PACK = declare("arithmetic", SIGNATURE, RULES, atlas_size=2)
entails, render = PACK.entails, PACK.render
