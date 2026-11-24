"""Modal logic S4, labelled: the K pack plus reflexive, transitive accessibility (a nest probe).

The frame conditions are theory axioms of the oracle; the calculus gains their rules: T (what is
necessary holds here), 4 (necessity persists along accessibility) and transitivity of reachability.
"""
from __future__ import annotations

from dataclasses import replace

from ..oracle import reflexive, transitive
from ..pack import declare
from ..signature import rule
from .modal import RULES as K_RULES, SIGNATURE as K_SIGNATURE

THEORY = ("Every situation is reachable from itself.",
          "For any situations X, Y and Z: if Y is reachable from X and Z is reachable from Y, then Z is "
          "reachable from X.")
SIGNATURE = replace(K_SIGNATURE, axioms=(reflexive("access", "World"), transitive("access", "World")), theory=THEORY)

RULES = K_RULES + [
    rule("necessity_here", "(box P W) |- holds P W"),
    rule("necessity_persists", "(box P W), (access W V) |- box P V"),
    rule("reachability_chain", "(access W V), (access V U) |- access W U"),
]

PACK = declare("modal_s4", SIGNATURE, RULES)
