"""FLD-style forward generation in any pack: the uncertified baseline recipe (FLD×2, RuleTaker, ProofWriter).

A random n-step proof tree plus n/2 random 1-3 step derivations as distractors, labelled by the pack's
oracle. Nothing about the tree is certified: its steps may be bypassable and its depth may not hold.
Same calculus, realizer and formula budget as construction, so the two differ only in how structure is
chosen and whether it is certified.
"""
from __future__ import annotations

from typing import Optional

from .build import Derivation, Fresh, derive
from .calculus import Universe
from .synth import Instance, negation_depth


def forward(pack, rng, n: int, max_negations: int = 2, label: Optional[str] = None):
    """(Instance, intended derivation) or None. `label` is entailment or contradiction (random if None)."""
    calc, fresh = pack.calculus, Fresh()
    ok = lambda f: negation_depth(f) <= max_negations
    d: Derivation = derive(calc, [None] * n, rng, fresh, admissible=ok, deep=rng.random() < 0.5,
                           max_depth=max(6, n // 3 + 4))
    if d is None:
        return None
    premises, roles = list(d.premises), {p: "proof" for p in d.premises}
    for _ in range(n // 2):
        e = derive(calc, [None] * rng.randint(1, 3), rng, fresh, admissible=ok)
        if e is not None:
            for p in e.premises:
                if p not in roles:
                    premises.append(p)
                    roles[p] = "irrelevant"
    rng.shuffle(premises)
    label = label or rng.choice(["entailment", "contradiction"])
    hyp = d.conclusion if label == "entailment" else pack.negate(d.conclusion)
    if pack.label(premises, hyp) != label:
        return None
    if label == "entailment" and pack.entails(premises, pack.negate(hyp)):  # inconsistent: entails anything
        return None
    universe = Universe([*premises, hyp], pack.negate, pack.closed)  # what a reader sees, nothing planted
    return Instance(premises, hyp, label, d.conclusion, universe, roles=roles), d
