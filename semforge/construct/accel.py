"""An optional accelerated oracle: settle entailment queries by a proof or a small countermodel before
asking z3 for the general answer.

Each query `premises ⊨ goal` goes through:
1. **Proof:** the pack's calculus derives the goal inside the usual analysis universe. The rules are
   sound for the pack's semantics (`semforge.construct.check` tests each against z3 on random
   derivations; a schematic soundness proof per rule is future work), so this answers "entailed".
2. **Countermodel:** z3 finds a model of premises ∧ ¬goal with every sort bounded to k elements, for k
   in `sizes`. A model of any size is a countermodel, so this answers "not entailed".
3. **z3**, as without acceleration (and `Undecided` if it times out).

Steps 1 and 2 only answer when their answer is certain, so labels are the same as z3's whenever z3
decides. Some queries z3 cannot settle within its timeout get settled here (quantified frame axioms
make z3 slow at both proving and model finding, which is where the S4 timeouts come from); the
others go to z3 exactly as before. `stats` counts which step settled each query.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import z3

from .calculus import Closure, Universe
from .oracle import query


def accelerated(pack, sizes=(1, 2, 3), timeout_ms: int = 2000, stats: Counter = None):
    """The pack with its oracle accelerated (see module docstring); `stats` collects step counts."""
    calc, inner, stats = pack.calculus, pack.entails, stats if stats is not None else Counter()

    def proved(premises, goal):
        premises, goal = [calc.canonical(p) for p in premises], calc.canonical(goal)
        return goal in Closure(calc, premises, Universe([*premises, goal], pack.negate, pack.closed)).round

    def refuted(premises, goal):
        for k in sizes:
            s = query(pack.signature, premises, goal, domain=k)
            s.set("timeout", timeout_ms)
            if s.check() == z3.sat:
                return True
        return False

    def entails(premises, goal):
        if proved(premises, goal):
            stats["proof"] += 1
            return True
        if pack.signature is not None and refuted(premises, goal):
            stats["countermodel"] += 1
            return False
        stats["z3"] += 1
        return inner(premises, goal)
    return replace(pack, entails=entails)
