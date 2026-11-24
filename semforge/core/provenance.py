"""Provenance chase: every way each atom is derived, as minimal sets of premises.

A plain chase keeps one derivation per atom. Tracking why-provenance (Green, Karvounarakis and
Tannen 2007, PosBool semiring) instead keeps, for each atom, a Pareto front of
premise supports and their shallowest proof depths. Supersets are retained only
when they offer a shallower proof. One fixpoint then
answers every subset question without re-chasing: entailment under any premise subset, all
minimal supports, necessary premises, and whether a shallower or smaller proof exists than the
one a generator planted.

Monotone rules only (no negation as failure); sign is part of the atom, as in logic_depth.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from math import prod
from typing import Callable, Iterable, Optional


class IncompleteProvenance(ValueError):
    """An exact query was requested from capped, incomplete provenance."""


@dataclass
class Provenance:
    """Known support/depth Pareto fronts; `truncated` includes dependent atoms.

    Truncation is conservative: even if later discoveries recover all proofs, the
    atom remains marked. Stored proofs are valid witnesses, but missing proofs
    cannot justify negative entailment or exact minimum/necessity claims.
    """
    supports: dict
    truncated: set = field(default_factory=set)

    def is_exact(self, atom) -> bool:
        return atom not in self.truncated

    def minimal_supports(self, atom, *, allow_incomplete=False) -> list:
        """Inclusion-minimal supports; opt-in returns minima among retained proofs only."""
        if not self.is_exact(atom) and not allow_incomplete:
            raise IncompleteProvenance(f"incomplete supports for {atom!r}")
        ss = self.supports.get(atom, {})
        return sorted((s for s in ss if not any(t < s for t in ss)),
                      key=lambda s: (len(s), sorted(map(repr, s))))

    def entails(self, atom, premises) -> Optional[bool]:
        """True from a known proof; None if truncation prevents deciding False."""
        premises = frozenset(premises)
        if any(s <= premises for s in self.supports.get(atom, ())):
            return True
        return False if self.is_exact(atom) else None

    def necessary(self, atom) -> Optional[frozenset]:
        """Premises in every proof; None when the support list was truncated (not exact)."""
        if atom in self.truncated:
            return None
        ss = list(self.supports.get(atom, ()))
        return frozenset.intersection(*ss) if ss else frozenset()

    def min_depth(self, atom) -> Optional[int]:
        return min(self.supports[atom].values()) if self.is_exact(atom) and self.supports.get(atom) else None

    def min_size(self, atom) -> Optional[int]:
        return min(map(len, self.supports[atom])) if self.is_exact(atom) and self.supports.get(atom) else None


def _add(antichain: dict, s: frozenset, depth: int, cap: int) -> int:
    """Insert (s, depth) unless some t <= s already has depth <= depth; drop entries it dominates.

    Entries are kept on the (inclusion, depth) Pareto front so that both the smallest proofs and
    the shallowest proofs survive. Returns 1 if added, 0 if dominated, -1 if refused by the cap.
    """
    for t, e in antichain.items():
        if t <= s and e <= depth:
            return 0
    for t in [t for t, e in antichain.items() if s <= t and depth <= e]:
        del antichain[t]
    if len(antichain) >= cap:
        return -1
    antichain[s] = depth
    return 1


def provenance_chase(facts: Iterable, rules: Iterable, instances: Callable, premise_id: Callable,
                     cap: int = 256, step_id: Optional[Callable] = None) -> Provenance:
    """Fixpoint over `instances(rule, closure) -> (head, parents)` with premise sets tracked.

    `premise_id(x)` names a fact or rule as a premise, or returns None for background statements
    that are always available (their provenance is the empty set). `step_id(rule, head, parents)`,
    if given, also puts each rule application in the supports it builds, so a minimal support is a
    minimal proof (its steps and leaves) rather than a minimal premise set.
    """
    if cap < 1:
        raise ValueError("cap must be positive")
    supports = defaultdict(dict)
    truncated = set()
    for f in facts:
        pid = premise_id(f)
        _add(supports[f], frozenset(() if pid is None else (pid,)), 0, cap)
    rules = list(rules)
    changed = True
    while changed:
        changed = False
        closure = set(supports)
        for rule in rules:
            rid = premise_id(rule)
            own = frozenset(() if rid is None else (rid,))
            for head, parents in instances(rule, closure):
                parents = tuple(parents)
                step = own if step_id is None else own | {step_id(rule, head, parents)}
                if head not in truncated and any(p in truncated for p in parents):
                    truncated.add(head)
                    changed = True
                options = [list(supports[p].items()) for p in parents]
                if prod(map(len, options)) > cap * cap:  # combining would explode (many-parent rules):
                    if head not in truncated:            # truncate, as when one atom exceeds the cap
                        truncated.add(head)
                        changed = True
                    continue
                for combo in product(*options):
                    s = step.union(*(c[0] for c in combo))
                    depth = 1 + max((c[1] for c in combo), default=0)
                    added = _add(supports[head], s, depth, cap)
                    changed |= added == 1
                    if added == -1 and head not in truncated:
                        truncated.add(head)
                        changed = True
    return Provenance(dict(supports), truncated)


def logic_depth_provenance(theory, source=None, background_rules=(), cap=256) -> Provenance:
    """Provenance over a reasoning_core logic_depth Theory, premises named by rendered line index."""
    from reasoning_core.tasks import logic_depth as L
    if source is None:
        _, source, _ = L.render(theory)
    if any(L.body_not_atoms(r.body) for r in theory.rules):
        raise ValueError("negation as failure is not monotone; use the oracle instead")
    background = set(background_rules)
    return provenance_chase(
        theory.facts, theory.rules, L._rule_instances,
        lambda x: None if x in background else source[x], cap,
    )
