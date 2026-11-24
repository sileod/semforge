"""Reasoning patterns of certified justifications, and how to diversify them. Domain-independent.

A justification of a label is a minimal support: an inclusion-minimal premise set on which the
domain's complete oracle still yields it (a minimal proof, a minimal clue set forcing a value). The
pattern of an instance is (required kinds, rung): the multiset of premise kinds common to every
justification -- a kind counts only when no justification avoids it -- and the weakest reasoner that
decides the instance. Domains (semforge.domains) supply justifications, kinds and rungs.

Diversifying patterns:
  select    greedy least-covered selection from a pool (reaches what the generator emits)
  enforce   delete a minimum hitting set of the shortcut justifications so a chosen justification's
            kinds become required (reaches patterns the generator shows but does not demand)
Each enforced instance is re-analysed and kept only if its label is unchanged and the kinds are
required, so certification never rests on the edit being correct by construction.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import z3


@dataclass
class Pattern:
    required: Counter     # kinds common to every justification (multiset intersection)
    rung: Optional[str]   # weakest deciding reasoner, None when the domain has one
    justifications: list  # minimal supports
    kinds: list           # kind multiset of each justification
    complete: bool        # justifications enumerated exhaustively; if False, `required` is only an upper
                          # bound (an unseen support may avoid a kind), so callers exclude it from statistics

    @property
    def signature(self) -> tuple:
        return tuple(sorted(self.required.items()))

    @property
    def key(self) -> tuple:
        return (self.signature, self.rung)


def kinds_of(support, kind) -> Counter:
    return Counter(k for k in map(kind, support) if k is not None)


def analyze(domain, inst, lab) -> Optional[Pattern]:
    """Pattern of an E/C instance; None for neutral or when justifications are unavailable."""
    if lab == "neutral":
        return None
    sups, complete = domain.justifications(inst, lab)
    if not sups:
        return None
    kinds = [kinds_of(s, domain.kind) for s in sups]
    required = kinds[0].copy()
    for k in kinds[1:]:
        required &= k
    return Pattern(required, domain.rung(inst, lab), sups, kinds, complete)


def enforce(p: Pattern, keep: frozenset, kind, want: Optional[Counter] = None) -> Optional[frozenset]:
    """Minimum premise set to delete so that `want` (default: `keep`'s kinds) is required, avoiding `keep`.

    Shortcuts are the justifications lacking some wanted kind. Deleting a hitting set of them that
    misses `keep` leaves `keep` intact (the label survives, oracles being monotone) and removes them.
    """
    want = kinds_of(keep, kind) if want is None else want
    bad = [s for s, k in zip(p.justifications, p.kinds) if want - k]
    if not bad:
        return frozenset()
    if any(not (s - keep) for s in bad):
        return None
    cand = sorted(set().union(*bad) - keep, key=repr)
    var = {x: z3.Bool(f"d{i}") for i, x in enumerate(cand)}
    opt = z3.Optimize()
    for s in bad:
        opt.add(z3.Or([var[x] for x in s - keep]))
    opt.minimize(z3.Sum([z3.If(v, 1, 0) for v in var.values()]))
    if opt.check() != z3.sat:
        return None
    model = opt.model()
    return frozenset(x for x in cand if z3.is_true(model.eval(var[x], model_completion=True)))


def enforced_variants(domain, inst, lab, p: Pattern):
    """Certified variants whose required pattern is new: one per justification that can be isolated."""
    if len(p.justifications) < 2 or not p.complete:
        return []
    out, seen = [], {p.key}
    for keep in p.justifications:
        drop = enforce(p, keep, domain.kind)
        if not drop:
            continue
        e = domain.without(inst, drop)
        if domain.label(e) != lab:
            continue
        q = analyze(domain, e, lab)
        if q is None or (kinds_of(keep, domain.kind) - q.required) or q.key in seen:
            continue
        seen.add(q.key)
        out.append((e, q))
    return out


def select_diverse(items, n, descriptor):
    """Greedy least-covered selection: novelty of the whole descriptor plus mean novelty of its parts.

    `descriptor(item)` returns (key, parts): the pattern key and an iterable of its components (kinds,
    rung, family) so that items sharing rare components are also favoured.
    """
    whole, part, out, pool = Counter(), Counter(), [], list(items)
    desc = {id(x): descriptor(x) for x in pool}
    while pool and len(out) < n:
        def score(x):
            k, parts = desc[id(x)]
            parts = list(parts)
            return 1 / (1 + whole[k]) + sum(1 / (1 + part[c]) for c in parts) / max(1, len(parts))
        best = max(pool, key=score)
        pool.remove(best)
        out.append(best)
        k, parts = desc[id(best)]
        whole[k] += 1
        part.update(parts)
    return out
