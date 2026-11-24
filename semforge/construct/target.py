"""What to construct: a target is a request for a certified structure, not a generator setting.

Every field is checked on the constructed instance by independent analysis (semantics for the label,
the calculus for structure); construction only proposes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Distractors:
    irrelevant: int = 0   # derivations over unrelated atoms
    near_miss: int = 0    # derivations of the hypothesis or its negation missing exactly one premise


@dataclass(frozen=True)
class Target:
    label: str                       # entailment | contradiction | neutral
    rules: tuple                     # rule applications of the intended proof (a multiset plan)
    proofs: int = 1                  # at least this many minimal proofs of the conclusion
    shared: Optional[tuple] = None   # rule applications every proof must make (a multiset); default:
                                     # all of `rules` for one proof, none for several
    min_depth: int = 1               # height of the *shallowest* proof
    max_depth: int = 6               # construction bound on the intended proof
    distractors: Distractors = field(default_factory=Distractors)
    shape: Optional[tuple] = None    # a proof skeleton (build.derive_shape): the intended proof has exactly
                                     # this shape, certified on the mandatory steps (spine, joins); `rules`
                                     # may then be empty (free) or a multiset of the skeleton's size
    prefer: Optional[frozenset] = None  # with a free shape: rule kinds to try first (a soft steer)
    exposed: int = 0                  # of the intended proof's applications, exactly this many must be
                                     # *bypassable*: some proof of the same conclusion avoids them (an
                                     # atlas nogood is one source, any other a legitimate one too). 0, the
                                     # default, keeps every planned application mandatory. The certificate
                                     # names the bypassed applications with a witness proof for each,
                                     # re-checked by `certcheck` like the mandatory claims.

    def __post_init__(self):
        if self.required - Counter(self.rules):
            raise ValueError("shared rules must be part of the intended proof")
        if self.shape is not None and self.proofs > 1:
            raise ValueError("shape targets describe a single proof")
        if self.exposed < 0:
            raise ValueError("exposed counts bypassable planned applications; it cannot be negative")

    @property
    def required(self) -> Counter:
        if self.shared is not None:
            return Counter(self.shared)
        return Counter(self.rules) if self.proofs == 1 else Counter()
