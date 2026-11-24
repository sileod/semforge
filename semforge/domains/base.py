"""The domain contract, and the registry of domains.

A domain produces entailment instances (premises, hypothesis) whose label is certified by a complete
oracle, and exposes what dependency-pattern analysis needs. The NLI pipelines (patterns,
enforcement, selection, transfer) use this contract. Answer-producing tasks such as proofs and plans
use `semforge.tasks` instead; they do not need selectable premises or three-valued labels.

Required:
  name                          registry key, also written on every row
  sample(rng, level)            -> list of (instance, label); one call = one world
  label(inst)                   -> "entailment" | "contradiction" | "neutral" | None (undecided)
  justifications(inst, label)   -> (minimal premise supports of the label, complete?); E/C only
  kind(premise)                 -> premise kind, or None for premises that do not count (plain facts)
  premises(inst)                -> the selectable premises, in order
  without(inst, drop)           -> the instance minus some premises
  render(inst)                  -> (premises text, hypothesis text)
Optional (defaults in `Domain`):
  rung(inst, label)             -> weakest deciding reasoner, None when the domain has one
  family(inst)                  -> sub-population name used for stratification
  world(inst)                   -> source id shared by all instances of one world (default: per sample call)
  annotations(inst, pattern)    -> extra row fields (e.g. a dataset's own gold annotations)
  expand(inst, label, rng, seen)-> a certified variant with an added required kind, or None
"""
from __future__ import annotations

E, C, N = "entailment", "contradiction", "neutral"
LABELS = (E, C, N)


class Domain:
    name = "domain"

    def rung(self, inst, label):
        return None

    def family(self, inst):
        return self.name

    def world(self, inst):
        return None

    def annotations(self, inst, pattern):
        return {}

    def expand(self, inst, label, rng, kinds_seen):
        return None

    def n_premises(self, inst):
        return len(self.premises(inst))


def get(name: str, **kw) -> Domain:
    """Instantiate a registered domain; imports are lazy so optional dependencies stay optional."""
    if name == "logic":
        from .logic import LogicDomain as D
    elif name == "csp":
        from .csp import CSPDomain as D
    elif name == "zebralogic":
        from .zebralogic import ZebraDomain as D
    elif name == "fld":
        from .fld import FLDDomain as D
    elif name.startswith("constructed"):  # constructed_<pack>; plain "constructed" is propositional
        from .constructed import ConstructedDomain
        pack, strategy = name.partition("_")[2], "plan"
        for s in STRATEGIES:  # a strategy is a suffix; pack names contain underscores themselves
            if pack.endswith("_" + s):
                pack, strategy = pack[:-len(s) - 1], s
        return ConstructedDomain(pack=pack or "propositional", strategy=strategy, **kw)
    else:
        raise KeyError(f"unknown domain {name!r}; known: {NAMES}")
    return D(**kw)


from ..construct.packs import NAMES as PACKS  # noqa: E402 -- names only, packs load on use

STRATEGIES = ("plan", "shape", "forward")
NAMES = ("logic", "csp", "zebralogic", "fld", "constructed",
         *(f"constructed_{p}" + ("" if s == "plan" else f"_{s}") for p in PACKS for s in STRATEGIES))
