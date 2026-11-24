"""semforge: generate and audit logical reasoning data with certified proof structure.

  generate, analyze, audit_premises, LOGICS, Problem   the public API (semforge.api)

  core      domain-independent machinery: premise lattices (MARCO), provenance chase, dependency patterns
            and the edits that diversify them, certified variation, reasoner ladders, generation stats
  domains   the Domain contract and its implementations (logic, csp, zebralogic, fld)
  tasks     answer-producing problems (proofs, plans), solving, checking and structured rows
  pipeline  rows -> matched arms -> transfer; diversity comparison across sources
"""
from .core.lattice import Oracle, deletion, enumerate_lattice, minimum_support, necessary, quickxplain
from .core.provenance import IncompleteProvenance, Provenance, logic_depth_provenance, provenance_chase

_API = {"LOGICS", "GenerationError", "Problem", "analyze", "audit_premises", "generate"}


def __getattr__(name):  # the public API, imported on first use (keeps `python -m semforge.certcheck` clean)
    if name in _API:
        from . import api
        return getattr(api, name)
    raise AttributeError(f"module 'semforge' has no attribute {name!r}")
