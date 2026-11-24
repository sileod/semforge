"""semforge: certified dependency patterns for procedural reasoning data.

  core      domain-independent machinery: premise lattices (MARCO), provenance chase, dependency patterns
            and the edits that diversify them, certified variation, reasoner ladders, generation stats
  domains   the Domain contract and its implementations (logic, csp, zebralogic, fld)
  tasks     answer-producing problems (proofs, plans), solving, checking and structured rows
  pipeline  rows -> matched arms -> transfer; diversity comparison across sources
"""
from .core.lattice import Oracle, deletion, enumerate_lattice, minimum_support, necessary, quickxplain
from .core.provenance import IncompleteProvenance, Provenance, logic_depth_provenance, provenance_chase
