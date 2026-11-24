"""Target-conditioned construction of reasoning problems with certified realized structure.

  terms     formulas, metavariables, unification
  calculus  rule schemas; exact analysis of required rules/premises, alternatives, depth (rules are
            counted as premises in the provenance chase)
  pack      a domain for construction: calculus (structure) + semantics (labels) + renderer
  target    what to construct: label, rules, alternatives, distractors
  build     backward construction of a derivation using requested rules
  synth     construct, add distractors, certify; failure accounting
  archive   least-covered niche next (diversity by construction)
  packs/    propositional (more: first-order, CSP propagation, planning)

Two guarantees, never mixed: labels are exact under the pack's complete semantics; structural claims
are exact relative to the pack's calculus over a bounded universe.
"""
