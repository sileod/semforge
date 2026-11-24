"""The front door: generate certified problems, and check them.

    import semforge as sf
    p = sf.generate("monadic", mandatory_steps=3, label="entailment", seed=0)
    print(p.text)
    assert p.verify() == []

A request becomes a construction target (label, a plan of `mandatory_steps` rule applications, proofs,
distractors), the synthesizer builds a problem for it, and only problems whose certified structure meets
the request are returned: every planned step is mandatory (each proof of the conclusion makes it) and
the label is decided by the logic's solver. Attempts that fail are retried a bounded number of times.
"""
from __future__ import annotations

from collections import Counter

import random
from dataclasses import dataclass, field
from typing import Optional

from .certcheck import check
from .construct.oracle import Undecided
from .construct.packs import NAMES as LOGICS
from .construct.packs import get as get_pack
from .construct.synth import Instance, Synthesizer, export_certificate
from .construct.target import Distractors, Target

LABELS = ("entailment", "contradiction", "neutral")


class GenerationError(RuntimeError):
    """No problem met the request within the attempt budget."""


@dataclass
class Problem:
    logic: str
    premises: list           # English sentences, in presentation order
    hypothesis: str
    label: str               # entailment | contradiction | neutral
    mandatory_steps: int     # rule applications every proof must make (for neutral: once `withheld` is restored)
    proofs: int              # minimal proofs of the conclusion
    depth: int               # height of the shallowest proof
    certificate: dict = field(repr=False)  # self-contained, checked by semforge.certcheck
    instance: Instance = field(repr=False)  # formulas, analysis and roles, for programmatic use

    @property
    def text(self) -> str:
        return "\n".join(self.premises) + f"\nHypothesis: {self.hypothesis}\nLabel: {self.label}"

    @property
    def mandatory_rules(self) -> dict:
        """Mandatory steps counted by inference rule (as in sf.analyze)."""
        return dict(sorted(Counter(s[1] for s in self.instance.analysis.mandatory_steps).items()))

    def verify(self) -> list:
        """Re-check the certificate with the independent checker; [] means every claim holds."""
        return check(self.certificate)

    def to_dict(self) -> dict:
        return {"logic": self.logic, "premises": self.premises, "hypothesis": self.hypothesis, "label": self.label,
                "mandatory_steps": self.mandatory_steps, "mandatory_rules": self.mandatory_rules,
                "proofs": self.proofs, "depth": self.depth, "certificate": self.certificate}


def _synth(logic: str) -> Synthesizer:
    # A fresh synthesizer per call, without nogood learning: learning would change later calls (the atlas is
    # shared per process), and the same seed must give the same problems. The atlas itself is built once
    # per process (about a second) and reused.
    return Synthesizer(get_pack(logic), strict=True, learn=False)


def _one(synth, logic, rng, mandatory_steps, label, distractors, require=None) -> Optional[Problem]:
    pack = synth.pack
    lab = label or rng.choice(LABELS)
    for _ in range(200 if require else 1):  # plans are cheap: steer them toward a required rule
        plan = synth.sample_plan(rng, mandatory_steps)
        if not require or require & set(plan):
            break
    if len(plan) != mandatory_steps or (require and not require & set(plan)):
        return None
    target = Target(label=lab, rules=plan, max_depth=max(6, mandatory_steps // 3 + 4),
                    distractors=Distractors(irrelevant=distractors, near_miss=distractors and rng.randint(0, 1)))
    try:
        inst = synth.realize(target, rng)
        if inst is None or inst.label != lab:
            return None
        a = inst.analysis
        if len(a.mandatory_steps) != mandatory_steps:
            return None
        if require and not require & {s[1] for s in a.mandatory_steps}:  # certified, not just planned
            return None
        premises, hypothesis = pack.render(inst.premises, inst.hypothesis, rng)
        cert = export_certificate(pack, inst)
    except (Undecided, ValueError):  # undecidable label, or more constants than the lexicon has words
        return None
    if cert is None:
        return None
    if isinstance(premises, str):
        premises = [s for s in premises.split("\n") if s.strip()]
    return Problem(logic, list(premises), hypothesis, inst.label, len(a.mandatory_steps), a.n_proofs, a.depth,
                   cert, inst)


def generate(logic: str = "propositional", mandatory_steps: int = 3, label: Optional[str] = None,
             n: Optional[int] = None, seed: int = 0, distractors: int = 1, max_attempts: int = 50,
             require_rules: Optional[set] = None):
    """Certified problem(s) of a logic with exactly `mandatory_steps` mandatory rule applications.

    logic: one of semforge.LOGICS. label: entailment, contradiction, neutral, or None for a random one
    per problem. n: None returns one Problem, an integer returns a list of n. distractors: irrelevant
    derivations added per problem (0 for none). require_rules: a set of rule names (see
    semforge.rules(logic)); every problem then has a mandatory step of one of them, so every proof
    applies it, e.g. the modal rules for problems that need modal reasoning. Deterministic for a given seed and code version; raises
    GenerationError when `max_attempts` attempts per problem all fail (an unplaceable request)."""
    if logic not in LOGICS:
        raise KeyError(f"unknown logic {logic!r}; known: {', '.join(LOGICS)}")
    if label is not None and label not in LABELS:
        raise ValueError(f"label must be one of {LABELS} or None")
    if mandatory_steps < 1:
        raise ValueError("mandatory_steps must be at least 1")
    require = set(require_rules) if require_rules else None
    if require and require - set(rules(logic)):
        raise ValueError(f"unknown rules for {logic}: {sorted(require - set(rules(logic)))}")
    synth, rng, out = _synth(logic), random.Random(seed), []
    for _ in range(1 if n is None else n):
        for _ in range(max_attempts):
            p = _one(synth, logic, rng, mandatory_steps, label, distractors, require)
            if p is not None:
                out.append(p)
                break
        else:
            raise GenerationError(f"no {logic} problem with {mandatory_steps} mandatory steps"
                                  f"{' and label ' + label if label else ''}"
                                  f"{' using ' + '/'.join(sorted(require)) if require else ''} in {max_attempts} attempts")
    return out[0] if n is None else out


def rules(logic: str) -> list:
    """Names of a logic's inference rules (its calculus), for require_rules and families."""
    return sorted(get_pack(logic).calculus.rules)


def analyze(logic: str, premises: list, hypothesis, families: Optional[dict] = None) -> dict:
    """Proof-level audit of one problem written in a logic's formulas (relative to that logic's calculus).

    premises and hypothesis are formulas in the logic's syntax, e.g. for propositional logic
    ["a", ("->", "a", "b")] and "b". Returns, for the hypothesis (entailment) or its negation
    (contradiction), whichever the calculus derives:

      proofs                 number of minimal proofs (a lower bound when exact is False)
      mandatory_steps        rule applications every proof makes; mandatory_rules: the same, by rule
      depth                  levels of the shallowest proof
      rules_in_every_proof   rules every proof applies, with the least number of times (a rule can be
                             needed by every proof at different places, so this can exceed mandatory_rules)
      rule_share             for each rule, the share of minimal proofs that apply it
      families_in_every_proof  with families={"name": {rule, ...}}: whether every proof applies some rule
                             of the family, e.g. {"disjunction": {"disjunctive_syllogism",
                             "constructive_dilemma"}}; this is how an advertised operator is checked
      mandatory_premises     premises every proof uses

    A step or rule found optional here stays optional under any richer calculus. sf.summarize
    aggregates these over a dataset."""
    from .construct.calculus import Universe, rule_counts
    from .construct.calculus import analyze as _analyze
    pack = get_pack(logic)
    for label, goal in (("entailment", hypothesis), ("contradiction", pack.negate(hypothesis))):
        a = _analyze(pack.calculus, list(premises), goal, Universe([*premises, goal], pack.negate, pack.closed))
        if a.derivable:
            uses = [set(rule_counts(p)) for p in a.proofs]
            out = {"derivable": label, "proofs": a.n_proofs, "mandatory_steps": len(a.mandatory_steps),
                   "depth": a.depth, "mandatory_rules": dict(sorted(Counter(s[1] for s in a.mandatory_steps).items())),
                   "rules_in_every_proof": dict(sorted(a.required_rules.items())),
                   "rule_share": {r: round(sum(r in u for u in uses) / len(uses), 3)
                                  for r in sorted(set().union(*uses))},
                   "mandatory_premises": len(a.mandatory_premises), "premises": len(premises), "exact": a.exact}
            if families:
                out["families_in_every_proof"] = {f: all(u & set(rs) for u in uses) for f, rs in families.items()}
            return out
    return {"derivable": None, "premises": len(premises)}


def summarize(analyses: list, labels: Optional[list] = None) -> dict:
    """Dataset-level audit from per-problem sf.analyze results (and, optionally, the dataset's labels).

    Returns the share of problems the calculus derives, means of proofs, mandatory steps and depth over
    derivable problems, the share with no mandatory step, and for each rule (and family, if analyze was
    given families) the share of derivable problems where every proof applies it: an advertised operator
    at 0% is never needed. With labels ("entailment"/"contradiction"/anything else for neutral), also the
    share of problems where the derived label disagrees with the dataset's."""
    der = [a for a in analyses if a.get("derivable")]
    out = {"problems": len(analyses), "derivable": round(len(der) / len(analyses), 3) if analyses else None}
    if der:
        mean = lambda k: round(sum(a[k] for a in der) / len(der), 2)  # noqa: E731
        out.update(proofs=mean("proofs"), mandatory_steps=mean("mandatory_steps"), depth=mean("depth"),
                   no_mandatory_step=round(sum(a["mandatory_steps"] == 0 for a in der) / len(der), 3),
                   exact=round(sum(a["exact"] for a in der) / len(der), 3))
        rules = sorted({r for a in der for r in a["rule_share"]})
        out["rule_in_every_proof"] = {r: round(sum(r in a["rules_in_every_proof"] for a in der) / len(der), 3)
                                      for r in rules}
        fams = sorted({f for a in der for f in a.get("families_in_every_proof", {})})
        if fams:
            out["family_in_every_proof"] = {f: round(sum(a["families_in_every_proof"].get(f, False) for a in der)
                                                     / len(der), 3) for f in fams}
    if labels is not None:
        norm = lambda x: x if x in ("entailment", "contradiction") else None  # noqa: E731
        out["label_disagreement"] = round(sum(norm(l) != a.get("derivable") for a, l in zip(analyses, labels))
                                          / len(analyses), 3)
    return out


def audit_premises(premises: list, entails, limit: Optional[int] = 64) -> dict:
    """Premise-level audit of one problem, calculus-free: works on any dataset given an entailment oracle.

    entails(subset) -> bool says whether a subset of the premises still yields the label (e.g. a z3 call
    on the dataset's formulas). Enumerates the minimal premise sets that do (up to `limit`) and returns how
    many there are (> 1: an alternative justification exists), the smallest one's size, the premises in
    every minimal set (indispensable) and the share of premises in none (never needed)."""
    from .core.lattice import Oracle, enumerate_lattice
    items = list(range(len(premises)))
    sups, _, complete = enumerate_lattice(items, Oracle(lambda s: entails([premises[i] for i in sorted(s)])),
                                          limit=limit)
    if not sups:
        return {"supports": 0, "premises": len(premises), "complete": complete}
    used, every = set().union(*sups), set.intersection(*map(set, sups))
    return {"supports": len(sups), "complete": complete, "min_support": min(map(len, sups)),
            "indispensable": sorted(every), "never_needed": round(1 - len(used) / len(premises), 3),
            "premises": len(premises)}
