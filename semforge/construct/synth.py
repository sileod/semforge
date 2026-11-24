"""Target-conditioned synthesis: construct toward a target, then certify what was realized.

  1. build the intended proof backwards from the target's rules; for several proofs, build alternatives
     of one node of it whose path to the conclusion already makes the shared rule applications, so
     those stay in every proof (a bottleneck) while the part below the node is duplicated
  2. pose the question: entailment asks the conclusion; contradiction asks its negation; neutral removes
     one premise of every proof
  3. add distractors: unrelated derivations, and near misses (a derivation of the hypothesis or of its
     negation with exactly one premise withheld, so the instance is one premise away from another label)
  4. certify: the label by semantics (complete), the structure by the calculus (exact relative to it).
     Entailment and contradiction get a *realized* certificate of the instance itself. Neutral has no
     proof to certify; it gets a *counterfactual* certificate of the instance with its withheld premises
     restored (the proof it is one fact away from), kept apart so the two are never pooled
  5. repair shortcuts: proofs that avoid a required rule application are cut by deleting a minimum
     hitting set of their premises outside the intended proof, then everything is certified again

Failure reasons are counted, so cost per realized target is measurable.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from ..core.patterns import Pattern, enforce, kinds_of
from .atlas import atlas_for, certificate_fields, to_json
from .build import Fresh, derive, derive_shape, shape_topology
from .calculus import Analysis, Universe, analyze, rule_of, step_id
from .oracle import Undecided, countermodel, query
from .pack import Pack
from .target import Target


def _order(steps, known) -> list:
    """A premise-first order for a derivation's steps (parents known before the step that uses them);
    the construction order is conclusion-first. Falls back to the given order if a parent is neither
    given nor derived by another step (a well-formed witness never needs it)."""
    out, todo = [], list(steps)
    while todo:
        e = next((s for s in todo if set(s[2]) <= known), None)
        if e is None:
            return steps
        out.append(e)
        known = known | {e[1]}
        todo.remove(e)
    return out


def _visible_plan(calc, main, universe, premises) -> list:
    """The part of the intended derivation the certificate can speak of: steps whose conclusion is
    inside the analysis universe and whose parents are premises or conclusions of other visible
    steps, in a premises-first valid order. A construction may build scaffolding the universe cannot
    see (e.g. an introduced conjunction used then immediately eliminated): structural claims are
    relative to the universe, so those steps are neither mandatory nor bypassable in the eyes of the
    certificate and are excluded, no differently from how the closure never sees them."""
    steps = [(calc.rules[n].family, main.nodes[at], tuple(main.nodes[i] for i in ids))
             for n, at, ids in reversed(main.steps)]
    bg = calc.background_set
    known, out = set(premises), []
    while True:
        add = [e for e in steps if e not in out and e[1] in universe
               and all(p in known or p in bg for p in e[2])]
        if not add:
            return out
        out.extend(add)
        known = known | {e[1] for e in add}


def negation_depth(f, neg="not") -> int:
    """Longest chain of directly nested negations (readability, not logic)."""
    if not isinstance(f, tuple):
        return 0
    inner = max(negation_depth(a, neg) for a in f[1:])
    return 1 + negation_depth(f[1], neg) if f[0] == neg else inner


def certificate(a: Optional[Analysis]) -> Optional[dict]:
    if a is None:
        return None
    return {"required_rules": dict(sorted(a.required_rules.items())), "proofs": a.n_proofs,
            "premise_supports": len(a.premise_supports), "depth": a.depth, "size": a.size}


def export_certificate(pack: Pack, inst: "Instance") -> Optional[dict]:
    """A self-contained certificate for `semforge.certcheck`: rules, premises, goal, universe, one proof
    and the mandatory steps and premises, plus the label queries as SMT-LIB. For a neutral instance the
    structural claims are counterfactual: they hold once the withheld premises are restored, and two
    finite countermodels certify the label itself; an entailment or contradiction carries the one
    countermodel that shows its premises consistent."""
    a = inst.analysis
    if a is None and inst.label != "neutral":  # e.g. forward instances, certified on demand
        a = analyze(pack.calculus, inst.premises, inst.conclusion, inst.universe)
    if a is None or not a.derivable:
        return None
    edges, _ = a._witness()
    rnd = a.closure.round
    order = sorted(edges, key=lambda k: rnd[a.edges[k][1]])
    proof = [{"kind": a.edges[k][0].family, "head": to_json(a.edges[k][1]), "parents": to_json(a.edges[k][2])}
             for k in order]
    ids = [step_id(*a.edges[k]) for k in order]
    sig = pack.signature
    cert = {
        **certificate_fields(pack, a.premises, a.goal, inst.universe), "label": inst.label,
        "hypothesis": to_json(inst.hypothesis),
        "proof": proof,
        "mandatory_steps": [n for n, x in enumerate(ids) if x in a.mandatory_steps],
        "mandatory_premises": to_json(tuple(sorted(a.mandatory_premises, key=repr))),
    }
    if inst.label == "neutral":
        cert["withheld"] = to_json(tuple(inst.withheld))  # premises the counterfactual proof adds
    if inst.bypassed:
        # requested exposure: the intended derivation, and for each planned application certified
        # bypassable a witness proof of the same conclusion that never makes it. certcheck verifies the
        # witnesses, and that the planned steps split exactly into mandatory and bypassed.
        cert["planned"] = [{"kind": k, "head": to_json(h), "parents": to_json(ps)} for k, h, ps in inst.planned]
        cert["bypassed"] = [{"step": i, "proof": [{"kind": k, "head": to_json(h), "parents": to_json(ps)}
                                               for k, h, ps in steps]} for i, steps in inst.bypassed]
    if sig is not None:
        negated = pack.negate(inst.hypothesis)
        cert["smt2"] = query(sig, inst.premises, inst.hypothesis).to_smt2()  # unsat iff entailment
        cert["smt2_contradiction"] = query(sig, inst.premises, negated).to_smt2()  # unsat iff contradiction
        # A neutral label needs both countermodels. An entailment or contradiction needs the one refuting the
        # opposite label: a model of the premises with the hypothesis true (false), so the premises are
        # consistent and the proof's label is not the vacuous one of an inconsistent premise set.
        refute = {"neutral": ("entailment", "contradiction"), "entailment": ("contradiction",),
                  "contradiction": ("entailment",)}[inst.label]
        goals = {"entailment": inst.hypothesis, "contradiction": negated}
        cms = [(r, countermodel(sig, inst.premises, goals[r])) for r in refute]
        cert["countermodels"] = [{"refutes": r, "model": cm["model"]} for r, cm in cms if cm is not None]
    return cert


@dataclass
class Instance:
    premises: list          # formulas, in presentation order
    hypothesis: object
    label: str
    conclusion: object      # what the certified proofs derive: H (E), ¬H (C), H once `withheld` is restored (N)
    universe: Universe
    target: Optional[Target] = None
    analysis: Optional[Analysis] = None        # E/C: of this instance; N: counterfactual (with `withheld`)
    roles: dict = field(default_factory=dict)  # premise -> "proof" | "alternative" | "irrelevant" | "near_miss"
    withheld: list = field(default_factory=list)
    near_misses: list = field(default_factory=list)  # (missing premise, formula it would complete)
    planned: list = field(default_factory=list)      # the intended derivation as (kind, head, parents), premises
                                                     # first, when exposure was requested; the derivation a
                                                     # target's claims are phrased over
    bypassed: list = field(default_factory=list)     # [(index into planned, witness steps of a proof that never
                                                     # makes that application)] — set when it certifies

    @property
    def counterfactual(self) -> bool:
        return self.label == "neutral"

    @property
    def exposed(self) -> int:
        """Planned applications certified bypassable (some proof of the conclusion avoids them)."""
        return len(self.bypassed)

    def realized(self) -> dict:
        roles = Counter(self.roles.values())
        cert = certificate(self.analysis)
        return {"label": self.label, "realized": None if self.counterfactual else cert,
                "counterfactual": cert if self.counterfactual else None, "withheld": len(self.withheld),
                "n_premises": len(self.premises), "near_miss": roles["near_miss"] > 0,
                "irrelevant": roles["irrelevant"] > 0, "exposed": self.exposed}


class Synthesizer:
    def __init__(self, pack: Pack, attempts: int = 8, repairs: int = 3, max_negations: int = 2, strict: bool = True,
                 atlas: bool = True, atlas_size: Optional[int] = None, learn: bool = True, policy=None, reward=None):
        """strict: reject instances whose certified structure misses the target; otherwise keep them
        (counted as "drift"), so the intended-vs-realized gap is measured rather than hidden.
        atlas: never build fragments the calculus can always bypass (construct.atlas), so planned
        steps are not lost to those shortcuts; the intended proof and sampled plans obey it.
        learn: when a planned step is still lost to a shortcut, lift the failure into a verified
        nogood (Atlas.lift), so no later construction repeats it.
        policy: orders rule choices in shape construction (construct.policy); a learning policy is
        updated after every attempt that built a derivation, with reward(instance or None), default 1
        if certified else 0. An attempt whose shape search fails builds nothing and updates nothing."""
        self.pack, self.attempts, self.repairs, self.max_negations = pack, attempts, repairs, max_negations
        self.strict = strict
        self.stats = Counter()
        self._scripts = {}  # plan -> construction script of a derivation that certifies it
        self.atlas = atlas_for(pack, atlas_size or pack.atlas_size) if atlas else None
        self.learn = learn and atlas
        self._forbid = self.atlas.violated if self.atlas else None
        self.policy, self.reward = policy, reward or (lambda inst: float(inst is not None))

    def realize(self, target: Target, rng) -> Optional[Instance]:
        if self.policy is not None:
            self.policy.begin(target)
        for _ in range(self.attempts):
            self.stats["attempts"] += 1
            self._main = None
            try:
                inst, why = self._attempt(target, rng)
            except Undecided:  # the label cannot be certified: a failed attempt
                inst, why = None, "undecided"
            self.stats[why] += 1
            if self.policy is not None and self._main is not None:
                self.policy.update(self.pack.calculus, self._main.steps, self.reward(inst))
            if inst is not None:
                return inst
        return None

    def _admissible(self, f) -> bool:
        return negation_depth(f) <= self.max_negations

    def _derive(self, plan, rng, fresh, **kw):
        return derive(self.pack.calculus, plan, rng, fresh, admissible=self._admissible, **kw)

    def sample_plan(self, rng, n, tries: int = 8) -> tuple:
        """Rules of a random n-step derivation in this pack that, on its own, certifies all of them as
        required: a plan that is placeable and realizable by construction. (Plans the calculus can
        shortcut, e.g. two instantiations next to Barbara, are skipped.) Its construction script is
        kept, so realizing the plan can replay that tree instead of searching for one."""
        pack, fallback = self.pack, None
        for _ in range(tries):
            d = self._derive([None] * n, rng, Fresh(), deep=rng.random() < 0.5, forbid=self._forbid,
                             max_depth=max(6, n // 3 + 4))  # long plans need room to grow
            if d is None:
                continue
            plan, fallback = tuple(d.rules), fallback or tuple(d.rules)
            universe = Universe([*d.premises, *d.intermediates], pack.negate, pack.closed)
            a = analyze(pack.calculus, d.premises, d.conclusion, universe)
            if a.requires(pack.calculus.kinds(plan)):
                if len(self._scripts) > 10000:
                    self._scripts.clear()
                self._scripts[plan] = d.steps
                return plan
        return fallback or tuple(self._random_plan(rng, n))

    def _random_plan(self, rng, k):
        return rng.sample(sorted(self.pack.calculus.rules), k=min(k, len(self.pack.calculus.rules)))

    def _attempt(self, target, rng):
        pack, fresh = self.pack, Fresh()
        deep = target.min_depth > 1 and rng.random() < 0.8
        forbid = self._forbid if not target.exposed else None  # pruning that removes shortcuts would
        # remove exactly the bypasses the target asks for, so an exposure request builds without it
        if target.shape is not None:
            main = self._main = derive_shape(pack.calculus, target.shape, rng, fresh, plan=target.rules or None,
                                             admissible=self._admissible, forbid=forbid,
                                             prefer=target.prefer, policy=self.policy)
        else:
            script = self._scripts.get(target.rules) if rng.random() < 0.5 else None  # keep shapes varied
            main = ((script and self._derive(target.rules, rng, fresh, script=script))
                    or self._derive(target.rules, rng, fresh, max_depth=target.max_depth, deep=deep,
                                    forbid=forbid))
        if main is None:
            return None, "unplaceable"
        # with a shape and free rules, the intended proof's own rules are what must stay necessary
        want = pack.calculus.kinds(target.required if target.rules else main.rules)
        roles = {p: "proof" for p in main.premises}
        derivations, withheld, near = [main], [], []
        if target.proofs > 1:
            # branch below a node whose path to the conclusion makes every shared rule application
            req, kinds = self.pack.calculus.kinds(target.required), self.pack.calculus.kinds
            branches = [i for i in range(len(main.nodes)) if not req - kinds(main.above(i))]
            if not branches:
                return None, "no_bottleneck"
            node = rng.choice(branches)
            k = max(1, len(target.rules) - sum(main.above(node).values()))
            for _ in range(target.proofs - 1):
                tries = (self._derive(self._random_plan(rng, k), rng, fresh, root=main.nodes[node],
                                      max_depth=target.max_depth) for _ in range(4))
                alt = next(filter(None, tries), None)
                if alt is None:
                    return None, "unplaceable_alternative"
                derivations.append(alt)
                roles.update({p: "alternative" for p in alt.premises if p not in roles})

        goal = main.conclusion
        if target.label == "entailment":
            hypothesis = goal
        elif target.label == "contradiction":
            hypothesis = pack.negate(goal)
        else:
            hypothesis = goal
            for d in derivations:  # withhold one premise of every proof
                p = rng.choice(d.premises)
                if roles.pop(p, None):
                    withheld.append(p)

        for _ in range(target.distractors.irrelevant):
            d = self._derive(self._random_plan(rng, rng.randint(1, 2)), rng, fresh)
            if d is not None:
                roles.update({p: "irrelevant" for p in d.premises if p not in roles})
                derivations.append(d)
        for _ in range(target.distractors.near_miss):
            aim = rng.choice([hypothesis, pack.negate(hypothesis)])
            d = self._derive(self._random_plan(rng, rng.randint(1, 2)), rng, fresh, root=aim)
            if d is None or len(d.premises) < 2:
                continue
            missing = rng.choice(d.premises)
            if missing in roles:
                continue
            roles.update({p: "near_miss" for p in d.premises if p != missing and p not in roles})
            derivations.append(d)
            near.append((missing, aim))

        if any(negation_depth(x) > self.max_negations for x in [*roles, hypothesis]):
            return None, "unreadable"
        premises = list(roles)
        rng.shuffle(premises)
        universe = Universe([*premises, hypothesis], pack.negate, pack.closed)  # what a reader sees, nothing planted
        plan_ids = None
        if target.exposed:  # exposure ranges over the applications certification can see
            inst_planned = _visible_plan(pack.calculus, main, universe, premises)
            plan_ids = [("step", *e) for e in inst_planned] \
                if any(e[1] == main.conclusion for e in inst_planned) else None  # scaffolding-covered goal:
            # e.g. arithmetic's computed conclusions stay outside the universe (shape
            # construction there fails), so no exposure claim can be phrased and the request fails
        for _ in range(self.repairs + 1):
            inst, why = self._certify(target, premises, hypothesis, goal, universe, roles, withheld, want,
                                      plan=plan_ids)
            if inst is not None:
                inst.near_misses = near
                if target.exposed and plan_ids:
                    inst.planned = inst_planned
                    inst.bypassed = self._bypassed(inst, plan_ids, inst_planned)
            if why != "shortcut":
                return inst, why
            drop = self._shortcut_cut(inst, want, main.premises)
            if not drop:
                break
            premises = [p for p in premises if p not in drop]
            self.stats["repaired"] += 1
        if self.learn and not target.exposed and inst is not None and inst.analysis.derivable:
            missing = want - inst.analysis.mandatory_kinds
            if self.atlas.lift(main.steps, set(missing), analysis=inst.analysis, nodes=main.nodes) is not None:
                self.stats["learned"] += 1
        return (None, "shortcut") if self.strict else (inst, "drift")

    @staticmethod
    def _bypassed(inst, plan_ids, planned):
        """The planned applications that are not mandatory, each with a witness proof that avoids it:
        the target's requested exposure, as identities and not as kind counts."""
        out = []
        for i, sid in enumerate(plan_ids):
            if sid not in inst.analysis.mandatory_steps:
                witness = inst.analysis.proof_without(sid)
                if witness:  # always: a non-mandatory application is avoided by some proof
                    steps = [(r, h, tuple(ps)) for _, r, h, ps in witness]
                    out.append((i, _order(steps, set(inst.analysis.given))))
        return out

    def _certify(self, target, premises, hypothesis, goal, universe, roles, withheld, want, plan=None):
        pack = self.pack
        label = pack.label(premises, hypothesis)
        if label == "entailment" and pack.entails(premises, pack.negate(hypothesis)):
            return None, "inconsistent"
        if label != target.label:
            return None, f"label_{label}"
        inst = Instance(premises, hypothesis, label, goal, universe, target,
                        roles={p: roles[p] for p in premises})
        if label == "neutral":
            inst.withheld = withheld
            premises = premises + withheld
        a = inst.analysis = analyze(pack.calculus, premises, goal, universe, cap=64)
        if not a.derivable:
            return None, "underivable"
        if a.depth < target.min_depth:
            return (None, "shallow") if self.strict else (inst, "drift")
        if target.exposed:  # identity-level: exactly k planned applications are bypassable, the rest mandatory
            n = sum(1 for sid in plan or () if sid not in a.mandatory_steps)
            if n != target.exposed:
                return (None, f"exposed_{n}") if self.strict else (inst, "drift")
        elif not a.requires(want):  # every planned application is a mandatory step (no enumeration)
            return inst, "shortcut"
        if target.shape is not None and a.topology != shape_topology(target.shape):
            return (None, "misshapen") if self.strict else (inst, "drift")
        if target.proofs > 1 and a.n_proofs < target.proofs:  # n_proofs enumerates: only when asked
            return (None, "merged_proofs") if self.strict else (inst, "drift")
        return inst, "ok"

    def _shortcut_cut(self, inst, want, proof):
        """Fewest premises to delete so every remaining proof makes the target's required rule
        applications, never touching the intended proof's premises; None when impossible."""
        a = inst.analysis
        if not a.exact:  # too many proofs to choose deletions from
            return None
        kinds = [kinds_of(p, rule_of) for p in a.proofs]
        keep = frozenset(proof) | {x for p in a.proofs for x in p if rule_of(x) is not None}
        return enforce(Pattern(None, None, a.proofs, kinds, a.exact), keep, rule_of,
                       want=want)
