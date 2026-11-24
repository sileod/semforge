"""Formal proof generation and grading using an existing deductive pack.

The answer is a sequence of rule applications, not an entailment label or a
free-text rationale. Problem rules and universe come from the task, never from
the proposed answer. Checking uses the independent certificate checker.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..certcheck import Checker
from ..construct.atlas import certificate_fields, to_json
from ..construct.calculus import Universe, analyze
from ..construct.synth import Synthesizer
from ..construct.target import Distractors, Target
from .base import Check, Example, Result


def _formula(f):
    if isinstance(f, str) and f and not f.startswith("?"):
        return f
    if isinstance(f, (list, tuple)) and f and isinstance(f[0], str) and not f[0].startswith("?"):
        return (f[0], *(_formula(x) for x in f[1:]))
    raise ValueError("expected a ground formula (string or nonempty application)")


@dataclass(frozen=True)
class ProofProblem:
    premises: tuple
    goal: object
    universe: Universe


class ProofTask:
    """Construct proof exercises or solve/check a supplied goal in a bounded calculus.

    A failed search means no derivation in this calculus/universe, not semantic
    non-entailment. A successful witness need not have the fewest applications.
    """

    def __init__(self, pack, synthesizer=None):
        """`synthesizer`: a construct.synth.Synthesizer for `generate` (e.g. with a policy); default plain."""
        self.pack = pack
        self.name = f"proof:{pack.name}"
        self.scope = f"derivability in {pack.name}'s declared calculus and the supplied formula universe"
        self._synth = synthesizer

    def problem(self, premises, goal, *, universe=None):
        calc = self.pack.calculus
        premises = tuple(calc.canonical(_formula(f)) for f in premises)
        goal = calc.canonical(_formula(goal))
        if self.pack.signature is not None:
            self.pack.signature.infer([*premises, goal])
        if universe is None:
            universe = Universe([*premises, goal], self.pack.negate, self.pack.closed)
        return ProofProblem(premises, goal, universe)

    def from_instance(self, inst):
        """Ask for the certified conclusion: H for entailment, not-H for contradiction.

        Neutral instances have no actual proof; do not silently restore withheld
        premises. A caller can construct an explicit counterfactual problem instead.
        """
        if inst.counterfactual:
            raise ValueError("neutral instances do not supply a proof task; choose a goal and premises explicitly")
        return self.problem(inst.premises, inst.conclusion, universe=inst.universe)

    def present(self, problem):
        """The question without its formula universe: the universe bounds the solver's search and lists
        every intermediate conclusion of the reference proof (among distractors), so it would be a hint."""
        fields = certificate_fields(self.pack, problem.premises, problem.goal, problem.universe)
        del fields["universe"]
        return {"instruction": 'Derive the goal. Return a JSON list of steps with "kind", "head", and '
                               '"parents". Each parent must be given or derived earlier. Use only the '
                               'provided rule schemas; an empty proof is allowed.', **fields}

    def prompt(self, problem):
        return json.dumps(self.present(problem), ensure_ascii=False, indent=2)

    def certificate(self, problem, answer):
        """Checked in the problem's universe extended by the answer's own formulas: the universe only
        bounds search, and every rule is sound for all instances, so an answer may pass through any
        formula its rule applications produce."""
        cert = certificate_fields(self.pack, problem.premises, problem.goal, problem.universe)
        used = [f for step in answer if isinstance(step, dict) for f in (step.get("head"), *step.get("parents", ()))]
        cert["universe"]["literals"] = cert["universe"]["literals"] + [f for f in used if f is not None]
        return {**cert, "proof": answer}

    def check(self, problem, answer):
        if not isinstance(answer, (list, tuple)):
            return Check(False, self.scope, errors=("a proof must be a list of rule applications",))
        try:
            for step in answer:
                if not isinstance(step, dict) or set(step) != {"kind", "head", "parents"}:
                    raise ValueError("each step must have exactly kind, head and parents")
                if not isinstance(step["kind"], str) or not isinstance(step["parents"], (list, tuple)):
                    raise ValueError("kind must be a string and parents a list")
                _formula(step["head"])
                for parent in step["parents"]:
                    _formula(parent)
            errors, _ = Checker(self.certificate(problem, answer)).proof(answer)
        except (ValueError, TypeError, KeyError, IndexError, RecursionError) as e:
            return Check(False, self.scope, errors=(str(e),))
        return Check(not errors, self.scope, ("valid_derivation",) if not errors else (), tuple(errors))

    def solve(self, problem):
        a = analyze(self.pack.calculus, problem.premises, problem.goal, problem.universe)
        if not a.derivable:
            return Result("no_solution", self.scope, reason="closure contains no derivation of this goal")
        edges, _ = a._witness()
        order = sorted(edges, key=lambda k: (a.closure.round[a.edges[k][1]], repr(a.edges[k])))
        answer = [{"kind": a.edges[k][0].family, "head": to_json(a.edges[k][1]),
                   "parents": to_json(a.edges[k][2])} for k in order]
        checked = self.check(problem, answer)
        if not checked.valid:
            raise RuntimeError(f"generated proof failed independent checking: {checked.errors}")
        return Result("solved", self.scope, answer, claims=("valid_derivation",),
                      metrics={"proof_steps": len(answer), "min_depth": a.depth},
                      evidence=self.certificate(problem, answer))

    def generate(self, rng, *, shape=None, rules=(), min_depth=1, max_depth=6, distractors=None):
        """Generate a certified exercise and reference proof, or None on construction failure.

        Shape/rules are optional construction recipes. The requested min_depth is
        checked by the existing synthesizer. No NLI label is required from callers.
        """
        if shape is None and not rules:
            raise ValueError("provide a shape or rule plan for proof construction")
        if self._synth is None:
            self._synth = Synthesizer(self.pack)
        target = Target("entailment", rules=tuple(rules), shape=shape, min_depth=min_depth,
                        max_depth=max_depth, distractors=distractors or Distractors())
        inst = self._synth.realize(target, rng)
        if inst is None:
            return None
        problem = self.from_instance(inst)
        return Example(problem, self.solve(problem))
