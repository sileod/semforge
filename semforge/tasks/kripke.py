"""Modal model checking: is a formula true at a world of a finite, fully stated Kripke model?

Unlike the modal packs, the model is closed: the listed worlds are all the worlds, and only the listed
links and facts hold. A box is then provable from the complete list of a world's successors, which open
entailment cannot do. The solver is z3 over `modal_hybrid`'s semantics plus closed-world axioms
(`oracle.domain_closure`, `exactly`, `true_exactly_at`); the checker is an independent recursive
evaluator over the model, so the two must agree. The answer is True or False; there is no unknown.

A partially observed problem (`hidden` non-empty, `generate(..., hidden=0.2)`) keeps the worlds and links
closed but states each fact's truth only outside `hidden`: true facts and false ones are both listed,
and the hidden ones are not mentioned. The answer is then UNDECIDED when the stated facts do not decide
the formula (the checker evaluates every completion of the hidden facts). This gives modal rows with a
neutral label, so the closed-world reading ("unstated means false") is not the only one trained.

    task = KripkeTask()
    problem = task.generate(random.Random(0), worlds=4, depth=2)
    task.solve(problem).answer           # True or False
    task.row(problem, random.Random(0))  # an NLI row: model as premises, formula at the world as hypothesis
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

from ..construct.oracle import domain_closure, exactly, true_exactly_at, z3_entails
from ..construct.packs.modal_hybrid import SIGNATURE as HYBRID
from ..construct.realize import Realizer
from .base import Check, Result

FRAMES = ("K", "T", "S4", "S5")
CLOSED = ("These are all the places: no other place exists, no place is reachable from another unless "
          "stated, and nothing holds in a place unless stated.")
UNDECIDED = "undecided"
OPEN_FACTS = "These are all the places: no other place exists, and no place is reachable from another unless stated."


@dataclass(frozen=True)
class KripkeProblem:
    worlds: tuple       # world constants
    edges: frozenset    # (w, v): v is reachable from w
    facts: frozenset    # (atom, w): atom holds at w
    world: str          # where the formula is evaluated
    formula: object     # over atoms, "not", "and", "or", "box", "dia"
    frame: str = "K"    # how the edges were generated; the edges are the model either way
    hidden: frozenset = frozenset()  # (atom, w) pairs whose truth is not stated; empty: every fact is closed


def modal_depth(f) -> int:
    if isinstance(f, str):
        return 0
    return (f[0] in ("box", "dia")) + max(map(modal_depth, f[1:]))


def reduce(f, frame):
    """f simplified by identities valid on every model of the frame class: double negation, negation pushed
    through a modality, and collapsed adjacent modalities (S4: the same one twice; S5: any two). An upper
    bound on the least modal depth of an equivalent formula, not that least depth (review5)."""
    if isinstance(f, str):
        return f
    op, args = f[0], tuple(reduce(x, frame) for x in f[1:])
    if op == "not":
        g = args[0]
        if isinstance(g, tuple) and g[0] == "not":
            return g[1]
        if isinstance(g, tuple) and g[0] in ("box", "dia"):
            return reduce(("dia" if g[0] == "box" else "box", ("not", g[1])), frame)
    if op in ("box", "dia"):
        g = args[0]
        if isinstance(g, tuple) and g[0] in ("box", "dia") and (frame == "S5" or (frame == "S4" and g[0] == op)):
            return g
    return (op, *args)


def evaluate(p: KripkeProblem, f, w) -> bool:
    """Truth of f at w by recursion on f: the checker, independent of z3."""
    if isinstance(f, str):
        return (f, w) in p.facts
    op, args = f[0], f[1:]
    if op == "not":
        return not evaluate(p, args[0], w)
    if op == "and":
        return all(evaluate(p, a, w) for a in args)
    if op == "or":
        return any(evaluate(p, a, w) for a in args)
    successors = [v for v in p.worlds if (w, v) in p.edges]
    if op == "box":
        return all(evaluate(p, args[0], v) for v in successors)
    if op == "dia":
        return any(evaluate(p, args[0], v) for v in successors)
    raise ValueError(f"unknown operator {op!r}")


def close(edges, worlds, frame):
    """The edges closed under the frame's conditions (reflexive; transitive; symmetric)."""
    edges = set(edges)
    if frame in ("T", "S4", "S5"):
        edges |= {(w, w) for w in worlds}
    if frame == "S5":
        edges |= {(v, w) for w, v in edges}
    if frame in ("S4", "S5"):
        while True:
            more = {(a, d) for a, b in edges for c, d in edges if b == c} - edges
            if not more:
                break
            edges |= more
    return frozenset(edges)


class KripkeTask:
    name = "kripke"
    scope = "truth at a world of the stated finite Kripke model (closed world)"

    def premises(self, p: KripkeProblem) -> list:
        edges = [("sees", w, v) for w, v in sorted(p.edges)]
        if not p.hidden:
            return edges + [("at", w, a) for a, w in sorted(p.facts)]
        stated = sorted((a, w) for a in _props(p) for w in p.worlds if (a, w) not in p.hidden)
        return edges + [("at", w, a) if (a, w) in p.facts else ("not", ("at", w, a)) for a, w in stated]

    def signature(self, p: KripkeProblem):
        axioms = (domain_closure("World", p.worlds), exactly("access", "World", p.edges))
        if not p.hidden:
            axioms += tuple(true_exactly_at(a, "World", [w for b, w in p.facts if b == a]) for a in _props(p))
        return replace(HYBRID, sorts={"World": replace(HYBRID.sorts["World"], distinct=True)},
                       axioms=axioms, theory=(OPEN_FACTS if p.hidden else CLOSED,))

    def present(self, p: KripkeProblem) -> dict:
        return {"worlds": list(p.worlds), "edges": sorted(map(list, p.edges)), "facts": sorted(map(list, p.facts)),
                "world": p.world, "formula": p.formula}

    def solve(self, p: KripkeProblem) -> Result:
        """z3 over the hybrid semantics and the closed-world axioms. A closed problem entails exactly one of
        f and not f; a partially observed one may entail neither (UNDECIDED), never both."""
        entails, goal = z3_entails(self.signature(p)), ("at", p.world, p.formula)
        prem = self.premises(p)
        true, false = entails(prem, goal), entails(prem, ("not", goal))
        if true and false or (true == false and not p.hidden):
            return Result("unknown", self.scope, reason="the stated model entails both or neither: not a model")
        return Result("solved", self.scope, answer=UNDECIDED if true == false else true,
                      metrics={"modal_depth": modal_depth(p.formula),
                               "reduced_depth": modal_depth(reduce(p.formula, p.frame)), "worlds": len(p.worlds),
                               "edges": len(p.edges), "hidden": len(p.hidden)})

    def check(self, p: KripkeProblem, answer) -> Check:
        """Recursive evaluation in every completion of the hidden facts (one completion when none is hidden):
        True if the formula holds in all, False if in none, UNDECIDED otherwise."""
        if not isinstance(answer, bool) and not (answer == UNDECIDED and p.hidden):
            return Check(False, self.scope, errors=("the answer must be True, False or (with hidden facts) UNDECIDED",))
        hidden, known = sorted(p.hidden), p.facts - p.hidden
        values = {evaluate(replace(p, facts=known | {h for h, bit in zip(hidden, bits) if bit}), p.formula, p.world)
                  for bits in itertools.product((False, True), repeat=len(hidden))}
        truth = values.pop() if len(values) == 1 else UNDECIDED
        return Check(answer == truth, self.scope, claims=("recursive evaluation over the model",),
                     errors=() if answer == truth else (f"the formula is {truth} at {p.world}",))

    def generate(self, rng, worlds=4, depth=2, atoms=3, frame="K", density=0.35, world_names=None,
                 hidden=0.0) -> KripkeProblem:
        """A random model (each edge with probability `density`, then closed under the frame) and a random
        formula of exactly `depth` nested modalities; each fact is hidden with probability `hidden`."""
        ws = tuple(world_names or [f"w{i}" for i in range(worlds)])
        props = [f"p{i}" for i in range(atoms)]
        edges = close({(w, v) for w in ws for v in ws if rng.random() < density}, ws, frame)
        facts = frozenset((a, w) for a in props for w in ws if rng.random() < 0.5)
        concealed = frozenset((a, w) for a in props for w in ws if hidden and rng.random() < hidden)  # hidden=0 draws nothing
        return KripkeProblem(ws, edges, facts, rng.choice(ws), _formula(rng, props, depth), frame, concealed)

    def row(self, p: KripkeProblem, rng) -> dict:
        """An NLI row: the closed model and its closure sentence as premises, the formula at the world as
        hypothesis; true is entailment, false contradiction, undecided neutral. `answer_checked` is the
        independent check."""
        result = self.solve(p)
        if result.status != "solved" or self.check(p, result.answer).valid is not True:
            raise ValueError(f"solver and checker disagree on {p}")
        realizer = Realizer(self.signature(p))
        premises, hypothesis = realizer.render(self.premises(p), ("at", p.world, p.formula), rng)
        gold = {True: "entailment", False: "contradiction", UNDECIDED: "neutral"}[result.answer]
        return {"gold": gold, "premises": premises,
                "hypothesis": hypothesis, "answer_checked": True, "frame": p.frame, **result.metrics}


def _props(p: KripkeProblem) -> list:
    return sorted({a for a, _ in p.facts | p.hidden} | _atoms(p.formula))


def _atoms(f) -> set:
    return {f} if isinstance(f, str) else set().union(*map(_atoms, f[1:]))


def _formula(rng, props, depth):
    """Exactly `depth` nested modalities on one branch; connectives and atoms elsewhere."""
    if depth == 0:
        a = rng.choice(props)
        return ("not", a) if rng.random() < 0.3 else a
    inner = (rng.choice(("box", "dia")), _formula(rng, props, depth - 1))
    roll = rng.random()
    if roll < 0.2:
        return ("not", inner)
    if roll < 0.45:
        other = _formula(rng, props, rng.randrange(depth))
        return (rng.choice(("and", "or")), *((inner, other) if rng.random() < 0.5 else (other, inner)))
    return inner
