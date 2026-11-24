"""Bounded planning (an extension probe): a robot moves between rooms over numbered steps.

The premises give a trace (at each step the robot tries to go from one room to another, or waits),
which doors exist or do not, and where the robot is at some steps. The hypothesis is where it is
(or is not) at a step, or whether a door exists. An action has a precondition and an effect: a move
from A to B succeeds when the robot is in A and a door connects A and B, and otherwise changes
nothing; waiting changes nothing. So conclusions follow the trace forward (projection), backward
(through waits), and diagnostically: a move that left the robot where it was shows the door is
missing, and one that brought it to B shows the door exists.

What the probe reuses and what it adds, against the other packs:
- reused: fixed sorts with fixed wording, background facts (successor, distinct rooms) as in grid,
  symbolic z3 semantics with a frame axiom, rule soundness by check.prove_rule;
- added: nothing in the library. State change is encoded with a time argument and a position
  function; the frame (what an action leaves unchanged) is in the axioms, so rules stay plain.
- not covered: plan existence (is there a trace reaching a goal?) is not entailment from premises,
  and optimality is a separate question; both need a search outside the calculus.
"""
from __future__ import annotations

from itertools import combinations

import z3

from ..pack import declare
from ..signature import Signature, Sort, Symbol, rule

K = 10  # steps 0..K-1
ROOMS = ["kitchen", "hall", "study", "garden", "cellar", "attic"]
STEPS = [f"t{i}" for i in range(K)]


def _num(m, t):
    return m.fn("step", "Time", "Int")(t)


def _pos(m, t):
    return m.fn("pos", "Time", "Room")(t)


def _door(m, a, b):
    return m.fn("door", "Room", "Room", "Prop")(a, b)


def _move(m, a, b, t):
    return m.fn("move", "Room", "Room", "Time", "Prop")(a, b, t)


def _wait(m, t):
    return m.fn("wait", "Time", "Prop")(t)


def _frame(m):
    """Steps are distinct integers 0..K-1 with their intended values; doors are symmetric; a move from
    A to B at step t sets the next position to B when the robot is in A and the door exists, and leaves
    it unchanged otherwise; waiting leaves it unchanged."""
    t, u = z3.Const("_t", m.sort("Time")), z3.Const("_u", m.sort("Time"))
    a, b = z3.Const("_a", m.sort("Room")), z3.Const("_b", m.sort("Room"))
    nxt = _num(m, u) == _num(m, t) + 1
    here = z3.And(_pos(m, t) == a, _door(m, a, b))
    return z3.And(
        z3.ForAll([t], z3.And(0 <= _num(m, t), _num(m, t) < K)),
        z3.ForAll([t, u], z3.Implies(_num(m, t) == _num(m, u), t == u)),
        z3.ForAll([a, b], _door(m, a, b) == _door(m, b, a)),
        z3.ForAll([a, b, t, u], z3.Implies(z3.And(nxt, _move(m, a, b, t), here), _pos(m, u) == b)),
        z3.ForAll([a, b, t, u], z3.Implies(z3.And(nxt, _move(m, a, b, t), z3.Not(here)), _pos(m, u) == _pos(m, t))),
        z3.ForAll([t, u], z3.Implies(z3.And(nxt, _wait(m, t)), _pos(m, u) == _pos(m, t))),
        *(_num(m, z3.Const(f"Time_{s}", m.sort("Time"))) == i for i, s in enumerate(STEPS)))


S = lambda name, args, templates=(), negated=(), z3f=None, unordered=(): \
    Symbol(name, args, templates, negated, z3f, unordered=unordered, leaf=False)

SYMBOLS = [
    Symbol("at", ("Room", "Time"), ("at {1}, the robot is in {0}", "the robot is in {0} at {1}"),
           ("at {1}, the robot is not in {0}", "the robot is not in {0} at {1}"),
           lambda m, a, t: _pos(m, t) == a),
    S("door", ("Room", "Room"), ("a door connects {0} and {1}", "there is a door between {0} and {1}"),
      ("no door connects {0} and {1}", "there is no door between {0} and {1}"), _door, ((0, 1),)),
    S("move", ("Room", "Room", "Time"), ("at {2}, the robot tries to go from {0} to {1}",),
      ("at {2}, the robot does not try to go from {0} to {1}",), _move),
    S("wait", ("Time",), ("at {0}, the robot waits",), ("at {0}, the robot does not wait",), _wait),
    # background theory
    S("succ", ("Time", "Time"), z3f=lambda m, t, u: _num(m, u) == _num(m, t) + 1),
    S("other", ("Room", "Room"), z3f=lambda m, a, b: a != b, unordered=((0, 1),)),
]


def _background():
    return tuple([("succ", a, b) for a, b in zip(STEPS, STEPS[1:])] +
                 [("other", a, b) for a, b in combinations(ROOMS, 2)])


THEORY = (f"The steps are numbered 0 to {K - 1}. At each step the robot is in exactly one room, and the rooms "
          f"({', '.join('the ' + r for r in ROOMS)}) are distinct.",
          "When the robot tries to go from one room to another at a step, it is in the second room at the next "
          "step if it was in the first room and a door connects the two; otherwise it stays where it was.",
          "When the robot waits at a step, it is in the same room at the next step.",
          "A door connects two rooms in both directions.")

SIGNATURE = Signature.of(
    sorts=[Sort("Room", fixed=tuple((r, f"the {r}") for r in ROOMS)),
           Sort("Time", fixed=tuple((s, f"step {i}") for i, s in enumerate(STEPS)))],
    symbols=SYMBOLS, connectives=("not",), axioms=(_frame,), background=_background(), theory=THEORY)

RULES = [
    rule("move", "(move A B T), (at A T), (door A B), (succ T T2) |- at B T2"),
    rule("blocked", "(move A B T), (at A T), not (door A B), (succ T T2) |- at A T2"),
    rule("stray", "(move A B T), (at C T), (other A C), (succ T T2) |- at C T2"),
    rule("wait", "(wait T), (at A T), (succ T T2) |- at A T2"),
    rule("wait_back", "(wait T), (at A T2), (succ T T2) |- at A T", kind="wait"),
    rule("wait_absent", "(wait T), not (at A T), (succ T T2) |- not (at A T2)"),
    rule("unique", "(at A T), (other A B) |- not (at B T)"),
    # diagnosis: the trace and two positions settle a door
    rule("no_door", "(move A B T), (at A T), (at A T2), (succ T T2), (other A B) |- not (door A B)"),
    rule("door", "(move A B T), (at A T), (at B T2), (succ T T2), (other A B) |- door A B"),
]

PACK = declare("planning", SIGNATURE, RULES, atlas_size=2)
entails, render = PACK.entails, PACK.render
