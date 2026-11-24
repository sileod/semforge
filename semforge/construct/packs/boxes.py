"""Boxes: objects put into, removed from and moved between boxes over numbered steps (entity tracking).

The premises say where every object is at step 0 (in a box or in none) and what happens at each later
step: an object is put into a box, removed from one, moved from one box to another, or the contents of
a box are moved to another. The hypothesis is where an object is (or is not) at a step. This is the
state-tracking setting of Kim & Schuster (2023) and of reasoning-core's tracking task.

What the pack reuses and what it adds, against the other packs:
- reused: fixed sorts with fixed wording, background facts (successor, distinct objects and places) as
  in planning, symbolic z3 semantics with frame axioms, rule soundness by check.prove_rule;
- added: nothing in the library. Being in no box is a place of its own (`nowhere`, named by a background
  fact), so removal is a move like the others. The frame (what an event leaves unchanged) is one rule per
  event type, so an object's position at a step depends on the event at that step even when the event is
  about something else: every event is a mandatory premise (it must be read). What a story requires is
  the events whose content matters: those whose replacement by `idle` (nothing happens) loses the
  conclusion (`required_events`). Depth, which every proof shares, says little here.
"""
from __future__ import annotations

from itertools import combinations

import z3

from ..pack import declare
from ..signature import Signature, Sort, Symbol, rule

K = 40  # steps 0..K-1
OBJECTS = ["fish", "shell", "bill", "file", "bowl", "cheese", "card", "bread", "fig", "camera", "mirror",
           "stone", "coat", "brick", "cake", "glass", "key", "ring", "book", "map", "leaf", "wire", "plate",
           "watch", "rope", "coin", "lamp", "bell", "drum", "flag"]
BOXES = [f"box{i}" for i in range(7)]
NOWHERE = "nowhere"
PLACES = BOXES + [NOWHERE]
STEPS = [f"t{i}" for i in range(K)]


def _num(m, t):
    return m.fn("step", "Time", "Int")(t)


def _loc(m, o, t):
    return m.fn("loc", "Obj", "Time", "Place")(o, t)


def _event(name, *sorts):
    return lambda m, *args: m.fn(name, *sorts, "Prop")(*args)


_put, _remove = _event("put", "Obj", "Place", "Time"), _event("remove", "Obj", "Place", "Time")
_move, _contents = _event("move", "Obj", "Place", "Place", "Time"), _event("contents", "Place", "Place", "Time")
_idle = _event("idle", "Time")


def _frame(m):
    """Steps are 0..K-1 with their intended values. At the next step: a put or moved object is in the
    target place, a removed one in no box, every object of a box whose contents are moved is in the target
    place; an event leaves every other object where it was."""
    t, u = z3.Const("_t", m.sort("Time")), z3.Const("_u", m.sort("Time"))
    o, o2 = z3.Const("_o", m.sort("Obj")), z3.Const("_o2", m.sort("Obj"))
    a, b = z3.Const("_a", m.sort("Place")), z3.Const("_b", m.sort("Place"))
    nxt = _num(m, u) == _num(m, t) + 1
    nowhere = z3.Const(f"Place_{NOWHERE}", m.sort("Place"))
    stay = _loc(m, o, u) == _loc(m, o, t)
    return z3.And(
        z3.ForAll([t], z3.And(0 <= _num(m, t), _num(m, t) < K)),
        z3.ForAll([t, u], z3.Implies(_num(m, t) == _num(m, u), t == u)),
        z3.ForAll([o, b, t, u], z3.Implies(z3.And(nxt, _put(m, o, b, t)), _loc(m, o, u) == b)),
        z3.ForAll([o, a, t, u], z3.Implies(z3.And(nxt, _remove(m, o, a, t)), _loc(m, o, u) == nowhere)),
        z3.ForAll([o, a, b, t, u], z3.Implies(z3.And(nxt, _move(m, o, a, b, t)), _loc(m, o, u) == b)),
        z3.ForAll([o, o2, b, t, u], z3.Implies(z3.And(nxt, _put(m, o2, b, t), o != o2), stay)),
        z3.ForAll([o, o2, a, t, u], z3.Implies(z3.And(nxt, _remove(m, o2, a, t), o != o2), stay)),
        z3.ForAll([o, o2, a, b, t, u], z3.Implies(z3.And(nxt, _move(m, o2, a, b, t), o != o2), stay)),
        z3.ForAll([o, t, u], z3.Implies(z3.And(nxt, _idle(m, t)), stay)),
        z3.ForAll([o, a, b, t, u], z3.Implies(z3.And(nxt, _contents(m, a, b, t)),
                                              z3.If(_loc(m, o, t) == a, _loc(m, o, u) == b, stay))),
        *(_num(m, z3.Const(f"Time_{s}", m.sort("Time"))) == i for i, s in enumerate(STEPS)))


S = lambda name, args, templates=(), negated=(), z3f=None, unordered=(): \
    Symbol(name, args, templates, negated, z3f, unordered=unordered, leaf=False)

SYMBOLS = [
    Symbol("in", ("Obj", "Place", "Time"), ("at {2}, {0} is in {1}", "{0} is in {1} at {2}"),
           ("at {2}, {0} is not in {1}", "{0} is not in {1} at {2}"),
           lambda m, o, p, t: _loc(m, o, t) == p),
    S("put", ("Obj", "Place", "Time"), ("at {2}, {0} is put into {1}",), ("at {2}, {0} is not put into {1}",), _put),
    S("remove", ("Obj", "Place", "Time"), ("at {2}, {0} is removed from {1}",),
      ("at {2}, {0} is not removed from {1}",), _remove),
    S("move", ("Obj", "Place", "Place", "Time"), ("at {3}, {0} is moved from {1} to {2}",),
      ("at {3}, {0} is not moved from {1} to {2}",), _move),
    S("contents", ("Place", "Place", "Time"), ("at {2}, the contents of {0} are moved to {1}",),
      ("at {2}, the contents of {0} are not moved to {1}",), _contents),
    S("idle", ("Time",), ("at {0}, nothing happens",), ("at {0}, something happens",), _idle),
    # background theory
    S("succ", ("Time", "Time"), z3f=lambda m, t, u: _num(m, u) == _num(m, t) + 1),
    S("nowhere", ("Place",), z3f=lambda m, p: p == z3.Const(f"Place_{NOWHERE}", m.sort("Place"))),
    S("other_obj", ("Obj", "Obj"), z3f=lambda m, a, b: a != b, unordered=((0, 1),)),
    S("other_place", ("Place", "Place"), z3f=lambda m, a, b: a != b, unordered=((0, 1),)),
]


def _background():
    return tuple([("succ", a, b) for a, b in zip(STEPS, STEPS[1:])] + [("nowhere", NOWHERE)] +
                 [("other_obj", a, b) for a, b in combinations(OBJECTS, 2)] +
                 [("other_place", a, b) for a, b in combinations(PLACES, 2)])


THEORY = (f"The steps are numbered 0 to {K - 1}. At each step every object is in exactly one box or in none.",
          "An object put into a box, or moved to it, is in that box at the next step; a removed object is in "
          "no box at the next step; when the contents of a box are moved to another box, everything in the "
          "first box is in the second at the next step.",
          "An event leaves every object it does not concern where it was.")

SIGNATURE = Signature.of(
    sorts=[Sort("Obj", fixed=tuple((o, f"the {o}") for o in OBJECTS)),
           Sort("Place", fixed=tuple((b, f"Box {b[3:]}") for b in BOXES) + ((NOWHERE, "no box"),)),
           Sort("Time", fixed=tuple((s, f"step {i}") for i, s in enumerate(STEPS)))],
    symbols=SYMBOLS, connectives=("not",), axioms=(_frame,), background=_background(), theory=THEORY)

RULES = [
    rule("put", "(put O B T), (succ T T2) |- in O B T2"),
    rule("remove", "(remove O A T), (nowhere N), (succ T T2) |- in O N T2"),
    rule("move", "(move O A B T), (succ T T2) |- in O B T2"),
    rule("contents", "(contents A B T), (in O A T), (succ T T2) |- in O B T2"),
    rule("frame_put", "(in O P T), (put O2 B T), (other_obj O O2), (succ T T2) |- in O P T2", kind="frame"),
    rule("frame_remove", "(in O P T), (remove O2 A T), (other_obj O O2), (succ T T2) |- in O P T2", kind="frame"),
    rule("frame_move", "(in O P T), (move O2 A B T), (other_obj O O2), (succ T T2) |- in O P T2", kind="frame"),
    rule("frame_contents", "(in O P T), (contents A B T), (other_place P A), (succ T T2) |- in O P T2",
         kind="frame"),
    rule("frame_idle", "(in O P T), (idle T), (succ T T2) |- in O P T2", kind="frame"),
    rule("unique", "(in O P T), (other_place P Q) |- not (in O Q T)"),
]

PACK = declare("boxes", SIGNATURE, RULES, atlas_size=2)


def closure(premises, goals=()):
    """Everything derivable from `premises`. The universe also admits the place `nowhere`, which the
    premises need not mention (a removed object goes there)."""
    from ..calculus import Closure, Universe
    calc = PACK.calculus
    ps = [calc.canonical(p) for p in premises]
    return Closure(calc, ps, Universe([*ps, *goals, ("nowhere", NOWHERE)], PACK.negate, PACK.closed)).round


def required_events(premises, goals):
    """Events whose replacement by `idle` at the same step makes some of `goals` underivable (all of them
    derivable from `premises`)."""
    goals = [PACK.calculus.canonical(g) for g in goals]

    def derives(ps):
        have = closure(ps, goals)
        return all(g in have for g in goals)
    events = [p for p in premises if p[0] in ("put", "remove", "move", "contents")]
    return [e for e in events if not derives([("idle", e[-1]) if p == e else p for p in premises])]


entails, render = PACK.entails, PACK.render
