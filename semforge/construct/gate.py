"""A learned gate in front of the oracle: skip entailment queries z3 is predicted not to settle.

Labels never depend on the gate. A query it skips is treated like a z3 timeout (`Undecided`: the attempt
fails and the instance is dropped); a query it lets through is decided by z3 as before. A wrong
prediction therefore only costs time (a timeout paid) or yield (a decidable instance skipped), and the
gate is judged on exactly that trade-off (benchmarks/bench_gate.py).

Features are generic, read off the formulas with the signature only: size, depth, nesting of symbols over
formulas (modal depth), and counts per head symbol and connective.
"""
from __future__ import annotations

import time
from dataclasses import replace

from .oracle import Undecided
from .signature import CONNECTIVES, PROP


def _walk(f, sig, depth=0, nest=0, acc=None):
    acc = acc if acc is not None else {"nodes": 0, "depth": 0, "nesting": 0}
    acc["nodes"] += 1
    acc["depth"] = max(acc["depth"], depth)
    if not isinstance(f, tuple):
        return acc
    acc[f"n_{f[0]}"] = acc.get(f"n_{f[0]}", 0) + 1
    over = f[0] not in CONNECTIVES and PROP in sig.symbols[f[0]].args
    acc["nesting"] = max(acc["nesting"], nest + over)
    for a in f[1:]:
        _walk(a, sig, depth + 1, nest + over, acc)
    return acc


def features(sig, premises, goal, heads) -> list:
    """A fixed-length vector: premise count, then nodes, depth, nesting, per-head counts (in `heads`
    order), for the premises together and for the goal."""
    def vec(fs):
        acc = {"nodes": 0, "depth": 0, "nesting": 0}
        for f in fs:
            _walk(f, sig, acc=acc)
        return [acc["nodes"], acc["depth"], acc["nesting"], *(acc.get(f"n_{h}", 0) for h in heads)]
    return [len(premises), *vec(premises), *vec([goal])]


def heads_of(sig) -> list:
    return [*CONNECTIVES, *sorted(sig.symbols)]


def logged(pack, log: list):
    """The pack with its oracle recording (features, outcome, seconds) per query; outcome is
    "yes" | "no" | "undecided"."""
    heads, inner = heads_of(pack.signature), pack.entails

    def entails(premises, goal):
        t, x = time.time(), features(pack.signature, premises, goal, heads)
        try:
            r = inner(premises, goal)
        except Undecided:
            log.append((x, "undecided", time.time() - t))
            raise
        log.append((x, "yes" if r else "no", time.time() - t))
        return r
    return replace(pack, entails=entails)


def gated(pack, model, threshold: float):
    """The pack with queries skipped (raised as Undecided) when model predicts undecided with
    probability above threshold. `model`: anything with predict_proba over `features` vectors."""
    heads, inner = heads_of(pack.signature), pack.entails

    def entails(premises, goal):
        if model.predict_proba([features(pack.signature, premises, goal, heads)])[0][1] > threshold:
            raise Undecided("predicted undecided")
        return inner(premises, goal)
    return replace(pack, entails=entails)
