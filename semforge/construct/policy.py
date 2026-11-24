"""Construction policies: in which order shape construction tries rules at an open node.

`derive_shape` backtracks over rule choices; a policy only orders the candidates. Pruning (atlas,
admissibility), backtracking and certification are untouched, so a policy can change cost, yield and
diversity but never what is certified. The default (`Policy`) shuffles, then tries the target's
preferred kinds first (kind steering). `FMPolicy` learns the order online: a factorization machine
scores (candidate rule, situation) pairs from sparse tokens, candidates are tried in a Plackett-Luce
sample of those scores (Gumbel noise, so search stays randomized), and after each attempt every step of
the derivation built is regressed toward the attempt's reward.
"""
from __future__ import annotations

import math
import zlib
from collections import Counter

import numpy as np


def situation(calculus, steps, node, kids: int, left: int) -> list:
    """Tokens describing an open node: the rule whose premise it is, its depth, how many derived
    premises it needs (`kids`), the size of the subtree still to build there (`left`), and the kinds
    already placed."""
    parent = {i: name for name, _, ids in steps for i in ids}
    at = {i: at for _, at, ids in steps for i in ids}
    depth, x = 0, node
    while x in at:
        depth, x = depth + 1, at[x]
    return [f"parent={parent.get(node, '-')}", f"depth={min(depth, 8)}", f"kids={kids}",
            f"left={min(left, 8)}", *{f"placed={calculus.rules[n].family}" for n, _, _ in steps}]


def _size(skeleton) -> int:
    return 1 + sum(map(_size, skeleton))


class Policy:
    """Shuffle, then preferred kinds first. Learns nothing."""

    def begin(self, target):
        self.prefer = target.prefer

    def order(self, calculus, pool, steps, node, skeleton, rng) -> list:
        pool = list(pool)
        rng.shuffle(pool)
        if self.prefer:
            pool.sort(key=lambda n: calculus.rules[n].family not in self.prefer)
        return pool

    def update(self, calculus, steps, reward: float):
        pass


class FMPolicy(Policy):
    """Online factorization machine over hashed tokens: r(x) = w0 + Σ w_i + Σ_{i<j} <v_i, v_j>."""

    def __init__(self, dim: int = 8, buckets: int = 1 << 12, lr: float = 0.05, temperature: float = 0.5, seed: int = 0):
        g = np.random.default_rng(seed)
        self.w0, self.w, self.v = 0.0, np.zeros(buckets), 0.01 * g.standard_normal((buckets, dim))
        self.buckets, self.lr, self.temperature = buckets, lr, temperature

    def begin(self, target):
        super().begin(target)
        self.goal = [f"label={target.label}", *(f"want={k}" for k in sorted(target.prefer or ()))]

    def _index(self, tokens) -> np.ndarray:
        # crc32, not hash(): str hashes are salted per process, so a fitted policy would read other buckets
        # in another process, and runs with the same seed would differ
        return np.array(sorted({zlib.crc32(t.encode()) % self.buckets for t in tokens}))

    def _tokens(self, calculus, name, situation_tokens) -> list:
        rule = [f"rule={name}", f"kind={calculus.rules[name].family}"]
        return rule + situation_tokens + self.goal + [f"{r}&{s}" for r in rule[1:] for s in situation_tokens[:1]]

    def score(self, idx) -> float:
        s = self.v[idx].sum(0)
        return self.w0 + self.w[idx].sum() + 0.5 * float(s @ s - (self.v[idx] ** 2).sum())

    def order(self, calculus, pool, steps, node, skeleton, rng) -> list:
        sit = situation(calculus, steps, node, len(skeleton), _size(skeleton))
        keyed = [(self.score(self._index(self._tokens(calculus, n, sit))) / self.temperature
                  - math.log(-math.log(rng.random() or 1e-12)), n) for n in pool]  # Plackett-Luce sample
        order = [n for _, n in sorted(keyed, reverse=True)]
        if self.prefer:  # the target's steer stays primary; the FM orders within each group
            order.sort(key=lambda n: calculus.rules[n].family not in self.prefer)
        return order

    def update(self, calculus, steps, reward: float):
        """Regress every placement of the derivation toward the attempt's reward (squared loss)."""
        below = {at: [i for i in ids if i in {a for _, a, _ in steps}] for _, at, ids in steps}

        def size(n):
            return 1 + sum(size(i) for i in below[n])
        for k, (name, node, _) in enumerate(steps):
            sit = situation(calculus, steps[:k], node, len(below[node]), size(node))
            idx = self._index(self._tokens(calculus, name, sit))
            err = self.score(idx) - reward
            s = self.v[idx].sum(0)
            self.w0 -= self.lr * err
            self.w[idx] -= self.lr * err
            self.v[idx] -= self.lr * err * (s - self.v[idx])


def kind_set(inst):
    """The multiset of mandatory step kinds, as a hashable key."""
    return tuple(sorted(inst.analysis.mandatory_kinds.items()))


def novelty_reward(key=kind_set, base: float = 1.0, bonus: float = 1.0):
    """Reward for a learning policy: 0 for a failed attempt; for a certified instance, base plus a
    count-based novelty bonus, bonus / sqrt(times its key was seen). The default key is the mandatory
    kind multiset; archive.structure is finer but nearly every large instance is new under it."""
    seen = Counter()

    def reward(inst) -> float:
        if inst is None:
            return 0.0
        k = key(inst)
        seen[k] += 1
        return base + bonus / math.sqrt(seen[k])
    return reward
