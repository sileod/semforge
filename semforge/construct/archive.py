"""Coverage archive: request the least-covered structure next, counted on what was *certified*.

Diversity comes from construction, not from filtering a pool: each request targets a niche (label,
required rule applications, number of proofs, depth, distractor types) that the archive has seen
least, among a random batch of candidates drawn at the requested difficulty level. Neutral niches come
from counterfactual certificates and are marked as such, so they never share a niche with realized ones.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from math import isfinite
from typing import Optional

from .calculus import analyze
from .target import Distractors, Target

LABELS = ("entailment", "contradiction", "neutral")


def _niche(label, required: Counter, proofs, depth, irrelevant, near_miss) -> tuple:
    kind = "counterfactual" if label == "neutral" else "realized"
    return (label, kind, tuple(sorted(required.items())), min(proofs, 3), min(depth, 4), irrelevant, near_miss)


def niche(inst) -> tuple:
    a, r = inst.analysis, inst.realized()
    return _niche(inst.label, a.required_rules, a.n_proofs, a.depth, r["irrelevant"], r["near_miss"])


class Archive:
    def __init__(self, rule_names, candidates: int = 32, sampler=None, kinds=Counter):
        """sampler(rng, n) -> a rule plan of about n applications that is known to be placeable
        (Synthesizer.sample_plan); without one, plans are drawn uniformly and may not be.
        kinds(names) -> Counter maps schema names to the kinds certificates count (Calculus.kinds)."""
        self.rules, self.candidates, self.sampler, self.kinds = sorted(rule_names), candidates, sampler, kinds
        self.coverage = Counter()

    def draw(self, rng, level: int = 1) -> Target:
        """A random target; level sets difficulty (plan length, proofs, depth, distractors)."""
        n = rng.randint(max(1, level), level + 2)
        rules = self.sampler(rng, n) if self.sampler else tuple(rng.choices(self.rules, k=n))
        proofs = rng.choice((1, 1, 2))
        shared = None
        if proofs > 1:  # a random sub-multiset of the plan stays a bottleneck
            shared = tuple(r for r in rules if rng.random() < 0.5)
        return Target(label=rng.choice(LABELS), rules=rules, proofs=proofs, shared=shared,
                      min_depth=rng.randint(1, max(1, min(n, level + 1))), max_depth=2 + 2 * level,
                      distractors=Distractors(irrelevant=rng.randint(0, level), near_miss=rng.randint(0, level)))

    def propose(self, rng, level: int = 1) -> Target:
        batch = [self.draw(rng, level) for _ in range(self.candidates)]
        return min(batch, key=lambda t: (self.coverage[self.expected(t)], rng.random()))

    def expected(self, t: Target) -> tuple:
        return _niche(t.label, self.kinds(t.required), t.proofs, t.min_depth, t.distractors.irrelevant > 0,
                      t.distractors.near_miss > 0)

    def add(self, inst):
        self.coverage[niche(inst)] += 1


# --- topology archive: invert descriptor cells into targets ---------------------------------------------

# A skeleton is a tree of steps, each a tuple of its derived children (leaves are steps from premises
# only). Steps have at most two derived children. Its topology, when every step is mandatory, is
# spine = height and joins = steps with two children (build.shape_topology). The functions below count
# ordered trees by (height, joins, size) and sample one uniformly, so a topology cell admits every
# arrangement: joins on the spine, on side branches, or balanced.

@lru_cache(maxsize=None)
def _exact(h: int, j: int, s: int) -> int:
    """Number of ordered trees of height exactly h with j joins and s steps."""
    if h < 1 or j < 0 or s < 1:
        return 0
    if h == 1:
        return int(j == 0 and s == 1)
    return _exact(h - 1, j, s - 1) + sum(n for _, n in _pairs(h, j, s))


@lru_cache(maxsize=None)
def _within(h: int, j: int, s: int) -> int:
    """Number of ordered trees of height at most h."""
    return sum(_exact(k, j, s) for k in range(1, h + 1))


@lru_cache(maxsize=None)
def _pairs(h: int, j: int, s: int) -> tuple:
    """A join step of height h: ((left, right) part summaries, count) for its two children, split so
    that each pair is counted once: the left child reaches height h-1, or else the right one does."""
    out = []
    for jl in range(j):
        for sl in range(1, s - 1):
            jr, sr = j - 1 - jl, s - 1 - sl
            for exact_left in (True, False):
                n = (_exact(h - 1, jl, sl) * _within(h - 1, jr, sr) if exact_left
                     else _within(h - 2, jl, sl) * _exact(h - 1, jr, sr))
                if n:
                    out.append(((exact_left, jl, sl, jr, sr), n))
    return tuple(out)


def max_joins(spine: int) -> int:
    return 2 ** (spine - 1) - 1


@lru_cache(maxsize=None)
def all_skeletons(h: int, j: int, s: int, exact: bool = True) -> tuple:
    """Every skeleton of height exactly (`exact`) h with j joins and s steps, listed once. The same
    recurrences as `_exact`/`_pairs`, so the count matches them (a test pins this): classifying a
    cell as impossible is a claim about every tree of the cell."""
    if not exact:
        return tuple(t for k in range(1, h + 1) for t in all_skeletons(k, j, s, True))
    if h < 1 or j < 0 or s < 1:
        return ()
    if h == 1:
        return ((),) if j == 0 and s == 1 else ()
    out = [(c,) for c in all_skeletons(h - 1, j, s - 1)]
    for (exact_left, jl, sl, jr, sr), _ in _pairs(h, j, s):
        lefts = all_skeletons(h - 1 if exact_left else h - 2, jl, sl, exact_left)
        rights = all_skeletons(h - 1, jr, sr, not exact_left)
        out += [(l, r) for l in lefts for r in rights]
    return tuple(out)


def impossible_reason(calculus, spine: int, joins: int, cap: int = 4096):
    """Why a retired topology cell is unfilled, when that can be proved rather than assumed: '' when
    trees exist and at least one admits a rule assignment on outer symbols (`build.shape_feasible`;
    a retired cell with this reason was unlucky or expensive, not impossible), 'tree' when no tree has
    this spine and join count, 'shape' when no tree of the cell does (exhaustive enumeration: every
    skeleton up to the shape's largest size, so it is a fact about the cell, not one tree), None when
    the cell holds more trees than `cap` or the archive knows no calculus. Selection-level fact,
    reported by `DistributionArchive.report()`; never part of a certificate."""
    from .build import shape_feasible
    if calculus is None:
        return None
    sizes = shape_sizes(spine, joins)
    if not sizes:
        return "tree"
    total = sum(_exact(spine, joins, s) for s in sizes)
    if total > cap:
        return None
    for s in sizes:
        for sk in all_skeletons(spine, joins, s):
            if shape_feasible(calculus, sk):
                return ""
    return "shape"


def shape_sizes(spine: int, joins: int) -> list:
    """Step counts a tree of this topology can have: from the smallest up to `spine` more."""
    lo = spine + joins
    return [s for s in range(lo, lo + spine + 1) if _exact(spine, joins, s)]


def skeleton_for(rng, spine: int, joins: int, size: Optional[int] = None):
    """A skeleton drawn uniformly among ordered trees with this spine, joins and size (default: a size
    drawn uniformly from shape_sizes)."""
    if not 0 <= joins <= max_joins(spine):
        raise ValueError(f"a spine of {spine} has at most {max_joins(spine)} joins")
    size = size if size is not None else rng.choice(shape_sizes(spine, joins))
    if not _exact(spine, joins, size):
        raise ValueError(f"no tree with spine {spine}, {joins} joins and {size} steps")

    def exact(h, j, s):
        if h == 1:
            return ()
        if rng.random() * _exact(h, j, s) < _exact(h - 1, j, s - 1):
            return (exact(h - 1, j, s - 1),)
        pairs = _pairs(h, j, s)
        (exact_left, jl, sl, jr, sr), = rng.choices([p for p, _ in pairs], weights=[n for _, n in pairs])
        if exact_left:
            return exact(h - 1, jl, sl), within(h - 1, jr, sr)
        return within(h - 2, jl, sl), exact(h - 1, jr, sr)

    def within(h, j, s):
        k, = rng.choices(range(1, h + 1), weights=[_exact(k, j, s) for k in range(1, h + 1)])
        return exact(k, j, s)

    return exact(spine, joins, size)


def topology_niche(inst) -> tuple:
    t = inst.analysis.topology
    return (inst.label, t["spine"], t["joins"])


class TopologyArchive:
    """Quality-diversity over certified proof shapes: each request goes to the least-covered cell of the
    descriptor grid (label × spine × joins, spine up to `max_spine`), inverted into a shape target that
    construction realizes exactly; cells that keep failing are tried less, and a cell that has failed
    `retire` times without ever succeeding is no longer proposed (some shapes are impossible in a
    calculus, e.g. nested joins in grid; selection only, never certification)."""

    def __init__(self, max_spine: int = 12, joins_cap: int = 6, calculus=None, candidates: int = 16,
                 retire: int = 8):
        """With a calculus, rule mix is steered too: each request also aims at the least-covered set of
        required kinds among `candidates` random ones, tried first when filling the skeleton (a soft
        steer; coverage counts the kind set actually certified)."""
        self.cells = [(lab, s, j) for lab in LABELS for s in range(1, max_spine + 1)
                      for j in range(0, min(max_joins(s), joins_cap) + 1)]
        self.coverage, self.failures, self.kind_sets = Counter(), Counter(), Counter()
        self.calculus, self.candidates, self.retire = calculus, candidates, retire
        self.last = None

    def live(self) -> list:
        """Cells still proposed: not retired (all of them, if every cell is)."""
        out = [c for c in self.cells if self.coverage[c] or self.failures[c] < self.retire]
        return out or self.cells

    def _prefer(self, rng):
        """The least-covered kind set among random candidates (sizes 1 to all kinds)."""
        kinds = sorted({r.family for r in self.calculus.rules.values()})
        cands = [frozenset(rng.sample(kinds, rng.randint(1, len(kinds)))) for _ in range(self.candidates)]
        return min(cands, key=lambda ks: (self.kind_sets[ks], rng.random()))

    def propose(self, rng, distractors: int = 2) -> Target:
        cell = min(self.live(), key=lambda c: (self.coverage[c] + self.failures[c] / 4, rng.random()))
        return self.target_for(cell, rng, distractors)

    def target_for(self, cell, rng, distractors: int = 2) -> Target:
        """Invert an explicitly selected topology cell into a shape target."""
        lab, spine, joins = cell
        self.last = cell
        sk = skeleton_for(rng, spine, joins)
        prefer = self._prefer(rng) if self.calculus is not None else None
        return Target(lab, rules=(), shape=sk, prefer=prefer,
                      distractors=Distractors(irrelevant=rng.randint(0, distractors), near_miss=rng.randint(0, distractors)))

    def add(self, inst):
        if inst is None:
            self.failures[self.last] += 1
        else:
            self.coverage[topology_niche(inst)] += 1
            if inst.label != "neutral":
                self.kind_sets[frozenset(inst.analysis.mandatory_kinds)] += 1


class DistributionArchive(TopologyArchive):
    """Track a requested distribution over certified E/C topology cells.

    Selection reduces the largest deficit between the requested and certified histograms; failures do
    not enter it. Penalizing failures would trade fidelity for throughput without saying so: with a
    50/50 target whose second cell succeeds one time in ten, a penalty of 0.25 per failure settles near
    76/24. A cell retired as in TopologyArchive (`retire` failures, no success) is no longer requested
    and is reported as unfilled; its mass is not renormalized away. A cell that succeeded once is
    feasible and stays requested whatever it costs. `retire` defaults higher than for coverage: a cell
    succeeding one time in ten survives 30 requests with probability 0.96.
    """

    def __init__(self, weights, max_spine=12, joins_cap=6, calculus=None, candidates=16, retire=30):
        super().__init__(max_spine, joins_cap, calculus, candidates, retire)
        valid = set(self.cells)
        if not weights or any(c not in valid or c[0] == "neutral" for c in weights):
            raise ValueError("weights must specify E/C cells in the topology grid")
        try:
            values = [float(w) for w in weights.values()]
        except (TypeError, ValueError, OverflowError) as e:
            raise ValueError("weights must be numeric") from e
        total = sum(values)
        if any(not isfinite(w) or w < 0 for w in values) or not isfinite(total) or total <= 0:
            raise ValueError("weights must be finite, nonnegative and not all zero")
        self.target = {c: w / total for c, w in zip(weights, values) if w > 0}
        self.cells = list(self.target)

    def propose(self, rng, distractors=2):
        """The target for the live cell furthest below its quota."""
        n = sum(self.coverage.values())
        cell = max(self.live(), key=lambda c: ((n + 1) * self.target[c] - self.coverage[c], rng.random()))
        return self.target_for(cell, rng, distractors)

    def report(self) -> list:
        """Requested vs achieved per cell; `unfilled` is the quota still missing at the current total.
        A retired cell's `reason` says why, when that is a fact about the calculus rather than the
        search: 'tree' (no tree has this spine and join count), 'shape' (every tree of the cell is
        infeasible on outer symbols), None (more trees than the classification cap), '' (feasible:
        impossible was luck or cost, not structure)."""
        n = sum(self.coverage.values())
        rows = []
        for c, p in self.target.items():
            retired = not self.coverage[c] and self.failures[c] >= self.retire
            rows.append(dict(label=c[0], spine=c[1], joins=c[2], requested=p,
                             achieved=self.coverage[c] / n if n else 0.0, certified=self.coverage[c],
                             failures=self.failures[c], unfilled=max(0, round(p * n) - self.coverage[c]),
                             retired=retired,
                             **({"reason": impossible_reason(self.calculus, c[1], c[2])} if retired else {})))
        return rows

    def total_variation(self):
        """Distance from requested to certified output proportions; None before a success."""
        n = sum(self.coverage.values())
        if not n:
            return None
        outside = sum(v for c, v in self.coverage.items() if c not in self.target)
        return (outside / n + sum(abs(self.coverage[c] / n - p) for c, p in self.target.items())) / 2


def structure(analysis) -> Optional[str]:
    """Canonical form of the certified mandatory proof structure, modulo renaming: the tree of mandatory
    steps (rule kind, children sorted per premise: a mandatory sub-derivation, "*" for a part derived in
    more than one way, "p" for a premise), AHU-style. A step whose conclusion two mandatory steps use is
    marked "&" at each use, so a reused lemma differs from the same lemma derived twice. Two instances
    share it iff their mandatory steps unfold to the same rule-labelled tree with the same steps
    marked shared (which of several identical shared steps coincide is not recorded)."""
    if analysis is None or not analysis.derivable:
        return None
    by_head = {x[2]: x for x in analysis.mandatory_steps}
    uses = Counter(p for x in analysis.mandatory_steps for p in set(x[3]) if p in by_head)
    given = set(analysis.given)

    def enc(f):
        if f in by_head:
            _, kind, _, parents = by_head[f]
            mark = "&" if uses[f] >= 2 else ""
            return f"{mark}{kind}({','.join(sorted(enc(p) for p in parents))})"
        return "p" if f in given else "*"
    return enc(analysis.goal)


def profile_cell(*fields, cap: Optional[int] = None, calculus=None):
    """A cell key over proof-profile fields (`Analysis.profile`), counts capped at `cap` (the open top
    bucket), with the label first. Neutral rows carry their counterfactual profile (the goal once the
    withheld premise is restored). A row built without analysis (forward rows) is analyzed in `calculus`
    on demand and keeps the result; None when it cannot be."""
    def key(inst):
        if inst.analysis is None and calculus is not None and not inst.counterfactual:
            inst.analysis = analyze(calculus, inst.premises, inst.conclusion, inst.universe)
        p = inst.analysis.profile if inst.analysis is not None else None
        if p is None:
            return None
        return (inst.label, *(min(p[f], cap) if cap is not None and isinstance(p[f], int) else p[f] for f in fields))
    return key


class ProfileQuota:
    """Select generated rows so a descriptor follows a requested distribution, for descriptors no target
    can request (minimum depth, number of proofs, sharing on forward rows). Selection only: generation is
    not steered, so a rare cell costs what it costs, and `report()` says how many rows were offered per
    cell. A row is kept when its cell is below quota once it is added; rows outside the requested cells are
    rejected. Same rule as DistributionArchive: never trade fidelity for throughput.
    """

    def __init__(self, weights, key):
        total = sum(weights.values())
        if not weights or total <= 0 or any(not isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError("weights must be finite, nonnegative and not all zero")
        self.target = {c: w / total for c, w in weights.items() if w > 0}
        self.key = key
        self.kept, self.offered = Counter(), Counter()

    def offer(self, inst) -> bool:
        cell = self.key(inst)
        self.offered[cell] += 1
        if cell not in self.target or self.kept[cell] >= (sum(self.kept.values()) + 1) * self.target[cell]:
            return False
        self.kept[cell] += 1
        return True

    def report(self) -> list:
        n = sum(self.kept.values())
        return [dict(cell=c, requested=p, achieved=self.kept[c] / n if n else 0.0, kept=self.kept[c],
                     offered=self.offered[c]) for c, p in self.target.items()]
