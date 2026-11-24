import random
from collections import Counter
from types import SimpleNamespace

import pytest

from semforge.construct.archive import DistributionArchive, topology_niche
from semforge.construct.build import shape_topology


def certified(cell):
    return SimpleNamespace(
        label=cell[0],
        analysis=SimpleNamespace(
            topology={"spine": cell[1], "joins": cell[2]},
            mandatory_kinds=Counter(),
        ),
    )


def test_distribution_tracks_certified_cells():
    weights = {("entailment", 2, 0): 1, ("contradiction", 3, 1): 3}
    arch, rng = DistributionArchive(weights), random.Random(7)
    for _ in range(200):
        t = arch.propose(rng)
        cell = (t.label, *([2, 0] if t.label == "entailment" else [3, 1]))
        assert cell == arch.last
        assert shape_topology(t.shape) == {"spine": cell[1], "joins": cell[2]}
        arch.add(certified(cell))
    assert arch.coverage[("entailment", 2, 0)] == 50
    assert arch.coverage[("contradiction", 3, 1)] == 150
    assert arch.total_variation() == 0


def test_failed_requests_are_not_certified():
    cells = {("entailment", 2, 0): 1, ("entailment", 3, 0): 1}
    arch, rng = DistributionArchive(cells), random.Random(0)
    assert arch.total_variation() is None
    arch.propose(rng)
    attempted = arch.last
    arch.add(None)
    assert arch.failures[attempted] == 1
    assert not arch.coverage and arch.total_variation() is None
    other = next(c for c in cells if c != attempted)
    arch.add(certified(other))
    assert topology_niche(certified(other)) == other
    assert arch.total_variation() == 0.5


def test_failures_do_not_bias_the_mixture():
    """A 50/50 target whose second cell succeeds one request in ten still ends 50/50 (review2, follow-up B)."""
    easy, hard = ("entailment", 2, 0), ("contradiction", 3, 1)
    arch, rng, tries = DistributionArchive({easy: 1, hard: 1}), random.Random(0), 0
    for _ in range(2000):
        arch.propose(rng)
        tries += arch.last == hard
        arch.add(certified(arch.last) if arch.last == easy or tries % 10 == 0 else None)
    assert abs(arch.coverage[easy] - arch.coverage[hard]) <= 1 and arch.failures[hard] > 0


def test_infeasible_cell_is_retired_and_reported():
    good, bad = ("entailment", 2, 0), ("entailment", 4, 1)
    arch, rng = DistributionArchive({good: 3, bad: 1}, retire=5), random.Random(0)
    for _ in range(100):
        arch.propose(rng)
        arch.add(certified(good) if arch.last == good else None)
    report = {(r["label"], r["spine"], r["joins"]): r for r in arch.report()}
    assert arch.failures[bad] == 5 and report[bad]["retired"] and not report[good]["retired"]
    assert report[bad]["unfilled"] == round(0.25 * arch.coverage[good])  # not renormalized away
    assert arch.total_variation() == 0.25


@pytest.mark.parametrize("weights", [
    {},
    {("entailment", 2, 0): 0},
    {("entailment", 2, 0): -1},
    {("entailment", 2, 0): float("nan")},
    {("entailment", 2, 0): float("inf")},
    {("entailment", 2, 0): 1e308, ("contradiction", 2, 0): 1e308},
    {("entailment", 0, 0): 1},
    {("neutral", 2, 0): 1},
])
def test_invalid_distributions_rejected(weights):
    with pytest.raises(ValueError):
        DistributionArchive(weights)


def test_shape_counts_match_enumeration():
    from semforge.construct.archive import _exact

    def trees(s):  # every ordered tree of s steps, at most two children per step
        if s == 1:
            yield ()
        else:
            yield from ((t,) for t in trees(s - 1))
            for sl in range(1, s - 1):
                yield from ((a, b) for a in trees(sl) for b in trees(s - 1 - sl))
    seen = Counter((shape_topology(t)["spine"], shape_topology(t)["joins"], s) for s in range(1, 10) for t in trees(s))
    assert all(_exact(h, j, s) == n for (h, j, s), n in seen.items())
    assert sum(_exact(h, j, 9) for h in range(1, 10) for j in range(5)) == sum(n for k, n in seen.items() if k[2] == 9)


def test_skeletons_are_uniform_and_exact():
    from semforge.construct.archive import _exact, skeleton_for
    rng = random.Random(0)
    draws = Counter(skeleton_for(rng, 4, 2, size=7) for _ in range(6000))
    assert len(draws) == _exact(4, 2, 7)
    assert all(shape_topology(t) == {"spine": 4, "joins": 2} for t in draws)
    assert max(draws.values()) < 1.5 * min(draws.values())
    assert skeleton_for(rng, 3, 3) == (((), ()), ((), ()))  # balanced: outside the old spine-with-side-chains grid


def test_balanced_cell_is_requested_and_certified():
    from semforge.construct.packs import get
    from semforge.construct.synth import Synthesizer
    pack, rng = get("propositional"), random.Random(0)
    arch = DistributionArchive({("entailment", 3, 3): 1}, calculus=pack.calculus)
    inst = next(i for i in (Synthesizer(pack).realize(arch.propose(rng), rng) for _ in range(10)) if i)
    assert topology_niche(inst) == ("entailment", 3, 3)


def test_profile_quota_selects_a_requested_depth_distribution():
    from semforge.construct.archive import ProfileQuota, profile_cell
    from semforge.construct.forward import forward
    from semforge.construct.packs import get
    pack, rng = get("propositional"), random.Random(0)
    cells = [(lab, d) for lab in ("entailment", "contradiction") for d in range(1, 7)]
    quota = ProfileQuota(dict.fromkeys(cells, 1), profile_cell("min_depth", cap=6, calculus=pack.calculus))
    kept = []
    while len(kept) < 36:
        out = forward(pack, rng, rng.randint(1, 20))
        if out and quota.offer(out[0]):
            kept.append(out[0])
    # forward rows mostly reach depth 6+; the quota keeps exactly 3 per cell, and says what it cost
    assert all(r["kept"] == 3 for r in quota.report())
    assert quota.offered[("entailment", 6)] > 3
    assert Counter((r.label, min(r.analysis.profile["min_depth"], 6)) for r in kept) == dict.fromkeys(cells, 3)
