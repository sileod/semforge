import itertools
import random

from semforge.core.lattice import (Oracle, deletion, enumerate_lattice, minimum_support, necessary,
                              quickxplain)


def dnf_oracle(terms):
    """holds(S) iff S contains one of the terms: the minimal supports are exactly the minimal terms."""
    return Oracle(lambda s: any(t <= s for t in terms), lambda s: next((t for t in terms if t <= s), None))


def minimal(terms):
    return {t for t in terms if not any(u < t for u in terms)}


def brute_minimal(items, holds):
    out = []
    for k in range(len(items) + 1):
        for c in itertools.combinations(items, k):
            c = frozenset(c)
            if holds(c) and not any(o <= c for o in out):
                out.append(c)
    return set(out)


def test_random_dnfs_match_brute_force():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(1, 9)
        items = list(range(n))
        terms = [frozenset(rng.sample(items, rng.randint(1, n))) for _ in range(rng.randint(1, 4))]
        truth = minimal(set(terms))
        for algo in (quickxplain, lambda i, o: deletion(i, o), lambda i, o: deletion(i, o, use_core=False)):
            s = algo(items, dnf_oracle(terms))
            assert s in truth
        sups, mcs, complete = enumerate_lattice(items, dnf_oracle(terms))
        assert complete and set(sups) == truth
        assert set(sups) == brute_minimal(items, lambda s: any(t <= s for t in terms))
        # minimal corrections are the minimal hitting sets of the supports
        for c in mcs:
            assert all(c & t for t in truth)
            assert all(not all((c - {x}) & t for t in truth) for x in c)
        assert necessary(items, dnf_oracle(terms)) == frozenset.intersection(*truth)
        assert len(minimum_support(items, dnf_oracle(terms))) == min(map(len, truth))


def test_not_holding():
    o = dnf_oracle([frozenset({5})])
    assert quickxplain([1, 2], o) is None and deletion([1, 2], o) is None
    assert minimum_support([1, 2], o) is None


def test_empty_support_with_background_only():
    for items in ([], [1, 2]):
        assert minimum_support(items, Oracle(lambda s: True)) == frozenset()
    assert minimum_support([], Oracle(lambda s: False)) is None
