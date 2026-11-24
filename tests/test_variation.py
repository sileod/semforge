import collections
import random

import pytest

import semforge.rc  # noqa: F401
L = pytest.importorskip("reasoning_core.tasks.logic_depth")
from semforge.domains.logic import variants as V


def orbits(level, n, seed=0):
    random.seed(seed)
    rng = random.Random(seed)
    cfg, state, out = L.MultistepNLIConfig(level=level), {}, []
    while len(out) < n:
        case, _ = L.generate_case(cfg, state=state)
        if case is not None:
            o = V.orbit(case, rng)
            if o is not None:
                out.append(o)
    return out


@pytest.mark.parametrize("level", [1, 3])
def test_orbit_contract(level):
    seen = collections.Counter()
    for o in orbits(level, 12, level):
        base_lines = collections.Counter(V.render(o.base.instance)[0])
        assert V.label(o.base.instance) == o.base.answer
        for v in o.variants:
            seen[v.name] += 1
            assert V.label(v.instance) == v.answer
            assert (v.answer == o.base.answer) == (v.kind == "invariant")
            lines = collections.Counter(V.render(v.instance)[0])
            added, removed = lines - base_lines, base_lines - lines
            if v.name.startswith("distractor:") or v.name == "twin:complete":
                assert sum(added.values()) == 1 and not removed
            if v.name == "twin:remove_necessary":
                assert sum(removed.values()) == 1 and not added
            if v.name == "shuffle":
                assert lines == base_lines
    assert seen["distractor:near_miss_1"] and seen["rename"]


def test_expand_fact_makes_its_rule_kind_required():
    import random
    from reasoning_core.tasks import logic_depth as L
    from semforge.core import patterns as P
    from semforge.domains.logic import LogicDomain
    from semforge.domains.logic import variants as V
    from semforge.domains.logic.generation import delta_neutral_checks
    random.seed(3)
    rng = random.Random(3)
    cfg, state, done = L.MultistepNLIConfig(level=2), {}, 0
    D = LogicDomain()
    req_of = lambda e: (lambda q: q.signature if q else None)(P.analyze(D, e, D.label(e)))
    while done < 5:
        with delta_neutral_checks():
            case, _ = L.generate_case(cfg, state=state)
        if case is None:
            continue
        inst = V.from_case(case)
        if req_of(inst) is None:
            continue
        for shape in V.SHAPES:
            e, meta = V.expand_fact(inst, rng, shape, req_of)
            if e is not None:
                assert V.label(e) == V.label(inst)
                assert meta["kind"] in dict(req_of(e))
                assert len(e.order) > len(inst.order)
                done += 1
                break
