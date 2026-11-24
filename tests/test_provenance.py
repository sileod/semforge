import random

import pytest

import semforge.rc  # noqa: F401  (puts reasoning_core on the path)
L = pytest.importorskip("reasoning_core.tasks.logic_depth")
from semforge.core.provenance import logic_depth_provenance


def sub_theory(theory, keep):
    return L.Theory([a for a in theory.facts if keep(a)], [r for r in theory.rules if keep(r)],
                    theory.denials, theory.pred_sigs, theory.entities, theory.domain_pack)


@pytest.mark.parametrize("level", [0, 3])
def test_provenance_matches_chase_on_random_subsets(level):
    random.seed(level)
    cfg = L.MultistepNLIConfig(level=level)
    rng = random.Random(1)
    checked = 0
    for _ in range(6):
        case, _ = L.generate_case(cfg)
        if case is None:
            continue
        _, source, _ = L.render(case.theory)
        prov = logic_depth_provenance(case.theory, source)
        full = L.chase(case.theory)
        assert set(prov.supports) == full.closure
        for a in full.closure:
            assert prov.min_depth(a) == full.derivations[a].depth
        ids = sorted(set(source.values()))
        for _ in range(15):
            keep_ids = {i for i in ids if rng.random() < 0.7}
            res = L.chase(sub_theory(case.theory, lambda x: source[x] in keep_ids))
            if res.inconsistent:
                continue
            for a in full.closure:
                assert prov.entails(a, keep_ids) == (a in res.closure)
            checked += 1
        if case.target is not None:
            assert prov.necessary(case.target) == set(L.necessary_indices(case))
    assert checked
