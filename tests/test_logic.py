import random

import pytest

import semforge.rc  # noqa: F401
L = pytest.importorskip("reasoning_core.tasks.logic_depth")
from semforge.domains.logic.generation import delta_inconsistent, delta_neutral_checks, repair


@pytest.mark.parametrize("level", [0, 3, 6])
def test_delta_matches_close_with(level):
    random.seed(level)
    cfg = L.MultistepNLIConfig(level=level)
    checked = 0
    for _ in range(40):
        theory = L.sample_theory(cfg)
        res = L.chase(theory)
        if res.inconsistent:
            continue
        atoms = list(L._all_ground_atoms(theory))
        for h in random.sample(atoms, min(25, len(atoms))):
            assert delta_inconsistent(theory, res, [h]) == L.close_with(theory, [h]).inconsistent
            checked += 1
    assert checked > 50


def test_delta_patch_leaves_cases_unchanged():
    def gen(fast):
        random.seed(7)
        cfg, state, out = L.MultistepNLIConfig(level=2), {}, []
        while len(out) < 15:
            if fast:
                with delta_neutral_checks(verify=True):
                    case, _ = L.generate_case(cfg, state=state)
            else:
                case, _ = L.generate_case(cfg, state=state)
            if case:
                out.append((case.label, case.hyp, tuple(case.lines)))
        return out
    assert gen(False) == gen(True)


def test_repair_yields_consistent_theories():
    random.seed(3)
    cfg = L.MultistepNLIConfig(level=6)
    fixed = 0
    for _ in range(60):
        theory = L.sample_theory(cfg)
        if not L.chase(theory).inconsistent:
            continue
        out = repair(theory)
        if out is not None:
            assert not L.chase(out).inconsistent
            assert all(r in out.rules for r in theory.rules if r.name in {"backbone", "spatial", "kin"})
            fixed += 1
    assert fixed
