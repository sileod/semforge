import itertools
import random

import pytest

import semforge.rc  # noqa: F401
sp = pytest.importorskip("sympy")
C = pytest.importorskip("reasoning_core.tasks._csp_utils")
A = pytest.importorskip("reasoning_core.tasks.arithmetics")
L = pytest.importorskip("reasoning_core.tasks.logic_depth")
P = pytest.importorskip("reasoning_core.tasks.planning")
from semforge.domains.adapters import (bounded_plans, linear_oracle, logic_derivation_audit,
                               logic_support_oracle, planning_cue_oracle, word_problem_oracle)
from semforge.domains.csp.oracles import core_guided_minimization, forces_value
from semforge.core.lattice import deletion, minimum_support, necessary


def subsets(n):
    for k in range(n + 1):
        yield from (frozenset(s) for s in itertools.combinations(range(n), k))


def test_linear_determination_matches_exact_solves_for_all_subsets():
    x, y, z = sp.symbols('x y z')
    equations = [sp.Eq(x + y, 3), sp.Eq(y, 1), sp.Eq(x, 2), sp.Eq(z, 4)]
    oracle = linear_oracle(equations, [x, y, z], x)
    for indices in subsets(len(equations)):
        selected = [equations[i] for i in indices]
        if selected:
            a, b = sp.linear_eq_to_matrix(selected, [x, y, z])
            sol = next(iter(sp.linsolve((a, b), [x, y, z])))
            expected = not sol[0].free_symbols
        else:
            expected = False
        assert oracle.holds(indices) == expected
    assert minimum_support(range(4), oracle) == frozenset({2})
    assert necessary(range(4), oracle) == frozenset()
    bad = [sp.Eq(x, 1), sp.Eq(x, 2), sp.Eq(y, 0)]
    with pytest.raises(ValueError):
        linear_oracle(bad, [x, y], x)
    unsat = linear_oracle(bad, [x, y], objective='inconsistent')
    assert minimum_support(range(3), unsat) == frozenset({0, 1})
    assert minimum_support([], linear_oracle([], [x], x)) is None
    assert minimum_support([0], linear_oracle([True], [x], x)) is None
    background_conflict = linear_oracle([], [x], background=[False], objective='inconsistent')
    assert minimum_support([], background_conflict) == frozenset()


def test_csp_requires_satisfiable_pool_and_checks_unique_value_on_all_subsets():
    x, y = C.Var('x', [0, 1]), C.Var('y', [0, 1])
    solver = C.CSPSolver([x, y])
    with pytest.raises(ValueError, match='satisfiable'):
        forces_value(solver, x, 0, pool=[C.Eq(x, 0), C.Eq(x, 1)])
    pool = [C.EqVar(x, y), C.Eq(y, 0), C.Eq(x, 0)]
    oracle = forces_value(solver, x, 0, pool=pool)
    for indices in subsets(len(pool)):
        clues = [pool[i] for i in indices]
        assert oracle.holds(clues) == C.UniqueValue(x, 0).holds(solver, clues)
    with pytest.raises(ValueError, match='subset'):
        oracle.holds([C.Eq(x, 1)])
    with pytest.raises(ValueError, match='domain'):
        forces_value(solver, x, 2, pool=pool)
    original = C.minimize_for_objective
    with core_guided_minimization():
        supports = C.minimize_for_objective(solver, pool, C.UniqueValue(x, 0), random.Random(0), 2)
        assert supports
        for support in supports:
            assert solver.unique_value(x, support) == 0
            assert all(solver.unique_value(x, [c for c in support if c != removed]) != 0
                       for removed in support)
        # Unsupported objectives retain the original implementation.
        assert C.minimize_for_objective(solver, pool, C.Consistent(), random.Random(0), 1) == \
            original(solver, pool, C.Consistent(), random.Random(0), 1)
    assert C.minimize_for_objective is original


def test_csp_unknown_does_not_become_a_negative_oracle_answer():
    from types import SimpleNamespace
    import z3
    from semforge.domains.csp.oracles import refutes
    unknown = SimpleNamespace(check=lambda *args: z3.unknown, reason_unknown=lambda: 'test budget')
    solver = SimpleNamespace(base=(), _guard=lambda f: z3.Bool(str(f)), _persistent=lambda: unknown)
    with pytest.raises(RuntimeError, match='unknown'):
        refutes(solver, ['probe']).holds(['clue'])


def test_planning_oracle_against_exhaustive_action_sequences():
    model = dict(horizon=2, initial_true=['s'], goal_true=['g'], goal_false=[], actions=[
        P._action('a', ['s'], [], ['m'], ['s']),
        P._action('b', ['s'], [], ['m'], ['s']),
        P._action('finish', ['m'], [], ['g'], []),
    ])
    reference = ('a', 'finish')
    oracle = planning_cue_oracle(model, reference)
    for indices in subsets(2):
        brute = []
        for actions in itertools.product(model['actions'], repeat=2):
            if any(actions[i].call != reference[i] for i in indices):
                continue
            state = set(model['initial_true'])
            for action in actions:
                if not P._applicable(action, state):
                    break
                state = P._apply(action, state)
            else:
                if set(model['goal_true']) <= state:
                    brute.append(tuple(a.call for a in actions))
        assert set(bounded_plans(model, ((i, reference[i]) for i in indices))) == set(brute)
        assert oracle.holds(indices) == (brute == [reference])
    assert minimum_support(range(2), oracle) == frozenset({0})
    with pytest.raises(ValueError):
        planning_cue_oracle(model, ('finish', 'a'))


def test_word_problem_adapter_agrees_with_existing_core_metric():
    random.seed(5)
    cfg = A.WordProblemMathConfig(level=2)
    checked = 0
    for _ in range(12):
        entry = A.gen_relational(cfg)
        if entry is None:
            continue
        m = entry.metadata
        oracle = word_problem_oracle(m)
        support = minimum_support(range(len(m.relations)), oracle)
        assert len(support) == m.proof_core_size
        assert oracle.holds(support)
        checked += 1
    assert checked >= 3


def test_logic_supports_match_chase_and_trace_uniqueness_stays_separate():
    # Two rules derive the same atom from the same fact: one minimal premise
    # support with background rules does not imply one proof trace.
    a, t = L.Atom('a', ('ann',)), L.Atom('t', ('ann',))
    r1 = L.Rule((a,), t, 'r1')
    r2 = L.Rule((a,), t, 'r2')
    theory = L.Theory([a], [r1, r2], [], {}, {}, 'unary')
    source = {a: 0, r1: 1, r2: 2}
    res = L.chase(theory)
    from types import SimpleNamespace
    case = SimpleNamespace(theory=theory, source=source, target=t, res=res,
                           derivation=res.derivations[t])
    oracle = logic_support_oracle(case)
    from semforge.core.provenance import logic_depth_provenance
    prov = logic_depth_provenance(theory, source)
    for indices in subsets(3):
        assert oracle.holds(indices) == prov.entails(t, indices)
    assert necessary(source.values(), oracle) == prov.necessary(t) == frozenset({0})
    audit = logic_derivation_audit(case)
    background = logic_depth_provenance(theory, {a: 0}, background_rules=[r1, r2])
    assert background.minimal_supports(t) == [frozenset({0})]
    assert audit['minimum_depth'] == 1
    assert audit['trace_unique'] is False
    assert audit['minimal_support_count'] == 2
    capped = logic_derivation_audit(case, cap=1)
    assert capped['minimum_premises'] is None
    assert capped['necessary'] is None
