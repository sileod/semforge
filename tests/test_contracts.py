import pytest

from semforge.core.ladder import Ladder, Reasoner
from semforge.core.provenance import IncompleteProvenance, provenance_chase
from semforge.core.stats import GenerationStats, profile_calls


def instances(rule, closure):
    head, parents = rule
    if set(parents) <= closure:
        yield head, parents


def chase(rules, cap=256):
    return provenance_chase(['a', 'b'], rules, instances,
                            lambda x: x if isinstance(x, str) else None, cap=cap)


def test_shallow_superset_is_not_a_minimal_support():
    p = chase([('x', ('a',)), ('y', ('x',)), ('t', ('y',)), ('t', ('a', 'b'))])
    assert p.minimal_supports('t') == [frozenset({'a'})]
    assert p.min_depth('t') == 1
    assert p.supports['t'] == {frozenset({'a'}): 3, frozenset({'a', 'b'}): 1}


def test_truncation_propagates_even_when_parent_is_truncated_after_child():
    p = chase([('x', ('a',)), ('t', ('x',)), ('u', ('t',)), ('x', ('b',))], cap=1)
    assert {'x', 't', 'u'} <= p.truncated
    assert p.necessary('u') is None
    assert p.entails('u', {'a'}) is True  # retained witnesses are still sound
    assert p.entails('u', {'b'}) is None
    assert p.entails('absent', {'a'}) is False
    assert p.min_size('u') is None and p.min_depth('u') is None
    with pytest.raises(IncompleteProvenance):
        p.minimal_supports('u')
    assert p.minimal_supports('u', allow_incomplete=True) == [frozenset({'a'})]
    with pytest.raises(ValueError):
        chase([], cap=0)


def test_lucky_heuristics_do_not_certify_rungs():
    ladder = Ladder([Reasoner('guess', lambda x: 'yes', sound=False),
                     Reasoner('solver', lambda x: x)])
    assert ladder.rung('yes', 'yes') == 'solver'
    report = ladder.audit(iter(['yes', 'no']), iter(['yes', 'no']))
    assert report['guess']['accuracy_when_answering'] == 0.5
    assert report['rungs'] == {'solver': 2}
    with pytest.raises(ValueError):
        ladder.audit([], [])


def test_accounting_records_failures_and_restores_patches():
    from types import SimpleNamespace
    module = SimpleNamespace(f=lambda x: x)
    original = module.f
    stats = GenerationStats()
    with pytest.raises(RuntimeError):
        with profile_calls(module, {'f': lambda x: 'accepted' if x else 'empty'}, stats):
            assert module.f(0) == 0
            assert module.f(1) == 1
            with stats.measure('generation'):
                raise RuntimeError('failed')
    assert module.f is original
    report = stats.report()
    assert report['f']['outcomes'] == {'empty': 1, 'accepted': 1}
    assert report['f']['attempts_per_accepted'] == 2
    assert report['generation']['outcomes'] == {'error:RuntimeError': 1}
