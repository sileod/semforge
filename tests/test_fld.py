from semforge.domains import fld as F
from semforge.core import patterns as P


def _inst(facts, hyp, gold="entailment"):
    formula = {f"fact{i}": x for i, x in enumerate(facts, 1)}
    return F.FLDInstance("t", list(formula.items()), formula, {}, hyp, "", gold, frozenset(), None, None)


def test_parser_handles_quantifiers_constants_and_connectives():
    env = {}
    for s in ["(x): ¬{A}x -> (¬{B}x & {C}x)", "¬({A}{aa} v {B}{aa})", "(Ex): {A}x", "{A} -> ¬{B}", "¬((x): ¬(¬{AA}x & ¬{AB}x))"]:
        F._Parser(s, env).parse()


def test_labels_and_justifications_are_certified_by_z3():
    d = F.FLDDomain(rows=[])  # synthetic instances: no Hub data needed
    inst = _inst(["(x): {A}x -> {B}x", "{A}{a}", "{C}{b}", "fake_formula"], "{B}{a}")
    assert d.label(inst) == "entailment"
    assert d.label(_inst(["(x): {A}x -> ¬{B}x", "{A}{a}"], "{B}{a}")) == "contradiction"
    assert d.label(_inst(["(x): {A}x -> {B}x"], "{B}{a}")) == "neutral"
    p = P.analyze(d, inst, "entailment")
    assert p.complete and len(p.justifications) == 1
    assert dict(p.required) == {"∀x: Px -> Px": 1}   # the literal {A}{a} is required but has no kind


def test_alternative_proofs_make_kinds_optional():
    d = F.FLDDomain(rows=[])  # synthetic instances: no Hub data needed
    inst = _inst(["(x): {A}x -> {B}x", "{A}{a}", "{C}{a} -> {B}{a}", "{C}{a}"], "{B}{a}")
    p = P.analyze(d, inst, "entailment")
    assert len(p.justifications) == 2 and not p.required
