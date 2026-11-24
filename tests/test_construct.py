import random
from dataclasses import replace

import pytest

from semforge.construct.archive import Archive
from semforge.construct.build import Fresh, derive
from semforge.construct.calculus import Calculus, Universe, analyze
from semforge.construct.packs.propositional import EXTRA, PACK, RULES, negate
from semforge.construct.synth import Instance, Synthesizer, negation_depth
from semforge.construct.target import Distractors, Target
from semforge.construct.terms import subst, unify

IMP = "->"


def universe(premises, goal):
    return Universe([*premises, goal], negate, (IMP,))


def test_unify_occurs_check_and_binding():
    s = unify(("->", "?A", "q"), ("->", "p", "?B"))
    assert subst("?A", s) == "p" and subst("?B", s) == "q"
    assert unify("?A", ("not", "?A")) is None


def test_proofs_are_not_premise_supports():
    # one premise set, two proofs: modus ponens twice, or hypothetical syllogism then modus ponens
    premises = ["a", (IMP, "a", "b"), (IMP, "b", "c")]
    a = analyze(PACK.calculus, premises, "c", universe(premises, "c"))
    assert a.derivable and a.exact
    assert a.n_proofs == 2 and len(a.premise_supports) == 1
    assert a.required_rules == {"modus_ponens": 1}  # multiset intersection of {MP: 2} and {HS: 1, MP: 1}
    assert set(premises) <= a.required_premises and a.depth == 2 and a.size == 2


def test_required_multiplicity():
    premises = ["a", (IMP, "a", "b"), (IMP, "b", "c"), (IMP, "c", "d")]
    a = analyze(Calculus([r for r in RULES if r.name != "hypothetical_syllogism"]), premises, "d",
                universe(premises, "d"))
    assert a.required_rules == {"modus_ponens": 3} and a.depth == 3


def test_disjoint_proofs_require_nothing():
    premises = ["a", (IMP, "a", "c"), ("and", "c", "d")]
    a = analyze(PACK.calculus, premises, "c", universe(premises, "c"))
    assert a.n_proofs == 2 and not a.required_rules


def test_redundant_calculus_hides_modus_tollens():
    premises = [(IMP, "a", "b"), ("not", "b")]
    goal = ("not", "a")
    assert "modus_tollens" in analyze(PACK.calculus, premises, goal, universe(premises, goal)).required_rules
    redundant = Calculus(RULES + EXTRA)
    assert "modus_tollens" not in analyze(redundant, premises, goal, universe(premises, goal)).required_rules


@pytest.mark.parametrize("rule", [r.name for r in RULES])
def test_every_default_rule_is_requireable(rule):
    rng = random.Random(0)
    hits = 0
    for _ in range(10):
        d = derive(PACK.calculus, [rule], rng, Fresh())
        assert d is not None and d.rules == [rule]
        a = analyze(PACK.calculus, d.premises, d.conclusion, Universe([*d.premises, *d.intermediates], negate, (IMP,)))
        hits += rule in a.required_rules
    assert hits > 0


def test_derivation_is_sound_and_uses_plan():
    rng = random.Random(1)
    plan = ["modus_tollens", "hypothetical_syllogism", "and_elimination"]
    for _ in range(20):
        d = derive(PACK.calculus, plan, rng, Fresh())
        assert d is not None and sorted(d.rules) == sorted(plan)
        assert PACK.entails(d.premises, d.conclusion)


def realize_all(target, seed, n=10):
    rng, syn = random.Random(seed), Synthesizer(PACK)
    return [i for i in (syn.realize(target, rng) for _ in range(n)) if i is not None]


@pytest.mark.parametrize("label", ["entailment", "contradiction", "neutral"])
def test_synthesis_certifies_label_and_structure(label):
    rng, syn = random.Random(2), Synthesizer(PACK)
    target = Target(label=label, rules=("modus_ponens", "disjunctive_syllogism"),
                    distractors=Distractors(irrelevant=1, near_miss=1))
    made = [i for i in (syn.realize(target, rng) for _ in range(10)) if i is not None]
    assert made
    for inst in made:
        assert PACK.label(inst.premises, inst.hypothesis) == label
        assert all(negation_depth(p) <= syn.max_negations for p in inst.premises)
        assert not target.required - inst.analysis.required_rules and inst.analysis.depth >= 1
        r = inst.realized()
        if label == "neutral":  # certified only counterfactually, and reported apart
            assert inst.withheld and PACK.entails(inst.premises + inst.withheld, inst.hypothesis)
            assert r["realized"] is None and r["counterfactual"]
        else:
            assert PACK.entails(inst.premises, inst.conclusion)
            assert r["counterfactual"] is None and r["realized"]


def test_requested_multiplicity_is_certified():
    made = realize_all(Target("entailment", rules=("modus_ponens", "modus_ponens", "and_elimination")), 5)
    assert made and all(i.analysis.required_rules["modus_ponens"] >= 2 for i in made)


def test_requested_depth_is_certified():
    made = realize_all(Target("contradiction", rules=("modus_ponens",) * 3 + ("disjunctive_syllogism",), min_depth=4), 6)
    assert made and all(i.analysis.depth >= 4 for i in made)


def test_proofs_sharing_a_bottleneck():
    target = Target("entailment", rules=("modus_ponens", "disjunctive_syllogism", "and_elimination"),
                    proofs=2, shared=("modus_ponens",))
    made = realize_all(target, 7)
    assert made
    for i in made:
        assert i.analysis.n_proofs >= 2 and i.analysis.required_rules["modus_ponens"] >= 1


def test_shared_rules_must_be_planned():
    with pytest.raises(ValueError):
        Target("entailment", rules=("modus_ponens",), proofs=2, shared=("and_elimination",))


def test_archive_spreads_over_niches():
    rng, syn = random.Random(3), Synthesizer(PACK)
    arch = Archive([r.name for r in RULES])
    for _ in range(40):
        inst = syn.realize(arch.propose(rng), rng)
        if inst is not None:
            arch.add(inst)
    assert len(arch.coverage) >= 15 and max(arch.coverage.values()) <= 4


def test_every_pack_is_a_constructed_domain():
    from semforge.construct.packs import NAMES as PACKS
    from semforge.domains.base import NAMES, get
    for pack in PACKS:
        for strategy in ("plan", "shape", "forward"):
            name = f"constructed_{pack}" + ("" if strategy == "plan" else f"_{strategy}")
            assert name in NAMES
            d = get(name)
            assert (d.pack.name, d.strategy, d.name) == (pack, strategy, name)
    with pytest.raises(KeyError, match="unknown pack"):
        get("constructed_nope")


def test_constructed_domain_patterns_match_realized():
    from semforge.core import patterns as P
    from semforge.domains import base
    dom, rng = base.get("constructed"), random.Random(4)
    for _ in range(15):
        for inst, label in dom.sample(rng, 1):
            assert dom.label(inst) == label
            prem, hyp = dom.render(inst)
            assert prem and hyp
            if label != "neutral":
                p = P.analyze(dom, inst, label)
                assert p.required == inst.analysis.required_rules


# --- monadic first-order pack ---------------------------------------------------------------------

from semforge.construct.packs import get as get_pack  # noqa: E402

MONADIC = get_pack("monadic")


@pytest.mark.parametrize("rule", sorted(MONADIC.calculus.rules))
def test_monadic_rules_sound_and_requireable(rule):
    rng, hits = random.Random(0), 0
    for _ in range(10):
        d = derive(MONADIC.calculus, [rule], rng, Fresh())
        assert MONADIC.entails(d.premises, d.conclusion)
        a = analyze(MONADIC.calculus, d.premises, d.conclusion,
                    Universe([*d.premises, *d.intermediates], MONADIC.negate, MONADIC.closed))
        hits += a.required_rules[rule] > 0
    assert hits > 0


def test_monadic_syllogism_chain():
    premises = [("all", "a", "b"), ("all", "b", "c"), ("is", "a", "r")]
    goal = ("is", "c", "r")
    a = analyze(MONADIC.calculus, premises, goal, Universe([*premises, goal], MONADIC.negate, MONADIC.closed))
    # instantiation twice, or Barbara then instantiation: two proofs, one premise set
    assert a.n_proofs == 2 and len(a.premise_supports) == 1 and a.required_rules == {"instantiation": 1}


@pytest.mark.parametrize("label", ["entailment", "contradiction", "neutral"])
def test_monadic_synthesis(label):
    rng, syn = random.Random(5), Synthesizer(MONADIC)
    target = Target(label, rules=syn.sample_plan(rng, 3), distractors=Distractors(irrelevant=1, near_miss=1))
    made = [i for i in (syn.realize(target, rng) for _ in range(8)) if i is not None]
    assert made
    for i in made:
        assert MONADIC.label(i.premises, i.hypothesis) == label and not target.required - i.analysis.required_rules
        prem, hyp = MONADIC.render(i.premises, i.hypothesis, rng)
        assert prem and hyp.endswith(".")


# --- CSP pack ---------------------------------------------------------------------------------------

CSP = get_pack("csp")


@pytest.mark.parametrize("rule", sorted(CSP.calculus.rules))
def test_csp_rules_sound_and_requireable(rule):
    rng, hits, kind = random.Random(0), 0, CSP.calculus.rules[rule].family
    for _ in range(10):
        d = derive(CSP.calculus, [rule], rng, Fresh())
        assert CSP.entails(d.premises, d.conclusion)
        a = analyze(CSP.calculus, d.premises, d.conclusion, Universe([*d.premises, *d.intermediates], CSP.negate, CSP.closed))
        hits += a.required_rules[kind] > 0
    assert hits > 0


def test_csp_unordered_arguments():
    # x in {a, b, c}, x != a, x != b: one naked-single schema matches any slot of the unordered set
    premises = [("one_of", "x", "a", "b", "c"), ("not", ("eq", "x", "a")), ("not", ("eq", "x", "b"))]
    goal = ("eq", "x", "c")
    a = analyze(CSP.calculus, premises, goal, Universe([*premises, goal], CSP.negate, CSP.closed))
    assert a.required_rules == {"naked_single": 1}
    assert len(CSP.calculus.rules) == 4
    # ne is symmetric: y = a and x != y give x != a whichever way the constraint is written
    for ne in (("ne", "x", "y"), ("ne", "y", "x")):
        p = [("eq", "y", "a"), ne]
        assert analyze(CSP.calculus, p, ("not", ("eq", "x", "a")),
                       Universe([*p, ("eq", "x", "a")], CSP.negate, CSP.closed)).required_rules == {"inequality": 1}


def test_csp_propagation_chain_and_semantics():
    # y = a, x != y, x in {a, b, c}, x != c  =>  x = b (inequality, then naked single)
    premises = [("eq", "y", "a"), ("ne", "x", "y"), ("one_of", "x", "a", "b", "c"), ("not", ("eq", "x", "c"))]
    goal = ("eq", "x", "b")
    assert CSP.entails(premises, goal) and not CSP.entails(premises[1:], goal)
    a = analyze(CSP.calculus, premises, goal, Universe([*premises, goal], CSP.negate, CSP.closed))
    assert a.required_rules == {"inequality": 1, "naked_single": 1} and a.depth == 2


@pytest.mark.parametrize("label", ["entailment", "contradiction", "neutral"])
def test_csp_synthesis(label):
    rng, syn = random.Random(6), Synthesizer(CSP)
    target = Target(label, rules=("inequality", "naked_single", "disjunctive_syllogism"),
                    distractors=Distractors(irrelevant=1, near_miss=1))
    made = [i for i in (syn.realize(target, rng) for _ in range(8)) if i is not None]
    assert made
    for i in made:
        assert CSP.label(i.premises, i.hypothesis) == label
        assert not CSP.calculus.kinds(target.required) - i.analysis.required_rules
        prem, hyp = CSP.render(i.premises, i.hypothesis, rng)
        assert "house" in hyp


# --- pack validation --------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["propositional", "monadic", "csp", "modal", "modal_s4", "modal_nested",
                                  "modal_nested_s4", "modal_hybrid"])
def test_pack_check(name):
    import io
    from semforge.construct.check import check
    assert check(get_pack(name), n=4, out=io.StringIO())


def test_nested_modal_semantics_and_rendering():
    from semforge.construct.signature import parse
    K, S4 = get_pack("modal_nested"), get_pack("modal_nested_s4")
    bp, bbp = ("box", "p"), ("box", ("box", "p"))
    assert not K.entails([bp], bbp) and S4.entails([bp], bbp)  # axiom 4 needs transitivity
    assert not K.entails([bp], "p") and S4.entails([bp], "p")    # axiom T needs reflexivity
    assert K.entails([("box", ("->", "p", "q")), bp], ("box", "q"))
    assert K.entails([("not", ("dia", ("dia", "p")))], ("box", ("box", ("not", "p"))))
    assert parse("box (dia A)") == ("box", ("dia", "?A"))
    premises, hyp = K.render([("box", ("dia", ("and", "p", "q")))], ("dia", "p"), random.Random(0))
    assert premises.count("(") == 2 and " and " in premises  # nested scopes bracketed
    leaves = [K.signature.leaf(Fresh(), random.Random(i), depth=2) for i in range(40)]
    assert None in leaves and any(isinstance(x, tuple) and isinstance(x[1], tuple) for x in leaves)


def test_hybrid_modal_formulas_at_named_worlds():
    H = get_pack("modal_hybrid")
    at, sees = (lambda w, f: ("at", w, f)), (lambda w, v: ("sees", w, v))
    chain = [at("u", ("box", ("box", "p"))), sees("u", "v"), sees("v", "x")]
    assert H.entails(chain, at("x", "p")) and not H.entails(chain[:2], at("x", "p"))
    assert H.entails([at("v", "p"), sees("u", "v")], at("u", ("dia", "p")))
    assert not H.entails([at("u", ("box", "p"))], at("u", "p"))  # K: no reflexivity
    assert not H.entails([at("u", "p")], at("v", "p"))           # truth is relative to the named world
    premises, hyp = H.render(chain, at("x", "p"), random.Random(0))
    assert "reachable" in premises and "(" in premises


def test_contextual_surface():
    from semforge.construct.check import unfaithful_surfaces
    from semforge.construct.realize import classical_surface as surface
    imp, rng = ("->", "p", ("and", "q", "r")), random.Random(0)
    assert surface(imp, "premise") == imp
    assert surface(imp, "hypothesis") == ("or", ("not", "p"), ("and", "q", "r"))
    assert surface(("not", imp), "hypothesis") == ("and", "p", ("not", ("and", "q", "r")))
    assert surface(("and", "p", imp), "hypothesis") == ("and", "p", imp)  # top level only
    assert surface(("->", ("not", "p"), "q"), "hypothesis") == ("or", "p", "q")  # no double negation
    premises, hyp = PACK.render([imp], imp, rng)
    assert "if" in premises.lower() and "if" not in hyp.lower() and " or " in hyp
    assert "does not" in hyp and "it is false" not in hyp  # the atom negates itself
    assert not unfaithful_surfaces(PACK, [imp, ("not", imp)])
    liar = replace(PACK, surface=lambda f, role: "p" if role == "hypothesis" else f)
    assert unfaithful_surfaces(liar, ["q"]) == [("hypothesis", "q", "p")]


def test_rule_dsl_and_sorts():
    from semforge.construct.signature import parse, rule
    r = rule("ns", "(one_of X A B C), not (eq X B), not (eq X C) |- eq X A")
    assert r.premises[0] == ("one_of", "?X", "?A", "?B", "?C") and r.conclusion == ("eq", "?X", "?A")
    assert parse("-> (not A) B") == ("->", ("not", "?A"), "?B")
    with pytest.raises(ValueError):
        CSP.signature.infer([("eq", "x", "a"), ("eq", "a", "x")])  # a: Colour then Person


def test_rule_check_rejects_rules_that_cannot_mean_what_they_say():
    """A malformed rule used to load and then silently never fire (a free conclusion variable) or fire on
    the wrong things (a metavariable at two sorts); now the calculus refuses it, naming the problem."""
    from semforge.construct.signature import rule
    sig = get_pack("arithmetic").signature
    bad = {"unknown symbol": "(score P A), (bigger A B) |- higher P P",
           "takes 2 arguments": "(score P A B) |- higher P P",
           "used as Num and as Person": "(score P A) |- higher P A",
           "in the conclusion but no premise": "(score P A) |- higher P Q",
           "concludes the interpreted": "(score P A), (score Q B) |- less A B",
           "bound only by interpreted": "(score P A), (plus A D C) |- score P A",
           "where a Num belongs": "(score P (plus A B C)) |- higher P P"}
    for message, text in bad.items():
        with pytest.raises(ValueError, match=message):
            sig.check_rule(rule("r", text))
    with pytest.raises(ValueError, match="bare formula premise"):
        sig.check_rule(rule("r", "A, (score P B) |- higher P P"))
    for r in get_pack("arithmetic").calculus.rules.values():
        sig.check_rule(r)


def test_interpreted_symbols_behave_as_their_table():
    """Computed theory facts (`Symbol.holds`) give the same closure, structure and certificates as the
    same relation listed as background facts, and the logic check confirms holds agrees with z3."""
    import io
    from semforge.certcheck import check as certcheck
    from semforge.construct.check import interpreted_disagreements
    from semforge.construct.forward import forward
    from semforge.construct.pack import declare
    from semforge.construct.synth import export_certificate
    p = get_pack("arithmetic")
    sig = p.signature
    table = tuple(sorted(p.calculus.background_set.facts(), key=repr))
    listed = replace(sig, symbols={n: replace(s, holds=None) for n, s in sig.symbols.items()}, background=table)
    ref = declare("arithmetic_listed", listed, list(p.calculus.rules.values()), atlas_size=2)
    assert ("plus", "n2", "n3", "n5") in p.calculus.background_set and ("less", "n3", "n2") not in p.calculus.background_set
    rng, seen = random.Random(1), 0
    while seen < 4:
        out = forward(p, rng, rng.randint(2, 4))
        if not out:
            continue
        inst, seen = out[0], seen + 1
        a, b = (analyze(q.calculus, inst.premises, inst.conclusion, inst.universe) for q in (p, ref))
        edges = lambda c: {h: sorted(map(repr, e)) for h, e in c.edges.items()}
        assert edges(a.closure) == edges(b.closure)
        assert (a.depth, a.mandatory_steps, a.mandatory_premises) == (b.depth, b.mandatory_steps, b.mandatory_premises)
        cert = export_certificate(p, replace(inst, analysis=a))
        assert cert is None or certcheck(cert) == []
    assert interpreted_disagreements(p, rng, cap=200) == []
    wrong = replace(sig, symbols={**sig.symbols, "less": replace(sig.symbols["less"], holds=lambda a, b: a != b)})
    assert interpreted_disagreements(declare("wrong", wrong, list(p.calculus.rules.values())), rng, cap=200)


def test_answers_bind_query_variables_with_certificates():
    kin = get_pack("kinship")
    facts = [("parent", "ann", "bob"), ("parent", "bob", "cy"), ("parent", "cy", "dee")]
    got = {b["X"]: a for b, a in kin.answers(facts, "ancestor ann X")}
    assert sorted(got) == ["bob", "cy", "dee"]
    assert got["dee"].depth == 3 and got["dee"].mandatory_premises == set(facts)
    assert kin.answers(facts, "ancestor dee X") == []


# --- the indexed chase computes exactly what the naive one does ---------------------------------------

from semforge.construct.calculus import rule_of  # noqa: E402


def _naive_analysis(calculus, premises, goal, universe, cap=256):
    from semforge.construct.calculus import Rule, step_id
    from semforge.core.provenance import provenance_chase
    premises, goal = [calculus.canonical(p) for p in premises], calculus.canonical(goal)
    prov = provenance_chase(premises, list(calculus.rules.values()),
                            lambda r, closure: calculus.instances(r, closure, universe),
                            lambda x: None if isinstance(x, Rule) else x, cap, step_id=step_id)
    if goal not in prov.supports:
        return set(), None
    return set(prov.minimal_supports(goal, allow_incomplete=True)), prov.min_depth(goal)


@pytest.mark.parametrize("name", ["propositional", "monadic", "csp"])
def test_indexed_chase_matches_naive(name):
    import semforge.construct.synth as S
    pack, rng, cases = get_pack(name), random.Random(7), []
    syn, orig = Synthesizer(pack), S.analyze
    S.analyze = lambda *a, **k: (cases.append(a), orig(*a, **k))[1]
    try:
        for _ in range(12):
            syn.realize(Target(rng.choice(["entailment", "contradiction", "neutral"]),
                               rules=syn.sample_plan(rng, rng.randint(2, 6)),
                               distractors=Distractors(rng.randint(0, 2), rng.randint(0, 2))), rng)
    finally:
        S.analyze = orig
    assert len(cases) >= 10
    for a in cases:
        got = analyze(*a)
        proofs, depth = _naive_analysis(*a)
        # steps compared modulo the order of unordered premises (permuted copies are one step, one proof)
        norm = lambda x: (x[1], x[2], frozenset(x[3])) if rule_of(x) is not None else x
        same = lambda ps: {frozenset(map(norm, p)) for p in ps}
        assert (same(got.proofs), got.depth) == (same(proofs), depth)
        assert len(got.proofs) == len(same(got.proofs))  # no proof counted twice
        if proofs:  # ablation gives exactly the steps and premises every proof contains
            steps = set.intersection(*({norm(x) for x in p if rule_of(x) is not None} for p in proofs))
            assert {norm(x) for x in got.mandatory_steps} == steps
            assert got.mandatory_premises == {x for x in frozenset.intersection(*proofs) if rule_of(x) is None}


# --- shortcut atlas -------------------------------------------------------------------------------------

def test_proof_guided_atlas_lift():
    from semforge.construct.atlas import Atlas
    pack = get_pack("propositional")
    steps = [("modus_ponens", 0, (1, 2)), ("modus_ponens", 1, (3, 4))]
    nodes = ["c", "b", ("->", "b", "c"), "a", ("->", "a", "b")]
    a = analyze(pack.calculus, [nodes[2], nodes[3], nodes[4]], nodes[0],
                Universe(nodes, pack.negate, pack.closed))
    atlas = Atlas(pack, max_size=0)
    assert a.derivable and not a.requires(pack.calculus.kinds([r for r, _, _ in steps]))
    pattern = atlas.lift(steps, {"modus_ponens"}, analysis=a, nodes=nodes)
    assert pattern == ("modus_ponens", (("modus_ponens", (None, None)), None))
    assert atlas.violated(steps, 1, nodes)


def test_proof_guided_atlas_does_not_learn_uncertified_pattern():
    from semforge.construct.atlas import Atlas
    pack = get_pack("propositional")
    steps = [("modus_ponens", 0, (1, 2))]
    nodes = ["b", "a", ("->", "a", "b")]
    a = analyze(pack.calculus, nodes[1:], nodes[0], Universe(nodes, pack.negate, pack.closed))
    atlas = Atlas(pack, max_size=0)
    assert a.derivable and a.requires(pack.calculus.kinds(["modus_ponens"]))
    assert atlas.lift(steps, {"modus_ponens"}, analysis=a, nodes=nodes) is None
    assert not atlas.nogoods


def test_atlas_contents():
    from semforge.construct.atlas import atlas_for
    prop = {atlas_for(get_pack("propositional")).describe(q) for q in atlas_for(get_pack("propositional")).nogoods}
    assert {"modus_ponens(modus_ponens, _)", "and_elimination(and_introduction)"} <= prop
    assert "instantiation(barbara, _)" in {atlas_for(MONADIC).describe(q) for q in atlas_for(MONADIC).nogoods}
    assert not atlas_for(CSP).nogoods


@pytest.mark.parametrize("name", ["propositional", "monadic"])
def test_atlas_nogoods_always_shortcut(name):
    # soundness: any derivation containing a nogood, built without the atlas, loses a planned step
    from semforge.construct.atlas import atlas_for
    pack, rng, seen = get_pack(name), random.Random(3), 0
    atlas, syn = atlas_for(pack), Synthesizer(pack, atlas=False)
    for _ in range(300):
        d = derive(pack.calculus, [None] * rng.randint(2, 6), rng, Fresh())
        if d is None or not any(atlas.violated(d.steps[:k + 1], d.steps[k][1], d.nodes) for k in range(len(d.steps))):
            continue
        seen += 1
        a = analyze(pack.calculus, d.premises, d.conclusion,
                    Universe([*d.premises, *d.intermediates], pack.negate, pack.closed))
        assert not a.requires(pack.calculus.kinds(d.rules))  # some planned application is not mandatory
    assert seen >= 20


@pytest.mark.parametrize("name", ["propositional", "monadic"])
def test_atlas_construction_has_no_shortcut_failures(name):
    rng, syn = random.Random(4), Synthesizer(get_pack(name))
    for _ in range(10):
        syn.realize(Target(rng.choice(["entailment", "contradiction"]), rules=syn.sample_plan(rng, 12),
                           distractors=Distractors(3, 3)), rng)
    assert syn.stats["shortcut"] == 0 and syn.stats["ok"] >= 8


# --- shape targets --------------------------------------------------------------------------------------

def test_topology_of_mandatory_steps():
    P = get_pack("propositional")
    pr = ["a", ("->", "a", "b"), "d", ("->", "d", "e"), ("->", ("and", "b", "e"), "g")]
    a = analyze(P.calculus, pr, "g", Universe(pr + [("and", "b", "e")], P.negate, P.closed))
    assert a.topology == {"spine": 3, "joins": 1}



def test_shared_subproof_is_not_a_copy():
    """One stem used by both branches vs two stems derived independently (review2, follow-up C)."""
    from semforge.construct.archive import structure
    from semforge.construct.calculus import Rule
    ground = lambda rules, premises, atoms: analyze(Calculus(rules), premises, "g", Universe(atoms, negate))
    shared = ground([Rule("stem", ("a",), "b"), Rule("left", ("b",), "c"), Rule("right", ("b",), "d"),
                     Rule("join", ("c", "d"), "g")], ["a"], ["a", "b", "c", "d", "g"])
    copied = ground([Rule("stem1", ("a1",), "b1", kind="stem"), Rule("stem2", ("a2",), "b2", kind="stem"),
                     Rule("left", ("b1",), "c"), Rule("right", ("b2",), "d"), Rule("join", ("c", "d"), "g")],
                    ["a1", "a2"], ["a1", "a2", "b1", "b2", "c", "d", "g"])
    assert (shared.shared, copied.shared) == (1, 0)
    assert structure(shared) == "join(left(&stem(p)),right(&stem(p)))"
    assert structure(copied) == "join(left(stem(p)),right(stem(p)))"
    assert shared.profile["shared"] == 1 and shared.topology == copied.topology


def test_profile_separates_overdetermined_from_shallow():
    # two disjoint 2-step proofs of g (more via hypothetical syllogism): no mandatory step, yet the
    # smallest proof has 2 steps
    from semforge.audit import summarize_profiles
    P = get_pack("propositional")
    pr = ["a", ("->", "a", "b"), ("->", "b", "g"), "c", ("->", "c", "d"), ("->", "d", "g")]
    a = analyze(P.calculus, pr, "g", Universe(pr, P.negate, P.closed))
    assert (a.profile["min_depth"], a.profile["min_size"], a.profile["mandatory"]) == (2, 2, 0)
    assert a.profile["proofs"] >= 2
    given = analyze(P.calculus, pr + ["g"], "g", Universe(pr + ["g"], P.negate, P.closed)).profile
    s = summarize_profiles([a.profile, given, None], planted=[2, 0, 5])
    assert (s["n"], s["zero_given"], s["zero_alternatives"], s["multi_proof"], s["min_size_below_planted"]) == \
        (2, 0.5, 0.5, 1.0, 0.0)  # a given goal still has its derived proofs too
    assert (a.profile["supports"], s["zero_overdetermined"], s["zero_rerouted"]) == (2, 0.5, 0.0)


def test_chain_under_hypothetical_syllogism_is_rerouted_not_overdetermined():
    # Figure 1a: a five-step chain has many proofs under HS and no mandatory step, but one premise set
    from semforge.audit import summarize_profiles
    P = get_pack("propositional")
    pr = ["a", ("->", "a", "b"), ("->", "b", "c"), ("->", "c", "d"), ("->", "d", "e"), ("->", "e", "f")]
    p = analyze(P.calculus, pr, "f", Universe(pr, P.negate, P.closed)).profile
    assert (p["mandatory"], p["min_size"], p["supports"]) == (0, 5, 1) and p["proofs"] > 1
    s = summarize_profiles([p])
    assert (s["zero_rerouted"], s["zero_overdetermined"]) == (1.0, 0.0)

@pytest.mark.parametrize("name", ["propositional", "monadic", "csp", "modal", "modal_s4", "modal_nested"])
def test_shape_targets(name):
    from semforge.construct.build import random_skeleton, shape_topology
    rng, syn = random.Random(2), Synthesizer(get_pack(name))
    made = 0
    for br in (0.0, 0.7):
        for _ in range(4):
            sk = random_skeleton(rng, 7, branch=br)
            i = syn.realize(Target(rng.choice(["entailment", "contradiction"]), rules=(), shape=sk,
                                   distractors=Distractors(2, 2)), rng)
            if i is not None:
                made += 1
                assert i.analysis.topology == shape_topology(sk)
    assert made >= 6


def test_guarded_nogood_excludes_negated_middle():
    # MT(HS, _) is bypassed by two modus tollens steps; with a negated middle the first one concludes a
    # double negation, which the universe admits (stacked=True), so HS is bypassed there too; without
    # stacked negations HS was needed
    from semforge.construct.atlas import atlas_for
    P = get_pack("propositional")
    atlas, q = atlas_for(P), ("modus_tollens", (("hypothetical_syllogism", (None, None)), None))
    steps = [("modus_tollens", 0, (1, 2)), ("hypothetical_syllogism", 1, (3, 4))]
    for b, guarded in (("b", True), (("not", "b"), True)):
        nodes = [("not", "a"), ("->", "a", "c"), ("not", "c"), ("->", "a", b), ("->", b, "c")]
        assert atlas.violated(steps, 0, nodes) == guarded
        a = analyze(P.calculus, nodes[2:], nodes[0], Universe(nodes, P.negate, P.closed))
        assert bool(P.calculus.kinds(["modus_tollens", "hypothetical_syllogism"]) - a.required_rules) == guarded


@pytest.mark.parametrize("strategy", ["shape", "forward"])
def test_constructed_domain_strategies(strategy):
    from semforge.domains import base
    from semforge.pipeline.rows import generate
    rows = generate(base.get(f"constructed_monadic_{strategy}"), 0, 0, 6, variants=False)
    assert rows and all(r["certified"] is not None for r in rows if r["label"] != "neutral")


@pytest.mark.parametrize("strategy", ["shape", "forward"])
def test_independent_certificate_check(strategy):
    import copy
    from semforge.certcheck import check
    from semforge.construct.synth import export_certificate
    from semforge.domains.constructed import ConstructedDomain
    d, rng, certs = ConstructedDomain("propositional", strategy=strategy), random.Random(0), []
    while len(certs) < 3:
        out = d.sample(rng, 3)
        if out and (c := export_certificate(d.pack, out[0][0])):
            certs.append(c)
    for c in certs:
        assert check(c) == [], check(c)
        bad = copy.deepcopy(c)
        bad["proof"][-1]["head"] = next(p for p in c["premises"] if p != c["proof"][-1]["head"])
        assert check(bad)  # not a rule instance, and the goal is no longer reached
        if c["mandatory_steps"] or c["mandatory_premises"]:
            silent = {**c, "mandatory_steps": [], "mandatory_premises": []}
            assert any("not claimed" in e for e in check(silent))  # omissions fail too, not only false claims
        if c["mandatory_premises"]:
            gone = copy.deepcopy(c)
            gone["premises"].remove(gone["mandatory_premises"][0])
            gone["mandatory_premises"] = []
            assert check(gone)  # the proof now uses a formula that is neither given nor derived


def test_fm_policy_orders_and_learns():
    from semforge.construct.build import random_skeleton
    from semforge.construct.policy import FMPolicy, novelty_reward
    pol = FMPolicy()
    syn, rng = Synthesizer(PACK, policy=pol, reward=novelty_reward()), random.Random(0)
    made = [syn.realize(Target("entailment", rules=(), shape=random_skeleton(rng, 6, branch=0.4)), rng)
            for _ in range(10)]
    assert all(i is not None and i.label == "entailment" for i in made)  # certification is unchanged
    assert pol.w0 != 0 and abs(pol.w).sum() > 0                          # and the policy learned


def test_nogood_certificates_check_independently():
    import copy
    from semforge.certcheck import check
    from semforge.construct.atlas import atlas_for, nogood_certificates
    atlas = atlas_for(get_pack("propositional"))
    certs = [c for q in atlas.nogoods for c in nogood_certificates(atlas, q)]
    assert certs and all(c is not None and check(c) == [] for c in certs)
    bad = copy.deepcopy(next(c for c in certs if c["proof"]))  # some bypasses are empty: the goal is given
    bad["avoids"] = bad["proof"][-1]  # claim to avoid a step the bypassing proof makes
    assert check(bad)


def test_rule_schemas_proved_for_all_instances():
    from semforge.construct.check import prove_rule
    from semforge.construct.packs import NAMES
    for name in NAMES:
        p = get_pack(name)
        for r in p.calculus.rules.values():
            assert prove_rule(p, r) == "proved", (name, r.name)


def test_unsound_rule_schemas_refuted():
    from semforge.construct.check import prove_rule
    from semforge.construct.signature import rule
    assert prove_rule(get_pack("propositional"), rule("bad", "(-> A B), B |- A")) == "refuted"
    assert prove_rule(get_pack("grid"), rule("bad", "(left A B), (at A H) |- at B H")) == "refuted"
    # two metavariables may denote one object: a rule sound only for distinct constants is refuted
    assert prove_rule(get_pack("grid"), rule("bad", "(at A H), (at B H) |- notsame A B")) == "refuted"


def test_datalog_import():
    from semforge.construct.check import prove_rule
    from semforge.construct.packs.datalog import from_datalog
    from semforge.construct.signature import rule
    p = from_datalog("reach", "path(X, Y) :- edge(X, Y).  path(X, Z) :- edge(X, Y), path(Y, Z).  % comment")
    assert sorted(r.name for r in p.calculus.rules.values()) == ["path_0", "path_1"]
    assert all(prove_rule(p, r) == "proved" for r in p.calculus.rules.values())
    assert prove_rule(p, rule("bad", "(edge X Y) |- edge Y X")) == "refuted"
    assert p.render([("edge", "a", "b")], ("path", "a", "b"), random.Random(0))


def test_countermodels_certify_neutral_labels():
    import copy
    from semforge.certcheck import ModelChecker, check_countermodels
    from semforge.construct.oracle import countermodel, query
    sig = get_pack("order").signature
    premises = [("taller", "Ada", "Bo"), ("taller", "Cy", "Bo")]
    assert countermodel(sig, premises, ("not", ("taller", "Bo", "Ada"))) is None  # entailed: no countermodel
    cms = [{"refutes": r, **countermodel(sig, premises, g)} for r, g in
           (("entailment", ("taller", "Ada", "Cy")), ("contradiction", ("not", ("taller", "Ada", "Cy"))))]
    assert all(len(cm["model"]["sorts"]["Person"]) <= 3 for cm in cms)  # small models first
    cert = {"label": "neutral", "countermodels": cms,
            "smt2": query(sig, premises, ("taller", "Ada", "Cy")).to_smt2(),
            "smt2_contradiction": query(sig, premises, ("not", ("taller", "Ada", "Cy"))).to_smt2()}
    assert check_countermodels(cert) == []
    assert check_countermodels({**cert, "countermodels": cms[:1]})  # one side is not enough
    assert check_countermodels({**cert, "smt2_contradiction": None})  # nothing to check it against
    swapped = [{**cms[0], "refutes": "contradiction"}, {**cms[1], "refutes": "entailment"}]
    assert check_countermodels({**cert, "countermodels": swapped})  # bound to the problem, not to itself
    bad = copy.deepcopy(cms[0]["model"])
    bad["functions"]["taller"] = {"else": False, "rows": []}  # Ada is no longer taller than Bo
    assert ModelChecker(bad).check(cms[0]["smt2"])


def test_entailment_certificates_show_their_premises_consistent():
    from semforge.certcheck import check_countermodels
    from semforge.construct.oracle import countermodel, query
    sig = get_pack("order").signature
    h, nh = ("taller", "Ada", "Cy"), ("not", ("taller", "Ada", "Cy"))
    ok = [("taller", "Ada", "Bo"), ("taller", "Bo", "Cy")]
    cert = {"label": "entailment", "smt2": query(sig, ok, h).to_smt2(), "smt2_contradiction": query(sig, ok, nh).to_smt2(),
            "countermodels": [{"refutes": "contradiction", "model": countermodel(sig, ok, nh)["model"]}]}
    assert check_countermodels(cert) == []
    assert check_countermodels({**cert, "countermodels": []})  # no witness: consistency not shown
    # inconsistent premises derive any label, and no model of them exists to witness consistency
    bad = [*ok, ("taller", "Cy", "Ada")]
    assert countermodel(sig, bad, nh) is None


def test_model_checker_validates_the_model_against_declarations():
    from semforge.certcheck import ModelChecker
    smt = ("(declare-sort S 0) (declare-fun c () S) (declare-fun P (S) Bool) "
           "(assert (forall ((x S)) (P x))) (assert (not (P c)))")  # unsatisfiable
    ok = {"sorts": {"S": ["a", "b"]}, "functions": {"c": {"rows": [], "else": "a"},
                                                    "P": {"rows": [], "else": True}}}
    assert ModelChecker(ok).check(smt) == ["assertion 1 is false in the model"]
    outside = {"sorts": {"S": ["a"]}, "functions": {"c": {"rows": [], "else": "b"},  # b is not in S
                                                    "P": {"rows": [["a", True]], "else": False}}}
    assert ModelChecker(outside).check(smt)
    empty = {"sorts": {"S": []}, "functions": {"c": {"rows": [], "else": "a"}, "P": {"rows": [], "else": False}}}
    assert any("no elements" in e for e in ModelChecker(empty).check(smt))
    arity = {"sorts": {"S": ["a"]}, "functions": {"c": {"rows": [], "else": "a"},
                                                  "P": {"rows": [["a", "a", True]], "else": True}}}
    assert ModelChecker(arity).check(smt)
    missing = {"sorts": {"S": ["a"]}, "functions": {"P": {"rows": [], "else": True}}}
    assert any("no table for c" in e for e in ModelChecker(missing).check(smt))


def test_certificate_label_is_bound_to_the_hypothesis():
    import copy
    from semforge.certcheck import check
    from semforge.construct.synth import export_certificate
    pack = get_pack("propositional")
    premises, h = ["a", ("->", "a", "b")], "b"
    universe = Universe([*premises, h], pack.negate, pack.closed)
    inst = Instance(premises, h, "entailment", h, universe, analysis=analyze(pack.calculus, premises, h, universe))
    cert = export_certificate(pack, inst)
    assert cert["hypothesis"] == "b" and check(cert) == []
    for label in ("contradiction", "potato"):
        assert check({**cert, "label": label})
    assert check({**cert, "hypothesis": "a"})
    assert check({**cert, "withheld": ["a"]})  # only neutral proofs may add premises
    assert check({k: v for k, v in cert.items() if k != "hypothesis"})
    neutral = {**copy.deepcopy(cert), "label": "neutral", "withheld": [["->", "a", "b"]]}
    assert any("countermodels" in e for e in check(neutral))  # the structure checks; the label needs models
    assert any("actual premises derive" in e for e in check({**neutral, "withheld": [["->", "a", "b"]],
                                                             "premises": ["a", ["->", "a", "b"], "b"]}))
    contra = Instance(["a", ("->", "a", ("not", "b"))], h, "contradiction", ("not", "b"), universe := Universe(
        ["a", ("->", "a", ("not", "b")), h, ("not", "b")], pack.negate, pack.closed))
    contra.analysis = analyze(pack.calculus, contra.premises, ("not", "b"), universe)
    c = export_certificate(pack, contra)
    assert check(c) == [] and check({**c, "label": "entailment"})


def test_exported_neutral_certificates_check():
    from semforge.certcheck import check
    from semforge.construct.synth import export_certificate
    from semforge.domains.constructed import ConstructedDomain
    d, rng, certs = ConstructedDomain("order"), random.Random(0), []
    while len(certs) < 2:
        out = d.sample(rng, 3)
        if out and out[0][0].label == "neutral" and (c := export_certificate(d.pack, out[0][0])):
            certs.append(c)
    for c in certs:
        assert c["withheld"] and check(c) == [], check(c)
        assert check({**c, "withheld": []})
        flip = {"entailment": "contradiction", "contradiction": "entailment"}
        assert check({**c, "countermodels": [{**cm, "refutes": flip[cm["refutes"]]} for cm in c["countermodels"]]})


def test_pack_validation_fails_on_a_refuted_schema():
    import io
    from semforge.construct.check import check as check_pack
    from semforge.construct.pack import declare
    from semforge.construct.signature import rule
    p = get_pack("propositional")
    bad = rule("bad", "(-> A B), B |- A")
    out = io.StringIO()
    assert not check_pack(declare("bad", p.signature, [*p.calculus.rules.values(), bad]), n=3, out=out)
    assert "UNSOUND SCHEMA" in out.getvalue()


def _mandatory_by_ablation(a):
    """Reference: remove each step (with its permuted copies) and each premise of one proof in turn."""
    from semforge.construct.calculus import step_id
    edges, prem = a._witness()
    bg = set(a.calculus.background)
    return (frozenset(step_id(*a.edges[k]) for k in edges if not a._reaches(skip_edge=k)),
            frozenset(p for p in prem if p not in bg and not a._reaches(skip_premise=p)))


def test_mandatory_matches_ablation_on_random_systems():
    from semforge.construct.calculus import Rule
    checked = 0
    for seed in range(300):
        rng = random.Random(seed)
        atoms = [f"p{i}" for i in range(rng.randint(4, 14))]
        rules = [Rule(f"r{i}", tuple(rng.sample(atoms, rng.randint(1, 3))), rng.choice(atoms))
                 for i in range(rng.randint(3, 40))]  # cycles, alternatives, given facts also derived
        a = analyze(Calculus(rules), rng.sample(atoms, rng.randint(1, 4)), rng.choice(atoms), Universe(atoms, negate))
        if a.derivable:
            assert (a.mandatory_steps, a.mandatory_premises) == _mandatory_by_ablation(a), seed
            checked += 1
    assert checked > 100


@pytest.mark.parametrize("name", ["propositional", "csp", "modal_s4", "order", "grid"])
def test_mandatory_matches_ablation_on_packs(name):
    from semforge.construct.build import random_skeleton
    pack, rng = get_pack(name), random.Random(1)
    syn, checked = Synthesizer(pack), 0
    for _ in range(12):
        inst = syn.realize(Target(rng.choice(["entailment", "contradiction"]), rules=(),
                                  shape=random_skeleton(rng, rng.randint(3, 12), branch=0.4),
                                  distractors=Distractors(2, 2)), rng)
        if inst is not None:
            a = inst.analysis
            a2 = analyze(pack.calculus, a.premises, a.goal, inst.universe)
            assert (a2.mandatory_steps, a2.mandatory_premises) == _mandatory_by_ablation(a2)
            checked += 1
    assert checked


@pytest.mark.parametrize("name", ["propositional", "csp", "grid", "kinship"])
def test_shape_feasibility_never_rejects_a_built_shape(name, monkeypatch):
    from semforge.construct import build
    from semforge.construct.build import Fresh, derive_shape, random_skeleton, shape_feasible
    calc, rng = get_pack(name).calculus, random.Random(0)
    monkeypatch.setattr(build, "shape_feasible", lambda c, sk: True)  # search without the filter
    built = [sk for sk in (random_skeleton(rng, rng.randint(1, 8), branch=0.5) for _ in range(60))
             if derive_shape(calc, sk, rng, Fresh()) is not None]
    monkeypatch.undo()
    assert built and all(shape_feasible(calc, sk) for sk in built)
    if name == "grid":  # a join (naked/hidden single) cannot feed another join: its inputs are "not at"
        assert not shape_feasible(calc, (((), ()), ()))


def test_theory_is_stated():
    """A label must not rest on theory the text leaves out: a pack whose oracle has axioms or background
    facts states them, first, in every rendering (S4's reflexivity was unstated, and models answered
    "neutral" correctly from the text)."""
    import random
    from dataclasses import replace
    from semforge.construct.packs import NAMES
    from semforge.construct.realize import Realizer
    for name in NAMES:
        sig = get_pack(name).signature
        assert bool(sig.theory) == bool(sig.axioms or sig.background or sig.interpreted), name
    sig = replace(get_pack("propositional").signature, theory=("Stated first.",))
    text, _ = Realizer(sig).render(["a", ("->", "a", "b")], "b", random.Random(0))
    assert text.split("\n")[0] == "Stated first." and len(text.split("\n")) == 3


def test_forward_rows_are_consistent():
    """Distractors are grounded apart from the main proof; in a functional pack (grid: one house per
    person) they often contradicted it, and inconsistent premises were labelled entailment."""
    import random
    from semforge.construct.forward import forward
    p, rng, rows = get_pack("grid"), random.Random(0), 0
    while rows < 15:
        out = forward(p, rng, rng.randint(1, 10))
        if out:
            inst, _ = out
            rows += 1
            assert not (p.entails(inst.premises, inst.hypothesis) and p.entails(inst.premises, p.negate(inst.hypothesis)))


def test_solved_grounding_keeps_functional_proofs_consistent():
    """Scores chained through differences: binding each number at random contradicts the other paths;
    solved grounding binds them so all nodes hold together, using only background facts that exist."""
    import random
    from semforge.construct.build import _bind_satisfiable
    from semforge.construct.terms import subst
    p = get_pack("arithmetic")
    nodes = [("more", "?P", "?R", "?F"), ("more", "?P", "?Q", "?D"), ("more", "?Q", "?R", "?E"),
             ("plus", "?D", "?E", "?F"), ("score", "?R", "?A"), ("score", "?P", "?C"), ("plus", "?A", "?F", "?C")]
    for seed in range(5):
        s = _bind_satisfiable(p.calculus, nodes, {}, random.Random(seed))
        ground = [subst(x, s) for x in nodes]
        assert all(x in p.calculus.background_set for x in ground if x[0] == "plus")
        facts = [x for x in ground if x[0] != "plus"]
        assert not p.entails(facts, p.negate(facts[0]))
        assert len({x[1] for x in facts if x[0] == "score"}) == 2  # distinct people where possible


def test_generation_is_deterministic_across_processes():
    """The same seed gives the same rows in a fresh process with another hash seed (set iteration order)."""
    import os
    import subprocess
    import sys
    code = ("import random; from semforge.domains.constructed import ConstructedDomain\n"
            "d, rng = ConstructedDomain('monadic', strategy='shape'), random.Random(7)\n"
            "print([d.render(i) for _ in range(4) for i, _ in d.sample(rng, 3)])")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": h}).stdout for h in ("1", "2")}
    assert len(outs) == 1 and len(next(iter(outs))) > 100


def test_certificate_checker_rejects_malformed_claims():
    """Regressions: out-of-range mandatory indices, repeated steps, the goal as an unclaimed mandatory premise
    of a zero-step proof, and malformed SMT-LIB must all fail."""
    import copy
    from semforge.certcheck import ModelChecker, check
    from semforge.construct.synth import export_certificate
    pack = get_pack("propositional")
    premises, h = ["a", ("->", "a", "b")], "b"
    universe = Universe([*premises, h], pack.negate, pack.closed)
    cert = export_certificate(pack, Instance(premises, h, "entailment", h, universe,
                                             analysis=analyze(pack.calculus, premises, h, universe)))
    cert = {k: v for k, v in cert.items() if k not in ("label", "hypothesis", "countermodels")}
    assert check(cert) == []
    for n in (999, -1):
        assert any("not a step of the proof" in e for e in check({**cert, "mandatory_steps": [*cert["mandatory_steps"], n]}))
    twice = copy.deepcopy(cert)
    twice["proof"].append(copy.deepcopy(twice["proof"][-1]))
    twice["mandatory_steps"] = [*cert["mandatory_steps"], len(twice["proof"]) - 1]
    assert any("repeats step" in e for e in check(twice))
    given = {**copy.deepcopy(cert), "premises": ["b"], "goal": "b", "proof": [], "mandatory_steps": [],
             "mandatory_premises": []}
    assert any("mandatory but not claimed" in e for e in check(given))
    assert check({**given, "mandatory_premises": ["b"]}) == []
    model = {"sorts": {"S": ["a"]}, "functions": {"P": {"rows": [], "else": True}}}
    decl = "(declare-sort S 0) (declare-fun P (S) Bool) "
    for bad in (decl + "(assert false", "", ")", decl + "(assert true))"):
        assert ModelChecker(model).check(bad), repr(bad)
    assert ModelChecker(model).check(decl + "(assert (forall ((x S)) (P x)))") == []
