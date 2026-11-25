"""The public API: generate meets the request and certifies it; analyze and audit on hand-made problems."""
import pytest

import semforge as sf


@pytest.mark.parametrize("label", ["entailment", "contradiction", "neutral"])
def test_generate_meets_request_and_verifies(label):
    p = sf.generate("propositional", mandatory_steps=3, label=label, seed=0)
    assert p.label == label and p.mandatory_steps == 3
    assert sum(p.mandatory_rules.values()) == 3
    assert p.premises and p.hypothesis
    assert p.verify() == []


def test_generate_many_is_deterministic():
    a = sf.generate("monadic", mandatory_steps=2, n=3, seed=7)
    b = sf.generate("monadic", mandatory_steps=2, n=3, seed=7)
    def stable(p):  # z3 names let-bindings and picks countermodels by its in-process state: compare the rest
        d = p.to_dict()
        d["certificate"] = {k: v for k, v in d["certificate"].items() if k not in ("smt2", "smt2_contradiction", "countermodels")}
        return d

    assert [stable(x) for x in a] == [stable(x) for x in b]
    assert all(x.mandatory_steps == 2 and x.verify() == [] for x in a)


def test_generate_rejects_bad_requests():
    with pytest.raises(KeyError):
        sf.generate("no_such_logic")
    with pytest.raises(ValueError):
        sf.generate("propositional", label="maybe")
    with pytest.raises(ValueError):
        sf.generate("propositional", mandatory_steps=0)


def test_tampered_certificate_fails():
    p = sf.generate("propositional", mandatory_steps=2, label="entailment", seed=1)
    p.certificate["label"] = "contradiction"
    assert p.verify() != []


def test_analyze_chain_has_no_mandatory_step():
    prem = ["a", ("->", "a", "b"), ("->", "b", "c"), ("->", "c", "d"), ("->", "d", "e"), ("->", "e", "f")]
    r = sf.analyze("propositional", prem, "f")
    assert r["derivable"] == "entailment" and r["proofs"] == 42 and r["mandatory_steps"] == 0


def test_audit_finds_alternative_premise_sets():
    z3 = pytest.importorskip("z3")
    a, b, c = z3.Bools("a b c")
    premises = [a, z3.Implies(a, b), z3.Implies(b, c), z3.Implies(a, c), c]

    def entails(sub):
        s = z3.Solver()
        s.add(*sub, z3.Not(c))
        return s.check() == z3.unsat

    r = sf.audit_premises(premises, entails)
    assert r["supports"] == 3 and r["min_support"] == 1 and r["indispensable"] == []


def test_neutral_certificate_lists_background_of_withheld_premises():
    # the withheld premise brings a number absent from the visible problem; its facts must be in the certificate
    p = sf.generate("arithmetic", mandatory_steps=1, label="neutral", seed=1)
    assert p.verify() == []


def test_analyze_families_and_summarize():
    # two routes to c, each through a disjunctive step at a different place: no step is mandatory,
    # but every proof applies disjunctive syllogism, so the operator is needed
    fam = {"disjunction": {"disjunctive_syllogism", "constructive_dilemma"}}
    two = [("or", "p", "q"), ("not", "p"), ("->", "q", "c"), ("or", "r", "s"), ("not", "r"), ("->", "s", "c")]
    x = sf.analyze("propositional", two, "c", families=fam)
    assert x["proofs"] == 2 and x["mandatory_steps"] == 0 and x["mandatory_rules"] == {}
    assert x["rules_in_every_proof"]["disjunctive_syllogism"] == 1 and x["families_in_every_proof"] == {"disjunction": True}
    y = sf.analyze("propositional", ["a", ("->", "a", "b")], "b", families=fam)
    assert y["mandatory_rules"] == {"modus_ponens": 1} and y["families_in_every_proof"] == {"disjunction": False}
    s = sf.summarize([x, y, sf.analyze("propositional", ["a"], "z")], labels=["entailment", "contradiction", "neutral"])
    assert s["derivable"] == 0.667 and s["family_in_every_proof"] == {"disjunction": 0.5}
    assert s["rule_in_every_proof"]["modus_ponens"] == 1.0 and s["label_disagreement"] == 0.333
    assert s["labelled_but_underivable"] == 0.0 and s["inconsistent"] == 0.0
    u = sf.summarize([sf.analyze("propositional", ["a"], "z")], labels=["entailment"])
    assert u["label_disagreement"] == 0.0 and u["labelled_but_underivable"] == 1.0
    assert sf.analyze("propositional", ["a", ("not", "a")], "a")["inconsistent"] is True
    with pytest.raises(ValueError):
        sf.summarize([x, y], labels=["entailment"])


def test_generate_require_rules():
    modal = {"necessity_elimination", "necessity_refutation", "impossibility_elimination", "possibility_introduction"}
    assert modal <= set(sf.rules("modal"))
    ps = sf.generate("modal", mandatory_steps=3, n=6, seed=1, require_rules=modal, label="entailment")
    for p in ps:
        assert set(p.mandatory_rules) & modal and sum(p.mandatory_rules.values()) == 3 and p.verify() == []
        a = sf.analyze("modal", list(p.instance.premises), p.instance.hypothesis, families={"modal": modal})
        assert a["families_in_every_proof"] == {"modal": True}
    with pytest.raises(ValueError):
        sf.generate("modal", require_rules={"no_such_rule"})


def test_rule_families_in_api():
    # grid's left_back is an argument-order variant of left: listed once, accepted as its family
    assert "left" in sf.rules("grid") and "left_back" not in sf.rules("grid")
    p = sf.generate("grid", mandatory_steps=3, seed=0, require_rules={"left_back"}, max_attempts=20)
    assert "left" in p.mandatory_rules
    with pytest.raises(ValueError):
        sf.analyze("propositional", ["a"], "a", families={"x": {"no_such_rule"}})


def test_family_necessity_by_removal():
    from semforge.api import _needed
    from semforge.construct.packs import get as get_pack
    pk = get_pack("propositional")
    prem = [("or", "a", "b"), ("not", "a"), ("->", "c", "b"), "c"]  # b by disjunctive syllogism or by modus ponens
    assert not _needed(pk, prem, "b", {"disjunctive_syllogism"}) and not _needed(pk, prem, "b", {"modus_ponens"})
    assert _needed(pk, prem, "b", {"disjunctive_syllogism", "modus_ponens"})
    x = sf.analyze("propositional", prem, "b", families={"disjunction": {"disjunctive_syllogism"}})
    assert x["families_in_every_proof"] == {"disjunction": False} and x["rules_in_every_proof"] == {}
