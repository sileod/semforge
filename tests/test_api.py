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
