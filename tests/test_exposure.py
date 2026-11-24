"""Requested shortcut exposure: `Target(exposed=k)` and its certificates.

An exposure request is the opposite of the atlas's default: of the intended proof's applications the
universe can see, exactly k must be *bypassable* — some proof of the same conclusion avoids them —
and the rest mandatory. The certificate names the bypassed applications with a witness proof each;
certcheck verifies both directions, so the number is exact. These tests pin the exactness and the
checker's rejection of false claims (CONTRIBUTING: a construction feature's test lives beside the
property it establishes).
"""
import copy
import random

from semforge.certcheck import check
from semforge.construct.atlas import atlas_for
from semforge.construct.packs import propositional
from semforge.construct.synth import Synthesizer, export_certificate
from semforge.construct.target import Target

PACK = propositional.PACK


def realize_exposed(rng, k):
    """A certified instance with exactly k exposed applications, via a nogood-seeded plan."""
    from semforge.construct.atlas import rules_of, size
    syn = Synthesizer(PACK, attempts=60)
    for _ in range(10):
        small = [q for q in atlas_for(PACK).nogoods if size(q) <= k]
        frag = tuple(rules_of(random.Random(0).choice(small))) if small else ()
        plan = frag + syn.sample_plan(rng, max(0, 7 - len(frag)))
        inst = syn.realize(Target(label="entailment", rules=plan, exposed=k), rng)
        if inst is not None and inst.exposed == k:
            return inst


def test_exposure_is_certified_exactly():
    inst = realize_exposed(random.Random(1), k=2)
    assert inst is not None and inst.exposed == 2
    a = inst.analysis
    # exact by identity: planned steps split into mandatory and bypassed, nothing else claimed
    assert len(inst.bypassed) == 2
    mandatory_of_planned = [("step", *inst.planned[i]) for i in range(len(inst.planned))
                           if i not in {j for j, _ in inst.bypassed}]
    for sid in [("step", *inst.planned[i]) for i, _ in inst.bypassed]:
        assert sid not in a.mandatory_steps
    assert all(p in a.mandatory_steps for p in mandatory_of_planned)


def test_exposure_certificate_checks_and_rejects_false_claims():
    inst = realize_exposed(random.Random(2), k=2)
    assert inst is not None
    cert = export_certificate(PACK, inst)
    assert cert["bypassed"] and check(cert) == []

    dropped = copy.deepcopy(cert)
    dropped["bypassed"] = dropped["bypassed"][:-1]  # hide one bypass: the step is bypassable but unreported
    assert any("not reported as such" in e for e in check(dropped))

    duplicated = copy.deepcopy(cert)
    duplicated["bypassed"].append(copy.deepcopy(cert["bypassed"][0]))  # same step claimed bypassed twice
    assert any("distinct steps" in e for e in check(duplicated))

    swapped = copy.deepcopy(cert)
    mandatory = [n for n in range(len(cert["planned"])) if n not in {b["step"] for b in cert["bypassed"]}]
    swapped["bypassed"] = [{"step": mandatory[0], "proof": cert["bypassed"][0]["proof"]}]
    assert check(swapped), "a mandatory step must not pass as bypassed"


def test_exposed_zero_rows_carry_no_exposure_fields():
    syn = Synthesizer(PACK)
    rng = random.Random(3)
    plan = syn.sample_plan(rng, 4)
    inst = syn.realize(Target(label="entailment", rules=plan), rng)
    assert inst is not None and inst.realized()["exposed"] == 0
    cert = export_certificate(PACK, inst)
    assert "planned" not in cert and "bypassed" not in cert
    assert check(cert) == []
