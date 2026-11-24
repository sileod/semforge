import random

from experiments.cue_reliability import arm_rows, negated


def row(gold, neg, i):
    h = f"Rex is {'not ' if neg else ''}a wumpus{i}."
    return {"gold": gold, "text": f"Premises:\nX.\n\nHypothesis: {h}\n\nLabel:"}


def test_negation_reads_the_hypothesis_only():
    assert negated("Premises:\nNo cat is a dog.\n\nHypothesis: Rex is a cat.\n\nLabel:") is False
    assert negated("Premises:\nRex is a cat.\n\nHypothesis: It is not the case that Rex is a cat.\n\nLabel:") is True
    assert negated("no hypothesis here") is None


def test_arm_rows_sets_polarity_share_per_label():
    pool = [row(g, n, i) for g in ("entailment", "contradiction", "neutral") for n in (True, False) for i in range(400)]
    rs = arm_rows(pool, 600, 0.9, random.Random(0))
    assert len(rs) == 600 and len({id(r) for r in rs}) == 600
    share = lambda g: sum(negated(r["text"]) for r in rs if r["gold"] == g) / 200
    assert share("contradiction") == 0.9 and share("entailment") == 0.1 and share("neutral") == 0.1
