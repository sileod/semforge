from semforge.construct.check import prove_rule
from semforge.construct.packs.boxes import NOWHERE, PACK, closure, required_events

STORY = [("in", "fish", "box0", "t0"), ("in", "shell", "box1", "t0"), ("in", "bill", NOWHERE, "t0"),
         ("move", "fish", "box0", "box1", "t0"), ("put", "bill", "box2", "t1"),
         ("contents", "box1", "box3", "t2"), ("remove", "bill", "box2", "t3")]


def test_rules_are_sound():
    assert all(prove_rule(PACK, r) == "proved" for r in PACK.calculus.rules.values())


def test_story_positions_match_z3():
    facts = [("in", "fish", "box3", "t4"), ("in", "shell", "box3", "t4"), ("in", "bill", NOWHERE, "t4"),
             ("not", ("in", "fish", "box0", "t4"))]
    have = closure(STORY, facts)
    for f in facts:
        assert PACK.calculus.canonical(f) in have and PACK.entails(STORY, f)


def test_removal_reaches_nowhere_unmentioned():
    story = [("in", "fish", "box0", "t0"), ("remove", "fish", "box0", "t0")]
    goal = ("not", ("in", "fish", "box0", "t1"))
    assert PACK.calculus.canonical(goal) in closure(story, [goal])


def test_required_events_skip_overwritten_ones():
    # shell's position needs only the contents move; fish's also needs its own move into box1
    assert required_events(STORY, [("in", "shell", "box3", "t4")]) == [("contents", "box1", "box3", "t2")]
    assert required_events(STORY, [("in", "fish", "box3", "t4")]) == [("move", "fish", "box0", "box1", "t0"),
                                                                    ("contents", "box1", "box3", "t2")]
