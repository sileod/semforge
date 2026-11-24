from experiments.llm_fld_wording import mcnemar, reword


def test_reword_makes_the_opening_indefinite_universal():
    assert reword("Something that adsorbs Tiflis is red.") == "Everything that adsorbs Tiflis is red."
    assert reword("Some person is a Bali.") == "Every person is a Bali."
    assert reword("Somebody does not traipse if it is wet.") == "Everybody does not traipse if it is wet."
    assert reword("If someone is baric then it is fatal.") == "For everyone, if it is baric then it is fatal."


def test_reword_leaves_other_openings():
    for s in ("Everything is red.", "That scanty is audile if some man is red.", "If something that is red is X then Y."):
        assert reword(s) is None


def test_mcnemar_exact():
    assert mcnemar(0, 0) == 1.0
    assert abs(mcnemar(0, 10) - 2 / 2 ** 10) < 1e-12
