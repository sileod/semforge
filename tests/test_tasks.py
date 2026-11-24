import json
import random
from copy import deepcopy
from dataclasses import replace

import pytest

from semforge.certcheck import check as check_certificate
from semforge.construct.packs.propositional import PACK
from semforge.construct.synth import Instance
from semforge.tasks import Check, Example, PlanTask, ProofTask, Result, row
from semforge.tasks.sokoban import Sokoban, SokobanBoard


def mp(a, b):
    return {"kind": "modus_ponens", "head": b, "parents": [a, ["->", a, b]]}


def test_proof_solving_and_alternative_answers():
    task = ProofTask(PACK)
    problem = task.problem(["a", ("->", "a", "b"), ("->", "b", "c")], "c")
    result = task.solve(problem)
    assert result.status == "solved"
    assert task.check(problem, result.answer).valid
    assert check_certificate(result.evidence) == []
    # Both derivations must be accepted, independent of the selected reference.
    assert task.check(problem, [mp("a", "b"), mp("b", "c")]).valid
    other = [{"kind": "hypothetical_syllogism", "head": ["->", "a", "c"],
              "parents": [["->", "a", "b"], ["->", "b", "c"]]}, mp("a", "c")]
    assert task.check(problem, other).valid
    exported = json.loads(json.dumps(row(task, Example(problem, result))))
    assert exported["task"] == "proof:propositional"
    assert "label" not in exported
    assert exported["input"]["goal"] == "c"
    assert "proof" not in exported["input"]
    assert exported["verification"]["claims"] == ["valid_derivation"]


def test_proof_problem_cannot_be_redefined_by_candidate():
    task = ProofTask(PACK)
    problem = task.problem(["a", ("->", "a", "b")], "b")
    proof = [mp("a", "b")]
    assert task.check(problem, proof).valid
    assert not task.check(problem, {"premises": ["b"], "proof": []}).valid
    assert not task.check(problem, [mp("missing", "b")]).valid
    assert not task.check(problem, [dict(proof[0], kind="invented_rule")]).valid
    assert not task.check(problem, []).valid
    assert not task.check(problem, [mp("a", "?B")]).valid
    assert not task.check(problem, [dict(proof[0], head=[])]).valid
    assert not task.check(problem, [None]).valid
    assert not task.check(task.problem(["a"], "b"), proof).valid
    assert not task.check(problem, [mp("b", "c"), mp("a", "b")]).valid


def test_empty_proof_and_scoped_absence():
    task = ProofTask(PACK)
    problem = task.problem(["a"], "a")
    result = task.solve(problem)
    assert result.answer == [] and task.check(problem, []).valid
    # Semantically valid but not derivable in the default bounded calculus.
    missing = task.problem([("->", "a", "b")], ("->", ("not", "b"), ("not", "a")))
    assert PACK.entails(missing.premises, missing.goal)
    result = task.solve(missing)
    assert result.status == "no_solution" and "calculus" in result.scope
    with pytest.raises(ValueError, match="cannot export"):
        row(task, Example(missing, result))


def test_generate_proof_exercise_and_neutral_rejection():
    task = ProofTask(PACK)
    example = task.generate(random.Random(0), shape=((), ()), min_depth=2)
    assert example is not None
    assert example.result.metrics["min_depth"] >= 2
    assert check_certificate(example.result.evidence) == []
    assert task.check(example.problem, example.result.answer).valid
    neutral = Instance(list(example.problem.premises), "unrelated", "neutral", example.problem.goal,
                       example.problem.universe)
    with pytest.raises(ValueError, match="neutral"):
        task.from_instance(neutral)
    with pytest.raises(ValueError, match="shape or rule plan"):
        task.generate(random.Random(0))


class Graph:
    """A second domain: state search with cycles, no formulas or premise deletion."""
    name = "graph"

    def initial(self, p):
        return p["start"]

    def actions(self, p, state):
        return p["edges"].get(state, ())

    def transition(self, p, state, action):
        return action

    def is_goal(self, p, state):
        return state == p["goal"]

    def present(self, p):
        return p


def test_plan_search_checks_alternatives_and_separates_optimality():
    problem = {"start": "s", "goal": "g", "edges": {"s": ["a", "b"], "a": ["g"], "b": ["a", "s"]}}
    task = PlanTask(Graph())
    result = task.solve(problem)
    assert result.answer == ["a", "g"]
    assert "minimum_action_count" in result.claims
    assert task.check(problem, ["b", "a", "g"]).valid  # valid but non-optimal
    assert "minimum_action_count" not in task.check(problem, result.answer).claims
    assert not task.check(problem, ["g"]).valid
    assert not task.check(problem, ["a"]).valid
    assert not task.check(problem, "a g").valid
    exported = json.loads(json.dumps(row(task, Example(problem, result))))
    assert "minimum_action_count" in exported["solver_claims"]
    assert "minimum_action_count" not in exported["verification"]["claims"]


def test_search_limits_are_unknown_not_impossibility():
    p = {"start": "s", "goal": "g", "edges": {"s": ["a"], "a": ["g"]}}
    assert PlanTask(Graph(), max_states=1).solve(p).status == "unknown"
    assert PlanTask(Graph(), max_depth=1).solve(p).status == "unknown"
    assert PlanTask(Graph(), max_depth=2).solve(p).status == "solved"
    dead = dict(p, edges={"s": ["a"], "a": ["s"]})
    assert PlanTask(Graph()).solve(dead).status == "no_solution"
    done = dict(p, goal="s")
    assert PlanTask(Graph(), max_states=1, max_depth=0).solve(done).answer == []
    with pytest.raises(ValueError):
        PlanTask(Graph(), max_states=0)
    with pytest.raises(ValueError):
        Result("solved", "test")


def test_export_rechecks_answer_and_rejects_unknown_verification():
    p = {"start": "s", "goal": "g", "edges": {"s": ["g"]}}
    task = PlanTask(Graph())
    result = task.solve(p)
    with pytest.raises(ValueError, match="checking"):
        row(task, Example(p, replace(result, answer=["illegal"])))

    class Unchecked(PlanTask):
        def check(self, problem, answer):
            return Check(None, self.scope, errors=("checker budget exhausted",))

    with pytest.raises(ValueError, match="checking"):
        row(Unchecked(Graph()), Example(p, result))


def test_sokoban_push_plans_replay_and_are_push_optimal():
    board = SokobanBoard.parse("#######\n# . . #\n# $ $ #\n#  @  #\n#######")
    task = PlanTask(Sokoban())
    result = task.solve(board)
    assert result.status == "solved" and len(result.answer) == 2
    assert task.check(board, result.answer).valid
    assert task.check(board, list(reversed(result.answer))).valid
    exported = json.loads(json.dumps(row(task, Example(board, result))))
    assert task.check(board, exported["answer"]).valid
    bad = deepcopy(result.answer)
    bad[0]["box"] = [0, 0]
    assert not task.check(board, bad).valid
    assert PlanTask(Sokoban(), max_depth=1).solve(board).status == "unknown"
    stuck = SokobanBoard.parse("#####\n#$ .#\n# @ #\n#####")
    assert task.solve(stuck).status == "no_solution"


def test_sokoban_reachable_player_positions_are_equivalent_for_pushes():
    system = Sokoban()
    a = SokobanBoard.parse("#######\n# . . #\n# $ $ #\n#  @  #\n#######")
    b = replace(a, player=(1, 3))
    assert system.initial(a) == system.initial(b)
    for text in ("", "###\n##", "###\n#@#\n###", "#####\n#@$?#\n#####"):
        with pytest.raises(ValueError):
            SokobanBoard.parse(text)


def test_task_cli_jsonl_and_bounded_failure(capsys, tmp_path):
    from semforge.tasks.__main__ import main
    assert main(["proof", "--steps", "3", "--count", "2"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 2 and all(check_certificate(r["evidence"]) == [] for r in rows)
    board = tmp_path / "board.txt"
    board.write_text("#####\n# . #\n# $ #\n# @ #\n#####")
    assert main(["sokoban", str(board)]) == 0
    assert len(json.loads(capsys.readouterr().out)["answer"]) == 1
    assert main(["sokoban", str(board), "--max-states", "1"]) == 1
    assert json.loads(capsys.readouterr().out)["result"]["status"] == "unknown"


def test_question_does_not_list_the_proof_and_double_negations_are_in_the_universe():
    task = ProofTask(PACK)
    # modus tollens on (not p -> q), not q gives not not p: the universe admits the double negation of a
    # negated subformula, so the bounded check accepts the proof
    problem = task.problem([("->", ("not", "p"), "q"), ("not", "q"), ("->", "p", "r")], "r")
    proof = [{"kind": "modus_tollens", "head": ["not", ["not", "p"]],
              "parents": [["->", ["not", "p"], "q"], ["not", "q"]]},
             {"kind": "double_negation", "head": "p", "parents": [["not", ["not", "p"]]]}, mp("p", "r")]
    from semforge.certcheck import Checker
    from semforge.construct.atlas import certificate_fields
    bounded = {**certificate_fields(PACK, problem.premises, problem.goal, problem.universe), "proof": proof}
    assert not Checker(bounded).proof(proof)[0]
    assert task.check(problem, proof).valid, task.check(problem, proof).errors
    assert "universe" not in task.present(problem)


def test_exported_proof_text_ends_in_the_conclusion_and_covers_mandatory_steps():
    """Proof supervision text (experiments/llm_depth_training): one line per witness step, in the row's lexicon."""
    import random
    import re
    from experiments.llm_common import proof_text
    from semforge.construct.calculus import analyze
    from semforge.domains.constructed import ConstructedDomain
    d, rng, seen = ConstructedDomain("monadic", strategy="shape"), random.Random(0), 0
    while seen < 5:
        for inst, label in d.sample(rng, 3):
            if label == "neutral":
                continue
            seen += 1
            lines = proof_text(d, inst).split("\n")
            a = analyze(d.pack.calculus, inst.premises, inst.conclusion, inst.universe)
            assert len(lines) >= len(a.mandatory_steps)
            premises, hyp = d.render(inst)
            last = re.sub(r"^\d+\. | \([a-z ]+\)\.$", "", lines[-1])
            if label == "entailment":  # a universal has two wordings: "every A is a B", "all As are Bs"
                same = lambda t: re.sub(r"^all (\w+)es are (\w+)es$", r"every \1 is a \2",
                                        t.rstrip(". ").lower()).replace(" an ", " a ")
                assert same(last) == same(hyp) or last in hyp, (last, hyp)
            names = set(re.findall(r"\b[A-Z][a-z]+\b", premises + " " + hyp))
            assert set(re.findall(r"\b[A-Z][a-z]+\b", " ".join(lines))) - {"Both", "Either", "At", "No", "Every", "All", "It", "Some", "Not", "Nothing", "Something"} <= names


def test_kripke_closed_model_proves_boxes_and_agrees_with_evaluation():
    from semforge.tasks.kripke import FRAMES, KripkeProblem, KripkeTask, close, evaluate, reduce
    assert reduce(("box", ("box", ("box", "p"))), "S4") == ("box", "p")
    assert reduce(("dia", ("box", "p")), "S5") == ("box", "p") and reduce(("box", ("box", "p")), "K") == ("box", ("box", "p"))
    task = KripkeTask()
    # u sees v and x only; p holds at both, so box p holds at u: provable only because the model is closed
    p = KripkeProblem(("u", "v", "x"), frozenset({("u", "v"), ("u", "x")}), frozenset({("p", "v"), ("p", "x")}),
                      "u", ("box", "p"))
    assert task.solve(p).answer is True and task.check(p, True).valid and not task.check(p, False).valid
    assert task.solve(replace(p, formula=("dia", ("not", "p")))).answer is False
    assert task.solve(replace(p, world="v", formula=("box", ("not", "p")))).answer is True  # no successors
    assert task.check(p, "yes").valid is False
    assert row(task, Example(p, task.solve(p)))["answer"] is True
    assert close({("a", "b"), ("b", "c")}, ("a", "b", "c"), "S4") >= {("a", "a"), ("a", "c")}
    rng = random.Random(0)
    for frame in FRAMES:
        for depth in (1, 2, 3):
            q = task.generate(rng, worlds=rng.randint(2, 5), depth=depth, frame=frame)
            result = task.solve(q)
            assert result.metrics["modal_depth"] == depth and task.check(q, result.answer).valid
            assert evaluate(q, reduce(q.formula, frame), q.world) == result.answer
            assert result.metrics["reduced_depth"] <= depth
    r = task.row(p, random.Random(0))
    assert r["gold"] == "entailment" and "all the places" in r["premises"]


def test_kripke_hidden_facts_leave_the_formula_undecided_unless_every_completion_agrees():
    from semforge.tasks.kripke import FRAMES, UNDECIDED, KripkeProblem, KripkeTask
    task = KripkeTask()
    # u sees v and x; p holds at v, and whether p holds at x is not stated
    p = KripkeProblem(("u", "v", "x"), frozenset({("u", "v"), ("u", "x")}), frozenset({("p", "v"), ("p", "x")}),
                      "u", ("box", "p"), hidden=frozenset({("p", "x")}))
    assert task.solve(p).answer == UNDECIDED and task.check(p, UNDECIDED).valid and not task.check(p, True).valid
    assert task.solve(replace(p, formula=("dia", "p"))).answer is True  # v alone decides it
    assert task.row(p, random.Random(0))["gold"] == "neutral"
    assert task.check(replace(p, hidden=frozenset()), UNDECIDED).valid is False  # closed: never undecided
    rng, golds = random.Random(1), set()
    for _ in range(40):
        q = task.generate(rng, worlds=rng.randint(2, 4), depth=rng.randint(0, 3), frame=rng.choice(FRAMES), hidden=0.3)
        golds.add(task.row(q, rng)["gold"])  # raises unless z3 and the completion check agree
    assert golds == {"entailment", "contradiction", "neutral"}


def test_closed_domain_does_not_depend_on_the_names_a_query_mentions():
    # review5 probe: "everyone is a" over the closed domain {a, b}; a tautology naming b must not change it
    import z3
    from semforge.construct.oracle import domain_closure, z3_entails
    from semforge.construct.signature import Signature, Sort, Symbol

    def sole(m, a):
        x = m.var("Person")
        return z3.ForAll([x], x == a)

    sig = Signature.of(sorts=[Sort("Person", ("Ann", "Bob"))], symbols=[Symbol("sole", ("Person",), z3=sole)],
                       axioms=(domain_closure("Person", ("a", "b")),))
    entails, h = z3_entails(sig), ("sole", "a")
    for premises in ([], [("or", ("sole", "b"), ("not", ("sole", "b")))]):
        assert not entails(premises, h) and entails(premises, ("not", h))
