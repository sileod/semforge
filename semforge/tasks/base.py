"""Tasks return answers, not necessarily NLI labels.

Solvers propose answers and describe their guarantees; checkers validate answers
against the supplied problem. A checker need not establish every solver claim
(e.g. replay proves a plan legal, not shortest). No premise, calculus or renderer
is required by this contract. Task-specific inputs and answers must be JSON-ready
when exported through `row`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Generic, Literal, Optional, Protocol, TypeVar

P = TypeVar("P")
A = TypeVar("A")


@dataclass(frozen=True)
class Check:
    valid: Optional[bool]  # None: could not check, not a rejection or an acceptance
    scope: str
    claims: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class Result(Generic[A]):
    status: Literal["solved", "no_solution", "unknown"]
    scope: str
    answer: Optional[A] = None
    claims: tuple[str, ...] = ()
    metrics: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    reason: Optional[str] = None

    def __post_init__(self):
        if self.status not in ("solved", "no_solution", "unknown"):
            raise ValueError(f"unknown result status: {self.status}")
        if (self.status == "solved") != (self.answer is not None):
            raise ValueError("only solved results carry an answer (an empty proof/plan is an answer)")


class Task(Protocol[P, A]):
    name: str

    def present(self, problem: P) -> dict:
        """Public question, including its rules/assumptions, without the answer."""
        ...

    def solve(self, problem: P) -> Result[A]: ...

    def check(self, problem: P, answer: A) -> Check:
        """Judge any proposed answer, rather than compare it with a reference."""
        ...


@dataclass(frozen=True)
class Example(Generic[P, A]):
    problem: P
    result: Result[A]


def row(task: Task, example: Example) -> dict:
    """Export a supervised row only after checking its answer.

    Solver claims and checked claims are deliberately separate. Unknown and
    no-solution results remain available from `solve`, but are not answer rows.
    Existing NLI rows/pipelines retain their original format.
    """
    result = example.result
    if result.status != "solved":
        raise ValueError(f"cannot export an answer for {result.status}: {result.reason}")
    checked = task.check(example.problem, result.answer)
    if checked.valid is not True:
        raise ValueError(f"answer did not pass checking: {checked.errors}")
    return {"task": task.name, "input": task.present(example.problem), "answer": result.answer,
            "verification": asdict(checked), "solver_claims": list(result.claims),
            "solver_scope": result.scope, "metrics": result.metrics, "evidence": result.evidence}
