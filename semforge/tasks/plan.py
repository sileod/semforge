"""Finite-state plan search, separate from deduction and NLI.

A system supplies the initial state, legal actions, deterministic transitions and
goal predicate. States must be immutable and hashable; actions and public inputs
must be JSON-ready for row export. Actions have unit cost and each action set is
finite. Chess strategies require an adversarial backend, not this existential
path search. No plan optimality claim is made by the replay checker.
"""
from __future__ import annotations

from collections import deque
from typing import Protocol

from .base import Check, Result


class TransitionSystem(Protocol):
    name: str

    def initial(self, problem): ...
    def actions(self, problem, state): ...
    def transition(self, problem, state, action): ...
    def is_goal(self, problem, state) -> bool: ...
    def present(self, problem) -> dict: ...


class PlanTask:
    def __init__(self, system: TransitionSystem, *, max_states=100_000, max_depth=None):
        if max_states < 1 or (max_depth is not None and max_depth < 0):
            raise ValueError("max_states must be positive and max_depth nonnegative")
        self.system, self.max_states, self.max_depth = system, max_states, max_depth
        self.name = f"plan:{system.name}"
        self.scope = f"finite legal-action paths in {system.name}; deterministic transitions, unit action cost"

    def present(self, problem):
        return {"instruction": "Find a legal sequence of actions reaching the goal. Return a JSON list of actions.",
                "system": self.system.name, "problem": self.system.present(problem)}

    def check(self, problem, answer):
        if not isinstance(answer, (list, tuple)):
            return Check(False, self.scope, errors=("a plan must be a list of actions",))
        s, state = self.system, self.system.initial(problem)
        for i, action in enumerate(answer):
            if action not in s.actions(problem, state):
                return Check(False, self.scope, errors=(f"action {i} is not legal in its state",))
            state = s.transition(problem, state, action)
        if not s.is_goal(problem, state):
            return Check(False, self.scope, errors=("plan does not reach the goal",))
        return Check(True, self.scope, ("legal_trace", "goal_reached"))

    def solve(self, problem):
        s, start = self.system, self.system.initial(problem)
        queue, parent, depths = deque([start]), {start: None}, {start: 0}
        truncated = False
        while queue:
            state = queue.popleft()
            if s.is_goal(problem, state):
                answer, at = [], state
                while parent[at] is not None:
                    at, action = parent[at]
                    answer.append(action)
                answer.reverse()
                checked = self.check(problem, answer)
                if not checked.valid:
                    raise RuntimeError(f"search produced an invalid plan: {checked.errors}")
                return Result("solved", self.scope, answer,
                              claims=("legal_trace", "goal_reached", "minimum_action_count"),
                              metrics={"plan_length": len(answer), "states_seen": len(parent)},
                              evidence={"method": "breadth_first_search", "max_states": self.max_states,
                                        "max_depth": self.max_depth})
            if self.max_depth is not None and depths[state] >= self.max_depth:
                truncated = True
                continue
            for action in s.actions(problem, state):
                nxt = s.transition(problem, state, action)
                if nxt in parent:
                    continue
                if len(parent) >= self.max_states:
                    return Result("unknown", self.scope, metrics={"states_seen": len(parent)},
                                  reason="state budget exhausted")
                parent[nxt], depths[nxt] = (state, action), depths[state] + 1
                queue.append(nxt)
        if truncated:
            return Result("unknown", self.scope, metrics={"states_seen": len(parent)},
                          reason="depth bound reached; no global impossibility claim")
        return Result("no_solution", self.scope, claims=("goal_unreachable",),
                      metrics={"states_seen": len(parent)}, reason="reachable state space exhausted")
