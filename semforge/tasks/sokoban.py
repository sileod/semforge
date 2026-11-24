"""Small Sokoban transition backend exercising plan tasks without a proof calculus.

Actions are pushes, not walking steps. Player positions in the same reachable
region are equivalent for this objective. The search is intended for small boards;
there is no deadlock atlas, heuristic search or level generator here yet.
"""
from __future__ import annotations

from dataclasses import dataclass

DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


@dataclass(frozen=True)
class SokobanBoard:
    rows: tuple[str, ...]
    floor: frozenset
    goals: frozenset
    boxes: frozenset
    player: tuple[int, int]

    @classmethod
    def parse(cls, text):
        rows = tuple(text.strip("\n").splitlines())
        if not rows or not rows[0] or len({len(r) for r in rows}) != 1:
            raise ValueError("a board must be a nonempty rectangle")
        if any(c not in "# .@$+*" for r in rows for c in r):
            raise ValueError("unknown Sokoban tile")
        positions = lambda chars: frozenset((x, y) for y, r in enumerate(rows)
                                           for x, c in enumerate(r) if c in chars)
        players, boxes, goals = positions("@+"), positions("$*"), positions(".*+")
        if len(players) != 1 or not boxes or len(boxes) != len(goals):
            raise ValueError("need one player and the same positive number of boxes and goals")
        return cls(rows, positions(" .@$+*"), goals, boxes, next(iter(players)))


def reachable(board, boxes, player):
    seen, stack = {player}, [player]
    while stack:
        x, y = stack.pop()
        for dx, dy in DIRECTIONS.values():
            p = x + dx, y + dy
            if p in board.floor and p not in boxes and p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


class Sokoban:
    name = "sokoban_pushes"

    def initial(self, problem):
        return problem.boxes, min(reachable(problem, problem.boxes, problem.player))

    def actions(self, problem, state):
        boxes, player = state
        area = reachable(problem, boxes, player)
        for x, y in sorted(boxes):
            for direction, (dx, dy) in DIRECTIONS.items():
                dest, behind = (x + dx, y + dy), (x - dx, y - dy)
                if behind in area and dest in problem.floor and dest not in boxes:
                    yield {"box": [x, y], "direction": direction}

    def transition(self, problem, state, action):
        boxes, _ = state
        x, y = action["box"]
        dx, dy = DIRECTIONS[action["direction"]]
        boxes = (boxes - {(x, y)}) | {(x + dx, y + dy)}
        return boxes, min(reachable(problem, boxes, (x, y)))

    def is_goal(self, problem, state):
        return state[0] == problem.goals

    def present(self, problem):
        return {"board": list(problem.rows),
                "legend": {"#": "wall", " ": "floor", ".": "goal", "$": "box", "@": "player",
                           "*": "box on goal", "+": "player on goal"},
                "rules": "Walk on unoccupied floor. Push one adjacent box into an unoccupied floor cell; "
                         "never pull. Put every box on a goal. Walking between pushes is implicit.",
                "action_format": {"box": "[x, y], zero-based from the top-left", "direction": list(DIRECTIONS)},
                "cost": "one per push; walking is free"}
