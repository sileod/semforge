"""Generate proof-task JSONL or solve a small Sokoban board.

python -m semforge.tasks proof --pack monadic --steps 6 --count 10
python -m semforge.tasks sokoban board.txt --max-states 100000
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

from . import Example, PlanTask, ProofTask, row
from .sokoban import Sokoban, SokobanBoard


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="task", required=True)
    proof = sub.add_parser("proof", help="generate exercises whose answers are checked formal proofs")
    proof.add_argument("--pack", default="propositional")
    proof.add_argument("--steps", type=int, default=4)
    proof.add_argument("--count", type=int, default=1)
    proof.add_argument("--seed", type=int, default=0)
    proof.add_argument("--max-attempts", type=int, default=100)
    plan = sub.add_parser("sokoban", help="find a shortest push plan for a small ASCII board")
    plan.add_argument("board", type=Path)
    plan.add_argument("--max-states", type=int, default=100_000)
    plan.add_argument("--max-depth", type=int, default=None)
    args = ap.parse_args(argv)
    if args.task == "proof":
        from ..construct.build import random_skeleton
        from ..construct.packs import get
        if min(args.steps, args.count, args.max_attempts) < 1:
            ap.error("steps, count and max-attempts must be positive")
        task, rng, made = ProofTask(get(args.pack)), random.Random(args.seed), 0
        for attempt in range(args.max_attempts):
            example = task.generate(rng, shape=random_skeleton(rng, args.steps))
            if example is not None:
                print(json.dumps({**row(task, example), "generation": {"seed": args.seed, "attempt": attempt}}))
                made += 1
                if made == args.count:
                    return 0
        print(f"constructed {made}/{args.count} proof tasks within {args.max_attempts} requests", file=sys.stderr)
        return 1
    if args.max_states < 1 or (args.max_depth is not None and args.max_depth < 0):
        ap.error("max-states must be positive and max-depth nonnegative")
    problem = SokobanBoard.parse(args.board.read_text())
    task = PlanTask(Sokoban(), max_states=args.max_states, max_depth=args.max_depth)
    result = task.solve(problem)
    if result.status == "solved":
        print(json.dumps(row(task, Example(problem, result))))
        return 0
    print(json.dumps({"task": task.name, "input": task.present(problem), "result": asdict(result)}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
