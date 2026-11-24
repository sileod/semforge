"""Kripke model-checking rows (semforge.tasks.kripke) for the LLM experiments: does training on a generic,
closed-model modal task transfer to ModalBench, and does modal depth in training matter?

Arms of `--n` rows each, balanced true/false, worlds 2-7, frames K/T/S4/S5 uniformly, 2-4 atoms:
  kshallow  modal depth 0-1      kdeep  modal depth 3      kmix  modal depth 0-3 uniformly
  kopen     modal depth 0-3, partially observed (each fact hidden with probability 0.3): E/C/N in thirds
Rows longer than `--max-chars` are redrawn, which favours smaller and sparser models. `val_kripke` holds
out `--val` rows per depth 0-3 (fresh seed). Lexicon and phrasing are the pack's (places, people's facts,
"reachable"), not ModalBench's (rooms, agents, "sees"). Test files of an earlier
build are copied with `--tests-from` (e.g. test_modalbench, test_folio, test_fld).

  python -m experiments.kripke_rows --out results/llm_mix/data_kripke --tests-from results/llm_mix/data
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import shutil

from experiments.llm_common import FORMAT
from semforge.tasks.kripke import FRAMES, KripkeTask

ARMS = {"kshallow": ((0, 1), 0.0), "kdeep": ((3,), 0.0), "kmix": ((0, 1, 2, 3), 0.0), "kopen": ((0, 1, 2, 3), 0.3)}


def rows(task, rng, depths, n, max_chars, hidden=0.0):
    """n rows, balanced over the labels (E/C, or E/C/N when facts are hidden), depth drawn uniformly from
    `depths`; longer texts are redrawn."""
    labels = ("entailment", "contradiction", "neutral") if hidden else ("entailment", "contradiction")
    want, out = {g: n // len(labels) + (i < n % len(labels)) for i, g in enumerate(labels)}, []
    while any(want.values()):
        p = task.generate(rng, worlds=rng.randint(2, 7), depth=rng.choice(depths), atoms=rng.randint(2, 4),
                          frame=rng.choice(FRAMES), hidden=hidden)
        r = task.row(p, rng)
        text = FORMAT.format(premises=r["premises"], hypothesis=r["hypothesis"])
        if want.get(r["gold"]) and len(text) <= max_chars:
            want[r["gold"]] -= 1
            out.append({**{k: v for k, v in r.items() if k not in ("premises", "hypothesis")}, "mandatory": None,
                        "text": text})
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/llm_mix/data_kripke")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--val", type=int, default=150, help="rows per modal depth in val_kripke")
    ap.add_argument("--max-chars", type=int, default=1400,
                    help="redraw longer rows (dense S5 models list many links), as long as the modal pack's")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tests-from")
    ap.add_argument("--arms", nargs="*", help="build only these arms (default: all, and val_kripke)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    task, stats = KripkeTask(), {}
    files = {f"train_{arm}": rows(task, random.Random(f"{args.seed}:{arm}"), d, args.n, args.max_chars, h)
             for arm, (d, h) in ARMS.items() if not args.arms or arm in args.arms}
    if not args.arms:
        files["val_kripke"] = [r for d in range(4) for r in
                               rows(task, random.Random(f"{args.seed}:val:{d}"), (d,), args.val, args.max_chars)]
    for name, rs in files.items():
        with open(f"{args.out}/{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)
        stats[name] = {"rows": len(rs), "labels": collections.Counter(r["gold"] for r in rs),
                       "frames": collections.Counter(r["frame"] for r in rs),
                       "modal_depth": collections.Counter(r["modal_depth"] for r in rs),
                       "mean_worlds": round(sum(r["worlds"] for r in rs) / len(rs), 2),
                       "mean_chars": round(sum(len(r["text"]) for r in rs) / len(rs))}
    for f in glob.glob(f"{args.tests_from}/test_*.jsonl") if args.tests_from else ():
        shutil.copy(f, args.out)
    old = json.load(open(f"{args.out}/stats.json")) if args.arms and os.path.exists(f"{args.out}/stats.json") else {}
    json.dump(old | {"args": vars(args), **stats}, open(f"{args.out}/stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
