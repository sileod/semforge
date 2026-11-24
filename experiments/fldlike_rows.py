"""A simplified FLD in our own packs: the baseline with neither certification nor distribution control.

FLD×2's recipe is a random proof tree of 1-n steps plus distractor derivations, labelled by construction,
with unknowns from a removed premise. semforge.construct.forward is that recipe in any pack (1-20 planned
steps, n/2 distractor derivations), and llm_common.shard(strategy="forward_neutral") gives its neutrals
(one mandatory premise dropped). Labels are still the oracle's, so the baseline is not label noise: what
it lacks is any selection on certified structure. Earlier forward arms ("forward rows
with forward-style neutrals") were filtered into certified depth buckets, so they still had the
distribution control; these rows are taken as the generator makes them.

  fldlike    monadic and modal forward rows, half each, a third per label
  fldlike_k  train_rmsel (the distribution search's choice) with its monadic and modal rows replaced by
             forward rows of the same pack and label, its Kripke rows kept: certification and the searched
             depth mixture removed from the derivation rows only

  python -m experiments.fldlike_rows --out results/llm_mix/data_packmix
"""
from __future__ import annotations

import argparse
import collections
import json
import random

from experiments.external_rows import write
from experiments.llm_common import LABELS, generate

PACKS = {"m": "monadic", "d": "modal"}


class AnyRow:
    """Every row with a label in `labels`, certified or not."""
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, r):
        return r["gold"] in self.labels


def pool(pack, per_label, seed):
    ec = generate(pack, "forward", seed, 2 * per_label, AnyRow(("entailment", "contradiction")))
    n = generate(pack, "forward_neutral", seed + 1, per_label, AnyRow(("neutral",)))
    by = collections.defaultdict(list)
    for r in ec + n:
        by[r["gold"]].append({**r, "cell": f"fwd_{pack}"})
    return by


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/llm_mix/data_packmix")
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    rng = random.Random(0)
    sel = [json.loads(line) for line in open(f"{args.out}/train_rmsel.jsonl")]
    need = collections.Counter((r["cell"][0], r["gold"]) for r in sel if r["cell"][0] in PACKS)
    per = {p: max(args.n // 6, *(need[p, lab] for lab in LABELS)) + 50 for p in PACKS}
    pools = {p: pool(PACKS[p], per[p], 30 + i) for i, p in enumerate(PACKS)}
    for p, by in pools.items():
        for lab in LABELS:
            rng.shuffle(by[lab])
        print(p, {lab: len(by[lab]) for lab in LABELS})
    plain = [r for p in PACKS for lab in LABELS for r in pools[p][lab][:args.n // 6]]
    rng.shuffle(plain)
    used = {(p, lab): args.n // 6 for p in PACKS for lab in LABELS}
    swapped = []
    for r in sel:  # fresh rows for fldlike_k where the pool allows, reusing the start of the pool otherwise
        p = r["cell"][0]
        if p not in PACKS:
            swapped.append(r)
            continue
        cands = pools[p][r["gold"]]
        swapped.append(cands[used[p, r["gold"]] % len(cands)])
        used[p, r["gold"]] += 1
    rng.shuffle(swapped)
    stats = {}
    for name, rows in (("fldlike", plain), ("fldlike_k", swapped)):
        write(f"{args.out}/train_{name}.jsonl", rows)
        m = [r["mandatory"] for r in rows if r.get("mandatory") is not None and r["cell"].startswith("fwd")]
        stats[name] = {"rows": len(rows), "labels": collections.Counter(r["gold"] for r in rows),
                       "cells": collections.Counter(r["cell"] for r in rows),
                       "mean_mandatory_fwd": round(sum(m) / max(len(m), 1), 2),
                       "uncertified_fwd": sum(r.get("mandatory") is None for r in rows if r["cell"].startswith("fwd")),
                       "mean_chars": round(sum(len(r["text"]) for r in rows) / len(rows))}
    print(json.dumps(stats, indent=1))
    json.dump(stats, open(f"{args.out}/fldlike_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
