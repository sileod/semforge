"""Cue reliability as a positive control for the knobs study: does this training setup (Qwen3-0.6B-Base, 3,000
rows) respond to a data knob whose effect is known from the literature (Lovering et al. 2021)?

The cue is hypothesis polarity: whether the hypothesis contains a negation word. In the flagship mixture pool
(`train_pmdk200x3`) contradictions are a little more often negated than entailments and neutrals (56% vs 46%
and 40%), and every label occurs with both polarities, so selection alone sets how well polarity predicts the
label. Arms (--n rows, a third per label, drawn from the same pool, shuffled):

  flat    every label 50% negated: polarity carries no information about the label
  cue70   contradictions 70% negated, entailments and neutrals 30%
  cue90   contradictions 90% negated, entailments and neutrals 10%

Pre-registered contrasts: cue90 - flat and cue70 - flat, two-way accuracy on *anti-cue* rows (negated
entailments, non-negated contradictions) of every test, and the gap between cue-aligned and anti-cue rows. A
setup that responds to data knobs should show cue90 below flat on anti-cue rows. If it does not, the null
results of the decisiveness knobs say little about decisiveness.

  python -m experiments.cue_reliability build
  DATA=results/llm_mix/data_cue OUT=results/llm_mix/cue bash scripts/queue_packmix.sh none flat:10 ...
  python -m experiments.cue_reliability analyze
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import shutil

import numpy as np

from experiments.llm_common import load_runs, seed_contrast
from experiments.external_rows import write
from experiments.llm_common import LABELS

ARMS = {"flat": 0.5, "cue70": 0.7, "cue90": 0.9}  # share of contradictions negated; E and N get 1 - share
CONTRASTS = [("cue90", "flat"), ("cue70", "flat"), ("cue90", "cue70")]
NEGATION = re.compile(r"\b(not|no|none|never|nobody|nothing|neither|nor)\b|n't", re.I)


def hypothesis(text):
    return text.split("Hypothesis:", 1)[1].split("\n", 1)[0].strip() if "Hypothesis:" in text else None


def negated(text):
    h = hypothesis(text)
    return None if h is None else bool(NEGATION.search(h))


def arm_rows(pool, n, share, rng):
    """n rows, a third per label; contradictions negated with probability `share`, the others `1 - share`."""
    cells = collections.defaultdict(list)
    for r in pool:
        cells[r["gold"], negated(r["text"])].append(r)
    out = []
    for lab in LABELS:
        k = round(n / 3 * (share if lab == "contradiction" else 1 - share))
        for neg, want in ((True, k), (False, n // 3 - k)):
            have = cells[lab, neg]
            assert len(have) >= want, f"pool has {len(have)} {lab} rows with negated={neg}, arm needs {want}"
            out += rng.sample(have, want)
    rng.shuffle(out)
    return out


def analyze(args):
    """Two-way accuracy on cue-aligned and anti-cue E/C rows of every test, per arm, mean over seeds, with
    seed-bootstrap contrasts. Aligned: negated contradictions and non-negated entailments."""
    runs = load_runs(args.runs)
    contrast = seed_contrast(np.random.default_rng(0))
    score = collections.defaultdict(lambda: collections.defaultdict(dict))  # test -> kind -> arm -> [seeds]
    for t in sorted({t for _, t in runs}):
        path = f"{args.data}/test_{t}.jsonl"
        if not os.path.exists(path):
            continue
        meta = [json.loads(line) for line in open(path)]
        for (a, test), seeds in runs.items():
            if test != t:
                continue
            for kind in ("aligned", "anti"):
                per = []
                for rows in seeds.values():
                    acc = []
                    for k, lab in enumerate(LABELS[:2]):
                        hits = [np.argmax(p["logp"][:2]) == k for r, p in zip(meta, rows)
                                if r["gold"] == lab and negated(r["text"]) is not None
                                and negated(r["text"]) == ((lab == "contradiction") == (kind == "aligned"))]
                        if hits:
                            acc.append(np.mean(hits))
                    per.append(np.mean(acc) if len(acc) == 2 else np.nan)
                score[t][kind][a] = per
    arms = sorted(ARMS)
    for t in score:
        print(f"\n{t}: two-way accuracy, aligned / anti-cue (mean over seeds)")
        for a in arms:
            if a in score[t]["anti"]:
                al, an = (np.nanmean(score[t][k][a]) for k in ("aligned", "anti"))
                print(f"  {a:8s} {al:.3f} / {an:.3f}   gap {al - an:+.3f}   ({len(score[t]['anti'][a])} seeds)")
        for a, b in CONTRASTS:
            if a in score[t]["anti"] and b in score[t]["anti"]:
                contrast(f"  {a} - {b}  anti-cue", score[t]["anti"][a], score[t]["anti"][b])
                gap = lambda x: np.array(score[t]["aligned"][x]) - np.array(score[t]["anti"][x])
                contrast(f"  {a} - {b}  gap", gap(a), gap(b))


def audit(args):
    """For any runs directory: per test, how often polarity alone gives the label (negated -> contradiction),
    and each arm's two-way accuracy on cue-aligned vs anti-cue E/C rows (mean over seeds). Anti-cue accuracy
    is the share of a test score that a polarity heuristic cannot explain."""
    runs = load_runs(args.runs)
    arms = sorted({a for a, _ in runs})
    if args.arms:
        arms = [a for a in arms if a in args.arms.split(",")]
    for t in sorted({t for _, t in runs}):
        path = f"{args.data}/test_{t}.jsonl"
        if not os.path.exists(path):
            continue
        meta = [json.loads(line) for line in open(path)]
        ec = [(i, r["gold"] == "contradiction", negated(r["text"])) for i, r in enumerate(meta)
              if r["gold"] in LABELS[:2] and negated(r["text"]) is not None]
        cue = np.mean([np.mean([c == g for _, c, g in ec if c == lab]) for lab in (False, True)])
        n_anti = sum(c != g for _, c, g in ec)
        print(f"\n{t}: polarity-only two-way accuracy {cue:.3f}; anti-cue E/C rows {n_anti} of {len(ec)}")
        for a in arms:
            seeds = runs.get((a, t))
            if not seeds:
                continue
            res = {}
            for kind in ("aligned", "anti"):
                per = []
                for rows in seeds.values():
                    acc = [np.mean([np.argmax(rows[i]["logp"][:2]) == int(c) for i, c, g in ec
                                    if c == lab and (g == c) == (kind == "aligned")] or [np.nan])
                           for lab in (False, True)]
                    per.append(np.mean(acc))
                res[kind] = np.nanmean(per)
            print(f"  {a:14s} aligned {res['aligned']:.3f}  anti {res['anti']:.3f}  ({len(seeds)} seeds)")
        contrast = seed_contrast(np.random.default_rng(0))
        for pair in filter(None, args.pairs.split(",")):
            a, b = pair.split(":")
            if (a, t) not in runs or (b, t) not in runs:
                continue
            for kind in ("aligned", "anti"):
                rows_k = [(i, c) for i, c, g in ec if (g == c) == (kind == "aligned")]
                if not rows_k:
                    continue
                acc = lambda arm: [np.mean([np.argmax(rows[i]["logp"][:2]) == int(c) for i, c in rows_k])
                                   for rows in runs[(arm, t)].values()]
                contrast(f"  {a} - {b}  {kind} ({len(rows_k)} rows, pooled)", acc(a), acc(b))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pool", default="results/llm_mix/data_packmix/train_pmdk200x3.jsonl")
    b.add_argument("--tests-from", default="results/llm_mix/data_packmix")
    b.add_argument("--out", default="results/llm_mix/data_cue")
    b.add_argument("--n", type=int, default=3000, help="rows per arm (a multiple of 3)")
    b.add_argument("--seed", type=int, default=97)
    z = sub.add_parser("analyze")
    z.add_argument("--runs", default="results/llm_mix/cue")
    z.add_argument("--data", default="results/llm_mix/data_cue")
    u = sub.add_parser("audit")
    u.add_argument("--runs", default="results/llm_mix/external")
    u.add_argument("--data", default="results/llm_mix/data_packmix")
    u.add_argument("--arms", default="", help="comma-separated subset of arms")
    u.add_argument("--pairs", default="", help="comma-separated a:b contrasts on aligned and anti-cue rows")
    args = ap.parse_args()
    if args.cmd == "analyze":
        return analyze(args)
    if args.cmd == "audit":
        return audit(args)
    os.makedirs(args.out, exist_ok=True)
    pool = [json.loads(line) for line in open(args.pool)]
    pool = [r for r in pool if negated(r["text"]) is not None]
    stats = {"pool": args.pool, "pool_rows": len(pool),
             "pool_negated_by_label": {lab: round(np.mean([negated(r["text"]) for r in pool if r["gold"] == lab]), 3)
                                       for lab in LABELS}}
    for arm, share in ARMS.items():
        rs = arm_rows(pool, args.n, share, random.Random(args.seed))
        write(f"{args.out}/train_{arm}.jsonl", rs)
        mean = lambda xs: round(float(np.mean([x for x in xs if x is not None])), 2)
        stats[arm] = {"rows": len(rs), "labels": collections.Counter(r["gold"] for r in rs),
                      "negated_by_label": {lab: mean([negated(r["text"]) for r in rs if r["gold"] == lab]) for lab in LABELS},
                      "mean_mandatory": mean([r.get("mandatory") for r in rs]), "mean_premises": mean([r.get("premises") for r in rs]),
                      "mean_chars": round(float(np.mean([len(r["text"]) for r in rs])))}
    for f in os.listdir(args.tests_from):
        if f.startswith(("test_", "val_")):
            shutil.copy(f"{args.tests_from}/{f}", args.out)
    print(json.dumps(stats, indent=1))
    json.dump(stats, open(f"{args.out}/stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
