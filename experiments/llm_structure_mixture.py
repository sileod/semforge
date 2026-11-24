"""Which mixture of certified proof structures trains best? A small, interpretable search, then confirmation.

The structure variable is certified depth (mandatory steps), in three buckets. Each bucket's rows are
matched to the others on label and premise count (and its neutrals to its own depth), so any mixture of
buckets has the same label mix and premise-count histogram; only the depth distribution changes.
Mixtures are a grid on the simplex (pure buckets, pairs, uniform).

Discovery and confirmation are separated:
  discovery     every mixture, 2 seeds, scored on validation sets only (val_*: in-pack rows of another seed,
                FLD x2 validation rows, propositional rows of another seed);
  confirmation  the selected mixture and the baselines on fresh seeds, scored on the untouched test sets
                (test_*: the same tests as experiments/llm_depth_training).
Training and scoring are experiments.llm_depth_training.train (it scores every test_*.jsonl and val_*.jsonl
in the data directory).

  python -m experiments.llm_structure_mixture build --pack monadic --out results/llm_mix/data
  python -m experiments.llm_structure_mixture select --runs results/llm_mix/runs
  python -m experiments.llm_structure_mixture forward --data results/llm_mix/data   # forward + certificate filter
  python -m experiments.llm_structure_mixture distract --out results/llm_mix/data_distract  # distractor axis

Forward + certificate filter is the selection baseline: FLD-style forward rows, certified after the fact,
filtered into the same buckets. Each mixture's E/C rows are replaced one for one by forward rows of the
same bucket and label with the nearest premise count (neutrals unchanged). Forward depth grows with
premise count, so the deep bucket cannot be matched at low premise counts; the achieved histograms are
reported in forward_stats.json. The fwdn arms keep those E/C rows and replace the shape neutrals by
forward-style neutrals (llm_common.shard, strategy "forward_neutral") matched on counterfactual depth and
premise count, removing the generator-style cue between E/C and neutral rows.

Distractor axis (`distract`): `build` matches buckets on premise count, so shallow rows carry more
irrelevant premises than deep ones, and depth and distractor load are confounded across mixtures. The
distract arms rebuild the 1:1:1 and 2:0:0 mixtures at two levels of irrelevant premises (DISTRACT:
none, and as in build), each bucket matched on label only, so premise count follows the level. If
2:0:0's transfer advantage is depth, it survives the lean level. (More padding barely adds premises:
most extra irrelevant derivations are rejected.)
"""
import argparse
import collections
import itertools
import json
import os
import random
import shutil

from experiments.llm_common import LABELS, Keep, commit, generate, like, match, reformat, report

BUCKETS = {"shallow": (1, 3), "mid": (4, 7), "deep": (8, 99)}
DISTRACT = {"lean": (0, 0), "dflt": (2, 6)}  # (distractors, pad) per shape proposal; dflt is build's


def grid(step=2):
    """Weights over BUCKETS in multiples of 1/step: (2,0,0), (1,1,0), ... plus the uniform mixture."""
    out = [w for w in itertools.product(range(step + 1), repeat=3) if sum(w) == step]
    return out + [(1, 1, 1)]


def name(w):
    return "mix" + "".join(str(x) for x in w)


def build(args):
    os.makedirs(args.out, exist_ok=True)
    pk, over = args.pack, 4 * args.n
    raw = {b: generate(pk, "shape", 10 + i, over, Keep(lo=lo, hi=hi), pad=args.pad, proof=args.proof)
           for i, (b, (lo, hi)) in enumerate(BUCKETS.items())}
    ec = match(raw, args.n, random.Random(0))
    pool = generate(pk, "shape", 14, 12 * args.n, Keep(labels=("neutral",)), pad=args.pad)
    rng = random.Random(1)
    sets = {b: rows + like(rows, pool, args.n // 2, rng) for b, rows in ec.items()}
    total = len(next(iter(sets.values())))
    for w in grid():
        rows = []
        for b, k in zip(BUCKETS, w):
            rows += rng.sample(sets[b], round(total * k / sum(w)))
        with open(f"{args.out}/train_{name(w)}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    vals = {f"val_{pk}": generate(pk, "shape", 15, args.test, Keep(labels=LABELS), pad=args.pad),
            "val_propositional": generate("propositional", "shape", 16, args.test // 2, Keep(labels=LABELS))}
    fld = [json.loads(line) for line in open(args.fld_val)]
    vals["val_fld"] = [{"gold": r["gold"], "mandatory": r["mandatory"], "text": reformat(r["prompt"])} for r in fld]
    for t, rows in vals.items():
        with open(f"{args.out}/{t}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    for t in ("shape_heldout", "fld", "propositional"):  # untouched tests, shared with llm_depth_training
        shutil.copy(f"{args.tests_from}/test_{t}.jsonl", args.out)
    stats = {b: {"rows": len(r), "labels": collections.Counter(x["gold"] for x in r),
                 "mean_premises": round(sum(x["premises"] for x in r) / len(r), 2),
                 "mean_mandatory": round(sum(x["mandatory"] for x in r) / len(r), 2)} for b, r in sets.items()}
    json.dump({"commit": commit(), "buckets": stats, "mixtures": [name(w) for w in grid()], "vals": {t: len(r) for t, r in vals.items()}},
              open(f"{args.out}/stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


def bucket_of(m):
    return next(b for b, (lo, hi) in BUCKETS.items() if lo <= m <= hi)


def forward(args):
    import time
    t = time.time()
    pool = generate(args.pack, "forward", 20, args.pool, Keep())
    cost = {"pool_rows": len(pool), "seconds": round(time.time() - t, 1)}
    by = collections.defaultdict(list)
    for r in pool:  # rows with no mandatory step (the hypothesis is given, or several disjoint proofs) fit no bucket
        if r["mandatory"] >= 1:
            by[(bucket_of(r["mandatory"]), r["gold"])].append(r)
    t = time.time()
    neutrals = generate(args.pack, "forward_neutral", 21, args.neutral_pool, Keep(labels=("neutral",)))
    cost["neutral_pool_rows"], cost["neutral_seconds"] = len(neutrals), round(time.time() - t, 1)
    stats = {"cost": cost, "pool": {f"{b}|{g}": len(v) for (b, g), v in sorted(by.items())}}
    for w in grid():
        rows = [json.loads(line) for line in open(f"{args.data}/train_{name(w)}.jsonl")]
        rng = random.Random(0)
        avail = {k: rng.sample(v, len(v)) for k, v in sorted(by.items())}
        out, gap = [], []
        for r in rows:
            if r["gold"] == "neutral":
                out.append(r)
                continue
            cands = avail[(bucket_of(r["mandatory"]), r["gold"])]
            i = min(range(len(cands)), key=lambda i: abs(cands[i]["premises"] - r["premises"]))
            gap.append(cands[i]["premises"] - r["premises"])
            out.append(cands.pop(i))
        with open(f"{args.data}/train_fwd{name(w)[3:]}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in out)
        ec = [r for r in out if r["gold"] != "neutral"]
        # fwdn: the same forward E/C rows with forward-style neutrals, so the generator's style is no cue
        own = ec + like(ec, neutrals, len(out) - len(ec), rng)
        with open(f"{args.data}/train_fwdn{name(w)[3:]}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in own)
        stats["fwd" + name(w)[3:]] = {"mean_premise_gap": round(sum(gap) / len(gap), 2),
                                      "exact_premises": round(sum(g == 0 for g in gap) / len(gap), 3),
                                      "mean_mandatory": round(sum(r["mandatory"] for r in ec) / len(ec), 2)}
    json.dump(stats, open(f"{args.data}/forward_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


def distract(args):
    os.makedirs(args.out, exist_ok=True)
    stats = {"commit": commit()}
    for k, (level, (dis, pad)) in enumerate(DISTRACT.items()):
        rng = random.Random(k)
        sets = {}
        for i, (b, (lo, hi)) in enumerate(BUCKETS.items()):
            ec = generate(args.pack, "shape", 40 + 10 * k + i, args.n, Keep(lo=lo, hi=hi), distractors=dis, pad=pad)
            pool = generate(args.pack, "shape", 44 + 10 * k + i, 3 * args.n, Keep(lo=lo, hi=hi, labels=("neutral",)),
                            distractors=dis, pad=pad)
            sets[b] = ec + like(ec, pool, args.n // 2, rng)
        for w in ((1, 1, 1), (2, 0, 0)):
            rows = []
            for b, x in zip(BUCKETS, w):
                rows += rng.sample(sets[b], round(len(sets[b]) * x / sum(w)))
            with open(f"{args.out}/train_{level}{name(w)}.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            stats[level + name(w)] = {"rows": len(rows), "labels": collections.Counter(r["gold"] for r in rows),
                                      "mean_premises": round(sum(r["premises"] for r in rows) / len(rows), 2),
                                      "mean_mandatory": round(sum(r["mandatory"] for r in rows) / len(rows), 2)}
    for f in os.listdir(args.tests_from):  # the same validation and test sets as the mixture runs
        if f.startswith(("val_", "test_")):
            shutil.copy(f"{args.tests_from}/{f}", args.out)
    json.dump(stats, open(f"{args.out}/stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


def select(args):
    """Macro accuracy on each validation set, mean over seeds; the selection criterion is their mean."""
    report(args.runs)
    rep = json.load(open(f"{args.runs}/report.json"))
    vals = sorted(t for t in rep if t.startswith("val_"))
    score = {arm: sum(rep[t][arm]["all"]["macro"][0] for t in vals) / len(vals)
             for arm in rep[vals[0]] if all(arm in rep[t] for t in vals)}
    for arm, s in sorted(score.items(), key=lambda x: -x[1]):
        print(f"{arm:8s} mean {s:.3f}  " + "  ".join(f"{t[4:]} {rep[t][arm]['all']['macro'][0]:.3f}" for t in vals))
    json.dump(score, open(f"{args.runs}/selection.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pack", default="monadic")
    b.add_argument("--n", type=int, default=2000, help="E/C rows per bucket (plus n/2 neutrals)")
    b.add_argument("--test", type=int, default=900)
    b.add_argument("--pad", type=int, default=6)
    b.add_argument("--fld-val", default="results/llm_depth/rows_val_s1_n120.jsonl",
                   help="FLD x2 validation rows (experiments/llm_realized_depth.py, split=validation)")
    b.add_argument("--tests-from", default="results/llm_train/data")
    b.add_argument("--out", default="results/llm_mix/data")
    b.add_argument("--proof", action="store_true", help="E/C rows carry a certified proof (train with arm+proof)")
    f = sub.add_parser("forward")
    f.add_argument("--pack", default="monadic")
    f.add_argument("--pool", type=int, default=200000)
    f.add_argument("--neutral-pool", type=int, default=60000)
    f.add_argument("--data", default="results/llm_mix/data")
    d = sub.add_parser("distract")
    d.add_argument("--pack", default="monadic")
    d.add_argument("--n", type=int, default=2000, help="E/C rows per bucket (plus n/2 neutrals)")
    d.add_argument("--tests-from", default="results/llm_mix/data")
    d.add_argument("--out", default="results/llm_mix/data_distract")
    s = sub.add_parser("select")
    s.add_argument("--runs", default="results/llm_mix/runs")
    args = ap.parse_args()
    {"build": build, "forward": forward, "distract": distract, "select": select}[args.cmd](args)


if __name__ == "__main__":
    main()
