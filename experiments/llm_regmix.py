"""Distribution search done properly: many cheap proxy runs on random mixtures of certified cells, a
regression of dev score on mixture weights, and the predicted optimum confirmed on untouched tests.

Earlier searches compared a handful of hand-picked mixtures with two seeds, so differences between
reasonable mixtures (2-4 points) were as large as seed noise. Here, following RegMix (Liu et al. 2025),
each proxy is one seed on a random mixture, and the regression pools them: seed noise averages out over
mixtures, and the fit says whether dev score is predictable from the distribution at all (leave-one-out
correlation), before anything is selected.

Cells are certified descriptors, each label-balanced within itself:
  m_s m_m m_d   monadic rows by certified depth (1-3, 4-7, 8+ mandatory steps)
  d_s d_m d_d   labelled modal rows, same buckets
  k_lo k_hi     closed Kripke model checking, modal depth 0-1 / 2-3 (E/C only)
  o             partially observed Kripke (E/C/N)
Dev objective, fixed before any proxy runs: mean macro accuracy over val_fld, val_folio, val_modalbench,
val_proverqa and val_proofwriter (the last two disjoint from test_proverqa / test_proofwriter).

  python -m experiments.llm_regmix build --out results/llm_mix/data_regmix --n 60
  OUT=results/llm_mix/regmix bash scripts/queue_packmix.sh none rm00:0 ...   (with DATA=results/llm_mix/data_regmix)
  python -m experiments.llm_regmix fit --runs results/llm_mix/regmix --confirm results/llm_mix/data_packmix

Certification ablation: the same mixtures (same seed, so the same weights) over FLD-style forward rows,
whose monadic and modal cells are bucketed either by certified mandatory depth or by the planned depth
the generator intended (uncertified); Kripke cells are unchanged. One forward pool serves both, so the
two searches differ only in the descriptor that defines the cells.
  python -m experiments.llm_regmix fwdpool --out results/llm_mix/data_fwdpool
  python -m experiments.llm_regmix build --cells planned --out results/llm_mix/data_regmix_planned --n 40
  python -m experiments.llm_regmix build --cells certified --out results/llm_mix/data_regmix_fwdcert --n 40

Pre-certification baseline for tab:search: the searched and uniform weights over forward rows binned by
planned depth (train_rmcalplan, train_rmuniplan). Cells the forward pool cannot fill (no modal row is planned
8+ steps deep) are dropped and the other weights renormalized.
  python -m experiments.llm_regmix precert --out results/llm_mix/data_packmix
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
import shutil

import numpy as np

from experiments.llm_common import FORMAT, LABELS

CELLS = ("m_s", "m_m", "m_d", "d_s", "d_m", "d_d", "k_lo", "k_hi", "o")
DEV = ("val_fld", "val_folio", "val_modalbench", "val_proverqa", "val_proofwriter")
ROWS = 3000


def _bucket(m):
    return "s" if m <= 3 else "m" if m <= 7 else "d"


def pool():
    """Every certified row once (deduplicated by text), by cell."""
    cells = collections.defaultdict(dict)
    for prefix, pat in (("m", "results/llm_mix/data/train_mix*.jsonl"), ("d", "results/llm_mix/data_modal/train_mix*.jsonl")):
        for f in sorted(glob.glob(pat)):
            for line in open(f):
                r = json.loads(line)
                cells[f"{prefix}_{_bucket(r['mandatory'])}"][r["text"]] = r
    for f in ("train_kshallow", "train_kdeep", "train_kmix"):
        for line in open(f"results/llm_mix/data_kripke/{f}.jsonl"):
            r = json.loads(line)
            cells["k_lo" if r["modal_depth"] <= 1 else "k_hi"][r["text"]] = r
    for line in open("results/llm_mix/data_kripke/train_kopen.jsonl"):
        r = json.loads(line)
        cells["o"][r["text"]] = r
    return {c: sorted(cells[c].values(), key=lambda r: r["text"]) for c in CELLS}


def forward_pool(descriptor, path="results/llm_mix/data_fwdpool"):
    """Monadic and modal cells from forward rows by `descriptor` (planned or certified depth), each cell
    label-balanced; Kripke cells as in pool()."""
    cells, rng = collections.defaultdict(list), random.Random(0)
    for prefix, pack in (("m", "monadic"), ("d", "modal")):
        for line in open(f"{path}/pool_{pack}.jsonl"):
            r = json.loads(line)
            depth = r["planned"] if descriptor == "planned" else (r["mandatory"] or 0)
            cells[f"{prefix}_{_bucket(depth)}"].append(r)
    out = {}
    for c, rows in cells.items():
        by = {lab: sorted((r for r in rows if r["gold"] == lab), key=lambda r: r["text"]) for lab in LABELS}
        k = min(map(len, by.values()))
        out[c] = sorted((r for lab in LABELS for r in rng.sample(by[lab], k)), key=lambda r: r["text"])
    kripke = pool()
    return {c: out.get(c, []) if c[0] in "md" else kripke[c] for c in CELLS}


def fwdpool(args):
    """One forward pool per pack: E/C rows and forward-style neutrals, with planned and certified depth."""
    from experiments.fldlike_rows import AnyRow
    from experiments.llm_common import generate
    os.makedirs(args.out, exist_ok=True)
    for i, pack in enumerate(("monadic", "modal")):
        rows = generate(pack, "forward", 40 + 10 * i, args.ec, AnyRow(("entailment", "contradiction")), jobs=48)
        rows += generate(pack, "forward_neutral", 41 + 10 * i, args.ec // 2, AnyRow(("neutral",)), jobs=48)
        write(f"{args.out}/pool_{pack}.jsonl", [{**r, "pack": pack} for r in rows])


def mixture_rows(cells, w, rng, n=ROWS):
    rows = []
    for c, x in zip(CELLS, w):
        k = min(round(n * x), len(cells[c]))
        rows += [{**r, "cell": c} for r in rng.sample(cells[c], k)]
    rng.shuffle(rows)
    return rows


def random_weights(rng, n):
    """Dirichlet weights with varied sparsity (RegMix style): some mixtures use few cells."""
    out = []
    for i in range(n):
        k = rng.choice((2, 3, 4, 6, len(CELLS)))
        on = rng.sample(range(len(CELLS)), k)
        g = [rng.gammavariate(1.0, 1.0) if j in on else 0.0 for j in range(len(CELLS))]
        out.append([x / sum(g) for x in g])
    return out


def _lines(context):
    return "\n".join(s.strip() for s in re.split(r"(?<=\.)\s+", context.strip()) if s.strip())


def external_dev(out, rng):
    """val_proverqa (ProverQA dev rows not in test_proverqa) and val_proofwriter (ProofWriter dev), 300 each."""
    from huggingface_hub import hf_hub_download
    from experiments.external_rows import PROOFWRITER, balanced, proofwriter_rows, proverqa_row
    test = {json.loads(line)["text"] for line in open("results/llm_mix/data_packmix/test_proverqa.jsonl")}
    rows = []
    for level in ("easy", "medium", "hard"):
        for x in json.load(open(hf_hub_download("opendatalab/ProverQA", f"dev/{level}.json", repo_type="dataset"))):
            r = proverqa_row(x["context"], x["question"].split("uncertain?", 1)[1].strip(), x["answer"],
                             source="proverqa", level=level)
            if r["text"] not in test:
                rows.append(r)
    write(f"{out}/val_proverqa.jsonl", rng.sample(rows, 300))
    write(f"{out}/val_proofwriter.jsonl", balanced(proofwriter_rows(f"{PROOFWRITER}/meta-dev.jsonl", 1, rng), 300, rng))


def write(path, rows):
    with open(path + ".tmp", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    os.replace(path + ".tmp", path)


def build(args):
    os.makedirs(args.out, exist_ok=True)
    cells, rng = (pool() if args.cells == "constructed" else forward_pool(args.cells)), random.Random(args.seed)
    print({c: len(rs) for c, rs in cells.items()})
    weights = random_weights(rng, args.n)
    for i, w in enumerate(weights):
        write(f"{args.out}/train_rm{i:02d}.jsonl", mixture_rows(cells, w, random.Random(f"{args.seed}:{i}")))
    for f in ("val_fld", "val_folio", "val_modalbench"):
        shutil.copy(f"results/llm_mix/data_packmix/{f}.jsonl", args.out)
    external_dev(args.out, random.Random(f"{args.seed}:dev"))
    json.dump({"cells": CELLS, "dev": DEV, "weights": {f"rm{i:02d}": w for i, w in enumerate(weights)}},
              open(f"{args.out}/mixtures.json", "w"), indent=1)


def macro(result):
    accs = [a for a in result["by_label"].values() if a is not None]
    return sum(accs) / len(accs)


def prior_free(path):
    """Macro accuracy that a shift of the label prior cannot buy, from a run's saved label log-probabilities:
    on a dev set with no neutral gold, the likelier of E and C; with three labels, the best per-label bias
    (two numbers fitted on the dev set itself). Plain macro accuracy on the binary dev sets counts every
    neutral answer as wrong, so it also rewards mixtures that merely make the model answer neutral less."""
    rows = [json.loads(line) for line in open(path)]
    lp, gold = np.array([r["logp"] for r in rows]), np.array([LABELS.index(r["gold"]) for r in rows])
    present = sorted(set(gold))
    score = lambda pred: np.mean([np.mean(pred[gold == k] == k) for k in present])
    if LABELS.index("neutral") not in present:
        return score(np.argmax(lp[:, :2], 1))
    grid = np.linspace(-4, 4, 17)
    return max(score(np.argmax(lp + np.array([0, b, c]), 1)) for b in grid for c in grid)


def features(W):
    W = np.asarray(W)
    return np.hstack([W, W ** 2])  # per-cell diminishing returns


def fit(args):
    meta = json.load(open(f"{args.data}/mixtures.json"))
    X, y, names = [], [], []
    for f in sorted(glob.glob(f"{args.runs}/rm*_s*.json")):
        r = json.load(open(f))
        if all(d in r for d in DEV):
            score = (lambda d: macro(r[d])) if args.metric == "macro" else (lambda d: prior_free(f"{f[:-5]}_{d}.preds.jsonl"))
            X.append(meta["weights"][r["arm"]]), y.append(np.mean([score(d) for d in DEV])), names.append(r["arm"])
    X, y = np.asarray(X), np.asarray(y)
    print(f"{len(y)} proxies; dev score mean {y.mean():.3f}, min {y.min():.3f}, max {y.max():.3f} ({names[y.argmax()]})")
    lam = args.ridge
    solve = lambda A, b: np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
    F = np.hstack([features(X), np.ones((len(y), 1))])
    loo = []
    for i in range(len(y)):  # leave-one-out: is dev score predictable from the mixture?
        keep = np.arange(len(y)) != i
        loo.append(F[i] @ solve(F[keep], y[keep]))
    rho = np.corrcoef(loo, y)[0, 1]
    print(f"leave-one-out correlation of predicted and observed dev score: {rho:.2f}")
    beta = solve(F, y)
    for c, b in zip(CELLS, beta[:len(CELLS)]):
        print(f"  {c:5s} linear {b:+.3f}")
    rng = random.Random(0)
    cand = np.asarray(random_weights(rng, 100_000))
    pred = np.hstack([features(cand), np.ones((len(cand), 1))]) @ beta
    top = cand[np.argsort(pred)[-100:]].mean(0)  # average of the top predictions (RegMix), less noisy than argmax
    top = top / top.sum()
    print("selected:", {c: round(float(x), 3) for c, x in zip(CELLS, top)},
          f"predicted {np.hstack([features([top])[0], 1]) @ beta:.3f}")
    uniform = np.ones(len(CELLS)) / len(CELLS)
    print(f"uniform predicted {np.hstack([features([uniform])[0], 1]) @ beta:.3f}")
    if args.objective == "maximin":
        top = robust_choice(args, meta, solve, cand)
        print("maximin choice:", {c: round(float(x), 3) for c, x in zip(CELLS, top)})
    if args.metric == "prior_free":  # its own files: fit.json and train_rmsel are the macro-accuracy search
        json.dump({"proxies": len(y), "loo_rho": rho, "beta": beta.tolist(), "selected": top.tolist(),
                   "across_tasks": across_tasks(args, meta, solve)},
                  open(f"{args.runs}/fit_prior_free.json", "w"), indent=1)
        if args.confirm:
            write(f"{args.confirm}/train_rmcal.jsonl", mixture_rows(pool(), top, random.Random("rmcal")))
            print(f"wrote train_rmcal to {args.confirm}")
        return
    across = across_tasks(args, meta, solve)
    json.dump({"proxies": len(y), "loo_rho": rho, "beta": beta.tolist(), "selected": top.tolist(),
               "uniform": uniform.tolist(), "observed_best": names[int(y.argmax())], "across_tasks": across},
              open(f"{args.runs}/fit.json", "w"), indent=1)
    if args.confirm:
        cells = pool()
        if args.objective == "maximin":
            write(f"{args.confirm}/train_rmrob.jsonl", mixture_rows(cells, top, random.Random("rmrob")))
            print(f"wrote train_rmrob to {args.confirm}")
        else:
            write(f"{args.confirm}/train_rmsel.jsonl", mixture_rows(cells, top, random.Random("rmsel")))
            write(f"{args.confirm}/train_rmuni.jsonl", mixture_rows(cells, uniform, random.Random("rmuni")))
            print(f"wrote train_rmsel and train_rmuni to {args.confirm}")


def robust_choice(args, meta, solve, cand):
    """The mixture whose worst task is best: one regression per dev task, predictions standardized per task
    over the candidates, the minimum over tasks maximized (average of the top 100, as for the mean)."""
    X, S = [], []
    for f in sorted(glob.glob(f"{args.runs}/rm*_s*.json")):
        r = json.load(open(f))
        if all(d in r for d in DEV):
            X.append(meta["weights"][r["arm"]]), S.append([macro(r[d]) for d in DEV])
    F = np.hstack([features(np.asarray(X)), np.ones((len(X), 1))])
    Fc = np.hstack([features(cand), np.ones((len(cand), 1))])
    P = np.stack([Fc @ solve(F, np.asarray(S)[:, j]) for j in range(len(DEV))], 1)
    z = (P - P.mean(0)) / P.std(0)
    top = cand[np.argsort(z.min(1))[-100:]].mean(0)
    return top / top.sum()


def across_tasks(args, meta, solve):
    """Patterns that hold across tasks: per-task linear effect of each cell (share of the mixture, fitted on
    the proxies), the cells whose sign agrees on every task, the correlation of tasks over proxies, and a
    leave-one-task-out check (choose the proxy best on the other tasks: its percentile on the held-out one)."""
    X, S = [], []
    for f in sorted(glob.glob(f"{args.runs}/rm*_s*.json")):
        r = json.load(open(f))
        if all(d in r for d in DEV):
            score = (lambda d: macro(r[d])) if args.metric == "macro" else (lambda d: prior_free(f"{f[:-5]}_{d}.preds.jsonl"))
            X.append(meta["weights"][r["arm"]]), S.append([score(d) for d in DEV])
    X, S = np.asarray(X), np.asarray(S)
    F = np.hstack([X, np.ones((len(X), 1))])
    effects = {d: solve(F, S[:, j])[:len(CELLS)] for j, d in enumerate(DEV)}
    print("per-task linear effect of each cell's share (centered per task):")
    print("       " + " ".join(f"{d[4:]:>10s}" for d in DEV) + "   agree")
    agree = {}
    for i, c in enumerate(CELLS):
        row = [effects[d][i] - np.mean(effects[d]) for d in DEV]
        agree[c] = all(v > 0 for v in row) or all(v < 0 for v in row)
        print(f"  {c:5s}" + " ".join(f"{v:+10.3f}" for v in row) + ("   yes" if agree[c] else ""))
    corr = np.corrcoef(S.T)
    print("task correlation over proxies:\n" + "\n".join(
        f"  {DEV[i][4:]:>10s} " + " ".join(f"{corr[i, j]:+.2f}" for j in range(len(DEV))) for i in range(len(DEV))))
    lotto = {}
    for j, d in enumerate(DEV):
        others = np.delete(S, j, axis=1).mean(1)
        pick = int(others.argmax())
        lotto[d] = float((S[:, j] < S[pick, j]).mean())
        print(f"  choose on the other tasks, held-out {d[4:]:>10s}: percentile {lotto[d]:.2f}")
    return {"effects": {d: e.tolist() for d, e in effects.items()}, "agree": agree,
            "task_corr": corr.tolist(), "leave_one_task_out_percentile": lotto}


def report(args):
    """Confirmation arms with the label prior factored out: accuracy on E/C-gold test rows when the model must
    choose between E and C (balanced over the two), how often it answers neutral, and how well its neutral
    score separates neutral rows from the rest (AUC). Means over the seeds found."""
    def auc(pos, neg):
        rank = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
        return (rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    print(f"{'arm':13s}{'test':13s}{'seeds':>6s}{'macro':>7s}{'E/C':>7s}{'N rate':>8s}{'N AUC':>7s}")
    for arm in args.arms:
        for test in args.tests:
            out = []
            for f in sorted(glob.glob(f"{args.runs}/{arm}_s*_{test}.preds.jsonl")):
                if not re.fullmatch(f"{arm}_s({args.seeds})_{test}.preds.jsonl", os.path.basename(f)):
                    continue  # also drops val_{test}, which the glob matches
                rows = [json.loads(line) for line in open(f)]
                lp, gold = np.array([r["logp"] for r in rows]), np.array([LABELS.index(r["gold"]) for r in rows])
                pred = np.array([LABELS.index(r["pred"]) for r in rows])
                margin = lp[:, 2] - lp[:, :2].max(1)
                out.append((np.mean([np.mean(pred[gold == k] == k) for k in set(gold)]),
                            np.mean([np.mean(np.argmax(lp[gold == k, :2], 1) == k) for k in (0, 1)]),
                            np.mean(pred == 2),
                            auc(margin[gold == 2], margin[gold != 2]) if (gold == 2).any() else np.nan))
            if out:
                print(f"{arm:13s}{test:13s}{len(out):6d}" + "".join(f"{x:7.3f} " for x in np.mean(out, 0)))


def compare(args):
    """Paired difference in forced E/C accuracy (as in `report`) between arm `--a` and each `--b` arm, on the
    seeds both have: test rows are resampled with the same draw for both arms and all seeds (2000 draws).
    With `--seed-bootstrap`, each draw also resamples each arm's seeds (independently: a seed number names
    a different training set in each arm), so the interval includes run-to-run variance; `wins` counts the
    seeds in which `a` beats `b`."""
    def correct(arm, test):
        out = {}
        for f in glob.glob(f"{args.runs}/{arm}_s*_{test}.preds.jsonl"):
            m = re.fullmatch(f"{arm}_s({args.seeds})_{test}.preds.jsonl", os.path.basename(f))
            if m:
                rows = [json.loads(line) for line in open(f)]
                lp, gold = np.array([r["logp"] for r in rows]), np.array([LABELS.index(r["gold"]) for r in rows])
                out[m.group(1)] = (np.argmax(lp[:, :2], 1) == gold)[gold < 2], gold[gold < 2]
        return out
    rng = np.random.default_rng(0)
    print(f"{'a - b':24s}{'test':13s}{'seeds':>6s}{'diff':>8s}{'95% interval':>18s}")
    for b in args.b:
        for test in args.tests:
            A, B = correct(args.a, test), correct(b, test)
            seeds = sorted(set(A) & set(B))
            if not seeds:
                continue
            gold = A[seeds[0]][1]
            d = np.mean([A[s][0].astype(float) - B[s][0] for s in seeds], 0)  # per row, mean over seeds
            bal = lambda idx: np.mean([d[idx][gold[idx] == k].mean() for k in (0, 1)])
            if args.seed_bootstrap:
                ca, cb = np.array([A[s][0] for s in seeds], float), np.array([B[s][0] for s in seeds], float)
                acc = lambda c, idx: np.mean([c[:, idx][:, gold[idx] == k].mean(1) for k in (0, 1)], 0)  # per seed
                def draw():
                    idx = rng.integers(0, len(d), len(d))
                    sa, sb = rng.integers(0, len(seeds), len(seeds)), rng.integers(0, len(seeds), len(seeds))
                    return acc(ca, idx)[sa].mean() - acc(cb, idx)[sb].mean()
                draws = [draw() for _ in range(2000)]
                every = np.arange(len(d))
                wins = f"  wins {int((acc(ca, every) > acc(cb, every)).sum())}/{len(seeds)}"
            else:
                draws, wins = [bal(rng.integers(0, len(d), len(d))) for _ in range(2000)], ""
            lo, hi = np.percentile(draws, [2.5, 97.5])
            print(f"{args.a + ' - ' + b:24s}{test:13s}{len(seeds):6d}{bal(np.arange(len(d))):+8.3f}   [{lo:+.3f}, {hi:+.3f}]{wins}")


def scale(args):
    """The prior-free selection, uniform weights and the FLD-style baseline at `--rows` rows, for a size curve:
    train_rmcal{k}k and train_rmuni{k}k from the certified pool, train_fldlike{k}k from the forward pool as
    generated (half per pack, a third per label)."""
    top = json.load(open(f"{args.runs}/fit_prior_free.json"))["selected"]
    cells, tag = pool(), f"{args.rows // 1000}k"
    write(f"{args.out}/train_rmcal{tag}.jsonl", mixture_rows(cells, top, random.Random("rmcal"), args.rows))
    write(f"{args.out}/train_rmuni{tag}.jsonl", mixture_rows(cells, [1 / len(CELLS)] * len(CELLS), random.Random("rmuni"), args.rows))
    rng, rows = random.Random("fldlike"), []
    for pack in ("monadic", "modal"):
        pack_rows = [json.loads(line) for line in open(f"{args.fwdpool}/pool_{pack}.jsonl")]
        for lab in LABELS:
            rows += [{**r, "cell": f"fwd_{pack}"} for r in rng.sample([r for r in pack_rows if r["gold"] == lab], args.rows // 6)]
    rng.shuffle(rows)
    write(f"{args.out}/train_fldlike{tag}.jsonl", rows)
    for name in ("rmcal", "rmuni", "fldlike"):
        rs = [json.loads(line) for line in open(f"{args.out}/train_{name}{tag}.jsonl")]
        print(name + tag, len(rs), dict(collections.Counter(r["gold"] for r in rs)), dict(collections.Counter(r["cell"] for r in rs)))


def precert(args):
    """Searched and uniform weights over forward cells binned by planned depth: what the mixture would be
    without certification."""
    top = json.load(open(f"{args.runs}/fit_prior_free.json"))["selected"]
    cells = forward_pool("planned", args.fwdpool)
    for name, w in (("rmcalplan", top), ("rmuniplan", [1 / len(CELLS)] * len(CELLS))):
        w = [x if cells[c] else 0.0 for c, x in zip(CELLS, w)]
        w = [x / sum(w) for x in w]
        rows = mixture_rows(cells, w, random.Random(name))
        write(f"{args.out}/train_{name}.jsonl", rows)
        print(name, len(rows), dict(collections.Counter(r["gold"] for r in rows)), dict(collections.Counter(r["cell"] for r in rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default="results/llm_mix/data_regmix")
    b.add_argument("--n", type=int, default=60)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--cells", default="constructed", choices=("constructed", "certified", "planned"),
                   help="constructed rows (certified cells), or forward rows by certified / planned depth")
    w = sub.add_parser("fwdpool")
    w.add_argument("--out", default="results/llm_mix/data_fwdpool")
    w.add_argument("--ec", type=int, default=12000)
    r = sub.add_parser("report")
    r.add_argument("--runs", default="results/llm_mix/external")
    r.add_argument("--arms", nargs="+", default=["rmsel", "rmcal", "rmuni", "fldlike", "xfld", "xproverqa", "xproofwriter"])
    r.add_argument("--tests", nargs="+", default=["folio", "modalbench", "fld", "proverqa", "proofwriter", "tptp"])
    r.add_argument("--seeds", default="1[0-6]", help="regular expression on the seed")
    p = sub.add_parser("compare")
    p.add_argument("--runs", default="results/llm_mix/external")
    p.add_argument("--a", required=True)
    p.add_argument("--b", nargs="+", required=True)
    p.add_argument("--tests", nargs="+", default=["folio", "modalbench", "fld", "proverqa", "proofwriter"])
    p.add_argument("--seeds", default=r"\d+")
    p.add_argument("--seed-bootstrap", action="store_true", help="also resample seeds (run-to-run variance)")
    c = sub.add_parser("scale")
    c.add_argument("--runs", default="results/llm_mix/regmix")
    c.add_argument("--fwdpool", default="results/llm_mix/data_fwdpool")
    c.add_argument("--out", default="results/llm_mix/data_packmix")
    c.add_argument("--rows", type=int, default=9000)
    f = sub.add_parser("fit")
    f.add_argument("--runs", default="results/llm_mix/regmix")
    f.add_argument("--data", default="results/llm_mix/data_regmix")
    f.add_argument("--ridge", type=float, default=1e-3)
    f.add_argument("--confirm", help="write the selected and uniform mixtures here as train_rmsel / train_rmuni")
    f.add_argument("--metric", default="macro", choices=("macro", "prior_free"),
                   help="dev score: macro accuracy, or accuracy with the label prior factored out (see prior_free)")
    f.add_argument("--objective", default="mean", choices=("mean", "maximin"),
                   help="mean dev score, or the worst standardized task (writes train_rmrob)")
    p = sub.add_parser("precert")
    p.add_argument("--runs", default="results/llm_mix/regmix")
    p.add_argument("--fwdpool", default="results/llm_mix/data_fwdpool")
    p.add_argument("--out", default="results/llm_mix/data_packmix")
    args = ap.parse_args()
    {"build": build, "fit": fit, "fwdpool": fwdpool, "report": report, "compare": compare, "scale": scale,
     "precert": precert}[args.cmd](args)


if __name__ == "__main__":
    main()
