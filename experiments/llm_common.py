"""Shared pieces of the LLM training experiments: the row format, certified row generation (optionally
with exported proofs), label/premise matching, and the evaluation report. Experiments import these
public names; anything private stays in its own experiment script."""
import collections
import glob
from dataclasses import replace
import json
import os
import random

LABELS = ("entailment", "contradiction", "neutral")
FORMAT = "Premises:\n{premises}\n\nHypothesis: {hypothesis}\n\nLabel (entailment, contradiction or neutral):"


def bucket(m):
    return "1-3" if m <= 3 else "4-7" if m <= 7 else "8-12" if m <= 12 else "13+"


def commit() -> str:
    """The code version a build used (rows change with construction code, not with load)."""
    import subprocess
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "semforge", "experiments"], capture_output=True, text=True).stdout
    return out + ("+dirty" if dirty.strip() else "")


def generate(pack, strategy, seed, want, keep, distractors=2, jobs=32, pad=0, proof=False, profile=False):
    """Rendered rows until `want` pass `keep`, from `jobs` generators with seeds derived from `seed`.
    With `profile`, rows carry their proof profile (min_size, proofs, supports, exact; for a forward
    neutral, the profile of the E/C row it was made from), available to `keep`."""
    from concurrent.futures import ProcessPoolExecutor
    shards = [(pack, strategy, 1000 * seed + j, -(-want // jobs), keep, distractors, pad, proof, profile)
              for j in range(jobs)]
    with ProcessPoolExecutor(jobs) as ex:
        return [r for part in ex.map(shard, *zip(*shards)) for r in part][:want]


def shard(pack, strategy, seed, want, keep, distractors, pad=0, proof=False, profile=False):
    """strategy "forward_neutral": neutral rows in the forward generator's style (it makes none): a
    forward E/C row with one mandatory premise dropped, kept when the oracle says neutral; like shape
    neutrals, it carries the depth of its counterfactual proof."""
    from semforge.construct.archive import profile_cell
    from semforge.domains.constructed import ConstructedDomain
    neutralize = strategy == "forward_neutral"
    strategy = "forward" if neutralize else strategy
    d, rng, out = ConstructedDomain(pack, strategy=strategy), random.Random(seed), []
    if strategy == "shape":
        def propose(rng, _propose=d.archive.propose):  # extra irrelevant derivations, never near misses
            t = _propose(rng, distractors=distractors)
            return replace(t, distractors=replace(t.distractors, irrelevant=t.distractors.irrelevant + rng.randint(0, pad)))
        d.archive.propose = propose
    key = profile_cell("mandatory", calculus=d.pack.calculus)
    tries = 0
    while len(out) < want and tries < 50 * want:
        tries += 1
        for inst, label in d.sample(rng, rng.randint(0, 5)):
            cell = key(inst)
            prof = inst.analysis.profile if profile and cell is not None else None
            if neutralize:
                if label == "neutral" or cell is None or not inst.analysis.mandatory_premises:
                    continue
                inst = d.without(inst, {rng.choice(sorted(inst.analysis.mandatory_premises, key=repr))})
                if d.label(inst) != "neutral":
                    continue
                label = "neutral"
            row = {"gold": label, "mandatory": cell[1] if cell else None, "premises": len(inst.premises),
                   "strategy": "forward_neutral" if neutralize else strategy}
            if strategy == "forward":
                row["planned"] = d.planned
            if prof:
                row.update({k: prof[k] for k in ("min_size", "proofs", "supports", "exact")})
            if keep(row):
                p, h = d.render(inst)
                out.append({**row, "text": FORMAT.format(premises=p, hypothesis=h)})
                if proof and label != "neutral":
                    out[-1]["proof"] = proof_text(d, inst)
    return out


def proof_text(d, inst):
    """One certified proof of the row's conclusion (H, or not-H for a contradiction), one numbered step per
    line, worded with the row's own lexicon: the earliest-parent witness proof of the analysis."""
    from semforge.construct.calculus import analyze
    a = analyze(d.pack.calculus, inst.premises, inst.conclusion, inst.universe)
    edges, _ = a._witness()
    realizer = d.pack.render.__self__
    rng = random.Random(repr((inst.premises, inst.hypothesis)))  # as in render: the same lexicon
    lex = realizer.lexicon([*inst.premises, inst.hypothesis], rng)
    steps = sorted(edges, key=lambda k: (a.closure.round[a.edges[k][1]], k))
    return "\n".join(f"{i + 1}. {realizer.statement(a.edges[k][1], 'premise', lex, rng)[:-1]} "
                     f"({a.edges[k][0].name.replace('_', ' ')})." for i, k in enumerate(steps))


class Keep:
    """Rows with a certified mandatory count in [lo, hi] and a label in `labels`."""
    def __init__(self, lo=0, hi=10 ** 9, labels=("entailment", "contradiction")):
        self.lo, self.hi, self.labels = lo, hi, labels

    def __call__(self, r):
        return r["gold"] in self.labels and r["mandatory"] is not None and self.lo <= r["mandatory"] <= self.hi


def match(arms, n, rng):
    """n rows per arm with equal label counts and equal premise-count histograms (2-premise bins)."""
    cell = lambda r: (r["gold"], r["premises"] // 2)
    pools = {a: collections.defaultdict(list) for a in arms}
    for a, rows in arms.items():
        for r in rows:
            pools[a][cell(r)].append(r)
    common = {c: min(len(pools[a][c]) for a in arms) for c in set.intersection(*(set(p) for p in pools.values()))}
    total = sum(common.values())
    if total < n:
        raise ValueError(f"only {total} matched rows per arm; generate more or widen distractors")
    picked = {a: [] for a in arms}
    for c, k in sorted(common.items()):
        take = round(k * n / total)
        for a in arms:
            picked[a] += rng.sample(pools[a][c], min(take, k))
    m = min(len(v) for v in picked.values())
    return {a: rng.sample(v, m) for a, v in picked.items()}



def like(rows, pool, k, rng):
    """k rows of `pool` with the (depth, premise-count) histogram of `rows` (2-wide bins), as near as the
    pool allows: a cell the pool lacks takes from the nearest premise bin at the same depth."""
    cell = lambda r: (r["mandatory"] // 2, r["premises"] // 2)
    by = collections.defaultdict(list)
    for r in pool:
        by[cell(r)].append(r)
    for v in by.values():
        rng.shuffle(v)
    out = []
    for r in rng.sample(rows, k):
        d, p = cell(r)
        near = sorted((c for c in by if c[0] == d and by[c]), key=lambda c: abs(c[1] - p))
        if near:
            out.append(by[near[0]].pop())
    return out


def reformat(prompt):
    """An API prompt (llm_realized_depth.PROMPT) in this experiment's format."""
    premises = prompt.split("Premises:\n", 1)[1].split("\n\nHypothesis: ", 1)
    return FORMAT.format(premises=premises[0], hypothesis=premises[1].split("\n\n", 1)[0])


def report(runs):
    """Per arm (mean and range over seeds), per test and depth bucket: macro accuracy over the three
    labels (an arm cannot gain by never answering neutral), polarity (E vs C, among E/C rows answered
    E or C) and how often E/C rows are called neutral."""
    import glob
    preds = collections.defaultdict(dict)
    for f in glob.glob(f"{runs}/*_s*_*.preds.jsonl"):
        name = os.path.basename(f)[:-len(".preds.jsonl")]
        arm, rest = name.split("_s", 1)
        seed, test = rest.split("_", 1)
        preds[(test, arm)][seed] = [json.loads(line) for line in open(f)]

    def metrics(rows):
        acc = [sum(r["pred"] == r["gold"] for r in rows if r["gold"] == lab) / n
               for lab in LABELS if (n := sum(r["gold"] == lab for r in rows))]
        ec = [r for r in rows if r["gold"] != "neutral"]
        answered = [r for r in ec if r["pred"] != "neutral"]
        return {"macro": sum(acc) / len(acc) if acc else None,
                "polarity": sum(r["pred"] == r["gold"] for r in answered) / len(answered) if answered else None,
                "ec_as_n": sum(r["pred"] == "neutral" for r in ec) / len(ec) if ec else None}

    out = {}
    for (test, arm), by_seed in sorted(preds.items()):
        for b in ("all", "1-3", "4-7", "8-12", "13+"):
            ms = [metrics([r for r in rows if b == "all" or (r["mandatory"] is not None and bucket(r["mandatory"]) == b)])
                  for rows in by_seed.values()]
            out.setdefault(test, {}).setdefault(arm, {})[b] = {
                k: (round(sum(v) / len(v), 3), round(min(v), 3), round(max(v), 3)) if (v := [m[k] for m in ms if m[k] is not None]) else None
                for k in ("macro", "polarity", "ec_as_n")}
    json.dump(out, open(f"{runs}/report.json", "w"), indent=1)
    for test, arms in out.items():
        print("==", test)
        for arm, bs in arms.items():
            print(f"  {arm:8s}", "  ".join(f"{b}: m{v['macro'][0] if v['macro'] else '-'} p{v['polarity'][0] if v['polarity'] else '-'} n{v['ec_as_n'][0] if v['ec_as_n'] else '-'}"
                                        for b, v in bs.items()))


def seed_contrast(rng):
    """Printer for a planned contrast a - b over per-seed scores, with a seed bootstrap (each arm's seeds
    resampled independently: a seed names a different training run in each arm)."""
    import numpy as np

    def contrast(label, A, B):
        A, B = np.asarray(A, float), np.asarray(B, float)
        if len(A) < 2 or len(B) < 2 or np.isnan(A).any() or np.isnan(B).any():
            return
        draws = [rng.choice(A, len(A)).mean() - rng.choice(B, len(B)).mean() for _ in range(2000)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        flag = " *" if lo > 0 or hi < 0 else ""
        print(f"{label:52s}{len(A):3d}/{len(B):<3d}{A.mean() - B.mean():+8.3f}   [{lo:+.3f}, {hi:+.3f}]{flag}")
    return contrast


def load_runs(runs_dir):
    """(arm, test) -> seed -> prediction rows, from `<arm>_s<seed>_<test>.preds.jsonl` files."""
    import re
    runs = collections.defaultdict(dict)
    for f in glob.glob(f"{runs_dir}/*.preds.jsonl"):
        m = re.fullmatch(r"(.+)_s(\d+)_(.+)\.preds\.jsonl", os.path.basename(f))
        runs[m.group(1), m.group(3)][int(m.group(2))] = [json.loads(line) for line in open(f)]
    return runs
